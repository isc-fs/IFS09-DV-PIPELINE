"""C++ detection backend (``ConeDetectionConfig.detector_backend = "cpp"``).

The fused kernels of :mod:`cone_detection.fused` ported 1:1 to C++
(``perception_cpp/cone_detection_core``, a pybind11 module). The scan runs in two
calls that release the GIL:

* ``Core.front``: crop, RANSAC, rotate, tall-column veto, voxel guard;
* ``Core.back``: grid clustering (unless another ``cluster_backend`` supplied the
  labels) and the per-cluster gates and fits.

Between them, Python applies the guard's seeded random fallback (``numpy``'s
``default_rng``) on the rare scan still above ``dbscan_max_points`` after the
largest voxel, and runs a non-grid ``cluster_backend`` if one is configured.

The output equals the fused backend's. RANSAC draws from a copy of Numba's RNG
inside ``Core`` (MT19937 with Numba's seeding and randint), so the per-scan seed
picks the same plane. ``test/test_golden.py`` and ``test/test_cpp_detect.py``
guard this.

When the module is not built, :func:`available` is False and the detector falls
back to "fused" with one warning.
"""

from __future__ import annotations

import time

import numpy as np

from cone_detection import fused
from cone_detection.clustering import fast_dbscan

try:  # built by colcon from perception_cpp/cone_detection_core
    import cone_detection_core as _core
except ImportError as _exc:  # pragma: no cover - depends on the build
    _core = None
    _IMPORT_ERROR = repr(_exc)
else:
    _IMPORT_ERROR = ""


def available() -> bool:
    return _core is not None


def unavailable_reason() -> str:
    return f"cone_detection_core not importable ({_IMPORT_ERROR})"


def new_core():
    """One C++ core (and RNG state) per detector."""
    return _core.Core()


def decode_xyz(data, n_points: int, point_step: int, ox: int, oy: int, oz: int) -> np.ndarray:
    """PointCloud2 bytes to ``(N, 3)`` float32, like ``pointcloud2_to_xyz``."""
    return _core.decode_xyz(data, n_points, point_step, ox, oy, oz)


def _guard_fallback(pts: np.ndarray, cap: int) -> np.ndarray:
    """``_bound_dbscan_input``'s last resort, after the voxel passes."""
    if cap <= 0 or len(pts) <= cap:
        return pts
    rng = np.random.default_rng(len(pts))
    keep = rng.choice(len(pts), cap, replace=False)
    return pts[np.sort(keep)]


def run_scan(
    core,
    data: np.ndarray,
    cfg,
    *,
    initial_plane: np.ndarray | None,
    ransac_iter_subsample_max: int,
    clustering_class: type,
    fast_clustering: bool,
    stage_timings: dict,
    keep_cropped: bool = False,
) -> fused.FusedScan:
    """Same contract as :func:`fused.run_scan`. ``scan.cropped`` holds the cropped
    cloud only when ``keep_cropped``; otherwise an empty array of the cropped
    length's truthiness (``len(scan.cropped) == 0`` still means "nothing left")."""
    st = stage_timings
    out = fused.FusedScan()
    dt = data.dtype.type
    crop2 = float(dt(cfg.input_range_crop_m**2)) if cfg.input_range_crop_m > 0 else 0.0
    min2 = float(dt(cfg.input_min_range_m**2)) if cfg.input_min_range_m > 0 else 0.0
    warm = None if initial_plane is None else np.asarray(initial_plane, dtype=np.float64)
    (n_cropped, cropped, coefs, clean, n_outliers, n_vetoed, n_before_guard, t) = core.front(
        data,
        warm,
        crop2,
        min2,
        float(cfg.ransac_prob),
        float(cfg.ransac_threshold),
        int(ransac_iter_subsample_max),
        bool(cfg.tall_column_veto),
        float(cfg.tall_column_veto_height_m),
        float(cfg.tall_column_veto_cell_m),
        int(cfg.dbscan_max_points),
        float(cfg.dbscan_guard_voxel_m),
        float(cfg.dbscan_guard_voxel_max_m),
        bool(keep_cropped),
    )
    out.cropped = cropped if cropped is not None else np.empty((n_cropped, 0))
    st["crop_ms"] = t["crop_ms"]
    if n_cropped == 0:
        return out
    out.coefs = coefs
    st["ransac_ms"] = t["ransac_ms"]
    st["rotate_ms"] = t["rotate_ms"]
    st["n_outliers"] = float(n_outliers)
    out.clean = clean
    if n_outliers == 0:
        st["dbscan_ms"] = 0.0
        st["n_dbscan_guard_dropped"] = 0.0
        return out
    if n_vetoed >= 0:
        st["column_veto_ms"] = t["column_veto_ms"]
        st["n_vetoed"] = float(n_vetoed)
        if len(clean) == 0:
            st["dbscan_ms"] = 0.0
            st["n_dbscan_guard_dropped"] = 0.0
            return out

    t_guard = time.perf_counter()
    sel = _guard_fallback(clean, int(cfg.dbscan_max_points))
    st["guard_ms"] = t["guard_ms"] + (time.perf_counter() - t_guard) * 1000.0
    out.guard_dropped = n_before_guard - len(sel)
    st["n_dbscan_guard_dropped"] = float(out.guard_dropped)
    out.clean = sel

    labels = None
    t4 = time.perf_counter()
    if not (cfg.cluster_backend == "grid" and fast_clustering and cfg.dbscan_min_samples == 2):
        if cfg.cluster_backend != "sklearn" and fast_clustering:
            labels = fast_dbscan(sel, cfg.dbscan_eps, cfg.dbscan_min_samples, cfg.cluster_backend)
        if labels is None:
            labels = clustering_class(
                eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples
            ).fit_predict(sel)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
    py_cluster_ms = (time.perf_counter() - t4) * 1000.0
    out.labels, out.rows, out.counters, dbscan_ms, fit_ms = core.back(
        sel, labels, float(cfg.dbscan_eps), fused.fit_params(cfg).tolist()
    )
    st["dbscan_ms"] = dbscan_ms + py_cluster_ms
    st["cluster_prep_ms"] = 0.0
    st["n_clusters"] = float(out.counters[fused.N_CLUSTERS])
    st["fit_ms"] = fit_ms
    return out
