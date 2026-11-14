"""LiDAR cone detection: one scan -> per-cone ``(x, y, height_m, sigma_xy)``.

:meth:`RealtimeConeDetector.detect` runs the stages in order; each lives in its
own package and has one entry point:

    preprocessing   crop_scan         raw scan -> cropped scan               (range crop)
    ground          separate_ground   cropped scan -> above-ground points    (ground removal, ground frame)
    clustering      cluster_scan      above-ground points -> DBSCAN labels   (pre-filters, DBSCAN)
    fitting         extract_cones     labelled points -> cones               (gates, cone fit, sigma_xy)

The ground stage already puts the above-ground points in the ground frame (z =
height above the floor) because pre-filters and DBSCAN read z that way; there is
no separate rotation stage. Rotating the *whole* scan is only for visualisation
and is done on demand (:meth:`Segmentation.ground_frame_scan`).

Tools that want the clusters rather than the cones read
``ScanResult.segmentation``, so they see exactly what the cones were fitted on.

``detector_backend`` picks who runs those stages. "python" runs the stage
packages above. "fused" (``fused.py``, about ten Numba calls) and "cpp"
(``cpp_backend.py``, the same kernels in C++) run the whole scan in one pass and
give the same cones, plane and counters, but only for the configs they reproduce
exactly (``fused.fused_unsupported_reason``); any other config, or an observer,
runs the python stages with one warning.
The ROS node uses the detector through
:class:`~cone_detection.strategies.BaseConeDetection`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from cone_detection import cpp_backend, fused
from cone_detection.clustering.dbscan import _warn_once
from cone_detection.clustering.stage import cluster_scan
from cone_detection.config import ConeDetectionConfig
from cone_detection.fitting.extraction import Cone, extract_cones
from cone_detection.fitting.fitters import cluster_fitter
from cone_detection.ground.alignment import rotate_xyz_to_ground
from cone_detection.ground.methods import make_ground_remover
from cone_detection.ground.methods.plane_methods import RansacGround
from cone_detection.ground.ransac import scan_seed, seed_numba_rng
from cone_detection.ground.separation import separate_ground
from cone_detection.instrumentation import ScanObserver, StageTimings
from cone_detection.preprocessing.range_crop import crop_scan


def _empty_xyz() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


@dataclass(frozen=True)
class Segmentation:
    """Everything up to clustering for one scan.

    Returned in :class:`ScanResult` so callers (strategy, benchmarks, tests) read
    the clusters the cones were fitted on instead of re-running the stages.
    Arrays are the ones the pipeline worked on, not copies: do not modify them.
    """

    # The cropped scan, sensor frame.
    scan: np.ndarray = field(default_factory=_empty_xyz)
    # The filtered above-ground returns DBSCAN saw, ground frame (z = height above floor).
    above_ground_xyz: np.ndarray = field(default_factory=_empty_xyz)
    # DBSCAN label per ``above_ground_xyz`` row (-1 = noise).
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Ground plane ``[bias, n_x, n_y, n_z]`` in the sensor frame; None when the
    # crop left nothing to fit.
    plane: np.ndarray | None = None

    def ground_frame_scan(self) -> np.ndarray:
        """The whole cropped scan in the ground frame, ground included (computed on call)."""
        if self.plane is None:
            return _empty_xyz()
        return rotate_xyz_to_ground(self.scan, self.plane)


@dataclass(frozen=True)
class ScanResult:
    """What :meth:`RealtimeConeDetector.detect` found in one scan."""

    cones: list[Cone] = field(default_factory=list)
    segmentation: Segmentation = field(default_factory=Segmentation)


class RealtimeConeDetector:
    """Configurable single-scan cone detector (see :class:`ConeDetectionConfig`).

    The only state it keeps between scans is the ground method's (see
    :class:`~cone_detection.ground.methods.GroundRemover`); every per-scan output
    is in the returned :class:`ScanResult`.
    """

    __slots__ = ("config", "_fit_cluster", "_ground", "_cpp_core")

    def __init__(self, config: ConeDetectionConfig | None = None) -> None:
        self.config = config or ConeDetectionConfig()
        self._fit_cluster = cluster_fitter(self.config.fit_backend)
        self._ground = make_ground_remover(self.config)
        # C++ core of detector_backend="cpp" (and its RNG state), made on first use.
        self._cpp_core = None

    def reset(self) -> None:
        """Forget state from previous scans (bag loop / seek, new run)."""
        self._ground.reset()

    def detect(
        self,
        data,
        *,
        debug_counters: dict | None = None,
        compare_logger=None,
        clustering_class: type = DBSCAN,
        stage_timings: dict[str, float] | None = None,
        observer: ScanObserver | None = None,
        keep_scan: bool = True,
    ) -> ScanResult:
        """Detect cones in a single LiDAR scan.

        Args:
            data: ``(N, 3)`` array of LiDAR returns in the sensor frame.
            debug_counters: Optional dict filled with per-stage cluster counts.
            compare_logger: Optional object with ``info(str)``; gets a fit summary line.
            clustering_class: sklearn-style clustering class (see ``cluster_points``).
            stage_timings: Optional dict filled with per-stage timings (ms) and counts.
            observer: Optional :class:`ScanObserver` told every stage's result.
            keep_scan: the cpp backend copies the cropped scan out of C++ only
                when asked (``Segmentation.scan``, for ``ground_frame_scan``);
                the other backends always keep it.

        Returns:
            The accepted cones (``(x, y, height_m, sigma_xy)`` each) and the
            segmentation they came from.
        """
        cfg = self.config
        # Counters and timings always go somewhere (a throwaway dict when the
        # caller passed none): the guard counters below read the timings back.
        counters = debug_counters if debug_counters is not None else {}
        timings = StageTimings(stage_timings)

        backend = self._backend(data, clustering_class, observer)
        if cfg.ransac_seed_per_scan and len(data):
            # The same scan always gives the same RANSAC plane (ransac.scan_seed).
            if backend == "cpp":
                self._cpp_core.seed(scan_seed(data))
            else:
                seed_numba_rng(scan_seed(data))
        if backend != "python":
            return self._detect_fused(
                data, counters, timings, compare_logger, clustering_class,
                cpp=backend == "cpp", keep_scan=keep_scan,
            )

        # 1. Preprocessing. Nothing left -> no plane to fit, and the ground
        # method's state (RANSAC's warm start) is left untouched.
        cropped = crop_scan(data, cfg) if len(data) else data
        if len(cropped) == 0:
            return ScanResult(segmentation=Segmentation(scan=cropped))
        counters["n_input_points"] = len(cropped)

        # 2. Ground.
        above, plane = separate_ground(cropped, self._ground, timings, observer=observer)

        # 3. Clustering.
        points, labels = cluster_scan(
            above, cfg, timings, observer=observer, clustering_class=clustering_class
        )
        # Same data as ScanResult.segmentation, for observers recording every stage.
        # No observer (the live node): this check is all it costs.
        if observer is not None:
            observer.clustered(labels, points, plane)
        dropped = int(timings.values.get("n_dbscan_guard_dropped", 0))
        counters["dbscan_guard_dropped"] = dropped
        counters["dbscan_guard_scans"] = int(dropped > 0)
        seg = Segmentation(scan=cropped, above_ground_xyz=points, labels=labels, plane=plane)
        if len(labels) == 0:
            return ScanResult(segmentation=seg)

        # 4. Fitting.
        cones = extract_cones(
            points,
            labels,
            cfg,
            self._fit_cluster,
            counters=counters,
            timings=timings,
            observer=observer,
            logger=compare_logger,
        )
        return ScanResult(cones=cones, segmentation=seg)

    # ------------------------------------------------------------------
    # fused / cpp backends
    # ------------------------------------------------------------------
    def _ransac(self) -> RansacGround | None:
        """The RANSAC remover whose warm-start plane the fused paths share."""
        g = getattr(self._ground, "_inner", self._ground)
        return g if isinstance(g, RansacGround) else None

    def _backend(self, data, clustering_class, observer) -> str:
        cfg = self.config
        backend = cfg.detector_backend
        if backend == "python":
            return backend
        if backend == "cpp" and not cpp_backend.available():
            _warn_once(
                "cpp:import",
                f"detector_backend=cpp: {cpp_backend.unavailable_reason()}; using fused",
            )
            backend = "fused"
        reason = fused.fused_unsupported_reason(cfg, data, clustering_class)
        if reason is None and observer is not None:
            reason = "an observer (stage hooks run on the python path)"
        if reason is None and self._ransac() is None:
            reason = "this ground remover (no shared RANSAC state)"
        if reason is not None:
            _warn_once(
                f"{backend}:" + reason.split(" ")[0],
                f"detector_backend={backend} does not support {reason}; using the python path",
            )
            return "python"
        if backend == "cpp" and self._cpp_core is None:
            self._cpp_core = cpp_backend.new_core()
        return backend

    def _detect_fused(
        self, data, counters, timings, compare_logger, clustering_class, *, cpp, keep_scan
    ) -> ScanResult:
        """``detect`` on the fused Numba backend, or on its C++ port when ``cpp``.

        Fills the same outputs as the python stages: cones, counters, timings,
        the segmentation and the RANSAC warm-start plane."""
        cfg = self.config
        ransac = self._ransac()
        common = dict(
            initial_plane=ransac._prev_plane,
            clustering_class=clustering_class,
            fast_clustering=clustering_class is DBSCAN,
            stage_timings=timings.values,
        )
        if cpp:
            scan = cpp_backend.run_scan(
                self._cpp_core, data, cfg, keep_cropped=keep_scan, **common
            )
        else:
            scan = fused.run_scan(data, cfg, **common)
        if len(scan.cropped) == 0:
            return ScanResult(segmentation=Segmentation(scan=_empty_xyz()))
        counters["n_input_points"] = len(scan.cropped)
        counters["dbscan_guard_dropped"] = int(scan.guard_dropped)
        counters["dbscan_guard_scans"] = int(scan.guard_dropped > 0)
        if len(scan.cropped) >= 3:  # a tiny cloud leaves the warm start alone (_TinyCloudGuard)
            ransac._prev_plane = scan.coefs
        cropped = scan.cropped if scan.cropped.shape[1:] == (3,) else _empty_xyz()
        seg = Segmentation(
            scan=cropped,
            above_ground_xyz=np.ascontiguousarray(scan.clean, dtype=np.float32),
            labels=scan.labels,
            plane=scan.coefs,
        )
        if len(scan.labels) == 0:
            return ScanResult(segmentation=seg)
        cnt = scan.counters
        counters["n_clusters"] = int(cnt[fused.N_CLUSTERS])
        counters["after_min_pts"] = int(cnt[fused.AFTER_MIN_PTS])
        counters["after_shape_gate"] = int(cnt[fused.AFTER_SHAPE_GATE])
        counters["after_residual_gate"] = int(cnt[fused.ACCEPTED])
        counters["accepted"] = int(cnt[fused.ACCEPTED])
        counters["far_dropped"] = int(cnt[fused.FAR_DROPPED])
        counters["small_count"] = int(cnt[fused.SMALL])
        counters["big_count"] = int(cnt[fused.BIG])
        counters["ambiguous_count"] = int(cnt[fused.AMBIGUOUS])
        rows = scan.rows
        n_shape = int(cnt[fused.SHAPE_REJECTED])
        n_residual = int(cnt[fused.RESIDUAL_REJECTED])
        if compare_logger is not None and (len(rows) or n_shape or n_residual):
            rmse_med = float(np.sqrt(np.median(rows[:, 4]))) if len(rows) else 0.0
            rmse_max = float(np.sqrt(rows[:, 4].max())) if len(rows) else 0.0
            compare_logger.info(
                f"[cone_fit backend={cfg.fit_backend} {'cpp' if cpp else 'fused'}] "
                f"kept={len(rows)} (small={int(cnt[fused.SMALL])} big={int(cnt[fused.BIG])} "
                f"amb={int(cnt[fused.AMBIGUOUS])}) "
                f"rejected: shape={n_shape} residual={n_residual} "
                f"| RMSE med={rmse_med * 1000:.1f}mm max={rmse_max * 1000:.1f}mm"
            )
        cones = [(float(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in rows]
        return ScanResult(cones=cones, segmentation=seg)
