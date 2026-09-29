"""
LiDAR cone detection — extract per-cone ``(x, y, height_m, σ_xy)`` from one scan.

Public surface:

* :class:`RealtimeConeDetector` — configurable pipeline (thresholds, DBSCAN,
  ``cone_detection.ransac.ransac2``, fit backend: template dispatch vs
  ``cone_fit_2params``).
* :func:`final_cone_result_rt` — thin wrapper around a module-default detector
  (backward-compatible call signature for scripts and tests).
* :func:`warmup_numba_functions` — eager-compile Numba/scipy on the hot path so
  the first real scan does not pay JIT cost (called from strategy ``configure()``).

Internal helper:

* :func:`clustering_separation_rt` — RANSAC ground plane + rotation + DBSCAN.

Default path: RANSAC ground removal + DBSCAN, then per-cluster
:func:`cone_detection.cone_fit.cone_fit_template_dispatch` unless
``ConeDetectionConfig.fit_backend == "two_param"``. The live ROS node uses
:class:`~cone_detection.strategies.BaseConeDetection` and publishes markers in
:class:`~cone_detection.cone_detection_node.ConeDetectionNode`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.cluster import DBSCAN

from cone_detection.cone_fit import (
    _CONE_BIG_C,
    _CONE_BIG_D,
    _CONE_SMALL_C,
    _CONE_SMALL_D,
    _LINEARITY_THRESHOLD,
    cluster_linearity,
    cone_fit_2params,
    cone_fit_collinear,
    cone_fit_template_dispatch,
)
from cone_detection.ransac import ransac2
from cone_detection.rotations import vectors2matrix

ConeFitBackend = Literal["template_dispatch", "two_param"]

# Cap for the /lidar_points/ground viz cloud. Full cropped scans are still
# tens of thousands of returns; Foxglove chokes before the multiply does.
VIZ_GROUND_CLOUD_MAX_POINTS = 15_000


def ground_rotation_matrix(plane_coefs: np.ndarray) -> np.ndarray:
    """3×3 rotation that maps the RANSAC plane normal onto +z.

    ``plane_coefs`` is ``[bias, n_x, n_y, n_z]`` from ``ransac2`` /
    ``clustering_separation_rt``. Degenerate normals return identity.
    """
    k = np.zeros(3, dtype=np.float64)
    k[-1] = 1.0
    normal = np.asarray(plane_coefs[1:], dtype=np.float64)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        return np.eye(3, dtype=np.float64)
    return vectors2matrix(k, normal / nrm)


def plane_floor_z(plane_coefs: np.ndarray) -> float:
    """Signed height of the LiDAR origin above the RANSAC plane."""
    normal = np.asarray(plane_coefs[1:], dtype=np.float64)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        return 0.0
    return float(
        np.dot(np.array([0.0, 0.0, -1.0 * float(plane_coefs[0])]), normal) / nrm
    )


def shift_rotated_ground_z0(
    xyz: np.ndarray,
    plane_coefs: np.ndarray,
) -> np.ndarray:
    """Shift an already-rotated cloud so the fitted floor sits at z=0."""
    if xyz is None or len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32)[:, :3], dtype=np.float32)
    out[:, 2] -= np.float32(plane_floor_z(plane_coefs))
    return out


def rotate_xyz_to_ground(
    xyz: np.ndarray,
    plane_coefs: np.ndarray,
    *,
    max_points: int = VIZ_GROUND_CLOUD_MAX_POINTS,
    align_ground_z0: bool = True,
) -> np.ndarray:
    """Rotate a cloud into the same frame as ``/Conos_raw`` (ground ≈ z=0).

    Detection rotates only above-ground outliers for DBSCAN. This helper
    rotates a (possibly decimated) copy of the full cropped scan with the
    same matrix, then optionally shifts z so the fitted floor sits at 0 —
    matching the markers, which are published at ``z=0`` in ``base_link``.
    """
    if xyz is None or len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    if max_points > 0 and len(pts) > max_points:
        stride = int(math.ceil(len(pts) / max_points))
        pts = pts[::stride]
    normal = np.asarray(plane_coefs[1:], dtype=np.float64)
    nrm = float(np.linalg.norm(normal))
    out = pts[:, :3] @ ground_rotation_matrix(plane_coefs)
    if align_ground_z0 and nrm >= 1e-12:
        out[:, 2] -= plane_floor_z(plane_coefs)
    return np.ascontiguousarray(out, dtype=np.float32)


@dataclass
class ConeDetectionConfig:
    """Tunable perception pipeline (clustering, gates, cone fit)."""

    # Drop a cluster if the chosen fit's MSE residual exceeds this (RMSE = √mse:
    # 0.2 → 447 mm). Loosened from the sim-tuned 0.002 (RMSE 45 mm), which
    # rejected 100% of shape-valid candidates on the real Hesai ATX track
    # (CONE_FILTER: ~20/scan pass shape, 0 pass residual → empty /Conos_raw →
    # no map/path → car won't move). Deliberately permissive to get the car
    # driving; expect more false positives. RE-TIGHTEN toward 0.02-0.03 once a
    # real-cloud residual histogram is measured (bag /lidar_points via #68).
    residual_gate_mse: float = 0.2

    # Coarse cluster-shape pre-filter (meters).
    cluster_height_min_m: float = 0.02
    cluster_height_max_m: float = 0.55

    # Range cutoff (m): beyond this, drop as too sparse / unreliable.
    range_gate_max_m: float = 20.0

    # Input range pre-crop (m): drop points beyond this horizontal radius
    # BEFORE RANSAC + DBSCAN. Everything past range_gate_max_m is discarded
    # downstream anyway (far_dropped), so cropping here is behavior-preserving
    # (radius > range_gate_max_m + cone extent) and removes most of the ~174k
    # ATX returns (far ground) before the O(n) clustering — the dominant cost.
    # Set to 0 to disable.
    input_range_crop_m: float = 25.0

    # Confidence margin (residual_other / residual_min) for template_dispatch
    # ambiguity. Ignored when ``res_other`` is not finite (e.g. two_param path).
    ambiguous_margin_ratio: float = 1.5

    # For ``fit_backend == "two_param"``: classify big orange when fitted apex
    # height ``d`` exceeds this (template path uses fixed 0.35 / 0.55 instead).
    big_orange_d_threshold_m: float = 0.45

    # Ground strip: drop returns closer than this above the estimated floor (m).
    floor_margin_m: float = 0.04

    # Minimum number of returns in a DBSCAN cluster before floor culling.
    min_cluster_points: int = 3

    # RANSAC plane inliers
    ransac_prob: float = 0.9999
    ransac_threshold: float = 0.05

    # DBSCAN on rotated above-ground cloud
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2

    # DBSCAN memory/time guard. sklearn's DBSCAN materialises EVERY point's
    # full neighbour list (radius_neighbors), so its cost is
    # O(N × neighbours-within-eps), i.e. quadratic in point density. A scan
    # whose above-ground cloud is both large and dense — tire walls, car
    # off-track facing terrain / an object at close range, or a mis-fitted
    # ground plane that turns the ground itself into "outliers" — blows
    # straight through the container: measured 90k pts @ 20k pts/m² → 4.7 GB
    # peak, 11 s, ~2.6 GB never returned to the OS. Two such scans OOM-kill
    # the node (8 GB limit) after stalling SLAM for seconds. A cone scene is
    # nowhere near: ~100 cones × ≤200 pts ≈ 20k worst case, typically 1-2k.
    #
    # Above dbscan_max_points the cloud is voxel-deduplicated, starting at
    # dbscan_guard_voxel_m and doubling the cell up to dbscan_guard_voxel_max_m
    # until the count fits, then uniformly subsampled to the cap as a last
    # resort. Normal scans never trigger it. Set dbscan_max_points to 0 to
    # disable.
    dbscan_max_points: int = 20_000
    dbscan_guard_voxel_m: float = 0.02
    dbscan_guard_voxel_max_m: float = 0.10

    # Tall-column veto (tire walls, fences, people): before DBSCAN, drop EVERY
    # point whose xy grid cell (3x3-dilated) contains a return more than
    # tall_column_veto_height_m above the fitted ground plane. No cone exceeds
    # cluster_height_max_m, so such a return proves non-cone structure in that
    # column; vetoing the whole column also removes the structure's cone-height
    # low band, which a plain z-crop would leave behind as truncated stumps.
    # Tire-wall scans (~40-90k DBSCAN points, seconds) collapse back to normal
    # (~ms) at zero measured cone change; scans with nothing tall skip the
    # masking entirely.
    tall_column_veto: bool = True
    tall_column_veto_height_m: float = 0.75
    tall_column_veto_cell_m: float = 0.30

    # Template (a, b) solver: "gn" = in-process Numba Gauss-Newton/LM
    # (production default); any scipy.optimize.minimize method name
    # (e.g. "L-BFGS-B", the previous default) selects the scipy path for A/B.
    cone_fit_solver: str = "gn"
    template_fit_maxiter: int = 12

    # Near-collinear clusters: closed-form collinear fit (no scipy).
    use_collinear_fit: bool = True

    # Skip the second template L-BFGS-B when cluster height strongly
    # indicates small vs big-orange (saves ~half of fit time).
    template_fit_early_exit: bool = True
    template_early_exit_height_small_m: float = 0.40
    template_early_exit_height_big_m: float = 0.50

    fit_backend: ConeFitBackend = "template_dispatch"


def _cluster_cone_mse(a: float, b: float, c: float, d: float, xyz: np.ndarray) -> float:
    """Mean squared error in z for z = d - c * ||(x,y) - (a,b)||."""
    x = xyz[:, 0].astype(np.float64, copy=False)
    y = xyz[:, 1].astype(np.float64, copy=False)
    z = xyz[:, 2].astype(np.float64, copy=False)
    gam = np.sqrt((x - a) ** 2 + (y - b) ** 2)
    z_pred = d - c * gam
    return float(np.mean((z - z_pred) ** 2))


def _fit_cluster(
    clean_cone: np.ndarray,
    cfg: ConeDetectionConfig,
    *,
    cluster_height_m: float,
    ground_z: float,
) -> tuple[float, float, float, float, float, float, bool]:
    """Return ``(a, b, c, d, res_min, res_other, is_big)`` like template dispatch."""
    if cfg.fit_backend == "two_param":
        a, b, c, d = cone_fit_2params(clean_cone, solver=cfg.cone_fit_solver)
        res_min = _cluster_cone_mse(a, b, c, d, clean_cone)
        res_other = float("nan")
        is_big = bool(d > cfg.big_orange_d_threshold_m)
        return float(a), float(b), float(c), float(d), res_min, res_other, is_big

    if cfg.use_collinear_fit and cluster_linearity(clean_cone) < _LINEARITY_THRESHOLD:
        a, b, c, d_class = cone_fit_collinear(
            clean_cone, lidar_xy=(0.0, 0.0), ground_z=ground_z
        )
        if np.isfinite(c):
            res_min = _cluster_cone_mse(a, b, c, d_class, clean_cone)
            is_big = bool(d_class >= _CONE_BIG_D - 1e-6)
            return (
                float(a),
                float(b),
                float(c),
                float(d_class),
                res_min,
                float("nan"),
                is_big,
            )

    only_small = only_big = None
    if cfg.template_fit_early_exit:
        if cluster_height_m < cfg.template_early_exit_height_small_m:
            only_small = True
        elif cluster_height_m > cfg.template_early_exit_height_big_m:
            only_big = True

    return cone_fit_template_dispatch(
        clean_cone,
        lidar_xy=(0.0, 0.0),
        solver=cfg.cone_fit_solver,
        maxiter=cfg.template_fit_maxiter,
        only_small=only_small,
        only_big=only_big,
    )


# Grid cells are keyed as i * stride + j, injective while |j| < stride / 2.
# With the 25 m input crop and the 0.3 m default cell, |i|, |j| <= ~90.
_VETO_KEY_STRIDE = 1 << 20


def _tall_column_veto_mask(
    xy: np.ndarray, height_above_ground: np.ndarray, cfg: ConeDetectionConfig
) -> np.ndarray | None:
    """Keep-mask over points, vetoing xy columns that contain a tall return.

    Returns ``None`` when no point exceeds ``tall_column_veto_height_m`` (the
    common case) so the caller can skip masking entirely.
    """
    tall = height_above_ground > cfg.tall_column_veto_height_m
    if not tall.any():
        return None
    ij = np.floor(xy / cfg.tall_column_veto_cell_m).astype(np.int64)
    keys = ij[:, 0] * _VETO_KEY_STRIDE + ij[:, 1]
    tall_keys = np.unique(keys[tall])
    # 3x3 dilation in key space: cell (i+di, j+dj) has key + di*stride + dj.
    offsets = np.array(
        [di * _VETO_KEY_STRIDE + dj for di in (-1, 0, 1) for dj in (-1, 0, 1)],
        dtype=np.int64,
    )
    veto_keys = np.unique((tall_keys[:, None] + offsets).ravel())
    return ~np.isin(keys, veto_keys)


# Voxel keys for the DBSCAN guard: per-axis index offset into [0, 2^21) and
# packed into 63 bits. Injective for |index| < 2^20, i.e. any coordinate
# within ±20 km at the 0.02 m default cell — far beyond LiDAR range.
_GUARD_KEY_OFFSET = 1 << 20
_GUARD_KEY_BITS = 21


def _voxel_first_points(data: np.ndarray, voxel_m: float) -> np.ndarray:
    """Keep the first point of every ``voxel_m`` cell (row order preserved)."""
    ijk = np.floor(data[:, :3] / voxel_m).astype(np.int64) + _GUARD_KEY_OFFSET
    keys = (
        (ijk[:, 0] << (2 * _GUARD_KEY_BITS))
        | (ijk[:, 1] << _GUARD_KEY_BITS)
        | ijk[:, 2]
    )
    _, first = np.unique(keys, return_index=True)
    return data[np.sort(first)]


def _bound_dbscan_input(
    data: np.ndarray, cfg: ConeDetectionConfig
) -> np.ndarray:
    """Cap the density and count of the cloud handed to DBSCAN.

    No-op (returns ``data`` itself) while ``len(data) <= dbscan_max_points``
    or the guard is disabled. See ``ConeDetectionConfig.dbscan_max_points``
    for the rationale. Row order is preserved so downstream label handling
    is unaffected.
    """
    cap = int(cfg.dbscan_max_points)
    if cap <= 0 or len(data) <= cap:
        return data
    voxel = float(cfg.dbscan_guard_voxel_m)
    while (
        voxel > 0.0
        and voxel <= cfg.dbscan_guard_voxel_max_m + 1e-9
        and len(data) > cap
    ):
        data = _voxel_first_points(data, voxel)
        voxel *= 2.0
    if len(data) > cap:
        # Seeded per scan size so a given input is reproducible; the choice
        # itself is uniform, so each cone keeps a proportional share.
        rng = np.random.default_rng(len(data))
        keep = rng.choice(len(data), cap, replace=False)
        data = data[np.sort(keep)]
    return data


def clustering_separation_rt(
    data: np.ndarray,
    config: ConeDetectionConfig | None = None,
    *,
    clustering_class: type = DBSCAN,
    stage_timings: dict[str, float] | None = None,
    ransac_iter_subsample_max: int = 5000,
    initial_plane: np.ndarray | None = None,
):
    """Ground removal + rotation correction + DBSCAN, single scan.

    RANSAC fits the ground plane (``ransac2`` on augmented ``[1,x,y,z]``), outliers
    are rotated so the plane normal aligns with +z, then DBSCAN clusters them.

    Args:
        initial_plane: optional previous-scan plane coefficients
            ``[bias, n_x, n_y, n_z]`` used to warm-start RANSAC (see
            ``ransac2``). Stale planes are harmless — they lose the support
            comparison and the full random search runs.

    Returns:
        ``(labels, rotated_outliers, plane_coefs)`` — per-point cluster labels,
        outlier points after rotation, and ``[bias, n_x, n_y, n_z]`` plane coeffs.
    """
    cfg = config or ConeDetectionConfig()
    A = np.c_[np.ones(data.shape[0]), data]
    warm = (
        np.asarray(initial_plane, dtype=np.float64)
        if initial_plane is not None
        else np.zeros(0)
    )
    t_ransac = time.perf_counter()
    inliers, def_coefs = ransac2(
        A,
        prob=cfg.ransac_prob,
        threshold=cfg.ransac_threshold,
        iter_subsample_max=ransac_iter_subsample_max,
        initial_coefs=warm,
    )
    if stage_timings is not None:
        stage_timings["ransac_ms"] = (time.perf_counter() - t_ransac) * 1000.0
    outliers = np.ones(data.shape[0], dtype=bool)
    outliers[inliers] = False
    t_rotate = time.perf_counter()
    # Select outliers BEFORE rotating: row selection commutes with the
    # right-multiplied rotation, and rotating only the ~1% above-ground
    # points instead of the full cloud removes the dominant cost here.
    data = data[outliers] @ ground_rotation_matrix(def_coefs)
    if stage_timings is not None:
        stage_timings["rotate_ms"] = (time.perf_counter() - t_rotate) * 1000.0
        stage_timings["n_outliers"] = float(len(data))
    if len(data) == 0:
        if stage_timings is not None:
            stage_timings["dbscan_ms"] = 0.0
            stage_timings["n_dbscan_guard_dropped"] = 0.0
        return np.array([]), data, def_coefs
    if cfg.tall_column_veto:
        t_veto = time.perf_counter()
        # Ground height in the rotated frame: distance from the LiDAR origin
        # to the plane (same formula as the per-cluster floor cull below).
        v = np.array([0, 0, -1 * def_coefs[0]])
        w = np.array(def_coefs[1:])
        floor_z = np.dot(v, w) / np.linalg.norm(w)
        keep = _tall_column_veto_mask(data[:, :2], data[:, 2] - floor_z, cfg)
        n_vetoed = 0
        if keep is not None:
            n_vetoed = int(len(keep) - keep.sum())
            data = data[keep]
        if stage_timings is not None:
            stage_timings["column_veto_ms"] = (
                time.perf_counter() - t_veto
            ) * 1000.0
            stage_timings["n_vetoed"] = float(n_vetoed)
        if len(data) == 0:
            if stage_timings is not None:
                stage_timings["dbscan_ms"] = 0.0
                stage_timings["n_dbscan_guard_dropped"] = 0.0
            return np.array([]), data, def_coefs
    # Memory guard — see ConeDetectionConfig.dbscan_max_points.
    n_before_guard = len(data)
    data = _bound_dbscan_input(data, cfg)
    if stage_timings is not None:
        stage_timings["n_dbscan_guard_dropped"] = float(n_before_guard - len(data))
    clust_model = clustering_class(
        eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples
    )
    t_dbscan = time.perf_counter()
    labels = clust_model.fit_predict(data)
    if stage_timings is not None:
        stage_timings["dbscan_ms"] = (time.perf_counter() - t_dbscan) * 1000.0
    return labels, data, def_coefs


class RealtimeConeDetector:
    """Configurable single-scan cone detector (see :class:`ConeDetectionConfig`)."""

    __slots__ = ("config", "_prev_plane", "last_rotated_xyz", "last_outlier_xyz")

    def __init__(self, config: ConeDetectionConfig | None = None) -> None:
        self.config = config or ConeDetectionConfig()
        # Previous scan's ground plane, used to warm-start RANSAC. At 10 Hz
        # the plane barely moves frame to frame, so the warm candidate
        # usually wins immediately and collapses the iteration budget.
        self._prev_plane: np.ndarray | None = None
        # Last scan's cropped cloud after the same RANSAC rotation used
        # for clustering, ground shifted to z=0. Only filled when
        # ``viz_full_cloud=True`` (the /lidar_points/ground path).
        self.last_rotated_xyz: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        # RANSAC outliers after rotation + z-shift — what DBSCAN sees.
        self.last_outlier_xyz: np.ndarray = np.zeros((0, 3), dtype=np.float32)

    def detect(
        self,
        data,
        *,
        debug_counters: dict | None = None,
        compare_logger=None,
        clustering_class: type = DBSCAN,
        stage_timings: dict[str, float] | None = None,
        ransac_iter_subsample_max: int = 5000,
        viz_full_cloud: bool = False,
    ) -> list[tuple[float, float, float, float]]:
        """Detect cones in a single LiDAR scan (same contract as ``final_cone_result_rt``)."""
        cfg = self.config
        empty = np.zeros((0, 3), dtype=np.float32)
        self.last_rotated_xyz = empty
        self.last_outlier_xyz = empty
        if len(data) == 0:
            return []
        # Range pre-crop: drop far points before the O(n) RANSAC + DBSCAN
        # instead of clustering ~1000 far groups just to discard them by the
        # range gate. Uses squared radius (no sqrt). Behavior-preserving for
        # crop radius > range_gate_max_m + cone extent.
        if cfg.input_range_crop_m > 0:
            r2 = data[:, 0] ** 2 + data[:, 1] ** 2
            data = data[r2 <= cfg.input_range_crop_m ** 2]
            if len(data) == 0:
                return []
        if debug_counters is not None:
            debug_counters["n_input_points"] = len(data)
        # Always collect stage bookkeeping so the DBSCAN guard can be
        # surfaced through debug_counters even when the caller does not ask
        # for timings (a handful of perf_counter calls — negligible).
        st = stage_timings if stage_timings is not None else {}
        labels, clean_data, def_coefs = clustering_separation_rt(
            data,
            cfg,
            clustering_class=clustering_class,
            stage_timings=st,
            ransac_iter_subsample_max=ransac_iter_subsample_max,
            initial_plane=self._prev_plane,
        )
        if debug_counters is not None:
            dropped = int(st.get("n_dbscan_guard_dropped", 0))
            debug_counters["dbscan_guard_dropped"] = dropped
            debug_counters["dbscan_guard_scans"] = int(dropped > 0)
        self._prev_plane = def_coefs
        # Outliers are already rotated inside clustering_separation_rt.
        # Only the z-shift is extra; the full-crop rotate is deferred
        # until a viz subscriber actually wants /lidar_points/ground.
        self.last_outlier_xyz = shift_rotated_ground_z0(clean_data, def_coefs)
        if viz_full_cloud:
            self.last_rotated_xyz = rotate_xyz_to_ground(data, def_coefs)
        if len(labels) == 0:
            return []
        t_cluster_prep = time.perf_counter()
        separated_data = [
            np.array(clean_data[labels == label]) for label in np.unique(labels)
        ]
        if stage_timings is not None:
            stage_timings["cluster_prep_ms"] = (
                time.perf_counter() - t_cluster_prep
            ) * 1000.0
            stage_timings["n_clusters"] = float(len(separated_data))
        if debug_counters is not None:
            debug_counters["n_clusters"] = len(separated_data)
            debug_counters["after_min_pts"] = 0
            debug_counters["after_shape_gate"] = 0
            debug_counters["after_residual_gate"] = 0
            debug_counters["accepted"] = 0
            debug_counters["far_dropped"] = 0
            debug_counters["small_count"] = 0
            debug_counters["big_count"] = 0
            debug_counters["ambiguous_count"] = 0

        cone_positions: list[tuple[float, float, float, float]] = []
        n_small = 0
        n_big = 0
        n_ambiguous = 0
        n_residual_rejected = 0
        n_shape_rejected = 0
        residuals_kept: list[float] = []
        t_fit = time.perf_counter()

        v = np.array([0, 0, -1 * def_coefs[0]])
        w = np.array(def_coefs[1:])
        lidar_distance_to_floor = np.dot(v, w) / np.linalg.norm(w)

        for cone in separated_data:
            if len(cone) < cfg.min_cluster_points:
                continue
            if debug_counters is not None:
                debug_counters["after_min_pts"] += 1

            clean_cone = cone[cone[:, 2] > cfg.floor_margin_m + lidar_distance_to_floor]
            if len(clean_cone) == 0:
                continue

            cluster_height = float(clean_cone[:, 2].max() - clean_cone[:, 2].min())
            n_pts = len(clean_cone)
            range_m_centroid = float(
                np.hypot(
                    float(clean_cone[:, 0].mean()),
                    float(clean_cone[:, 1].mean()),
                )
            )

            cluster_shape_ok = (
                cfg.cluster_height_min_m <= cluster_height <= cfg.cluster_height_max_m
                and range_m_centroid <= cfg.range_gate_max_m
            )
            if not cluster_shape_ok:
                n_shape_rejected += 1
                if debug_counters is not None and range_m_centroid > cfg.range_gate_max_m:
                    debug_counters["far_dropped"] += 1
                continue
            if debug_counters is not None:
                debug_counters["after_shape_gate"] += 1

            a_xy, b_xy, _c_chosen, d_chosen, res_min, res_other, is_big = _fit_cluster(
                clean_cone,
                cfg,
                cluster_height_m=cluster_height,
                ground_z=float(lidar_distance_to_floor),
            )
            if not (np.isfinite(res_min) and res_min <= cfg.residual_gate_mse):
                n_residual_rejected += 1
                continue
            if debug_counters is not None:
                debug_counters["after_residual_gate"] += 1

            ambiguous = (
                np.isfinite(res_other)
                and res_other / max(res_min, 1e-12) < cfg.ambiguous_margin_ratio
            )
            if ambiguous:
                n_ambiguous += 1
                if debug_counters is not None:
                    debug_counters["ambiguous_count"] += 1
            if is_big:
                n_big += 1
                if debug_counters is not None:
                    debug_counters["big_count"] += 1
            else:
                n_small += 1
                if debug_counters is not None:
                    debug_counters["small_count"] += 1
            residuals_kept.append(res_min)

            if debug_counters is not None:
                debug_counters["accepted"] += 1

            range_m = float(np.hypot(a_xy, b_xy))
            base_sigma = 0.05 + 0.005 * range_m
            sigma_xy = 1.5 * base_sigma / math.sqrt(max(1, n_pts) / 10.0)
            if ambiguous:
                sigma_xy *= 1.5
            cone_positions.append((float(a_xy), float(b_xy), float(d_chosen), sigma_xy))

        if stage_timings is not None:
            stage_timings["fit_ms"] = (time.perf_counter() - t_fit) * 1000.0

        if compare_logger is not None and (
            cone_positions or n_residual_rejected or n_shape_rejected
        ):
            if residuals_kept:
                res_arr = np.asarray(residuals_kept, dtype=float)
                rmse_med = float(np.sqrt(np.median(res_arr)))
                rmse_max = float(np.sqrt(res_arr.max()))
            else:
                rmse_med = 0.0
                rmse_max = 0.0
            compare_logger.info(
                f"[cone_fit backend={cfg.fit_backend}] kept={len(cone_positions)} "
                f"(small={n_small} big={n_big} amb={n_ambiguous}) "
                f"rejected: shape={n_shape_rejected} residual={n_residual_rejected} "
                f"| RMSE med={rmse_med * 1000:.1f}mm max={rmse_max * 1000:.1f}mm"
            )
        return cone_positions


_DEFAULT_DETECTOR = RealtimeConeDetector(ConeDetectionConfig())


def final_cone_result_rt(
    data, model=DBSCAN, debug_counters=None, compare_logger=None
):
    """Detect cones in a single LiDAR scan (real-time path).

    This uses the module-default :class:`ConeDetectionConfig`. For experiments
    (e.g. ``fit_backend="two_param"``), build a :class:`RealtimeConeDetector`
    with a custom config instead.

    Args:
        data: ``(N, 3)`` array of LiDAR returns in the sensor frame.
        model: Clustering class with sklearn-style ``fit_predict``; instantiated
            as ``model(eps=..., min_samples=...)`` from the default config
            (same as historical behavior when this was ``DBSCAN``).
        debug_counters: Optional dict populated with per-stage cluster counts.
        compare_logger: Optional object with ``info(str)`` (e.g. a ROS logger).

    Returns:
        List of ``(x, y, height_m, sigma_xy)`` — one tuple per accepted cone.
    """
    return _DEFAULT_DETECTOR.detect(
        data,
        debug_counters=debug_counters,
        compare_logger=compare_logger,
        clustering_class=model,
    )


def rect2polars(x, y):
    """Cartesian ``(x, y)`` → ``(radius, angle)``."""
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def warmup_numba_functions(
    config: ConeDetectionConfig | None = None,
    *,
    also_warm_two_param: bool = False,
) -> None:
    """Warm every Numba kernel on the live hot path at configure time.

    All perception kernels compile with ``cache=True``, so the compiled
    machine code persists in ``__pycache__`` next to the sources (or in
    ``NUMBA_CACHE_DIR`` when set). Warmup still runs on every configure: the
    first call per process is what loads the disk cache, and it transparently
    recompiles on a cache miss (numba/Python upgrade, source change, cold
    volume). Cached start ≈ 1–2 s; first-ever start pays the full 10–20 s JIT.

    Coverage matters more than volume here. Numba specializes per *call
    pattern*: an argument left to its default compiles a different
    specialization (``Omitted`` type) than the same value passed explicitly,
    so hand-written kernel calls alone can miss the signature the node
    actually uses. The end-to-end block below therefore replays the real
    call chain — ``RealtimeConeDetector.detect`` on a synthetic scan, with
    the caller's ``config`` — twice, covering both the no-previous-plane and
    warm-plane RANSAC entries exactly as the live node hits them.

    Args:
        config: The production config the node will run with (solver choice,
            fit backend, RANSAC parameters). ``None`` = module defaults.
        also_warm_two_param: Force-warm ``cone_fit_2params`` even when the
            config does not select ``fit_backend="two_param"``.
    """
    cfg = config or ConeDetectionConfig()
    rng = np.random.default_rng(0)

    def _synthetic_cone_cloud(
        n: int, noise_m: float, c_true: float, d_true: float
    ) -> np.ndarray:
        """N points on a cone-like sweep; float32 like decoded PointCloud2."""
        th = np.linspace(0, 2 * np.pi, n)
        rr = np.linspace(0.05, 0.10, n)
        a0, b0 = 5.0, 0.5
        x = (a0 + rr * np.cos(th)).astype(np.float32)
        y = (b0 + rr * np.sin(th)).astype(np.float32)
        z = (d_true - c_true * rr + rng.normal(0.0, noise_m, size=n)).astype(
            np.float32
        )
        return np.column_stack([x, y, z])

    def _near_face_cluster(
        cx: float, cy: float, c_true: float, d_true: float, n: int
    ) -> np.ndarray:
        """One-sided near-face cluster like a vertical-beam LiDAR sees."""
        z = rng.uniform(0.06, d_true - 0.03, size=n)
        gamma = (d_true - z) / c_true
        norm = math.hypot(cx, cy)
        ux, uy = cx / norm, cy / norm
        psi = rng.uniform(-1.0, 1.0, size=n)
        px = cx - gamma * (np.cos(psi) * ux - np.sin(psi) * uy)
        py = cy - gamma * (np.cos(psi) * uy + np.sin(psi) * ux)
        px += rng.normal(0.0, 0.003, size=n)
        py += rng.normal(0.0, 0.003, size=n)
        z = z + rng.normal(0.0, 0.008, size=n)
        return np.column_stack([px, py, z])

    # --- direct kernel warms: both solver paths (gn = Numba kernel, scipy
    # kept for A/B) plus whatever non-default method the config selects, and
    # the collinear closed-form dispatch branch.
    solvers = {"gn", "L-BFGS-B", cfg.cone_fit_solver}
    for n in (8, 24, 48):
        for solver in solvers:
            cone_fit_template_dispatch(
                _synthetic_cone_cloud(n, 0.012, 5.0, 0.35), solver=solver
            )
            cone_fit_template_dispatch(
                _synthetic_cone_cloud(n, 0.012, 5.5, 0.55), solver=solver
            )

    line_cluster = _synthetic_cone_cloud(6, 0.012, 5.0, 0.35)
    line_cluster[:, :2] = np.linspace(
        (5.0, 0.0), (5.2, 0.03), num=line_cluster.shape[0]
    )
    cone_fit_template_dispatch(line_cluster)

    if also_warm_two_param or cfg.fit_backend == "two_param":
        for n in (8, 24, 48):
            cone_fit_2params(_synthetic_cone_cloud(n, 0.012, 5.0, 0.35))

    # --- end-to-end warm: replay the live call chain (decode-shaped float32
    # cloud -> RANSAC -> rotate -> DBSCAN -> floor cull -> gates -> fit) with
    # the production config. A tilted ground plane at z ~ 0 plus one cluster
    # per template keeps every gate passable so the fit stage really runs.
    n_ground = 2000
    gx = rng.uniform(1.0, 14.0, n_ground)
    gy = rng.uniform(-4.0, 4.0, n_ground)
    gz = 0.02 * gx + rng.normal(0.0, 0.01, size=n_ground)
    small = _near_face_cluster(5.0, 1.0, _CONE_SMALL_C, _CONE_SMALL_D, 24)
    big = _near_face_cluster(8.0, -2.0, _CONE_BIG_C, _CONE_BIG_D, 24)
    small[:, 2] += 0.02 * small[:, 0]
    big[:, 2] += 0.02 * big[:, 0]
    scan = np.vstack(
        [np.column_stack([gx, gy, gz]), small, big]
    ).astype(np.float32)
    detector = RealtimeConeDetector(cfg)
    detector.detect(scan)  # first scan: no previous plane
    detector.detect(scan)  # second scan: warm-plane RANSAC branch


if __name__ == "__main__":
    from cone_detection.lidar_io import read_lidar_data

    frames = read_lidar_data("puntos_lidar.txt")
    print(final_cone_result_rt(frames[0]))
