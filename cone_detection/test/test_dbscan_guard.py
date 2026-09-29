"""Tests for the DBSCAN memory guard (cone_detection_node OOM fix).

sklearn's DBSCAN materialises every point's neighbour list, so a scan whose
above-ground cloud is large AND dense (car off-track facing terrain or an
object, or a mis-fitted ground plane) costs O(N × neighbours): measured
90k pts @ 20k pts/m² -> 4.7 GB peak / 11 s, which OOM-killed the node twice
(8 GB container limit) after stalling SLAM for ~8 s. The guard bounds density
(voxel dedup) and count (uniform subsample) only when the cloud exceeds
``dbscan_max_points``; ordinary cone scenes are untouched.
"""
import math

import numpy as np

from cone_detection.cone_detection import (
    ConeDetectionConfig,
    RealtimeConeDetector,
    _bound_dbscan_input,
    clustering_separation_rt,
)
from cone_detection.cone_fit import _CONE_SMALL_C, _CONE_SMALL_D

RNG = np.random.default_rng(11)


def _ground(n: int = 8000) -> np.ndarray:
    x = RNG.uniform(0.5, 20.0, n)
    y = RNG.uniform(-8.0, 8.0, n)
    z = RNG.normal(0.0, 0.008, n)
    return np.column_stack([x, y, z])


def _cone_cluster(cx: float, cy: float, n: int = 30) -> np.ndarray:
    z = RNG.uniform(0.06, _CONE_SMALL_D - 0.03, size=n)
    gamma = (_CONE_SMALL_D - z) / _CONE_SMALL_C
    norm = math.hypot(cx, cy)
    ux, uy = cx / norm, cy / norm
    psi = RNG.uniform(-1.0, 1.0, size=n)
    px = cx - gamma * (np.cos(psi) * ux - np.sin(psi) * uy)
    py = cy - gamma * (np.cos(psi) * uy + np.sin(psi) * ux)
    px += RNG.normal(0.0, 0.003, size=n)
    py += RNG.normal(0.0, 0.003, size=n)
    z = z + RNG.normal(0.0, 0.008, size=n)
    return np.column_stack([px, py, z])


def _low_dense_terrain(n: int, pts_per_m2: float) -> np.ndarray:
    """Dense, bumpy surface 0.15-0.55 m above ground (a grass bank / rubble):
    below the tall-column veto, and non-planar so it cannot win the RANSAC
    ground vote."""
    side = math.sqrt(n / pts_per_m2)
    x = RNG.uniform(6.0, 6.0 + side, n)
    y = RNG.uniform(-side / 2, side / 2, n)
    z = 0.35 + 0.2 * np.sin(x * 4.0) * np.sin(y * 4.0) + RNG.normal(0.0, 0.01, n)
    return np.column_stack([x, y, z])


class _CapturingDBSCAN:
    """Stand-in clustering class recording how many points reached it."""

    seen: list[int] = []

    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        _CapturingDBSCAN.seen.append(len(X))
        return np.zeros(len(X), dtype=int)


def test_guard_is_noop_below_cap():
    cfg = ConeDetectionConfig(dbscan_max_points=1000)
    pts = RNG.uniform(-1.0, 1.0, (999, 3))
    assert _bound_dbscan_input(pts, cfg) is pts


def test_guard_disabled_with_zero_cap():
    cfg = ConeDetectionConfig(dbscan_max_points=0)
    pts = RNG.uniform(-1.0, 1.0, (50_000, 3))
    assert _bound_dbscan_input(pts, cfg) is pts


def test_guard_bounds_count_and_density_and_keeps_order():
    cfg = ConeDetectionConfig(dbscan_max_points=2000, dbscan_guard_voxel_m=0.05)
    # 30k points crammed into 1 m² -> 30k/m²; voxel dedup alone leaves
    # <= 400 cells per 0.05 m layer x few layers, then the cap applies.
    pts = np.column_stack(
        [RNG.uniform(0, 1, 30_000), RNG.uniform(0, 1, 30_000), RNG.uniform(0, 0.2, 30_000)]
    )
    pts = pts[np.argsort(pts[:, 0])]  # sorted input to check order preservation
    out = _bound_dbscan_input(pts, cfg)
    assert len(out) <= 2000
    # Density is bounded: no two kept points share a 0.05 m voxel.
    ijk = np.floor(out / 0.05).astype(np.int64)
    assert len(np.unique(ijk, axis=0)) == len(out)
    assert np.all(np.diff(out[:, 0]) >= 0)  # row order preserved


def test_dbscan_never_sees_more_than_cap_on_pathological_scan():
    _CapturingDBSCAN.seen = []
    cfg = ConeDetectionConfig(dbscan_max_points=5000)
    scan = np.vstack(
        [_ground(30_000), _cone_cluster(5.0, 1.0), _low_dense_terrain(60_000, 10_000.0)]
    ).astype(np.float32)
    st: dict = {}
    clustering_separation_rt(
        scan, cfg, clustering_class=_CapturingDBSCAN, stage_timings=st
    )
    assert _CapturingDBSCAN.seen and max(_CapturingDBSCAN.seen) <= 5000
    assert st["n_dbscan_guard_dropped"] > 40_000


def test_guard_untouched_on_normal_scene_and_counter_zero():
    scan = np.vstack([_ground(), _cone_cluster(5.0, 1.0), _cone_cluster(8.0, -2.0)])
    scan = scan.astype(np.float32)
    counters: dict = {}
    st: dict = {}
    cones = RealtimeConeDetector(ConeDetectionConfig()).detect(
        scan, debug_counters=counters, stage_timings=st
    )
    assert st["n_dbscan_guard_dropped"] == 0
    assert counters["dbscan_guard_scans"] == 0
    assert counters["dbscan_guard_dropped"] == 0
    assert len(cones) == 2


def test_cone_still_detected_when_guard_fires():
    cfg = ConeDetectionConfig(dbscan_max_points=8000)
    cone = _cone_cluster(5.0, 1.0, n=60)
    # Terrain far from the cone so the clusters stay separable.
    terrain = _low_dense_terrain(40_000, 5000.0)
    terrain[:, 0] += 6.0  # push to x >= 12 m
    scan = np.vstack([_ground(30_000), cone, terrain]).astype(np.float32)
    counters: dict = {}
    cones = RealtimeConeDetector(cfg).detect(scan, debug_counters=counters)
    assert counters["dbscan_guard_scans"] == 1
    assert counters["dbscan_guard_dropped"] > 30_000
    near = [c for c in cones if math.hypot(c[0] - 5.0, c[1] - 1.0) < 0.3]
    assert len(near) == 1
