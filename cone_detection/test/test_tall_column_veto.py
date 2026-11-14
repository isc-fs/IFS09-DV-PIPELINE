"""Tests for the tall-column veto (tire-wall DBSCAN spike fix).

Tire-wall returns span ~0.04-1.15 m above ground, so a plain z-crop above cone
height leaves the wall's cone-height low band as DBSCAN input (and potential
truncated-stump false positives). The veto instead drops every 0.3 m xy-column
whose 3x3-dilated cell contains any return above tall_column_veto_height_m —
removing the wall wholesale while leaving free-standing cones untouched.
Measured on bag sim_benchmark_20260527_135117: 43-94k DBSCAN points -> 380-760,
identical accepted cones.
"""
import math

import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.detector import RealtimeConeDetector
from cone_detection.clustering.prefilters import column_veto_mask
from cone_detection.fitting.cone_classes import CONE_SMALL_C, CONE_SMALL_D

RNG = np.random.default_rng(7)


def _ground(n: int = 8000) -> np.ndarray:
    x = RNG.uniform(0.5, 20.0, n)
    y = RNG.uniform(-8.0, 8.0, n)
    z = RNG.normal(0.0, 0.008, n)
    return np.column_stack([x, y, z])


def _cone_cluster(cx: float, cy: float, n: int = 30) -> np.ndarray:
    """One-sided near-face small-cone cluster (as in warmup_numba_functions)."""
    z = RNG.uniform(0.06, CONE_SMALL_D - 0.03, size=n)
    gamma = (CONE_SMALL_D - z) / CONE_SMALL_C
    norm = math.hypot(cx, cy)
    ux, uy = cx / norm, cy / norm
    psi = RNG.uniform(-1.0, 1.0, size=n)
    px = cx - gamma * (np.cos(psi) * ux - np.sin(psi) * uy)
    py = cy - gamma * (np.cos(psi) * uy + np.sin(psi) * ux)
    px += RNG.normal(0.0, 0.003, size=n)
    py += RNG.normal(0.0, 0.003, size=n)
    z = z + RNG.normal(0.0, 0.008, size=n)
    return np.column_stack([px, py, z])


def _wall(x0: float = 6.0, x1: float = 14.0, y: float = 5.0, n: int = 2500) -> np.ndarray:
    """Vertical slab with contiguous returns from near-ground to 1.15 m."""
    x = RNG.uniform(x0, x1, n)
    z = RNG.uniform(0.05, 1.15, n)
    yy = y + RNG.normal(0.0, 0.03, n)
    return np.column_stack([x, yy, z])


def test_mask_is_none_when_nothing_tall():
    cfg = ConeDetectionConfig()
    xy = RNG.uniform(-5.0, 5.0, (500, 2))
    h = RNG.uniform(0.0, 0.5, 500)
    assert column_veto_mask(xy, h, cfg) is None


def test_mask_drops_whole_column_including_low_band():
    cfg = ConeDetectionConfig()
    # One column inside cell (0, 0): nine cone-height points plus one tall one.
    col_xy = RNG.uniform(0.01, 0.09, (10, 2))
    col_h = np.full(10, 0.15)
    col_h[3] = 1.0
    # A cone-height point one cell over (dilation reach) and one 2 m away.
    near_xy = np.array([[0.35, 0.05]])
    far_xy = np.array([[2.0, 2.0]])
    xy = np.vstack([col_xy, near_xy, far_xy])
    h = np.concatenate([col_h, [0.2], [0.2]])
    keep = column_veto_mask(xy, h, cfg)
    assert keep is not None
    assert not keep[:10].any()  # entire column vetoed, low band included
    assert not keep[10]  # adjacent cell caught by the 3x3 dilation
    assert keep[11]  # distant point untouched


def test_veto_removes_wall_before_dbscan():
    scan = np.vstack([_ground(), _cone_cluster(5.0, 1.0), _wall()]).astype(np.float32)
    st_on: dict = {}
    st_off: dict = {}
    pts_on = RealtimeConeDetector(ConeDetectionConfig()).detect(
        scan, stage_timings=st_on
    ).segmentation.above_ground_xyz
    pts_off = RealtimeConeDetector(ConeDetectionConfig(tall_column_veto=False)).detect(
        scan, stage_timings=st_off
    ).segmentation.above_ground_xyz
    assert st_on["n_vetoed"] > 2000  # the wall went away
    assert "n_vetoed" not in st_off
    assert len(pts_off) - len(pts_on) > 2000
    # The cone cluster survives the veto.
    near_cone = np.hypot(pts_on[:, 0] - 5.0, pts_on[:, 1] - 1.0) < 0.3
    assert near_cone.sum() >= 20


def test_detect_same_cone_with_and_without_wall():
    cone = _cone_cluster(5.0, 1.0)
    base = np.vstack([_ground(), cone]).astype(np.float32)
    walled = np.vstack([_ground(), cone, _wall()]).astype(np.float32)

    cones_base = RealtimeConeDetector(ConeDetectionConfig()).detect(base).cones
    st: dict = {}
    cones_wall = RealtimeConeDetector(ConeDetectionConfig()).detect(
        walled, stage_timings=st
    ).cones

    assert st["n_vetoed"] > 2000
    assert len(cones_base) == len(cones_wall) == 1
    bx, by = cones_base[0][0], cones_base[0][1]
    wx, wy = cones_wall[0][0], cones_wall[0][1]
    assert math.hypot(wx - bx, wy - by) < 0.10
    assert math.hypot(wx - 5.0, wy - 1.0) < 0.30
