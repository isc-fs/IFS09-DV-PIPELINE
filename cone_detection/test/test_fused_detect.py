"""The fused Numba backend (detector_backend="fused") must equal the Python path.

Both draw RANSAC samples from the Numba RNG, so seeding it before each run makes
the two detectors see the same plane; everything downstream is deterministic.
Cones must match to 1e-9 and every counter, stage count, published cloud and the
warm-start plane exactly (to float rounding).
"""
import dataclasses
import logging
import math

import numba
import numpy as np
import pytest

from cone_detection import fused
from cone_detection.cone_detection import (
    ConeDetectionConfig,
    RealtimeConeDetector,
    _tall_column_veto_mask,
    _voxel_first_points,
)
from cone_detection.cone_fit import _CONE_BIG_C, _CONE_BIG_D, _CONE_SMALL_C, _CONE_SMALL_D

RNG = np.random.default_rng(23)


@numba.njit
def _seed_numba(seed):
    np.random.seed(seed)


def _cone(cx, cy, c, d, n=30):
    """One-sided near-face cluster, z above ground (as in warmup_numba_functions)."""
    z = RNG.uniform(0.06, d - 0.03, size=n)
    gamma = (d - z) / c
    norm = math.hypot(cx, cy)
    ux, uy = cx / norm, cy / norm
    psi = RNG.uniform(-1.0, 1.0, size=n)
    px = cx - gamma * (np.cos(psi) * ux - np.sin(psi) * uy)
    py = cy - gamma * (np.cos(psi) * uy + np.sin(psi) * ux)
    px += RNG.normal(0.0, 0.003, size=n)
    py += RNG.normal(0.0, 0.003, size=n)
    z = z + RNG.normal(0.0, 0.008, size=n)
    return np.column_stack([px, py, z])


def _scene(*, wall=False, blob=0, tilt=0.02, height=-1.05, n_ground=12000):
    """Tilted ground at the sensor's mounting height, cones, optional wall/blob."""
    x = RNG.uniform(-25.0, 25.0, n_ground)
    y = RNG.uniform(-25.0, 25.0, n_ground)
    parts = [np.column_stack([x, y, RNG.normal(0.0, 0.01, n_ground)])]
    for k in range(12):
        ang = 2 * math.pi * k / 12
        r = 3.0 + 1.4 * k
        if k % 3 == 2:
            parts.append(_cone(r * math.cos(ang), r * math.sin(ang), _CONE_BIG_C, _CONE_BIG_D))
        else:
            parts.append(_cone(r * math.cos(ang), r * math.sin(ang), _CONE_SMALL_C, _CONE_SMALL_D))
    # a line of collinear cones along a ray, and a far cluster for the range gate
    for k in range(4):
        parts.append(_cone(5.0 + 0.9 * k, 0.05, _CONE_SMALL_C, _CONE_SMALL_D, n=8))
    parts.append(_cone(22.0, 3.0, _CONE_SMALL_C, _CONE_SMALL_D))
    if wall:
        xw = RNG.uniform(6.0, 14.0, 3000)
        parts.append(
            np.column_stack([xw, 5.0 + RNG.normal(0.0, 0.03, 3000), RNG.uniform(0.05, 1.15, 3000)])
        )
    if blob:
        parts.append(RNG.normal([-4.0, -4.0, 0.3], [0.4, 0.4, 0.06], (blob, 3)))
    pts = np.vstack(parts)
    # tilt the ground and lift the sensor; add no-return points at the origin
    pts[:, 2] += tilt * pts[:, 0] + height
    pts = np.vstack([pts, np.zeros((200, 3))])
    return pts[RNG.permutation(len(pts))].astype(np.float32)


def _run(cfg, scans, **kw):
    det = RealtimeConeDetector(cfg)
    out = []
    for k, scan in enumerate(scans):
        _seed_numba(1000 + k)
        dc, st = {}, {}
        cones = det.detect(scan, debug_counters=dc, stage_timings=st, viz_full_cloud=True, **kw)
        out.append(
            dict(
                cones=np.array(cones, dtype=float).reshape(-1, 4),
                counters=dc,
                stages={k: v for k, v in st.items() if not k.endswith("_ms")},
                outliers=det.last_outlier_xyz.copy(),
                rotated=det.last_rotated_xyz.copy(),
                plane=np.array(det._prev_plane, dtype=float),
            )
        )
    return out


def _assert_same(cfg, scans, **kw):
    py = _run(dataclasses.replace(cfg, detector_backend="python"), scans, **kw)
    fu = _run(dataclasses.replace(cfg, detector_backend="fused"), scans, **kw)
    for a, b in zip(py, fu):
        assert a["counters"] == b["counters"]
        assert a["stages"] == b["stages"]
        np.testing.assert_allclose(b["plane"], a["plane"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(b["cones"], a["cones"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(b["outliers"], a["outliers"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(b["rotated"], a["rotated"], rtol=0, atol=1e-5)
    return py


@pytest.fixture(scope="module")
def scans():
    return [_scene(), _scene(wall=True), _scene(tilt=-0.03, height=-0.9)]


def _has_pypi_dbscan():
    try:
        import dbscan  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize(
    "backend",
    [
        "grid",
        "sklearn",
        pytest.param(
            "pypi",
            marks=pytest.mark.skipif(not _has_pypi_dbscan(), reason="dbscan not installed"),
        ),
    ],
)
def test_fused_equals_python(scans, backend):
    py = _assert_same(ConeDetectionConfig(cluster_backend=backend), scans)
    # the scenes really exercise the gates
    assert all(r["counters"]["accepted"] >= 8 for r in py)
    assert py[1]["stages"]["n_vetoed"] > 1000  # the wall
    assert any(r["counters"]["far_dropped"] for r in py)


def test_fused_equals_python_without_early_exit_or_collinear(scans):
    cfg = ConeDetectionConfig(
        cluster_backend="grid", template_fit_early_exit=False, use_collinear_fit=False
    )
    _assert_same(cfg, scans)


def test_fused_equals_python_through_the_guard():
    cfg = ConeDetectionConfig(cluster_backend="grid", dbscan_max_points=1500)
    scans = [_scene(blob=6000), _scene(blob=400)]
    py = _assert_same(cfg, scans)
    assert py[0]["counters"]["dbscan_guard_dropped"] > 0


def test_fused_handles_empty_and_tiny_scans():
    cfg = ConeDetectionConfig(cluster_backend="grid")
    far = np.full((50, 3), 100.0, dtype=np.float32)  # all cropped
    tiny = np.array([[1.0, 0.0, 0.3], [1.0, 0.1, 0.3]], dtype=np.float32)
    _assert_same(cfg, [far, tiny, np.zeros((0, 3), dtype=np.float32)])


def test_unsupported_config_falls_back_to_python(scans, caplog):
    cfg = ConeDetectionConfig(cluster_backend="grid", ground_method="gpf")
    assert fused.fused_unsupported_reason(cfg, scans[0], None) is not None
    with caplog.at_level(logging.WARNING):
        _assert_same(cfg, scans[:1])
    assert "detector_backend=fused" in caplog.text


def test_column_veto_matches_reference():
    cfg = ConeDetectionConfig()
    for _ in range(5):
        pts = np.column_stack(
            [RNG.uniform(-20, 20, 4000), RNG.uniform(-20, 20, 4000), RNG.uniform(0, 1.0, 4000)]
        )
        keep = _tall_column_veto_mask(pts[:, :2], pts[:, 2], cfg)
        out, n_vetoed = fused.column_veto(
            pts, cfg.tall_column_veto_height_m, cfg.tall_column_veto_cell_m
        )
        np.testing.assert_array_equal(out, pts[keep])
        assert n_vetoed == len(pts) - keep.sum()
    low = pts.copy()
    low[:, 2] = 0.1
    out, n_vetoed = fused.column_veto(low, cfg.tall_column_veto_height_m, cfg.tall_column_veto_cell_m)
    assert _tall_column_veto_mask(low[:, :2], low[:, 2], cfg) is None
    assert out is low or np.array_equal(out, low)
    assert n_vetoed == 0


@pytest.mark.parametrize("voxel", [0.02, 0.04, 0.08])
def test_voxel_first_matches_reference(voxel):
    pts = RNG.normal(0.0, 0.5, (20000, 3))
    idx = fused.voxel_first(pts, voxel, fused._GUARD_KEY_OFFSET, fused._GUARD_KEY_BITS)
    np.testing.assert_array_equal(pts[idx], _voxel_first_points(pts, voxel))


def _plain_run(cfg, scans):
    """Detect without seeding anything here: what the live node does."""
    det = RealtimeConeDetector(cfg)
    out = []
    for scan in scans:
        dc = {}
        cones = det.detect(scan, debug_counters=dc)
        out.append((np.array(cones, dtype=float).reshape(-1, 4), dc, det._prev_plane.copy()))
    return out


@pytest.mark.parametrize("backend", ["python", "fused"])
def test_per_scan_seed_makes_runs_reproducible(scans, backend):
    cfg = ConeDetectionConfig(cluster_backend="grid", detector_backend=backend)
    _seed_numba(1)
    a = _plain_run(cfg, scans)
    _seed_numba(2)  # a different RNG state, as in another process
    b = _plain_run(cfg, scans)
    for (ca, da, pa), (cb, db, pb) in zip(a, b):
        assert da == db
        np.testing.assert_array_equal(pa, pb)
        np.testing.assert_array_equal(ca, cb)


def test_python_and_fused_agree_without_external_seeding(scans):
    py = _plain_run(ConeDetectionConfig(cluster_backend="grid"), scans)
    _seed_numba(99)
    fu = _plain_run(ConeDetectionConfig(cluster_backend="grid", detector_backend="fused"), scans)
    for (ca, da, pa), (cb, db, pb) in zip(py, fu):
        assert da == db
        np.testing.assert_allclose(pb, pa, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cb, ca, rtol=0, atol=1e-9)


def test_scan_seed_depends_on_content():
    from cone_detection.ransac import scan_seed

    a = RNG.normal(size=(50000, 3)).astype(np.float32)
    assert scan_seed(a) == scan_seed(a.copy())
    b = a.copy()
    b[0, 0] += 1.0
    assert scan_seed(a) != scan_seed(b)
    assert scan_seed(a) != scan_seed(a[:-1])
    assert 0 <= scan_seed(a) < 2**32
