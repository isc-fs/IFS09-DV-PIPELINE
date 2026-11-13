"""The C++ backend (detector_backend="cpp") must equal the fused Numba backend.

Same scenes and the same bar as test_fused_detect.py: every counter and stage
count equal, plane to 1e-12, cones to 1e-9, published clouds to float32 rounding.
Kernel-level checks compare each C++ kernel with its Numba twin. Skipped when
the cone_detection_core module is not built.
"""
import dataclasses

import numba
import numpy as np
import pytest

core = pytest.importorskip("cone_detection_core")

from cone_detection import cpp_backend, fused  # noqa: E402
from cone_detection.clustering import _grid_components  # noqa: E402
from cone_detection.cone_detection import ConeDetectionConfig, RealtimeConeDetector  # noqa: E402
from cone_detection.ransac import sample_without_replacement  # noqa: E402
from test_fused_detect import _scene  # noqa: E402

RNG = np.random.default_rng(41)


@numba.njit
def _numba_sample(seed, n, k):
    np.random.seed(seed)
    return sample_without_replacement(n, k)


@numba.njit
def _numba_ransac(seed, xyz, prob, thr, max_iter, n_sub, warm):
    np.random.seed(seed)
    return fused.ransac_plane(xyz, prob, thr, max_iter, n_sub, warm)


def _run(cfg, scans, viz=True):
    det = RealtimeConeDetector(cfg)
    out = []
    for scan in scans:
        dc, st = {}, {}
        cones = det.detect(scan, debug_counters=dc, stage_timings=st, viz_full_cloud=viz)
        out.append(
            dict(
                cones=np.array(cones, dtype=float).reshape(-1, 4),
                counters=dc,
                stages={k: v for k, v in st.items() if not k.endswith("_ms")},
                outliers=det.last_outlier_xyz.copy(),
                rotated=det.last_rotated_xyz.copy(),
                plane=None if det._prev_plane is None else np.array(det._prev_plane, dtype=float),
            )
        )
    return det, out


def _assert_same(cfg, scans, viz=True):
    _, fu = _run(dataclasses.replace(cfg, detector_backend="fused"), scans, viz)
    det, cp = _run(dataclasses.replace(cfg, detector_backend="cpp"), scans, viz)
    assert det._cpp_core is not None  # really ran the C++ path
    for a, b in zip(fu, cp):
        assert a["counters"] == b["counters"]
        assert a["stages"] == b["stages"]
        if a["plane"] is None:
            assert b["plane"] is None
        else:
            np.testing.assert_allclose(b["plane"], a["plane"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(b["cones"], a["cones"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(b["outliers"], a["outliers"], rtol=0, atol=1e-5)
        np.testing.assert_allclose(b["rotated"], a["rotated"], rtol=0, atol=1e-5)
    return fu


@pytest.fixture(scope="module")
def scans():
    return [_scene(), _scene(wall=True), _scene(tilt=-0.03, height=-0.9)]


@pytest.mark.parametrize("backend", ["grid", "sklearn"])
def test_cpp_equals_fused(scans, backend):
    fu = _assert_same(ConeDetectionConfig(cluster_backend=backend), scans)
    assert all(r["counters"]["accepted"] >= 8 for r in fu)
    assert fu[1]["stages"]["n_vetoed"] > 1000


def test_cpp_equals_fused_without_viz_veto_early_exit_or_collinear(scans):
    cfg = ConeDetectionConfig(
        template_fit_early_exit=False, use_collinear_fit=False, tall_column_veto=False
    )
    _assert_same(cfg, scans, viz=False)


def test_cpp_equals_fused_on_float64_input(scans):
    _assert_same(ConeDetectionConfig(), [s.astype(np.float64) for s in scans[:2]])


def test_cpp_equals_fused_through_the_guard_and_its_random_fallback():
    cfg = ConeDetectionConfig(dbscan_max_points=1500)
    fu = _assert_same(cfg, [_scene(blob=6000), _scene(blob=400)])
    assert fu[0]["counters"]["dbscan_guard_dropped"] > 0
    # a cap the voxel passes cannot reach: the seeded numpy fallback runs
    _assert_same(ConeDetectionConfig(dbscan_max_points=50), [_scene(blob=6000)])


def test_cpp_handles_empty_and_tiny_scans():
    far = np.full((50, 3), 100.0, dtype=np.float32)
    tiny = np.array([[1.0, 0.0, 0.3], [1.0, 0.1, 0.3]], dtype=np.float32)
    _assert_same(ConeDetectionConfig(), [far, tiny, np.zeros((0, 3), dtype=np.float32)])


def test_cpp_without_per_scan_seed_is_valid(scans):
    # Its own RNG stream: not the fused one, but a correct ground plane.
    cfg = ConeDetectionConfig(detector_backend="cpp", ransac_seed_per_scan=False)
    det, out = _run(cfg, scans[:1])
    assert out[0]["counters"]["accepted"] >= 8


def test_unbuilt_module_falls_back_to_fused(scans, monkeypatch):
    monkeypatch.setattr(cpp_backend, "_core", None)
    det, cp = _run(ConeDetectionConfig(detector_backend="cpp"), scans[:1])
    _, fu = _run(ConeDetectionConfig(detector_backend="fused"), scans[:1])
    assert det._cpp_core is None
    np.testing.assert_array_equal(cp[0]["cones"], fu[0]["cones"])


# --------------------------------------------------------------------------
# kernels against their Numba twins
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 12345, 2**32 - 1])
def test_sampler_matches_numba(seed):
    np.testing.assert_array_equal(
        core.sample_without_replacement(seed, 174000, 5000), _numba_sample(seed, 174000, 5000)
    )


@pytest.mark.parametrize("warm", [None, np.array([1.0, 0.02, 0.0, 1.0])])
def test_ransac_matches_numba(scans, warm):
    xyz = scans[1].astype(np.float64)
    for seed in (3, 77, 90210):
        w = np.zeros(0) if warm is None else warm
        o_nb, c_nb = _numba_ransac(seed, xyz, 0.9999, 0.05, 200, 5000, w)
        o_cp, c_cp = core.ransac_plane(xyz, 0.9999, 0.05, 200, 5000, warm, seed)
        np.testing.assert_array_equal(o_cp, o_nb)
        np.testing.assert_allclose(c_cp, c_nb, rtol=0, atol=1e-15)


def test_grid_dbscan_matches_numba():
    for n in (1, 2, 500, 20000):
        pts = RNG.uniform(-10, 10, (n, 3)) * [1, 1, 0.05]
        np.testing.assert_array_equal(core.grid_dbscan(pts, 0.3), _grid_components(pts, 0.3))


def test_veto_and_voxel_match_numba():
    pts = np.column_stack(
        [RNG.uniform(-20, 20, 4000), RNG.uniform(-20, 20, 4000), RNG.uniform(0, 1.0, 4000)]
    )
    out_nb, n_nb = fused.column_veto(pts, 0.75, 0.3)
    out_cp, n_cp = core.column_veto(pts, 0.75, 0.3)
    np.testing.assert_array_equal(out_cp, out_nb)
    assert n_cp == n_nb
    for voxel in (0.02, 0.08):
        np.testing.assert_array_equal(
            core.voxel_first(pts, voxel, fused._GUARD_KEY_OFFSET, fused._GUARD_KEY_BITS),
            fused.voxel_first(pts, voxel, fused._GUARD_KEY_OFFSET, fused._GUARD_KEY_BITS),
        )


def test_fit_clusters_matches_numba(scans):
    cfg = ConeDetectionConfig()
    det = RealtimeConeDetector(dataclasses.replace(cfg, detector_backend="fused"))
    det.detect(scans[0])
    pts = np.ascontiguousarray(det.last_outlier_xyz, dtype=np.float64)
    labels = _grid_components(pts, cfg.dbscan_eps)
    p = fused.fit_params(cfg)
    rows_nb, cnt_nb = fused.fit_clusters(pts, labels, p)
    rows_cp, cnt_cp = core.fit_clusters(pts, labels, p.tolist())
    np.testing.assert_array_equal(cnt_cp, cnt_nb)
    np.testing.assert_allclose(rows_cp, rows_nb, rtol=0, atol=1e-12)
