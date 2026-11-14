"""Every ground_method keeps the (non_ground_mask, plane) contract on a known scene."""

import numpy as np
import pytest

from cone_detection.config import ConeDetectionConfig
from cone_detection.ground.methods import remove_ground, GROUND_METHODS


def _scene(seed=0, tilt=0.03):
    """Tilted floor (z = -1.1 + tilt*x) + 6 cones + a 1 m tall wall of returns."""
    rng = np.random.default_rng(seed)
    n = 30_000
    x = rng.uniform(-20, 20, n)
    y = rng.uniform(-20, 20, n)
    z = -1.1 + tilt * x + rng.normal(0, 0.008, n)
    ground = np.c_[x, y, z]
    cones = []
    for cx, cy in [(5, 1), (7, -2), (10, 3), (12, -3), (15, 0), (3, -1.5)]:
        r = rng.uniform(0.01, 0.12, 60)
        th = rng.uniform(0, 2 * np.pi, 60)
        h = 0.3 * (1 - r / 0.12)
        cones.append(np.c_[cx + r * np.cos(th), cy + r * np.sin(th), -1.1 + tilt * cx + 0.04 + h])
    wall = np.c_[rng.uniform(-3, 3, 800), np.full(800, 8.0), -1.1 + rng.uniform(0.0, 1.0, 800)]
    cones = np.vstack(cones)
    return ground, cones, wall


# Methods backed by optional packages; their tests skip when the package is missing.
_OPTIONAL = {"csf": "CSF", "patchworkpp": "pypatchworkpp"}


def _need(method):
    if method in _OPTIONAL:
        pytest.importorskip(_OPTIONAL[method])


def test_registry_lists_expected_methods():
    assert {
        "ransac", "gpf", "grid_resid", "ray_slope", "czm", "ransac_zones", "ransac_rings",
        "height_jump", "linefit", "range_image", "groundgrid", "pmf", "gp_insac", "mrf", "csf",
        "patchworkpp",
    } <= set(GROUND_METHODS)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown ground_method"):
        remove_ground("nope", np.zeros((10, 3)), ConeDetectionConfig())


@pytest.mark.parametrize("method", GROUND_METHODS)
def test_method_separates_ground_from_cones_and_wall(method):
    _need(method)
    if method == "range_image":
        pytest.skip("needs laser rows; covered by test_method_on_raycast_scan")
    ground, cones, wall = _scene()
    data = np.vstack([ground, cones, wall])
    cfg = ConeDetectionConfig(ground_method=method)
    outlier, plane = remove_ground(method, data, cfg)

    assert outlier.shape == (len(data),) and outlier.dtype == bool
    assert plane.shape == (4,)
    assert plane[3] > 0 and abs(np.linalg.norm(plane[1:]) - 1) < 1e-6

    n_g, n_c = len(ground), len(cones)
    ground_kept = 1 - outlier[:n_g].mean()          # ground labelled ground
    cones_found = outlier[n_g:n_g + n_c].mean()     # cone returns labelled non-ground
    wall_found = outlier[n_g + n_c:].mean()
    assert ground_kept > 0.95, (method, ground_kept)
    assert cones_found > 0.6, (method, cones_found)
    assert wall_found > 0.5, (method, wall_found)
    # tilt 0.03 rad plane: normal_x ~ -0.03
    assert abs(plane[1] + 0.03) < 0.02, (method, plane)


def _scan_scene(tilt=0.03, n_channels=64, n_az=1800):
    """Ray-cast spinning-LiDAR scan of the same floor, cones and wall.

    Returns ``(points, kind)`` with kind 0 = floor, 1 = cone, 2 = wall. Unlike
    ``_scene`` the returns lie on laser rows and objects occlude the floor,
    which the range-image method needs.
    """
    el = np.radians(np.linspace(-25.0, 2.0, n_channels))
    az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
    E, A = np.meshgrid(el, az, indexing="ij")
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)
    # Floor z = -1.1 + tilt * x: t * (dz - tilt * dx) = -1.1
    den = d[:, 2] - tilt * d[:, 0]
    with np.errstate(divide="ignore"):
        t = np.where(den < -1e-6, -1.1 / den, np.inf)
    kind = np.zeros(len(d), dtype=np.int64)
    t[t > 30] = np.inf

    def hit_cylinder(cx, cy, rad, z0, z1, k):
        dx, dy = d[:, 0], d[:, 1]
        a = dx * dx + dy * dy
        b = -2 * (dx * cx + dy * cy)
        c = cx * cx + cy * cy - rad * rad
        disc = b * b - 4 * a * c
        ok = disc > 0
        tc = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * a), np.inf)
        z = tc * d[:, 2]
        tc[(tc <= 0) | (z < z0) | (z > z1)] = np.inf
        closer = tc < t
        t[closer] = tc[closer]
        kind[closer] = k

    for cx, cy in [(5, 1), (7, -2), (10, 3), (12, -3), (15, 0), (3, -1.5)]:
        z0 = -1.1 + tilt * cx
        hit_cylinder(cx, cy, 0.08, z0, z0 + 0.3, 1)
    # Wall: a row of thin posts at y = 8, x in [-3, 3], 1 m tall.
    for wx in np.arange(-3, 3.01, 0.1):
        hit_cylinder(wx, 8.0, 0.06, -1.1 + tilt * wx, -1.1 + tilt * wx + 1.0, 2)
    keep = np.isfinite(t)
    pts = d[keep] * t[keep, None]
    pts += np.random.default_rng(0).normal(0, 0.008, pts.shape)
    return pts, kind[keep]


@pytest.mark.parametrize("method", GROUND_METHODS)
def test_method_on_raycast_scan(method):
    _need(method)
    pts, kind = _scan_scene()
    outlier, plane = remove_ground(method, pts, ConeDetectionConfig(ground_method=method))
    ground_kept = 1 - outlier[kind == 0].mean()
    cones_found = outlier[kind == 1].mean()
    wall_found = outlier[kind == 2].mean()
    assert ground_kept > 0.9, (method, ground_kept)
    assert cones_found > 0.6, (method, cones_found)
    assert wall_found > 0.5, (method, wall_found)
    assert abs(plane[1] + 0.03) < 0.02, (method, plane)


@pytest.mark.parametrize("method", GROUND_METHODS)
def test_degenerate_clouds_do_not_crash(method):
    _need(method)
    cfg = ConeDetectionConfig(ground_method=method)
    for pts in (np.zeros((0, 3)), np.array([[1.0, 0.0, -1.0]]), np.random.default_rng(1).normal(size=(5, 3))):
        outlier, plane = remove_ground(method, pts, cfg)
        assert len(outlier) == len(pts) and plane.shape == (4,)


def test_flat_cell_filter_drops_sidewalk_keeps_cone():
    from cone_detection.clustering.prefilters import flat_cell_mask

    rng = np.random.default_rng(0)
    cfg = ConeDetectionConfig(flat_cell_filter=True)
    sidewalk = np.c_[rng.uniform(2, 6, 3000), rng.uniform(3, 4, 3000), 0.12 + rng.normal(0, 0.005, 3000)]
    th = rng.uniform(0, 2 * np.pi, 60)
    r = rng.uniform(0.01, 0.11, 60)
    cone = np.c_[10 + r * np.cos(th), r * np.sin(th), 0.3 * (1 - r / 0.12)]
    pts = np.vstack([sidewalk, cone])
    keep = flat_cell_mask(pts[:, :2], pts[:, 2], cfg)
    assert keep[:3000].mean() < 0.02          # sheet gone
    assert keep[3000:].mean() > 0.95          # cone kept


def test_flat_cell_mask_none_when_nothing_flat():
    from cone_detection.clustering.prefilters import flat_cell_mask

    pts = np.random.default_rng(1).uniform([0, 0, 0], [30, 30, 1], (200, 3))
    assert flat_cell_mask(pts[:, :2], pts[:, 2], ConeDetectionConfig()) is None


def _mrf_energy(lab, cg, co, wd, wr):
    return (np.where(lab, cg, co).sum()
            + (wd[:-1, :] * (lab[:-1, :] != lab[1:, :])).sum()
            + (wr[:, :-1] * (lab[:, :-1] != lab[:, 1:])).sum())


@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (5, 1), (2, 2), (3, 3), (3, 4), (4, 3)])
def test_mrf_scipy_cut_is_optimal_by_brute_force(shape):
    """The scipy cut reaches the minimum energy over every labelling of a small grid,
    including empty cells (both unaries 0) and zero neighbour weights."""
    import itertools

    from cone_detection.ground.methods.mrf_methods import _grid_mincut

    rng = np.random.default_rng(sum(shape))
    n = shape[0] * shape[1]
    for _ in range(10):
        cg = rng.uniform(0, 3, shape)
        co = np.ones(shape)
        empty = rng.random(shape) < 0.3
        cg[empty] = co[empty] = 0.0
        wd = rng.uniform(0, 0.8, shape) * (rng.random(shape) > 0.2)
        wr = rng.uniform(0, 0.8, shape) * (rng.random(shape) > 0.2)
        best = min(
            _mrf_energy(np.array(bits, dtype=bool).reshape(shape), cg, co, wd, wr)
            for bits in itertools.product((False, True), repeat=n)
        )
        got = _mrf_energy(_grid_mincut(cg, co, wd, wr, solver="scipy"), cg, co, wd, wr)
        assert got == pytest.approx(best, abs=1e-4)


def test_mrf_scipy_cut_matches_pymaxflow():
    """Both min-cut backends reach the same optimal energy (ties may pick different cuts)."""
    pytest.importorskip("maxflow")
    from cone_detection.ground.methods.mrf_methods import _grid_mincut

    def energy(lab, cg, co, wd, wr):
        return (np.where(lab, cg, co).sum()
                + (wd[:-1, :] * (lab[:-1, :] != lab[1:, :])).sum()
                + (wr[:, :-1] * (lab[:, :-1] != lab[:, 1:])).sum())

    rng = np.random.default_rng(0)
    for _ in range(5):
        shape = (60, 80)
        args = (rng.uniform(0, 3, shape), np.ones(shape), rng.uniform(0, 0.8, shape), rng.uniform(0, 0.8, shape))
        a = _grid_mincut(*args, solver="pymaxflow")
        b = _grid_mincut(*args, solver="scipy")
        assert abs(energy(a, *args) - energy(b, *args)) < 1e-6 * energy(a, *args)


# --- plane-fit primitives: numba kernels vs the numpy formulation they replaced ---------

def _fit_plane_ref(points):
    c = points.mean(axis=0)
    u = points - c
    _, vecs = np.linalg.eigh(u.T @ u)
    n = vecs[:, 0]
    if n[2] < 0:
        n = -n
    return np.array([-float(n @ c), n[0], n[1], n[2]], dtype=np.float64)


def _gpf_plane_ref(pts, thr, n_iter=10, seed_band_m=0.12, seed_pct=2.0):
    from cone_detection.ground.methods.plane_fitting import _DEFAULT_PLANE

    lpr = np.percentile(pts[:, 2], seed_pct)
    seeds = pts[pts[:, 2] < lpr + seed_band_m]
    plane = _DEFAULT_PLANE.copy()
    for _ in range(n_iter):
        if len(seeds) < 3:
            break
        plane = _fit_plane_ref(seeds)
        grown = pts[np.abs(pts @ plane[1:] + plane[0]) < thr]
        converged = len(grown) == len(seeds)
        seeds = grown
        if converged:
            break
    return plane


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fit_plane_matches_numpy_reference(dtype):
    from cone_detection.ground.methods.plane_fitting import fit_plane

    ground, cones, wall = _scene(seed=2)
    pts = np.vstack([ground, cones, wall]).astype(dtype)
    for view in (pts, pts[::7], pts[:50]):  # contiguous and strided layouts
        got, want = fit_plane(view), _fit_plane_ref(view)
        np.testing.assert_allclose(got, want, atol=1e-6)


def test_fit_plane_keeps_precision_with_large_xy_extent():
    """cm-level height spread next to tens of metres of x / y must not cancel."""
    from cone_detection.ground.methods.plane_fitting import fit_plane

    rng = np.random.default_rng(0)
    xy = rng.uniform(-25, 25, (20_000, 2))
    z = -1.1 + 0.02 * xy[:, 0] + rng.normal(0, 0.005, 20_000)
    pts = np.c_[xy, z].astype(np.float32)
    got = fit_plane(pts)
    assert abs(got[1] + 0.02) < 1e-3 and got[3] > 0
    np.testing.assert_allclose(got, _fit_plane_ref(pts), atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gpf_plane_matches_gather_reference(seed):
    from cone_detection.ground.methods.plane_fitting import _gpf_plane, _subsample

    ground, cones, wall = _scene(seed=seed)
    data = np.vstack([ground, cones, wall]).astype(np.float32)
    pts = _subsample(data)
    got, want = _gpf_plane(pts, 0.05), _gpf_plane_ref(pts, 0.05)
    assert np.degrees(np.arccos(min(1.0, float(got[1:] @ want[1:])))) < 0.05
    assert abs(got[0] - want[0]) < 1e-3


def test_gpf_plane_tiny_clouds():
    from cone_detection.ground.methods.plane_fitting import _DEFAULT_PLANE, _gpf_plane

    np.testing.assert_array_equal(_gpf_plane(np.zeros((2, 3)), 0.05), _DEFAULT_PLANE)
    flat = np.c_[np.random.default_rng(0).uniform(0, 5, (20, 2)), np.zeros(20)]
    np.testing.assert_allclose(_gpf_plane(flat, 0.05)[1:], [0, 0, 1], atol=1e-9)


# --- MRF graph reduction ---------------------------------------------------------------

def _mrf_like_grid(rng, shape, p_empty=0.6, p_object=0.05):
    """Unaries like the real ones: ground cells cost ~0, objects cost a lot, empty cells 0 / 0."""
    cg = np.zeros(shape)
    co = np.ones(shape)
    obj = rng.random(shape) < p_object
    cg[obj] = rng.uniform(1, 10, obj.sum())
    cg[~obj] = rng.uniform(0, 0.3, (~obj).sum()) * (rng.random((~obj).sum()) < 0.3)
    empty = rng.random(shape) < p_empty
    cg[empty] = co[empty] = 0.0
    wd = 0.5 * np.exp(-rng.uniform(0, 3, shape) ** 2) + rng.uniform(0, 1e-3, shape)
    wr = 0.5 * np.exp(-rng.uniform(0, 3, shape) ** 2) + rng.uniform(0, 1e-3, shape)
    return cg, co, wd, wr, ~empty


@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (5, 1), (2, 2), (3, 3), (3, 4), (4, 3)])
def test_mrf_reduced_cut_is_optimal_by_brute_force(shape):
    """With every cell active, fixing provably-decided cells keeps the optimal energy."""
    import itertools

    from cone_detection.ground.methods.mrf_methods import _grid_mincut_reduced

    rng = np.random.default_rng(sum(shape) + 100)
    n = shape[0] * shape[1]
    for _ in range(10):
        cg = rng.uniform(0, 3, shape) * (rng.random(shape) < 0.5)
        co = np.ones(shape)
        empty = rng.random(shape) < 0.3
        cg[empty] = co[empty] = 0.0
        wd = rng.uniform(0, 0.8, shape) * (rng.random(shape) > 0.2)
        wr = rng.uniform(0, 0.8, shape) * (rng.random(shape) > 0.2)
        best = min(
            _mrf_energy(np.array(bits, dtype=bool).reshape(shape), cg, co, wd, wr)
            for bits in itertools.product((False, True), repeat=n)
        )
        got = _grid_mincut_reduced(cg, co, wd, wr, np.ones(shape, bool))
        assert _mrf_energy(got, cg, co, wd, wr) == pytest.approx(best, abs=1e-4)


def test_mrf_persistent_cells_are_fixed_in_every_optimal_labelling():
    """Each cell the proof fixes takes that label in all minimum-energy labellings."""
    import itertools

    from cone_detection.ground.methods.mrf_methods import _persistent_states

    rng = np.random.default_rng(7)
    shape = (3, 4)
    n = 12
    n_fixed = 0
    for _ in range(25):
        cg, co, wd, wr, has = _mrf_like_grid(rng, shape, p_empty=0.2, p_object=0.2)
        state = _persistent_states(cg, co, wd, wr, np.ones(shape, bool), 1e-4)
        labs = [np.array(b, bool).reshape(shape) for b in itertools.product((False, True), repeat=n)]
        en = np.array([_mrf_energy(lab, cg, co, wd, wr) for lab in labs])
        optimal = [lab for lab, e in zip(labs, en) if e <= en.min() + 1e-9]
        for lab in optimal:
            assert (lab[state == 1]).all() and not (lab[state == 0]).any()
        n_fixed += int(((state == 0) | (state == 1)).sum())
    assert n_fixed > 0  # the proof must actually decide something here


def test_mrf_exact_mode_matches_full_cut():
    from cone_detection.ground.methods.mrf_methods import _grid_mincut, _grid_mincut_reduced

    rng = np.random.default_rng(11)
    for _ in range(6):
        cg, co, wd, wr, has = _mrf_like_grid(rng, (40, 55))
        full = _grid_mincut(cg, co, wd, wr, solver="scipy")
        exact = _grid_mincut_reduced(cg, co, wd, wr, np.ones(cg.shape, bool))
        assert _mrf_energy(exact, cg, co, wd, wr) == pytest.approx(
            _mrf_energy(full, cg, co, wd, wr), rel=1e-6
        )
        assert (exact != full).sum() <= 2  # ties only


def test_mrf_reduced_graph_handles_fully_decided_and_empty_grids():
    from cone_detection.ground.methods.mrf_methods import _grid_mincut_reduced

    shape = (6, 7)
    flat = (np.zeros(shape), np.ones(shape), np.full(shape, 0.5), np.full(shape, 0.5))
    assert not _grid_mincut_reduced(*flat, np.zeros(shape, bool)).any()  # nothing active
    # an isolated flat patch: every cell is provably ground, so no flow is needed
    active = np.zeros(shape, bool)
    active[2:4, 2:5] = True
    got = _grid_mincut_reduced(*flat, active)
    assert (got == active).all()
    # all object
    obj = (np.full(shape, 9.0), np.ones(shape), np.full(shape, 0.5), np.full(shape, 0.5))
    assert not _grid_mincut_reduced(*obj, np.ones(shape, bool)).any()


def test_mrf_graph_modes_agree_on_a_scene(monkeypatch):
    import cone_detection.ground.methods as gm
    from cone_detection.ground.methods import mrf_methods

    ground, cones, wall = _scene(seed=3)
    data = np.vstack([ground, cones, wall])
    cfg = ConeDetectionConfig(ground_method="mrf")
    out = {}
    for mode in ("full", "exact", "reduced"):
        monkeypatch.setattr(mrf_methods, "_MRF_GRAPH", mode)
        out[mode] = gm.remove_ground("mrf", data, cfg)[0]
    assert (out["exact"] != out["full"]).mean() < 2e-4
    assert (out["reduced"] != out["full"]).mean() < 2e-3
    monkeypatch.setattr(mrf_methods, "_MRF_GRAPH", "bogus")
    with pytest.raises(ValueError, match="_MRF_GRAPH"):
        gm.remove_ground("mrf", data, cfg)


def test_mrf_warmup_compiles_every_kernel_a_real_scan_uses():
    """After the configure-time warm-up, a real scan must not trigger a JIT compile
    (it cost ~1 s on the first scan when the warm-up scene was fully decided)."""
    import numba

    import cone_detection.ground.methods as gm
    from cone_detection.detector import RealtimeConeDetector
    from cone_detection.warmup import warmup_numba_functions

    modules = [m for name, m in vars(gm).items() if type(m) is type(gm)]

    cfg = ConeDetectionConfig(ground_method="mrf")
    warmup_numba_functions(cfg)

    def n_signatures():
        return {
            f"{m.__name__}.{k}": len(v.signatures)
            for m in modules
            for k, v in vars(m).items()
            if isinstance(v, numba.core.registry.CPUDispatcher)
        }

    before = n_signatures()
    ground, cones, wall = _scene(seed=5)
    data = np.vstack([ground, cones, wall]).astype(np.float32)
    RealtimeConeDetector(cfg).detect(data)
    late = {k for k, n in n_signatures().items() if n != before[k]}
    assert not late, f"compiled on the first real scan: {sorted(late)}"


# --- compiled replacements must reproduce what they replaced -----------------


def _tilted_scene(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 20.0, n)
    y = rng.uniform(-8.0, 8.0, n)
    z = 0.05 * x - 0.02 * y - 1.0 + rng.normal(0.0, 0.01, n)
    z[: n // 20] += rng.uniform(0.2, 0.6, n // 20)  # objects
    return np.column_stack([x, y, z]).astype(np.float32)


def test_ransac_ground_mask_matches_ransac2_plane_test():
    """Same plane and the same vertical-distance labels as the ransac2 path."""
    from cone_detection.ground.methods.plane_methods import _ground_ransac_legacy, ground_ransac

    data = _tilted_scene()
    cfg = ConeDetectionConfig()
    out_new, plane_new = ground_ransac(data, cfg)
    out_old, plane_old = _ground_ransac_legacy(data, cfg, np.zeros(0), 5000)
    assert out_new.dtype == bool and plane_new[3] > 0
    assert abs(float(np.linalg.norm(plane_new[1:])) - 1.0) < 1e-9
    # Both are random searches; they must agree on (nearly) every label.
    assert (out_new != out_old).mean() < 0.02
    # The mask is exactly "vertical distance to the returned plane >= threshold".
    dz = np.abs(-(plane_new[0] + data[:, :2].astype(np.float64) @ plane_new[1:3]) / plane_new[3] - data[:, 2])
    assert np.array_equal(out_new, ~(dz < cfg.ground_threshold_m))
    # Warm start from the previous plane keeps the same ground.
    out_warm, _ = ground_ransac(data, cfg, warm_plane=plane_new)
    assert (out_warm != out_new).mean() < 0.02


def test_refit_on_ground_matches_gathered_fit():
    from cone_detection.ground.methods import plane_fitting as gm

    data = _tilted_scene(50_000)
    outlier = np.zeros(len(data), dtype=bool)
    outlier[::7] = True
    idx = np.flatnonzero(~outlier)
    idx = idx[:: int(np.ceil(len(idx) / gm._FIT_MAX_POINTS))]
    assert np.array_equal(
        gm._refit_on_ground(data, outlier, gm._DEFAULT_PLANE), gm.fit_plane(data[idx])
    )
    few = np.ones(len(data), dtype=bool)
    few[:2] = False
    assert gm._refit_on_ground(data, few, gm._DEFAULT_PLANE) is gm._DEFAULT_PLANE


def test_level_cells_matches_level_then_numpy_cells():
    from cone_detection.ground.methods import scan_geometry as gm

    data = _tilted_scene()
    uv, h, plane = gm._level(data, 0.05)
    ij = np.floor(uv / 0.3).astype(np.int64)
    ij -= ij.min(axis=0)
    nx, ny = (ij.max(axis=0) + 1).tolist()
    cell2, nx2, ny2, h2, plane2 = gm._level_cells(data, 0.05, 0.3)
    assert (nx, ny) == (nx2, ny2)
    np.testing.assert_array_equal(cell2, ij[:, 0] * ny + ij[:, 1])
    assert np.array_equal(h, h2) and np.array_equal(plane, plane2)


def test_grey_opening_matches_scipy():
    from scipy import ndimage

    from cone_detection.ground.methods.height_grid import _grey_opening

    rng = np.random.default_rng(0)
    for shape in [(1, 1), (2, 7), (5, 40), (17, 4), (60, 83)]:
        for w in (3, 5, 9, 17):
            g = rng.normal(size=shape)
            assert np.array_equal(_grey_opening(g, w), ndimage.grey_opening(g, size=(w, w)))


@pytest.mark.parametrize("method", ["ransac", "ransac_zones", "ransac_rings"])
def test_ransac_remover_owns_its_warm_start_plane(method):
    """The previous scan's plane lives on the remover: reset() drops it and two
    removers never share it."""
    from cone_detection.ground.methods import make_ground_remover

    data = _tilted_scene()
    cfg = ConeDetectionConfig(ground_method=method)
    a = make_ground_remover(cfg)
    b = make_ground_remover(cfg)
    inner_a, inner_b = a._inner, b._inner
    assert inner_a._prev_plane is None
    a.remove(data)
    assert inner_a._prev_plane is not None and inner_b._prev_plane is None
    a.reset()
    assert inner_a._prev_plane is None


def test_tiny_cloud_does_not_touch_remover_state():
    from cone_detection.ground.methods import make_ground_remover

    remover = make_ground_remover(ConeDetectionConfig(ground_method="ransac"))
    remover.remove(_tilted_scene())
    plane = remover._inner._prev_plane.copy()
    outlier, _ = remover.remove(np.zeros((2, 3), dtype=np.float32))
    assert outlier.all()
    assert np.array_equal(remover._inner._prev_plane, plane)


def test_stateless_methods_are_plain_functions():
    """Only methods that carry state between scans are classes."""
    from cone_detection.ground.methods import METHODS

    stateful = {name for name, obj in METHODS.items() if isinstance(obj, type)}
    assert stateful == {"ransac", "ransac_zones", "ransac_rings", "patchworkpp"}
