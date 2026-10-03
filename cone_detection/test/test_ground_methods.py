"""Every ground_method keeps the (outlier_mask, plane) contract on a known scene."""

import numpy as np
import pytest

from cone_detection.config import ConeDetectionConfig
from cone_detection.ground_methods import GROUND_METHODS, remove_ground


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


def test_registry_lists_expected_methods():
    assert {"ransac", "gpf", "grid_resid", "ray_slope", "czm", "ransac_zones", "ransac_rings"} <= set(GROUND_METHODS)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown ground_method"):
        remove_ground("nope", np.zeros((10, 3)), ConeDetectionConfig())


@pytest.mark.parametrize("method", GROUND_METHODS)
def test_method_separates_ground_from_cones_and_wall(method):
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


@pytest.mark.parametrize("method", GROUND_METHODS)
def test_degenerate_clouds_do_not_crash(method):
    cfg = ConeDetectionConfig(ground_method=method)
    for pts in (np.zeros((0, 3)), np.array([[1.0, 0.0, -1.0]]), np.random.default_rng(1).normal(size=(5, 3))):
        outlier, plane = remove_ground(method, pts, cfg)
        assert len(outlier) == len(pts) and plane.shape == (4,)


def test_flat_cell_filter_drops_sidewalk_keeps_cone():
    from cone_detection.cone_detection import _flat_cell_mask

    rng = np.random.default_rng(0)
    cfg = ConeDetectionConfig(flat_cell_filter=True)
    sidewalk = np.c_[rng.uniform(2, 6, 3000), rng.uniform(3, 4, 3000), 0.12 + rng.normal(0, 0.005, 3000)]
    th = rng.uniform(0, 2 * np.pi, 60)
    r = rng.uniform(0.01, 0.11, 60)
    cone = np.c_[10 + r * np.cos(th), r * np.sin(th), 0.3 * (1 - r / 0.12)]
    pts = np.vstack([sidewalk, cone])
    keep = _flat_cell_mask(pts[:, :2], pts[:, 2], cfg)
    assert keep[:3000].mean() < 0.02          # sheet gone
    assert keep[3000:].mean() > 0.95          # cone kept


def test_flat_cell_mask_none_when_nothing_flat():
    from cone_detection.cone_detection import _flat_cell_mask

    pts = np.random.default_rng(1).uniform([0, 0, 0], [30, 30, 1], (200, 3))
    assert _flat_cell_mask(pts[:, :2], pts[:, 2], ConeDetectionConfig()) is None
