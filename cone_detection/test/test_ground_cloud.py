"""Ground-plane viz cloud: same ground-frame rotation as /Conos_raw, floor at z=0."""

import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.detector import RealtimeConeDetector
from cone_detection.ground.alignment import ground_rotation_matrix, rotate_xyz_to_ground
from cone_detection.fitting.cone_classes import CONE_SMALL_C, CONE_SMALL_D


def test_tilted_plane_lands_at_z0():
    # Plane z = 0.2 x  →  0.2 x - z = 0  →  coefs [bias, nx, ny, nz]
    coefs = np.array([0.0, 0.2, 0.0, -1.0])
    x = np.linspace(1.0, 10.0, 40)
    y = np.linspace(-2.0, 2.0, 40)
    xx, yy = np.meshgrid(x, y)
    zz = 0.2 * xx
    cloud = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    out = rotate_xyz_to_ground(cloud, coefs, max_points=0)
    assert out.shape[1] == 3
    assert float(np.std(out[:, 2])) < 1e-6
    assert abs(float(np.mean(out[:, 2]))) < 1e-6


def test_decimate_caps_count():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(20_000, 3))
    coefs = np.array([0.0, 0.0, 0.0, 1.0])
    out = rotate_xyz_to_ground(cloud, coefs, max_points=1500)
    assert len(out) <= 1500
    assert len(out) >= 1400


def test_ground_frame_scan_aligned_with_cones():
    rng = np.random.default_rng(3)
    n = 4000
    gx = rng.uniform(0.5, 12.0, n)
    gy = rng.uniform(-4.0, 4.0, n)
    # Tilted ground: z = 0.15 x, plus a cone standing on that plane.
    gz = 0.15 * gx + rng.normal(0.0, 0.005, n)
    ground = np.column_stack([gx, gy, gz])
    cx, cy = 6.0, 1.2
    cz0 = 0.15 * cx
    cone_z = rng.uniform(0.06, 0.30, 40)
    cone = np.column_stack(
        [
            np.full(40, cx) + rng.normal(0.0, 0.02, 40),
            np.full(40, cy) + rng.normal(0.0, 0.02, 40),
            cz0 + cone_z,
        ]
    )
    scan = np.vstack([ground, cone]).astype(np.float32)
    det = RealtimeConeDetector(ConeDetectionConfig(input_range_crop_m=20.0))
    res = det.detect(scan)
    cones = res.cones
    full = res.segmentation.ground_frame_scan()
    above = res.segmentation.above_ground_xyz
    assert len(full) > 100
    # Ground band should sit near z=0 after the shift.
    z = full[:, 2]
    assert float(np.median(z)) < 0.08
    assert cones, "expected at least one cone on the synthetic cluster"
    # Cone XY in the rotated frame should land near the rotated cluster.
    ax, ay = cones[0][0], cones[0][1]
    d = np.hypot(full[:, 0] - ax, full[:, 1] - ay)
    assert float(d.min()) < 0.35
    # Above-ground points are a strict subset and still contain the cone.
    assert 0 < len(above) < len(full)
    d_out = np.hypot(above[:, 0] - ax, above[:, 1] - ay)
    assert float(d_out.min()) < 0.35


def test_each_scan_result_keeps_its_own_clouds():
    """Results are per scan: a later detect() does not change an earlier result."""
    det = RealtimeConeDetector(ConeDetectionConfig(input_range_crop_m=20.0))
    rng = np.random.default_rng(1)
    scan = rng.normal(size=(200, 3)).astype(np.float32)
    scan[:, 2] *= 0.01
    first = det.detect(scan)
    n_first = len(first.segmentation.ground_frame_scan())
    assert n_first > 0
    empty = det.detect(np.zeros((0, 3), dtype=np.float32))
    assert len(empty.segmentation.ground_frame_scan()) == 0
    assert empty.segmentation.plane is None
    assert len(first.segmentation.ground_frame_scan()) == n_first


def test_ground_rotation_matrix_is_orthonormal():
    R = ground_rotation_matrix(np.array([0.0, 0.2, 0.1, -1.0]))
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_elevated_sensor_cone_fit_uses_height_above_ground():
    # Car-mounted LiDAR: sensor ~1.07 m above flat ground, so ground returns
    # sit at z = -1.07 in the sensor frame. The cone templates fix the apex
    # at d ABOVE GROUND; if z were left relative to the sensor the
    # fit would chase an apex ~1 m above the cone and misplace (a, b).
    rng = np.random.default_rng(7)
    sensor_h = 1.07
    n = 6000
    gx = rng.uniform(0.5, 15.0, n)
    gy = rng.uniform(-5.0, 5.0, n)
    ground = np.column_stack([gx, gy, rng.normal(-sensor_h, 0.005, n)])

    # Near face of a small cone (template shape, z = d - c r above ground), as a
    # LiDAR sees it: an arc of the surface facing the sensor.
    cx, cy, c_small, d_small = 7.0, 1.5, CONE_SMALL_C, CONE_SMALL_D
    m = 60
    h = rng.uniform(0.06, 0.30, m)
    r = (d_small - h) / c_small
    ux, uy = np.array([cx, cy]) / np.hypot(cx, cy)
    psi = rng.uniform(-1.0, 1.0, m)
    px = cx - r * (np.cos(psi) * ux - np.sin(psi) * uy)
    py = cy - r * (np.cos(psi) * uy + np.sin(psi) * ux)
    cone = np.column_stack([px, py, h - sensor_h])

    scan = np.vstack([ground, cone]).astype(np.float32)
    det = RealtimeConeDetector(ConeDetectionConfig())
    res = det.detect(scan)
    cones = res.cones

    # Points handed to clustering are expressed as height above ground.
    assert float(res.segmentation.above_ground_xyz[:, 2].min()) > 0.0
    assert len(cones) == 1
    ax, ay = cones[0][0], cones[0][1]
    assert np.hypot(ax - cx, ay - cy) < 0.05


def test_no_return_points_at_origin_are_dropped_on_input():
    # The Hesai driver encodes "no return" as (0, 0, 0). With the sensor 1 m
    # up, those points would sit ~1 m above the fitted ground after the shift
    # and get swallowed by the tall-column veto; they must never get that far.
    rng = np.random.default_rng(11)
    sensor_h = 1.07
    n = 4000
    ground = np.column_stack(
        [rng.uniform(0.5, 12.0, n), rng.uniform(-4.0, 4.0, n), rng.normal(-sensor_h, 0.005, n)]
    )
    no_return = np.zeros((5000, 3))
    scan = np.vstack([ground, no_return]).astype(np.float32)

    det = RealtimeConeDetector(ConeDetectionConfig())
    counters: dict = {}
    timings: dict = {}
    det.detect(scan, debug_counters=counters, stage_timings=timings)

    assert counters["n_input_points"] == n
    assert timings.get("n_vetoed", 0.0) == 0.0
