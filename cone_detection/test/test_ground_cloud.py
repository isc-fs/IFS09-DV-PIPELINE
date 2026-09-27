"""Ground-plane viz cloud: same RANSAC rotation as /Conos_raw, floor at z=0."""

import numpy as np

from cone_detection.cone_detection import (
    ConeDetectionConfig,
    RealtimeConeDetector,
    ground_rotation_matrix,
    rotate_xyz_to_ground,
)


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


def test_detect_fills_last_rotated_xyz_aligned_with_cones():
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
    cones = det.detect(scan, viz_full_cloud=True)
    assert len(det.last_rotated_xyz) > 100
    # Ground band should sit near z=0 after the shift.
    z = det.last_rotated_xyz[:, 2]
    assert float(np.median(z)) < 0.08
    assert cones, "expected at least one cone on the synthetic cluster"
    # Cone XY in the rotated frame should land near the rotated cluster.
    ax, ay = cones[0][0], cones[0][1]
    d = np.hypot(det.last_rotated_xyz[:, 0] - ax, det.last_rotated_xyz[:, 1] - ay)
    assert float(d.min()) < 0.35
    # RANSAC outliers are a strict subset and still contain the cone.
    assert 0 < len(det.last_outlier_xyz) < len(det.last_rotated_xyz)
    d_out = np.hypot(
        det.last_outlier_xyz[:, 0] - ax, det.last_outlier_xyz[:, 1] - ay
    )
    assert float(d_out.min()) < 0.35


def test_full_ground_cloud_skipped_unless_requested():
    det = RealtimeConeDetector(ConeDetectionConfig(input_range_crop_m=20.0))
    rng = np.random.default_rng(1)
    scan = rng.normal(size=(200, 3)).astype(np.float32)
    scan[:, 2] *= 0.01
    det.detect(scan)
    assert len(det.last_rotated_xyz) == 0
    det.detect(scan, viz_full_cloud=True)
    assert len(det.last_rotated_xyz) > 0


def test_ground_rotation_matrix_is_orthonormal():
    R = ground_rotation_matrix(np.array([0.0, 0.2, 0.1, -1.0]))
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-6
