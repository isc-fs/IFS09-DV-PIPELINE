"""The ground-aligned frame ``/Conos_raw`` lives in: plane normal rotated onto +z,
fitted floor shifted to z=0 (so z is height above ground).
"""

from __future__ import annotations

import math

import numba
import numpy as np
from scipy.spatial.transform import Rotation as R


# Cap for the /lidar_points/ground viz cloud. Full cropped scans are still
# tens of thousands of returns; Foxglove chokes before the multiply does.
VIZ_GROUND_CLOUD_MAX_POINTS = 15_000


def ground_rotation_matrix(plane_coefs: np.ndarray) -> np.ndarray:
    """3×3 rotation that maps the ground plane normal onto +z.

    ``plane_coefs`` is ``[bias, n_x, n_y, n_z]`` from ``ransac2`` /
    ``separate_ground``. Degenerate normals return identity.
    """
    k = np.zeros(3, dtype=np.float64)
    k[-1] = 1.0
    normal, nrm = _plane_normal(plane_coefs)
    if nrm < 1e-12:
        return np.eye(3, dtype=np.float64)
    return vectors2matrix(k, normal / nrm)


def _plane_normal(plane_coefs: np.ndarray) -> tuple[np.ndarray, float]:
    """``(normal, |normal|)`` of ``[bias, n_x, n_y, n_z]``, as float64."""
    normal = np.asarray(plane_coefs[1:], dtype=np.float64)
    return normal, float(np.linalg.norm(normal))


def plane_floor_z(plane_coefs: np.ndarray) -> float:
    """Signed height of the LiDAR origin above the ground plane (0 for a degenerate normal)."""
    normal, nrm = _plane_normal(plane_coefs)
    if nrm < 1e-12:
        return 0.0
    return float(
        np.dot(np.array([0.0, 0.0, -1.0 * float(plane_coefs[0])]), normal) / nrm
    )


def rotate_xyz_to_ground(
    xyz: np.ndarray,
    plane_coefs: np.ndarray,
    *,
    max_points: int = VIZ_GROUND_CLOUD_MAX_POINTS,
    align_ground_z0: bool = True,
) -> np.ndarray:
    """Rotate a cloud into the same frame as ``/Conos_raw`` (ground ≈ z=0).

    Detection rotates only the non-ground points for DBSCAN. This helper
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
    out = pts[:, :3] @ ground_rotation_matrix(plane_coefs)
    if align_ground_z0:
        out[:, 2] -= plane_floor_z(plane_coefs)
    return np.ascontiguousarray(out, dtype=np.float32)


def rotate_non_ground_to_ground(
    data: np.ndarray, non_ground: np.ndarray, plane_coefs: np.ndarray
) -> np.ndarray:
    """``data[non_ground]`` rotated upright and shifted so the plane is z=0.

    Select the non-ground rows BEFORE rotating: row selection commutes with the
    right-multiplied rotation, and rotating only the ~1% above-ground
    points instead of the full cloud removes the dominant cost here.
    Shift so the fitted plane IS the xy plane: every consumer downstream
    (floor cull, column veto, template fit) reads z as height above ground.
    The templates fix the apex at d = 0.35 / 0.55 m above ground, so leaving
    the sensor at z=0 fits the cone ~1 m too high on the car (Hesai mounted
    ~1.07 m up); the sim LiDAR sits at ground level, which hid this.
    Select, rotate and shift happen in one pass over the cloud.
    """
    rot = ground_rotation_matrix(plane_coefs)
    floor_z = plane_floor_z(plane_coefs)
    if data.ndim == 2 and data.shape[1] == 3:
        return rotate_masked(data, np.asarray(non_ground, dtype=np.bool_), rot, floor_z)
    out = data[non_ground] @ rot
    out[:, 2] -= floor_z
    return out


@numba.njit(cache=True)
def rotate_masked(data, mask, rot, floor_z):
    """``data[mask] @ rot`` with ``floor_z`` taken off z, as float64, in one pass.

    Counts the selected rows first so the result is allocated once at its final size;
    no gathered float32 copy and no upcast copy for the matrix product.
    """
    n = data.shape[0]
    n_out = 0
    for i in range(n):
        if mask[i]:
            n_out += 1
    out = np.empty((n_out, 3))
    r00, r01, r02 = rot[0, 0], rot[0, 1], rot[0, 2]
    r10, r11, r12 = rot[1, 0], rot[1, 1], rot[1, 2]
    r20, r21, r22 = rot[2, 0], rot[2, 1], rot[2, 2]
    k = 0
    for i in range(n):
        if mask[i]:
            x = np.float64(data[i, 0])
            y = np.float64(data[i, 1])
            z = np.float64(data[i, 2])
            out[k, 0] = x * r00 + y * r10 + z * r20
            out[k, 1] = x * r01 + y * r11 + z * r21
            out[k, 2] = x * r02 + y * r12 + z * r22 - floor_z
            k += 1
    return out


def vectors2matrix(v1, v2):
    """Rotation matrix that maps unit vector v1 onto unit vector v2.

    Both inputs must be normalised.
    """
    v_rot = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))
    v_norm = np.linalg.norm(v_rot)
    if v_norm == 0:
        # v1 ≡ ±v2 — return identity or flip.
        return np.eye(v1.shape[0]) if angle == 0 else -np.eye(v1.shape[0])
    v_rot /= v_norm
    return R.from_rotvec(v_rot * angle).as_matrix()
