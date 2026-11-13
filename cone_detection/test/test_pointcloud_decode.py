"""Regression tests for ConeDetectionNode.pointcloud2_to_xyz.

Guards the packed-`point_step` decode bug: the Hesai ATX driver publishes a
26-byte packed point (x,y,z,intensity f32 + ring u16 + timestamp f64). The old
`frombuffer(...).reshape(N, point_step // 4)` raised ValueError on every scan
(26 not divisible by 4), so cone_detection silently emitted zero cones on the
car while the 16-byte sim cloud worked. See fix/cone-pointcloud-stride.
"""
import struct
from types import SimpleNamespace

import numpy as np

from cone_detection.cone_detection_node import ConeDetectionNode


def _field(name, offset, datatype):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype)


def _cloud(points, point_step, layout):
    """Build a fake PointCloud2 (only the fields pointcloud2_to_xyz reads)."""
    buf = bytearray()
    for (x, y, z) in points:
        row = bytearray(b"\x00" * point_step)
        row[0:4] = struct.pack("<f", x)
        row[4:8] = struct.pack("<f", y)
        row[8:12] = struct.pack("<f", z)
        buf += row
    return SimpleNamespace(
        width=len(points), height=1, point_step=point_step,
        data=bytes(buf), fields=layout,
    )


# x,y,z at 0/4/8 in every real ROS cloud; the trailing fields differ per source.
XYZ = [_field("x", 0, 7), _field("y", 4, 7), _field("z", 8, 7)]
ATX = XYZ + [_field("intensity", 12, 7), _field("ring", 16, 4),
             _field("timestamp", 18, 8)]  # packed -> point_step 26

POINTS = [(1.0, 2.0, 3.0), (7.9, -3.5, 0.37), (0.0, 0.0, 0.0), (4.0, 5.0, 6.0)]


def test_packed_26_atx():
    """The exact ATX layout that used to raise ValueError every scan."""
    xyz = ConeDetectionNode.pointcloud2_to_xyz(_cloud(POINTS, 26, ATX))
    assert xyz.shape == (4, 3)
    assert xyz.dtype == np.float32
    assert np.allclose(xyz, np.array(POINTS, dtype=np.float32))


def test_aligned_32():
    xyz = ConeDetectionNode.pointcloud2_to_xyz(_cloud(POINTS, 32, ATX))
    assert np.allclose(xyz, np.array(POINTS, dtype=np.float32))


def test_sim_16_xyzi():
    xyz = ConeDetectionNode.pointcloud2_to_xyz(_cloud(POINTS, 16, XYZ))
    assert np.allclose(xyz, np.array(POINTS, dtype=np.float32))


def test_cpp_decode_matches_numpy():
    """detector_backend=cpp decodes in C++: same array for every layout."""
    import pytest

    core = pytest.importorskip("cone_detection_core")
    for step, layout in ((26, ATX), (32, ATX), (16, XYZ)):
        msg = _cloud(POINTS, step, layout)
        want = ConeDetectionNode.pointcloud2_to_xyz(msg)
        got = ConeDetectionNode.pointcloud2_to_xyz(msg, core.decode_xyz)
        assert got.dtype == np.float32 and got.shape == want.shape
        np.testing.assert_array_equal(got, want)
