"""/cones_raw and /cones_orange (dv_msgs/ConeArray) carry the same cones as the MarkerArrays.

SLAM and control read the ConeArrays; the MarkerArrays are for viewers only and are built
only when something subscribes (a 36-cone MarkerArray took 18.7 ms to decode in rclpy).
"""
from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("dv_msgs")
from builtin_interfaces.msg import Time  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from cone_detection.cone_detection_node import ConeDetectionNode  # noqa: E402
from cone_detection.strategies.cone_detection_strategy import ConeObservation  # noqa: E402

CONES = [
    ConeObservation(x=4.0, y=1.5, height_m=0.32, sigma_xy=0.04),
    ConeObservation(x=6.0, y=-1.5, height_m=0.05),            # tiny: unknown sigma
    ConeObservation(x=9.0, y=1.6, height_m=0.52, sigma_xy=0.08),  # big orange
]


def test_arrays_match_markers():
    stamp = Time(sec=7, nanosec=5)
    arr, orange = ConeDetectionNode._cones_to_arrays(CONES, stamp=stamp, big_orange_threshold_m=0.45)
    ma, mo = ConeDetectionNode._cones_to_markers(CONES, stamp=stamp, big_orange_threshold_m=0.45)
    adds = [m for m in ma.markers if m.action == Marker.ADD]
    assert list(arr.x) == pytest.approx([m.pose.position.x for m in adds])
    assert list(arr.y) == pytest.approx([m.pose.position.y for m in adds])
    assert list(arr.height) == pytest.approx([0.32, 0.05, 0.52])   # raw, not display-clamped
    assert list(arr.sigma_xy) == pytest.approx([0.04, -1.0, 0.08])  # unknown is < 0
    assert list(arr.id) == [-1, -1, -1]
    assert arr.header.frame_id == "base_link" and arr.header.stamp == stamp
    o_adds = [m for m in mo.markers if m.action == Marker.ADD]
    assert list(orange.x) == pytest.approx([m.pose.position.x for m in o_adds]) == [9.0]


def test_empty_scan_still_stamped():
    stamp = Time(sec=9)
    arr, orange = ConeDetectionNode._cones_to_arrays([], stamp=stamp, big_orange_threshold_m=0.45)
    assert len(arr.x) == 0 and len(orange.x) == 0
    assert arr.header.stamp == stamp and orange.header.stamp == stamp
