"""Per-mission settings: the sight-distance cap is off in accel only, and
accel overrides v_max / throttle_max.

Requires rclpy — drives the real ControlNode, not a stub.
"""
from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("transforms3d")

from control.control_node import ControlNode  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = ControlNode()
    yield n
    n.destroy_node()


def test_sight_cap_off_in_accel(node):
    node._mode_name = "accel"
    assert node._unseen_corner_radius() == 0.0


@pytest.mark.parametrize("mode", ["autocross", "trackdrive", "skidpad", None])
def test_sight_cap_on_elsewhere(node, mode):
    node._mode_name = mode
    assert node._unseen_corner_radius() == pytest.approx(
        node.get_parameter("unseen_corner_radius").value)


def test_accel_overrides_speed_and_throttle(node):
    node._mode_name = "accel"
    assert node._mode_p("v_max") == pytest.approx(12.0)
    assert node._mode_p("throttle_max") == pytest.approx(0.5)


@pytest.mark.parametrize("mode", ["autocross", "trackdrive", "skidpad", None])
def test_other_missions_use_the_shared_speed(node, mode):
    node._mode_name = mode
    assert node._mode_p("v_max") == pytest.approx(5.0)
    assert node._mode_p("throttle_max") == pytest.approx(0.2)
