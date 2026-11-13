"""SLAM's latched mission-start values must reach subscribers on activate.

THE BUG THIS PINS
-----------------
on_activate published /slam/finished, /slam/final_lap and /slam/stop_request
BEFORE calling the base on_activate, while the lifecycle publishers were still
inactive, so publish() silently dropped them. Subscribers kept the previous
run's latched values: after an autocross (final_lap=true), a trackdrive's
control_node still read final_lap=true and braked to a stop at the end of lap 1.

Requires rclpy: these drive the real ConeGraphSlamNode through its lifecycle.
"""
from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("gtsam")

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402

from cone_slam.cone_graph_slam_node import ConeGraphSlamNode  # noqa: E402

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def rig():
    slam = ConeGraphSlamNode()
    probe = rclpy.create_node("final_lap_probe")
    got: list[bool] = []
    probe.create_subscription(Bool, "/slam/final_lap", lambda m: got.append(m.data), LATCHED)
    ex = SingleThreadedExecutor()
    ex.add_node(slam)
    ex.add_node(probe)
    yield slam, got, ex
    ex.shutdown()
    probe.destroy_node()
    slam.destroy_node()


def _start(slam, mission: str) -> None:
    slam._mode_name = mission
    slam._behavior = mission
    assert slam.trigger_configure() == rclpy.lifecycle.TransitionCallbackReturn.SUCCESS
    assert slam.trigger_activate() == rclpy.lifecycle.TransitionCallbackReturn.SUCCESS


def _spin(ex, n: int = 20) -> None:
    for _ in range(n):
        ex.spin_once(timeout_sec=0.05)


def test_trackdrive_publishes_final_lap_false_on_activate(rig):
    slam, got, ex = rig
    _start(slam, "trackdrive")
    _spin(ex)
    assert got and got[-1] is False, f"final_lap never reached subscribers: {got}"


def test_trackdrive_after_autocross_overrides_the_stale_true(rig):
    """The failure seen in the full sim check: autocross latches true, then a
    trackdrive must replace it with false before the car reaches the gate."""
    slam, got, ex = rig
    _start(slam, "autocross")
    _spin(ex)
    assert got and got[-1] is True
    slam.trigger_deactivate()
    slam.trigger_cleanup()
    _start(slam, "trackdrive")
    _spin(ex)
    assert got[-1] is False, f"trackdrive kept the autocross value: {got}"
