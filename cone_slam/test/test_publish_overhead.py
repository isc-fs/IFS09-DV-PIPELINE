"""SLAM's per-scan overhead on the RK3588: what it may skip, and what it must not.

THE COST THIS PINS
------------------
In the full sim check on the Orange Pi, SLAM's core sat at ~97 % while its scan
work was ~26 ms per 100 ms scan, ~14 ms of that publishing. The rest was the
single-threaded executor waking for inputs SLAM drops: with motion_model=odom
(the default) /imu and /motor_rpm only feed the start-up calibration and
/steering_angle is never read. On the car that is 600 messages/s.

The cuts, and the contract each keeps:
  * /steering_angle: subscribed only with motion_model=imu.
  * /imu, /motor_rpm: dropped once SLAM runs (odom), recreated on activate.
  * /Conos markers: reused per landmark, content identical to a fresh build.
  * /Conos_full (viewer only): every Nth scan, skipped with no subscriber,
    keyframe when one appears. /Conos itself still goes out every scan.

Requires rclpy and gtsam: these drive the real ConeGraphSlamNode.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("gtsam")

from builtin_interfaces.msg import Time  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from cone_slam.cone_graph_slam_node import ConeGraphSlamNode, State  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def slam():
    n = ConeGraphSlamNode()
    yield n
    n.destroy_node()


def _activate(n, motion_model: str = "odom"):
    n.set_parameters([Parameter("motion_model", value=motion_model)])
    assert n.trigger_configure() == rclpy.lifecycle.TransitionCallbackReturn.SUCCESS
    assert n.trigger_activate() == rclpy.lifecycle.TransitionCallbackReturn.SUCCESS


# ------------------------------------------------------------ inputs

def test_odom_model_never_subscribes_to_steering(slam):
    _activate(slam, "odom")
    assert slam._sub_steering is None
    assert slam._sub_imu is not None and slam._sub_rpm is not None, \
        "calibration needs /imu and /motor_rpm until SLAM runs"


def test_odom_model_drops_imu_and_rpm_once_running(slam):
    _activate(slam, "odom")
    slam._drop_init_only_subs()                 # still calibrating: keep them
    assert slam._sub_imu is not None
    slam._state = State.SLAM_RUNNING
    slam._drop_init_only_subs()
    assert slam._sub_imu is None and slam._sub_rpm is None
    slam._drop_init_only_subs()                 # idempotent


def test_next_run_subscribes_again(slam):
    _activate(slam, "odom")
    slam._state = State.SLAM_RUNNING
    slam._drop_init_only_subs()
    slam.trigger_deactivate()
    assert slam.trigger_activate() == rclpy.lifecycle.TransitionCallbackReturn.SUCCESS
    assert slam._sub_imu is not None and slam._sub_rpm is not None


def test_imu_model_keeps_every_input(slam):
    _activate(slam, "imu")
    assert slam._sub_steering is not None
    slam._state = State.SLAM_RUNNING
    slam._drop_init_only_subs()
    assert slam._sub_imu is not None and slam._sub_rpm is not None


# ------------------------------------------------------------ markers

def _lm(i, x, y, n_obs=5):
    return SimpleNamespace(id=i, position=(x, y, 0.1), n_observations=n_obs)


def _reference_marker(lm, stamp, frame) -> Marker:
    """The marker as _cone_marker built it before reuse."""
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame
    m.id = lm.id
    m.type = Marker.CYLINDER
    m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, lm.position)
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = 0.2, 0.2, 0.3
    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 1.0
    return m


def test_reused_marker_matches_a_fresh_build(slam):
    _activate(slam)
    t1, t2 = Time(sec=1), Time(sec=2)
    a = slam._cone_marker(_lm(7, 1.0, 2.0), t1)
    assert a == _reference_marker(_lm(7, 1.0, 2.0), t1, slam.map_frame)
    b = slam._cone_marker(_lm(7, 1.5, 2.5), t2)       # the landmark moved
    assert b is a, "not reused"
    assert b == _reference_marker(_lm(7, 1.5, 2.5), t2, slam.map_frame)


def test_marker_cache_is_per_run(slam):
    _activate(slam)
    a = slam._cone_marker(_lm(7, 1.0, 2.0), Time(sec=1))
    slam.trigger_deactivate()
    slam.trigger_activate()
    assert slam._cone_marker(_lm(7, 1.0, 2.0), Time(sec=1)) is not a


# ------------------------------------------------------------ cone map cadence

class _Pub:
    def __init__(self, subs: int = 1):
        self.subs = subs
        self.sent = []

    def get_subscription_count(self):
        return self.subs

    def publish(self, msg):
        self.sent.append([(m.action, m.id) for m in msg.markers])


def _map_rig(slam, full_subs: int = 1):
    _activate(slam)
    slam._db = [_lm(i, float(i), 0.0) for i in range(5)]
    slam._latest_result = None                        # no crop
    slam._cones_pub = _Pub()
    slam._cones_full_pub = _Pub(full_subs)
    slam._graph = SimpleNamespace(step=0)
    slam._loc_scan = 0
    return slam._cones_pub, slam._cones_full_pub


def _scans(slam, n, move_every_scan=True):
    for k in range(1, n + 1):
        slam._graph.step = k
        if move_every_scan:
            slam._db[0] = _lm(0, 0.0, 0.1 * k)        # one cone keeps moving
        slam._publish_cone_map(Time(sec=k))


def test_conos_still_goes_out_every_scan(slam):
    conos, _ = _map_rig(slam)
    _scans(slam, 12)
    assert len(conos.sent) == 12
    assert all(len(s) == 1 + 5 for s in conos.sent), "DELETEALL + every cone, every scan"


def test_full_map_every_nth_scan_after_a_keyframe(slam):
    _, full = _map_rig(slam)
    _scans(slam, 12)
    # scan 1: first publish with a subscriber → keyframe; then scans 5 and 10.
    assert len(full.sent) == 3
    assert full.sent[0][0] == (Marker.DELETEALL, 0) and len(full.sent[0]) == 6
    assert full.sent[1] == [(Marker.ADD, 0)], "only the moved cone"


def test_full_map_skipped_without_subscribers_then_resyncs(slam):
    _, full = _map_rig(slam, full_subs=0)
    _scans(slam, 7)
    assert full.sent == []
    full.subs = 1                                      # a viewer connects
    slam._graph.step = 8
    slam._publish_cone_map(Time(sec=8))
    assert full.sent and full.sent[0][0][0] == Marker.DELETEALL and len(full.sent[0]) == 6


def test_delta_every_scan_restores_the_old_cadence(slam):
    _, full = _map_rig(slam)
    slam.set_parameters([Parameter("cone_map_full_delta_every_n_scans", value=1)])
    _scans(slam, 6)
    assert len(full.sent) == 6
