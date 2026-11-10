"""Stop-anchor lap gating — the trackdrive fix (#384 follow-up).

THE BUG THIS PINS
-----------------
`_on_orange` latched the stop anchor on the FIRST big-orange gate seen past
`stop_latch_min_travel` (30 m) and never unlatched. The 30 m guard only exists
to avoid latching on the start gate — it was never a lap counter. So on
trackdrive the car braked to a stop at the end of lap 1 and could never reach
its 10 laps, no matter what the lap counter said.

The fix gates the latch on `/slam/final_lap` (SLAM owns the lap count; control
stays dumb about mission rules). Two properties matter and are pinned here:

  * trackdrive: the anchor does NOT latch until SLAM says it is the closing lap.
  * everything else: behaviour is UNCHANGED — autocross/accel/skidpad get
    final_lap=true immediately, and a stack with no publisher at all keeps the
    historical "stop at the first gate" rather than "never stop".

That last one is the safety-relevant direction: defaulting the gate to False
would mean a missing/late topic leaves the car unable to ever stop itself.

Requires rclpy — these drive the real ControlNode, not a stub.
"""
from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("transforms3d")

from geometry_msgs.msg import Point, Pose  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from control.control_node import ControlNode  # noqa: E402

MIN_TRAVEL = 30.0


@pytest.fixture(scope="module", autouse=True)
def _ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def node():
    n = ControlNode()
    # _on_orange needs an absolute pose to project the gate into odom frame.
    odom = Odometry()
    odom.pose.pose.orientation.w = 1.0
    n._latest_pose = odom
    yield n
    n.destroy_node()


def _gate(n_cones: int = 2) -> MarkerArray:
    """A big-orange gate `n_cones` wide, 10 m ahead in base_link."""
    ma = MarkerArray()
    for i in range(n_cones):
        m = Marker()
        m.pose = Pose()
        m.pose.position = Point(x=10.0, y=-1.5 + 3.0 * i, z=0.0)
        ma.markers.append(m)
    return ma


def _past_start(node) -> None:
    node._travelled = MIN_TRAVEL + 5.0


# ------------------------------------------------------- the default

def test_final_lap_defaults_true(node):
    """A stack with no /slam/final_lap publisher must keep the historical
    behaviour. Defaulting False would mean the car never stops itself."""
    assert node._final_lap is True


def test_latches_with_no_slam_publisher(node):
    """Unchanged behaviour when nothing publishes final_lap."""
    _past_start(node)
    node._on_orange(_gate())
    assert node._stop_latched


# ------------------------------------------------------ the trackdrive fix

def test_does_not_latch_when_not_final_lap(node):
    """THE FIX. Trackdrive laps 1..9: the gate is seen, travel guard passed,
    but SLAM says this is not the closing lap → must not stop."""
    _past_start(node)
    node._on_final_lap(Bool(data=False))
    node._on_orange(_gate())
    assert not node._stop_latched, "braked on a non-final lap — trackdrive dies at lap 1"
    assert node._stop_anchor_xy is None


def test_latches_once_final_lap_arrives(node):
    """Lap 10: SLAM raises final_lap → the next gate latches the anchor."""
    _past_start(node)
    node._on_final_lap(Bool(data=False))
    for _ in range(9):                      # nine laps' worth of gate sightings
        node._on_orange(_gate())
    assert not node._stop_latched
    node._on_final_lap(Bool(data=True))     # closing lap
    node._on_orange(_gate())
    assert node._stop_latched
    assert node._stop_anchor_xy is not None


def test_travel_guard_still_applies_on_the_final_lap(node):
    """The lap gate ADDS to the travel guard, it does not replace it — the
    start gate is big-orange too."""
    node._travelled = 1.0
    node._on_final_lap(Bool(data=True))
    node._on_orange(_gate())
    assert not node._stop_latched


def test_final_lap_alone_does_not_latch_without_a_gate(node):
    """final_lap is a permission, not a trigger."""
    _past_start(node)
    node._on_final_lap(Bool(data=True))
    node._on_orange(_gate(n_cones=1))       # only one cone — not a gate
    assert not node._stop_latched


# ----------------------------------------------------------- latching

def test_anchor_never_unlatches_if_final_lap_drops(node):
    """Once stopped, a flickering topic must not release the stop."""
    _past_start(node)
    node._on_final_lap(Bool(data=True))
    node._on_orange(_gate())
    assert node._stop_latched
    node._on_final_lap(Bool(data=False))
    assert node._stop_latched, "stop released by a final_lap drop"


def test_anchor_position_projected_into_odom(node):
    """Sanity: the gate lands ~10 m ahead in world frame, not at the car,
    and the stop target is stop_past_anchor_m further along the approach."""
    _past_start(node)
    node._on_orange(_gate())
    past = node.get_parameter("stop_past_anchor_m").value
    ax, ay = node._stop_anchor_xy
    assert ax == pytest.approx(10.0 + past, abs=0.2)
    assert ay == pytest.approx(0.0, abs=0.2)


def test_skidpad_keeps_the_old_stop_on_the_anchor(node):
    """Skidpad latches on the centre gate in lap 1: no target past it, and a
    straight-line distance, so circling past it never reads as 'passed'."""
    node._mode_name = "skidpad"
    _past_start(node)
    node._on_orange(_gate())
    ax, ay = node._stop_anchor_xy
    assert ax == pytest.approx(10.0, abs=0.2)
    assert node._stop_dir == (0.0, 0.0)
