"""Unit tests for the pure watchdog health core.

No ROS, no sleeping — time is injected, so every timing branch is exercised
deterministically. Layered the way the failures actually matter:
  * arming semantics (must never trip a car nobody armed)
  * liveness (the documented runaway)
  * pose progress (the runaway's nastier twin)
  * latching (an intermittent fault must not rearm the car)
"""
from __future__ import annotations

import os
import sys

# Put the package dir (parent of test/) on the path so
# `import pipeline_watchdog.health_monitor` resolves without ROS.
# Mirrors the import shim used by mission_control's and bringup's tests.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from pipeline_watchdog.health_monitor import (  # noqa: E402
    HealthMonitor,
    PoseProgressSpec,
    TopicSpec,
    Verdict,
)

POSE = TopicSpec("/slam/pose", 0.6, "SLAM stopped solving")
ODOM = TopicSpec("/odom", 0.5, "odometry filter died")
SPECS = (POSE, ODOM)

GRACE = 3.0


def _mon(specs=SPECS, grace=GRACE, pose_cfg=None):
    return HealthMonitor(specs, grace_period_s=grace, pose_progress=pose_cfg)


def _feed_all(mon, t):
    for s in SPECS:
        mon.record(s.topic, t)


# ---------------------------------------------------------------- arming

def test_disarmed_never_trips_even_with_everything_silent():
    """A pipeline that was never armed must never fire the EBS."""
    mon = _mon()
    assert not mon.armed
    # Hours of total silence.
    assert mon.evaluate(10_000.0) == Verdict(False)
    assert not mon.tripped


def test_grace_window_absorbs_spin_up():
    """Nodes take time to produce their first message; that is not a fault."""
    mon = _mon()
    mon.set_running(True, 0.0)
    # Nothing has ever published, but we are inside the grace window.
    assert not mon.evaluate(GRACE - 0.01).tripped


def test_topic_that_never_publishes_trips_after_grace():
    """'SLAM never started' is as fatal as 'SLAM stopped'."""
    mon = _mon()
    mon.set_running(True, 0.0)
    v = mon.evaluate(GRACE + 1.0)
    assert v.tripped
    assert "never published" in v.summary()


def test_arming_is_idempotent_grace_window_does_not_restart():
    """Re-asserting DV_RUNNING every tick must not slide the grace window
    forward forever — that would disable the watchdog entirely."""
    mon = _mon()
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        mon.set_running(True, t)
    # Grace was armed at t=0, so by t=4.0 budgets apply despite re-arming.
    assert mon.evaluate(4.0).tripped


def test_disarm_clears_latch_and_memory():
    """A fresh run starts clean — mirrors mission_control resetting its own
    emergency latch on a new cycle."""
    mon = _mon()
    mon.set_running(True, 0.0)
    assert mon.evaluate(GRACE + 1.0).tripped
    mon.set_running(False, 10.0)
    assert not mon.armed
    assert not mon.tripped
    # New run: grace applies again from the new arm time.
    mon.set_running(True, 20.0)
    assert not mon.evaluate(20.0 + GRACE - 0.01).tripped


# -------------------------------------------------------------- liveness

def test_healthy_pipeline_does_not_trip():
    """The case that must never false-positive: everything publishing."""
    mon = _mon()
    mon.set_running(True, 0.0)
    t = 0.0
    while t < 30.0:
        _feed_all(mon, t)
        assert not mon.evaluate(t).tripped, f"false trip at t={t}"
        t += 0.05


def test_silent_pose_trips_after_budget():
    """The documented runaway: /slam/pose freezes, control drives blind."""
    mon = _mon()
    mon.set_running(True, 0.0)
    _feed_all(mon, 5.0)
    # /odom keeps flowing; only /slam/pose stops.
    mon.record("/odom", 5.4)
    assert not mon.evaluate(5.4).tripped          # 0.4s < 0.6s budget
    mon.record("/odom", 5.7)
    v = mon.evaluate(5.7)                          # 0.7s > 0.6s budget
    assert v.tripped
    assert "/slam/pose" in v.summary()
    assert "SLAM stopped solving" in v.summary()   # rationale reaches the log


def test_budget_boundary_is_strict_greater_than():
    """Exactly at budget is still healthy; a hair past is not."""
    mon = _mon(specs=(POSE,))
    mon.set_running(True, 0.0)
    mon.record("/slam/pose", 5.0)
    assert not mon.evaluate(5.0 + POSE.max_silence_s).tripped
    assert mon.evaluate(5.0 + POSE.max_silence_s + 1e-6).tripped


def test_multiple_stale_topics_all_reported():
    """The trip reason must name every broken topic, not just the first."""
    mon = _mon()
    mon.set_running(True, 0.0)
    _feed_all(mon, 5.0)
    v = mon.evaluate(7.0)
    assert v.tripped
    assert "/slam/pose" in v.summary() and "/odom" in v.summary()


# --------------------------------------------------------- pose progress

def _pose_mon(**kw):
    cfg = PoseProgressSpec(
        min_speed_mps=1.0, min_travel_m=0.5, window_s=1.5, **kw)
    # Only /odom liveness, so pose-progress is what trips (not pose silence).
    return HealthMonitor((ODOM,), grace_period_s=GRACE, pose_progress=cfg)


def test_frozen_pose_while_moving_trips():
    """SLAM still publishing, but the solve is stale and the car is moving."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    mon.record_pose(0.0, 0.0, 4.0)
    t = 4.0
    while t < 6.0:
        mon.record("/odom", t)
        mon.record_speed(3.0, t)       # moving
        mon.record_pose(0.0, 0.0, t)   # but pose never advances
        t += 0.1
    v = mon.evaluate(t)
    assert v.tripped
    assert "not advancing" in v.summary()


def test_frozen_pose_while_stopped_is_fine():
    """A legitimately stopped car has a frozen pose. Must not trip."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    t = 4.0
    while t < 20.0:
        mon.record("/odom", t)
        mon.record_speed(0.0, t)       # stopped
        mon.record_pose(0.0, 0.0, t)
        assert not mon.evaluate(t).tripped, f"false trip at t={t}"
        t += 0.1


def test_advancing_pose_while_moving_is_fine():
    """The nominal driving case must never trip."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    t, x = 4.0, 0.0
    while t < 30.0:
        mon.record("/odom", t)
        mon.record_speed(3.0, t)
        x += 3.0 * 0.1                 # 3 m/s of honest progress
        mon.record_pose(x, 0.0, t)
        assert not mon.evaluate(t).tripped, f"false trip at t={t}"
        t += 0.1


def test_driving_off_after_a_long_wait_is_fine():
    """#31: the car waited ~15 s for its first path, standing still with a
    still pose, then drove off. The stall must count from when it started
    moving, not from the pose's last advance before the wait."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    t = 4.0
    while t < 18.0:                    # waiting: stopped, pose still
        mon.record("/odom", t)
        mon.record_speed(0.0, t)
        mon.record_pose(0.0, 0.0, t)
        assert not mon.evaluate(t).tripped
        t += 0.1
    x, v = 0.0, 0.0
    while t < 24.0:                    # drive off, accelerating at 2 m/s^2
        v = min(3.0, v + 0.2)
        x += v * 0.1
        mon.record("/odom", t)
        mon.record_speed(v, t)
        mon.record_pose(x, 0.0, t)     # SLAM keeps up
        assert not mon.evaluate(t).tripped, f"false trip at t={t:.1f}"
        t += 0.1


def test_frozen_pose_after_driving_off_still_trips():
    """The same drive-off, but SLAM never follows: still trips, once the car
    has been moving for the window."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    t = 4.0
    while t < 18.0:
        mon.record("/odom", t)
        mon.record_speed(0.0, t)
        mon.record_pose(0.0, 0.0, t)
        t += 0.1
    tripped_at = None
    while t < 22.0 and tripped_at is None:
        mon.record("/odom", t)
        mon.record_speed(2.0, t)
        mon.record_pose(0.0, 0.0, t)   # frozen
        if mon.evaluate(t).tripped:
            tripped_at = t
        t += 0.1
    assert tripped_at is not None
    assert 18.0 + 1.5 <= tripped_at < 18.0 + 2.0


def test_slowing_below_the_floor_restarts_the_count():
    """Dipping under min_speed_mps resets the moving clock."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    mon.record_pose(0.0, 0.0, 4.0)
    t = 4.0
    for speed in [2.0] * 10 + [0.5] * 5 + [2.0] * 10:   # 1.0 s, dip, 1.0 s
        mon.record("/odom", t)
        mon.record_speed(speed, t)
        mon.record_pose(0.0, 0.0, t)
        assert not mon.evaluate(t).tripped, f"false trip at t={t:.1f}"
        t += 0.1


def test_slow_creep_below_min_travel_still_trips():
    """Pose drifting a few mm is not progress — must still trip."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    t, x = 4.0, 0.0
    while t < 6.5:
        mon.record("/odom", t)
        mon.record_speed(3.0, t)
        x += 0.001                     # 1 mm/tick: never reaches min_travel
        mon.record_pose(x, 0.0, t)
        t += 0.1
    assert mon.evaluate(t).tripped


def test_pose_progress_can_be_disabled():
    """Escape hatch: liveness alone must still work."""
    mon = _pose_mon(enabled=False)
    mon.set_running(True, 0.0)
    t = 4.0
    while t < 20.0:
        mon.record("/odom", t)
        mon.record_speed(3.0, t)
        mon.record_pose(0.0, 0.0, t)   # frozen, but the check is off
        assert not mon.evaluate(t).tripped
        t += 0.1


def test_pose_progress_needs_a_pose_first():
    """No /slam/pose yet → progress check has nothing to say (liveness owns
    that case). Must not trip on an empty anchor."""
    mon = _pose_mon()
    mon.set_running(True, 0.0)
    mon.record("/odom", 4.0)
    mon.record_speed(3.0, 4.0)
    assert not mon.evaluate(4.0).tripped


# -------------------------------------------------------------- latching

def test_trip_latches_even_if_data_recovers():
    """An intermittent sensor that recovers must NOT silently rearm the car."""
    mon = _mon()
    mon.set_running(True, 0.0)
    _feed_all(mon, 5.0)
    assert mon.evaluate(7.0).tripped
    # Everything comes back, healthily, for a long time.
    t = 7.0
    while t < 20.0:
        _feed_all(mon, t)
        assert mon.evaluate(t).tripped, "watchdog un-tripped itself"
        t += 0.1


def test_reasons_are_stable_after_latch():
    """The latched reason must keep describing the ORIGINAL fault, so the
    post-run bag analysis says what actually broke."""
    mon = _mon()
    mon.set_running(True, 0.0)
    _feed_all(mon, 5.0)
    first = mon.evaluate(5.7).summary()
    _feed_all(mon, 100.0)
    assert mon.evaluate(100.0).summary() == first


def test_verdict_summary_healthy():
    assert Verdict(False).summary() == "healthy"
