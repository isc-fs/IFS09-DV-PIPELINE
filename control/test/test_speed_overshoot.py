"""The car must not overshoot v_max with the throttle uncapped (#41).

Before the fix the sim check showed 6.2 m/s at v_max 5: the PI tracked the
lagging EMA speed and the throttle could only fall at 2/s. The closed-loop test
reproduces that with a 1-D car (full-throttle 6 m/s², 0.1 s speed delay, as
fitted to the sim check).
"""
import collections

import pytest

from control.control_node import _slew
from control.controllers.pi_velocity import PIVelocity

DT = 1.0 / 40.0


def _ctrl(v_max):
    return PIVelocity(v_max=v_max, kp=0.5, ki=0.05, deadband=0.05, throttle_max=1.0)


def test_tracking_speed_has_no_filter_lag_while_accelerating():
    c = _ctrl(5.0)
    v = 0.0
    for _ in range(40):          # 1 s at 6 m/s²
        v += 6.0 * DT
        tracked = c._tracking_speed(v)
    assert tracked == pytest.approx(v)
    assert c._v_filt < v - 0.5   # the filter alone reads ~0.9 m/s low


def test_tracking_speed_ignores_a_spurious_zero():
    c = _ctrl(5.0)
    for _ in range(40):
        c._tracking_speed(5.0)
    assert c._tracking_speed(0.0) > 4.0


def test_slew_releases_faster_than_it_rises():
    assert _slew(0.0, 1.0, 0.05, 0.2) == pytest.approx(0.05)
    assert _slew(1.0, 0.0, 0.05, 0.2) == pytest.approx(0.8)
    assert _slew(1.0, 0.0, 0.05) == pytest.approx(0.95)


def _peak(v_set, rise=2.0, release=8.0, delay_s=0.1, seconds=14.0, old_speed=False):
    c = _ctrl(v_set)
    v = cmd = act = peak = 0.0
    lag = collections.deque([0.0] * (int(delay_s / DT) + 1))
    hold = []
    for i in range(int(seconds / DT)):
        lag.append(v)
        v_meas = lag.popleft()
        speed = c._condition_velocity(v_meas) if old_speed else c._tracking_speed(v_meas)
        thr, regen = c._track(speed, v_set)
        target = thr - regen
        cmd = _slew(cmd, target, rise * DT, release * DT if cmd > 0 else rise * DT)
        act += (cmd - act) * DT / (0.05 + DT)                       # motor lag
        a_avail = min(6.0, 80e3 / (275 * max(v, 0.5))) if act > 0 \
            else min(4.0, 11e3 / (275 * max(v, 0.5)))
        v = max(0.0, v + (act * a_avail - 0.196 - 0.72 * v * v / 275) * DT)
        peak = max(peak, v)
        if i * DT > seconds - 3.0:
            hold.append(v)
    return peak, sum(hold) / len(hold)


@pytest.mark.parametrize("v_set", [5.0, 12.0])
def test_full_throttle_does_not_overshoot_v_max(v_set):
    peak, hold = _peak(v_set)
    assert peak < v_set + 0.3
    assert hold == pytest.approx(v_set, abs=0.15)


def test_old_settings_did_overshoot():
    # Guards the model: the old controller (filtered speed, symmetric 2/s
    # slew) reproduces the sim's ~1 m/s overshoot.
    peak, _ = _peak(5.0, release=2.0, old_speed=True)
    assert peak > 5.5
