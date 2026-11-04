"""OdomHistory tests (#33)."""
from __future__ import annotations

import math

import pytest

from cone_slam.odom_history import OdomHistory


def _hist():
    h = OdomHistory(horizon_s=1.0)
    for i in range(11):  # 100 Hz, driving +x at 3 m/s, yawing 0.5 rad/s
        t = 10.0 + i * 0.01
        h.add(t, 3.0 * (t - 10.0), 0.0, 0.5 * (t - 10.0))
    return h


def test_interpolates_between_samples():
    x, y, yaw = _hist().at(10.055)
    assert x == pytest.approx(0.165)
    assert yaw == pytest.approx(0.0275)


def test_yaw_interpolates_across_pi():
    h = OdomHistory()
    h.add(0.0, 0.0, 0.0, math.pi - 0.1)
    h.add(0.1, 0.0, 0.0, -math.pi + 0.1)
    assert abs(h.at(0.05)[2]) == pytest.approx(math.pi)


def test_outside_window_returns_none():
    h = _hist()
    assert h.at(9.0) is None
    assert h.at(10.5) is None           # far past the newest sample
    assert h.at(10.12) == h.at(10.1)    # just past: newest sample


def test_old_samples_are_dropped_and_clock_restart_clears():
    h = _hist()
    h.add(12.0, 0.0, 0.0, 0.0)
    assert len(h) == 1
    h.add(1.0, 0.0, 0.0, 0.0)           # clock went backwards
    assert len(h) == 1 and h.at(1.0) == (0.0, 0.0, 0.0)
