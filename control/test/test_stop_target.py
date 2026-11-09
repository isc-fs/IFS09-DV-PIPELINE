"""Stop target past the finish gate (control/stop_target.py)."""
from __future__ import annotations

import pytest

from control.stop_target import distance_to_stop, stop_target


def test_target_is_pushed_past_the_anchor_along_the_approach():
    target, u = stop_target((0.0, 0.0), (0.0, 10.0), 8.0)
    assert target == pytest.approx((0.0, 18.0))
    assert u == pytest.approx((0.0, 1.0))


def test_distance_goes_negative_past_the_target():
    target, u = stop_target((0.0, 0.0), (10.0, 0.0), 8.0)
    assert distance_to_stop((5.0, 0.0), target, u) == pytest.approx(13.0)
    assert distance_to_stop((18.0, 0.0), target, u) == pytest.approx(0.0)
    # A straight-line distance would grow again here and let the car re-accelerate.
    assert distance_to_stop((20.0, 0.5), target, u) == pytest.approx(-2.0)


def test_car_on_the_anchor_falls_back_to_straight_line():
    target, u = stop_target((3.0, 4.0), (3.0, 4.0), 8.0)
    assert target == (3.0, 4.0) and u == (0.0, 0.0)
    assert distance_to_stop((0.0, 0.0), target, u) == pytest.approx(5.0)
