"""PIVelocity setpoint: regen-limited braking back from corners and stops.

Regen is the car's only service brake and it is power limited, so braking
falls with speed: a(v) = min(a_dec_max, P / (m·v)). These pin that the
setpoint brakes back from a corner along the path with that model, instead
of dropping to the corner speed only once the corner is reached.
"""
from __future__ import annotations

import math

import pytest

from control.controllers.pi_velocity import PIVelocity
from control.reference import ReferenceTrajectory
from control.state import VehicleState

P, M, A = 11000.0, 275.0, 4.0          # v* = P / (m·a) = 10 m/s


def _ctrl(**kw) -> PIVelocity:
    # Sight cap and latency off unless a test turns them on.
    args = dict(v_max=12.0, a_lat_max=3.0, a_dec_max=A,
                regen_power_w=P, vehicle_mass=M,
                unseen_corner_radius=0.0, brake_latency_s=0.0)
    args.update(kw)
    return PIVelocity(**args)


def _simulate_braking(v0: float, v_end: float, a_max: float, p: float, m: float) -> float:
    """Distance to brake v0 → v_end with a(v) = min(a_max, p/(m·v)), by Euler."""
    v, s, dt = v0, 0.0, 1e-4
    while v > v_end:
        a = min(a_max, p / (m * v)) if p > 0 else a_max
        s += v * dt
        v -= a * dt
    return s


@pytest.mark.parametrize("v_end,s", [(0.0, 5.0), (0.0, 30.0), (4.5, 8.0),
                                     (4.5, 25.0), (10.5, 6.0)])
def test_brake_entry_speed_matches_integrated_braking(v_end, s):
    c = _ctrl()
    v0 = c.brake_entry_speed(v_end, s)
    assert _simulate_braking(v0, v_end, A, P, M) == pytest.approx(s, rel=2e-3)


def test_constant_decel_below_crossover():
    c = _ctrl()
    # 0 → 8 m/s stays under v* = 10 m/s: plain v² = 2·a·s.
    assert c.brake_entry_speed(0.0, 8.0) == pytest.approx(math.sqrt(2 * A * 8.0))


def test_power_limit_lengthens_braking_at_speed():
    c_power = _ctrl()
    c_const = _ctrl(regen_power_w=0.0)       # old model: constant a_dec_max
    # From 30 m out, the power limit allows less entry speed than 4 m/s² would.
    assert c_power.brake_entry_speed(4.5, 30.0) < c_const.brake_entry_speed(4.5, 30.0)
    assert c_const.brake_entry_speed(4.5, 30.0) == pytest.approx(math.sqrt(4.5**2 + 2 * A * 30))


def _straight_into_corner(straight_m: float, radius: float, step: float = 0.5):
    """Straight along +x for straight_m, then a left arc of the given radius."""
    xs, ys = [], []
    n_straight = int(straight_m / step)
    for i in range(n_straight):
        xs.append(i * step)
        ys.append(0.0)
    x0 = n_straight * step
    for j in range(40):
        a = j * step / radius
        xs.append(x0 + radius * math.sin(a))
        ys.append(radius * (1 - math.cos(a)))
    kappa = [0.0] * n_straight + [1.0 / radius] * 40
    return ReferenceTrajectory.from_xy(xs, ys, kappa)


def test_setpoint_starts_braking_before_the_corner():
    c = _ctrl()
    ref = _straight_into_corner(straight_m=12.0, radius=6.75)   # v_corner = 4.5 m/s
    v_set = c._setpoint(VehicleState(x=0.0, y=0.0, vx=12.0), ref)
    expected = c.brake_entry_speed(4.5, ref.s[24])               # corner starts at index 24
    assert v_set == pytest.approx(expected, rel=1e-6)
    assert 4.5 < v_set < 12.0


def test_setpoint_unchanged_when_corner_is_far():
    c = _ctrl()
    ref = _straight_into_corner(straight_m=60.0, radius=6.75)
    assert c._setpoint(VehicleState(vx=12.0), ref) == pytest.approx(12.0)


def test_setpoint_at_corner_is_corner_speed():
    c = _ctrl()
    ref = _straight_into_corner(straight_m=0.0, radius=6.75)
    assert c._setpoint(VehicleState(vx=4.5), ref) == pytest.approx(4.5, rel=1e-3)


def test_stop_cap_uses_braking_curve():
    c = _ctrl()
    ref = ReferenceTrajectory.from_xy([i * 0.5 for i in range(200)], [0.0] * 200)
    ref.stop_distance = 15.0      # past v* = 10 m/s: power limited
    v_set = c._setpoint(VehicleState(vx=12.0), ref)
    assert v_set == pytest.approx(c.brake_entry_speed(0.0, 15.0))
    assert v_set < math.sqrt(2 * A * 15.0)
    # Short of v*, identical to the old sqrt(2·a·d) cap.
    ref.stop_distance = 5.0
    assert c._setpoint(VehicleState(vx=5.0), ref) == pytest.approx(math.sqrt(2 * A * 5.0))


def _straight(length_m: float, step: float = 0.5) -> ReferenceTrajectory:
    n = int(length_m / step) + 1
    return ReferenceTrajectory.from_xy([i * step for i in range(n)], [0.0] * n)


def test_sight_cap_on_a_short_straight():
    c = _ctrl(unseen_corner_radius=4.5)
    ref = _straight(12.0)
    v_unseen = math.sqrt(3.0 * 4.5)
    v_set = c._setpoint(VehicleState(vx=10.0), ref)
    assert v_set == pytest.approx(c.brake_entry_speed(v_unseen, 12.0), rel=1e-6)
    assert v_set < 12.0


def test_sight_cap_lifts_on_a_long_straight():
    c = _ctrl(unseen_corner_radius=4.5)
    assert c._setpoint(VehicleState(vx=12.0), _straight(60.0)) == pytest.approx(12.0)


def test_latency_shortens_braking_distance():
    c = _ctrl(brake_latency_s=0.3)
    ref = _straight_into_corner(straight_m=12.0, radius=6.75)
    v = 10.0
    v_set = c._setpoint(VehicleState(vx=v), ref)
    assert v_set == pytest.approx(c.brake_entry_speed(4.5, ref.s[24] - v * 0.3), rel=1e-6)
