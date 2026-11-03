"""Stanley tests: polyline projection and yaw-rate damping (#33)."""
from __future__ import annotations

import math

import pytest

from control.controllers.stanley import Stanley, _project_onto_polyline
from control.models.bicycle import KinematicBicycle
from control.reference import ReferenceTrajectory
from control.state import VehicleState

BIKE = KinematicBicycle(wheelbase=1.627, max_steer_rad=math.radians(18.2))


def _straight(n: int = 25, step: float = 0.5) -> ReferenceTrajectory:
    return ReferenceTrajectory.from_xy([i * step for i in range(n)], [0.0] * n)


def _arc(radius: float, n: int = 40, step: float = 0.5) -> ReferenceTrajectory:
    """Left-turning arc starting at the origin heading +x."""
    xs, ys = [], []
    for i in range(n):
        a = i * step / radius
        xs.append(radius * math.sin(a))
        ys.append(radius * (1.0 - math.cos(a)))
    return ReferenceTrajectory.from_xy(xs, ys)


@pytest.mark.parametrize("along", [0.0, 0.1, 0.25, 0.4])
def test_centred_car_has_no_cross_track_between_vertices(along):
    # The old nearest-vertex distance read up to 0.25 m here, with a
    # random sign, on a car sitting exactly on the path.
    p = _project_onto_polyline(3.0 + along, 0.0, *_xy(_straight()))
    assert p.offset == pytest.approx(0.0, abs=1e-9)
    assert p.yaw == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("lateral", [0.3, -0.3])
def test_offset_is_perpendicular_and_signed(lateral):
    # Car to the right of the path (lateral < 0) ⇒ path is LEFT ⇒ +offset.
    p = _project_onto_polyline(3.2, lateral, *_xy(_straight()))
    assert p.offset == pytest.approx(-lateral, abs=1e-9)


def test_curvature_sign_and_size_on_arc():
    p = _project_onto_polyline(*_xy_at(_arc(8.0), 10), *_xy(_arc(8.0)))
    assert p.curvature == pytest.approx(1.0 / 8.0, rel=0.05)
    q = _project_onto_polyline(*_xy_at(_mirror(_arc(8.0)), 10), *_xy(_mirror(_arc(8.0))))
    assert q.curvature == pytest.approx(-1.0 / 8.0, rel=0.05)


def test_centred_car_on_straight_steers_zero():
    c = Stanley(k=2.0, k_soft=1.5, k_yaw_rate=0.5, k_damp_steer=0.2, model=BIKE)
    for i in range(20):
        # Front axle walks across vertices; the command must stay at zero.
        st = VehicleState(x=1.0 + 0.07 * i, y=0.0, yaw=0.0, vx=3.0)
        assert c.compute(st, _straight()) == pytest.approx(0.0, abs=1e-9)


def test_yaw_rate_term_is_neutral_when_turning_at_path_rate():
    # On the arc, at the path's own yaw rate, damping must not pull the
    # command away from what the geometry asks for.
    ref = _arc(8.0)
    i = 6
    x, y = ref.x[i], ref.y[i]
    yaw = ref.yaw[i]
    # Put the REAR axle so the FRONT axle lands on the path.
    bx, by = x - BIKE.wheelbase * math.cos(yaw), y - BIKE.wheelbase * math.sin(yaw)
    v = 3.0
    plain = Stanley(k=2.0, k_soft=1.5, k_yaw_rate=0.0, model=BIKE)
    damped = Stanley(k=2.0, k_soft=1.5, k_yaw_rate=0.5, model=BIKE)
    st = VehicleState(x=bx, y=by, yaw=yaw, vx=v, yaw_rate=v / 8.0)
    assert damped.compute(st, ref) == pytest.approx(plain.compute(st, ref), abs=0.02)


def test_yaw_rate_term_opposes_excess_rotation():
    plain = Stanley(k=2.0, k_soft=1.5, k_yaw_rate=0.0, model=BIKE)
    damped = Stanley(k=2.0, k_soft=1.5, k_yaw_rate=0.5, model=BIKE)
    st = VehicleState(x=1.0, y=0.0, yaw=0.0, vx=3.0, yaw_rate=0.4)  # spinning left
    assert damped.compute(st, _straight()) < plain.compute(st, _straight()) - 0.1


def test_no_tick_to_tick_sign_alternation():
    # The old term fed back its own previous command (≈ −0.9× at 3 m/s),
    # so a constant input produced an alternating output.
    c = Stanley(k=2.0, k_soft=1.5, k_yaw_rate=0.5, k_damp_steer=0.2, model=BIKE)
    st = VehicleState(x=1.0, y=-0.2, yaw=0.0, vx=3.0)
    out = [c.compute(st, _straight()) for _ in range(10)]
    diffs = [b - a for a, b in zip(out, out[1:])]
    assert all(d >= -1e-9 for d in diffs) or all(d <= 1e-9 for d in diffs)


def _xy(ref):
    return ref.x, ref.y


def _xy_at(ref, i):
    return ref.x[i], ref.y[i]


def _mirror(ref):
    return ReferenceTrajectory.from_xy(ref.x, [-v for v in ref.y])
