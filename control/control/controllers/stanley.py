"""Stanley lateral controller.

Geometric path follower that projects the front axle onto the path and
combines two terms:

    δ = ψ_e + atan(k · e_y / (k_soft + v)) - k_yaw_rate · (ω_meas - ω_ref)

where:
    ψ_e             heading error between path tangent and vehicle yaw,
                    wrapped to [-π, π]
    e_y             signed perpendicular distance from the front axle
                    to the path polyline (m, positive = path is to the
                    left of the car)
    v               longitudinal speed (m/s)
    k               cross-track gain (1/s scaling)
    k_soft          softening term (m/s) — keeps the denominator
                    bounded at v→0, preventing the cross-track term
                    from saturating the actuator on a stopped car
    k_yaw_rate      yaw-rate damping gain (rad / (rad/s)); 0.0 disables
    ω_meas          measured yaw rate (state.yaw_rate, from /odom)
    ω_ref           yaw rate the path asks for at the projection point,
                    v · κ_path

The damping term is Hoffmann §3.3: it opposes the car's actual yaw
rate in excess of what the path needs, so it damps the heading loop
without biasing the steady-state corner command. An earlier version
fed back the controller's own previous command instead of the
measured yaw rate; that only scaled the gain down (by ~half at 3 m/s)
and added a tick-to-tick sign alternation, and did nothing about the
car's real yaw overshoot (#33).

Reference: Hoffmann et al., "Autonomous Automobile Trajectory
Tracking for Off-Road Driving" (Stanley, 2007),
https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
Port adapted from alt_pipeline/control/control/controlador_stanley.py
onto the LateralController ABC used by control_node.

Sign convention matches the rest of `controllers/`: positive δ → LEFT
(math convention). control_node._tick flips the sign at the actuator
boundary to match UE5/Chaos automotive convention.

Why Stanley alongside Pure Pursuit:
  - Sharper response on constant-curvature paths (skidpad figure-8) —
    cross-track and heading terms have well-defined signs across the
    full circle, no chase-target geometry to tune.
  - Picks the nearest projection rather than a forward chase target,
    so on a tight hairpin it doesn't risk chasing a past-apex
    point. Pure Pursuit needs the β·R radius cap (#260 follow-up)
    to avoid that; Stanley doesn't.
  - At v→0 the softening_gain holds the cross-track term stable; the
    Pure Pursuit radius cap also holds in that regime, so both are
    safe for tight low-speed starts.

When NOT to use Stanley:
  - On near-straight paths with noisy planner yaw, the heading term
    can dominate and produce small oscillations. Acceleration mission
    keeps Pure Pursuit for this reason.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

from control.controllers.base import LateralController
from control.state import VehicleState
from control.reference import ReferenceTrajectory
from control.models.bicycle import KinematicBicycle


class Stanley(LateralController):
    """Stanley lateral controller behind the LateralController ABC."""

    def __init__(
        self,
        k: float = 4.0,
        k_soft: float = 6.0,
        k_yaw_rate: float = 0.0,
        k_damp_steer: float = 0.0,
        model: KinematicBicycle | None = None,
    ) -> None:
        self.k = k
        self.k_soft = k_soft
        self.k_yaw_rate = k_yaw_rate
        # Steering-delay damping: small fraction of the previous tick's
        # commanded steering is subtracted from the new command. Pure
        # additive smoothing — 0.0 disables, ~0.5 visibly damps. The
        # actuator slew limiter in control_node._tick handles the
        # rate-of-change side; this term reduces single-tick swings on
        # noisy paths.
        self.k_damp_steer = k_damp_steer
        self.model = model or KinematicBicycle()
        # Previous tick's normalized steering — input to the damping
        # term. Reset() clears it.
        self._prev_steer_norm: float = 0.0

    def reset(self) -> None:
        self._prev_steer_norm = 0.0

    def compute(self, state: VehicleState, ref: ReferenceTrajectory) -> float:
        if ref.empty:
            return 0.0

        # Project the front axle onto the path. Stanley operates on
        # the front axle (not the CG/rear axle Pure Pursuit uses)
        # because the steering authority is at the front; the
        # geometric error there is what the controller can directly
        # null out.
        fx, fy = self.model.front_axle_xy(state.x, state.y, state.yaw)

        proj = _project_onto_polyline(fx, fy, ref.x, ref.y)

        # Heading error: path tangent at the projection point minus
        # vehicle yaw, wrapped to [-π, π].
        yaw_error = _wrap_pi(proj.yaw - state.yaw)

        # Cross-track term — atan keeps the contribution bounded
        # below ±π/2 even when (k_soft + v) is small. proj.offset is
        # positive when the path is to the LEFT of the car, which is
        # the same sign as a positive (left) steering command.
        cte_term = math.atan2(self.k * proj.offset, self.k_soft + state.speed)

        # Yaw-rate damping (Hoffmann §3.3) on the MEASURED yaw rate,
        # relative to the yaw rate the path itself asks for.
        omega_ref = state.speed * proj.curvature
        yaw_rate_term = self.k_yaw_rate * (state.yaw_rate - omega_ref)

        steer_rad = yaw_error + cte_term - yaw_rate_term

        # Optional single-tick damping against the previous command.
        # k_damp_steer is dimensionless; 0.0 = off, 0.5 = blends in
        # half of the previous tick's steering as inertia. Helps on
        # noisy planner yaw.
        prev_steer_rad = self._prev_steer_norm * self.model.max_steer_rad
        if self.k_damp_steer > 0.0:
            steer_rad -= self.k_damp_steer * (steer_rad - prev_steer_rad)

        # Clamp and convert to normalized [-1, 1]. The actuator slew
        # limiter in control_node._tick handles rate-of-change at the
        # publish boundary; this just bounds the steady-state command.
        steer_rad = max(
            -self.model.max_steer_rad, min(self.model.max_steer_rad, steer_rad)
        )
        steer_norm = self.model.normalize_steer(steer_rad)
        self._prev_steer_norm = steer_norm
        return steer_norm


@dataclass(frozen=True)
class _Projection:
    offset: float     # signed perpendicular distance, + = path is LEFT of the point
    yaw: float        # path tangent at the projection
    curvature: float  # path curvature at the projection (1/m, + = turns left)


def _project_onto_polyline(x: float, y: float, xs, ys) -> _Projection:
    """Project (x, y) onto the polyline through (xs, ys).

    Uses the closest point on any SEGMENT, not the closest vertex: with
    ~0.5 m waypoint spacing the vertex distance carries up to 0.25 m of
    along-track offset, and on a centred car its sign is noise, which
    turned the cross-track term into a ±0.1 rad sawtooth (#33).

    The tangent is blended between neighbouring segment headings so it
    doesn't step at every vertex, and the curvature is the heading
    change per metre around the projection (pure geometry, so its sign
    always matches the path).
    """
    n = len(xs)
    seg_yaw = [
        math.atan2(ys[i + 1] - ys[i], xs[i + 1] - xs[i]) for i in range(n - 1)
    ]
    seg_len = [
        math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(n - 1)
    ]

    best_i, best_t, best_d2 = 0, 0.0, float("inf")
    for i in range(n - 1):
        ax, ay = xs[i], ys[i]
        dx, dy = xs[i + 1] - ax, ys[i + 1] - ay
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 < 1e-12 else ((x - ax) * dx + (y - ay) * dy) / l2
        t = min(1.0, max(0.0, t))
        px, py = ax + t * dx, ay + t * dy
        d2 = (x - px) ** 2 + (y - py) ** 2
        if d2 < best_d2:
            best_i, best_t, best_d2 = i, t, d2

    i, t = best_i, best_t
    ax, ay = xs[i], ys[i]
    dx, dy = xs[i + 1] - ax, ys[i + 1] - ay
    px, py = ax + t * dx, ay + t * dy
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        offset = 0.0
    else:
        # cross(tangent, point - projection) > 0 ⇒ point is LEFT of the
        # path ⇒ path is to its RIGHT ⇒ negative offset.
        cross = (dx * (y - py) - dy * (x - px)) / norm
        offset = -math.copysign(math.sqrt(best_d2), cross) if best_d2 > 0 else 0.0

    # Tangent: blend toward the neighbouring segment on the near side
    # of the projection so the heading changes smoothly across vertices.
    yaw = seg_yaw[i]
    if t > 0.5 and i + 1 < n - 1:
        yaw = seg_yaw[i] + (t - 0.5) * _wrap_pi(seg_yaw[i + 1] - seg_yaw[i])
    elif t < 0.5 and i > 0:
        yaw = seg_yaw[i] - (0.5 - t) * _wrap_pi(seg_yaw[i] - seg_yaw[i - 1])
    yaw = _wrap_pi(yaw)

    # Curvature: heading change over the segments either side.
    lo, hi = max(0, i - 1), min(n - 2, i + 1)
    ds = 0.5 * seg_len[lo] + sum(seg_len[lo + 1:hi]) + 0.5 * seg_len[hi]
    curvature = (
        _wrap_pi(seg_yaw[hi] - seg_yaw[lo]) / ds if hi > lo and ds > 1e-6 else 0.0
    )
    return _Projection(offset=offset, yaw=yaw, curvature=curvature)


def _wrap_pi(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
