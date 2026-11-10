"""Where the car stops after the finish gate.

The stop anchor is the centroid of the first big-orange cones seen after
stop_latch_min_travel. That is the finish gate itself, and its projection
lands 3-5 m short of the line (72.5-75 m on the 78 m acceleration gate in
every sim run). With weak braking the car always overshot it and crossed
the line; with the power-limited regen profile it stops on it, short of the
line, and the run is not finished. So the target is pushed past the anchor
along the direction the car was heading for it, and the distance to it is
measured along that direction: it goes to zero and negative once the car
is past it, instead of growing again like a straight-line distance would.
"""
from __future__ import annotations

import math
from typing import Tuple

XY = Tuple[float, float]


def stop_target(car_xy: XY, anchor_xy: XY, past_m: float) -> Tuple[XY, XY]:
    """(target, unit direction): the anchor moved past_m further along the
    car→anchor direction at the moment of latching."""
    dx, dy = anchor_xy[0] - car_xy[0], anchor_xy[1] - car_xy[1]
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return anchor_xy, (0.0, 0.0)
    ux, uy = dx / d, dy / d
    return (anchor_xy[0] + past_m * ux, anchor_xy[1] + past_m * uy), (ux, uy)


def distance_to_stop(car_xy: XY, target_xy: XY, direction: XY) -> float:
    """Signed distance left to the target along `direction`: negative once
    the car is past it. Falls back to the straight-line distance when there
    is no direction."""
    dx, dy = target_xy[0] - car_xy[0], target_xy[1] - car_xy[1]
    if direction == (0.0, 0.0):
        return math.hypot(dx, dy)
    return dx * direction[0] + dy * direction[1]
