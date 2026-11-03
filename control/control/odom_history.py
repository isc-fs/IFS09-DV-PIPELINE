"""Short history of /odom poses, looked up by timestamp — no rclpy import.

Copy of cone_slam/odom_history.py (separate ROS package, kept in sync by
hand). control_node uses it to carry the 10 Hz /slam/pose, which is the
pose at the LiDAR scan time, forward to the newest /odom sample on every
40 Hz tick (#33).
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import deque
from typing import Optional


class OdomHistory:
    """Planar (x, y, yaw) samples, newest last, interpolated by time."""

    def __init__(self, horizon_s: float = 2.0) -> None:
        self._horizon = horizon_s
        self._t: deque[float] = deque()
        self._pose: deque[tuple[float, float, float]] = deque()

    def clear(self) -> None:
        self._t.clear()
        self._pose.clear()

    def __len__(self) -> int:
        return len(self._t)

    def add(self, t: float, x: float, y: float, yaw: float) -> None:
        # A stamp at or before the newest one means the clock restarted
        # (new sim session, bag loop): start the history over.
        if self._t and t <= self._t[-1]:
            if t < self._t[-1]:
                self.clear()
            else:
                return
        self._t.append(t)
        self._pose.append((x, y, yaw))
        while self._t and self._t[0] < t - self._horizon:
            self._t.popleft()
            self._pose.popleft()

    def at(self, t: float, max_gap_s: float = 0.25) -> Optional[tuple[float, float, float]]:
        """Pose at time t, linearly interpolated (yaw on the circle).

        Returns None when t is outside the buffered window or the two
        samples around it are more than max_gap_s apart, so the caller
        can fall back to the newest sample instead of trusting a guess.
        A t up to max_gap_s past the newest sample returns the newest.
        """
        if not self._t:
            return None
        ts = self._t
        if t >= ts[-1]:
            return self._pose[-1] if t - ts[-1] <= max_gap_s else None
        if t < ts[0]:
            return None
        i = bisect_left(ts, t)
        if ts[i] == t:
            return self._pose[i]
        t0, t1 = ts[i - 1], ts[i]
        if t1 - t0 > max_gap_s:
            return None
        a = (t - t0) / (t1 - t0)
        x0, y0, h0 = self._pose[i - 1]
        x1, y1, h1 = self._pose[i]
        dh = (h1 - h0 + math.pi) % (2.0 * math.pi) - math.pi
        h = (h0 + a * dh + math.pi) % (2.0 * math.pi) - math.pi
        return (x0 + a * (x1 - x0), y0 + a * (y1 - y0), h)


def yaw_from_quat(qw: float, qz: float) -> float:
    """Yaw of a z-axis-only quaternion."""
    return 2.0 * math.atan2(qz, qw)


def project_pose(
    pose: tuple[float, float, float],
    odom_then: tuple[float, float, float],
    odom_now: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Move a map-frame pose by the odom-frame motion odom_then → odom_now.

    The motion is taken in odom's own frame and rotated by the yaw
    difference between the two frames at `odom_then`, i.e. map→odom is
    held constant over the interval.
    """
    x, y, yaw = pose
    ox0, oy0, oh0 = odom_then
    ox1, oy1, oh1 = odom_now
    dh = yaw - oh0
    c, s = math.cos(dh), math.sin(dh)
    ddx, ddy = ox1 - ox0, oy1 - oy0
    yaw1 = yaw + (oh1 - oh0)
    yaw1 = (yaw1 + math.pi) % (2.0 * math.pi) - math.pi
    return (x + c * ddx - s * ddy, y + s * ddx + c * ddy, yaw1)
