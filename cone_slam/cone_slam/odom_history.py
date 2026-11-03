"""Short history of /odom poses, looked up by timestamp — no rclpy import.

A cone scan reaches SLAM ~80-120 ms after the LiDAR captured it. Pairing
its cones with the NEWEST /odom pose puts them where the car is now, not
where it was when it saw them: at 0.5 rad/s of yaw that is ~3 deg, ~0.8 m
at 15 m range — enough to break data association and to make the
map→odom correction jump by that much every scan (#33). Looking up /odom
at the scan's own stamp removes that error.
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
