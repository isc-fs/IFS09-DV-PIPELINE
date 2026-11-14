"""Position uncertainty handed to SLAM with each cone (``sigma_xy``, metres)."""

from __future__ import annotations

import math


def sigma_xy(a: float, b: float, n_points: int, ambiguous: bool) -> float:
    """Grows with range, shrinks with the number of returns; ambiguous class calls x1.5."""
    base_sigma = 0.05 + 0.005 * math.hypot(a, b)
    sigma = 1.5 * base_sigma / math.sqrt(max(1, n_points) / 10.0)
    return sigma * 1.5 if ambiguous else sigma
