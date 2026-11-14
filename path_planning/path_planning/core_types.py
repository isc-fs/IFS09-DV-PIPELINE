"""Shared dataclasses for the path planning package.

These types form the input/output contract between the ROS node
(`path_planning.py`) and the planner adapter (`fasttube_adapter.py`).
Kept dependency-free (numpy only) so the node and tests can import them
without pulling in the planner library itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cone:
    """Position-only cone observation in world frame.

    The pipeline used to carry a per-cone `color` field, populated by
    the cone_slam body_y heuristic. That classifier was removed when
    we found it had no real colour signal underlying it; the planner
    sorts cones by corridor geometry and ignores colour entirely.
    """

    x: float
    y: float


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float        # radians


@dataclass(frozen=True)
class PathPoint:
    x: float           # world frame
    y: float           # world frame
    yaw: float         # radians, tangent direction
    # Signed curvature (1/m) at this path point. Positive = left turn,
    # negative = right turn, 0 on straights. Sourced from the planner
    # library (circle fit over its path spline) when available; controller-side
    # finite-difference fallback when not.
    curvature: float = 0.0


def world_to_body(pts_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Project (N, 2) world-frame points into body frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy.copy()
    cos_y, sin_y = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    R = np.array([[cos_y, sin_y], [-sin_y, cos_y]], dtype=np.float64)
    return (pts_xy - np.array([pose.x, pose.y], dtype=np.float64)) @ R.T


def body_to_world(pts_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Project (N, 2) body-frame points into world frame."""
    if pts_xy.shape[0] == 0:
        return pts_xy.copy()
    cos_y, sin_y = float(np.cos(pose.yaw)), float(np.sin(pose.yaw))
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
    return pts_xy @ R.T + np.array([pose.x, pose.y], dtype=np.float64)
