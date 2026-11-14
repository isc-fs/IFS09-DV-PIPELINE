"""Abstract base for cone-detection behavior strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ConeObservation:
    """Single detected cone in the LiDAR / body frame.

    Attributes:
        x, y: Apex position in base_link (metres).
        height_m: Template apex height or fitted ``d`` (metres); used for
            big-orange vs small classification and marker.scale.z.
        sigma_xy: Position uncertainty in metres for SLAM; ``< 0`` means
            unknown (node publishes 0.1 m placeholder on marker.scale.x).
    """

    x: float
    y: float
    height_m: float
    sigma_xy: float = -1.0


# Cone height above which a detection goes to /Conos_Orange. The templates fix the
# fitted apex at 0.43 m (small) and 0.65 m (big orange, see fitting.cone_classes),
# so this sits just above the small template; the two_param backend fits d freely.
BIG_ORANGE_HEIGHT_THRESHOLD_M = 0.45


def _empty_xyz() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


@dataclass
class DetectionResult:
    """Per-scan output from a strategy (no ROS types)."""

    cones: list[ConeObservation] = field(default_factory=list)
    debug_counters: dict[str, int] = field(default_factory=dict)
    # Whole cropped scan in the ground frame used for clustering (floor at z=0,
    # lines up with /Conos_raw). Only filled when ``viz_full_cloud`` is asked.
    rotated_xyz: np.ndarray = field(default_factory=_empty_xyz)
    # Above-ground returns only (what clustering saw), same frame as the cones.
    above_ground_xyz: np.ndarray = field(default_factory=_empty_xyz)
    # DBSCAN label per ``above_ground_xyz`` row (-1 = noise).
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    stage_timings: dict[str, float] = field(default_factory=dict)


class ConeDetectionStrategy(ABC):
    """Per-behavior perception algorithm; the node owns ROS I/O and markers.

    Mode manager passes ``behavior`` via ``~/setup``; :class:`ConeDetectionNode`
    maps it with ``CONE_DETECTION_STRATEGY_MAP`` and instantiates the strategy
    in ``on_configure``, same pattern as path_planning / control.
    """

    @abstractmethod
    def configure(self) -> None:
        """One-shot setup (e.g. Numba JIT warmup) during lifecycle configure."""

    @abstractmethod
    def detect_cones(
        self,
        point_cloud: np.ndarray,
        *,
        viz_full_cloud: bool = False,
    ) -> DetectionResult:
        """Detect cones from an ``(N, 3)`` point cloud in the sensor frame.

        ``viz_full_cloud`` asks for the whole crop in the ground frame on
        ``DetectionResult.rotated_xyz`` (the /lidar_points/ground path).
        Skip it when nobody is subscribed — the 15 k-point rotate is
        otherwise wasted. ``above_ground_xyz`` is always filled.
        """

    def big_orange_height_threshold_m(self) -> float:
        """Height above which a cone goes to /Conos_Orange; strategies may override."""
        return BIG_ORANGE_HEIGHT_THRESHOLD_M
