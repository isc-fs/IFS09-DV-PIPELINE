"""
LiDAR cone detection for behavior ``base``: ground removal + DBSCAN + template-dispatch fit.

Returns :class:`DetectionResult` only; the ROS node builds and publishes markers.
"""

from __future__ import annotations

import traceback
from typing import Any, ClassVar

import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.detector import RealtimeConeDetector
from cone_detection.warmup import warmup_numba_functions
from cone_detection.strategies.cone_detection_strategy import (
    ConeDetectionStrategy,
    ConeObservation,
    DetectionResult,
    _empty_xyz,
)


class BaseConeDetection(ConeDetectionStrategy):
    """Default LiDAR cone pipeline (mode-manager behavior key ``base``)."""

    CONE_DETECTION_CONFIG: ClassVar[ConeDetectionConfig | None] = None

    # Log a per-scan template-dispatch summary every Nth scan (small/big counts,
    # residual stats). Set to 0 to disable. At 20 Hz, N=20 ≈ 1 Hz.
    LOG_FIT_COMPARISON_EVERY_N = 20

    def __init__(self, logger: Any, config: ConeDetectionConfig | None = None) -> None:
        """``config``: the node's settings (its ROS parameters over this strategy's own
        ``CONE_DETECTION_CONFIG``); None keeps the strategy's."""
        self._log = logger
        self._numba_warmup_done = False
        self._scan_count = 0
        self._config = config or self.CONE_DETECTION_CONFIG
        self._detector = RealtimeConeDetector(self._config)

    def configure(self) -> None:
        if not self._numba_warmup_done:
            # Passing the config lets warmup replay the live detect() call
            # chain with the production solver/backend, and (with cache=True
            # on the kernels) load the on-disk JIT cache instead of compiling.
            self._log.info(
                "warming up Numba kernels (~1-2 s cached, 10-20 s first ever)"
            )
            warmup_numba_functions(config=self._config)
            self._numba_warmup_done = True
            self._log.info("Numba warmup complete")

    def detect_cones(
        self,
        point_cloud: np.ndarray,
        *,
        stage_timings: dict[str, float] | None = None,
        viz_full_cloud: bool = False,
    ) -> DetectionResult:
        """Run ground removal + DBSCAN + template-dispatch fit; no ROS types."""
        compare_logger = None
        if self.LOG_FIT_COMPARISON_EVERY_N > 0:
            self._scan_count += 1
            if self._scan_count % self.LOG_FIT_COMPARISON_EVERY_N == 0:
                compare_logger = self._log

        debug_counters: dict[str, int] = {}
        cones: list[ConeObservation] = []
        rotated_xyz = above_ground_xyz = _empty_xyz()
        labels = np.zeros(0, dtype=np.int64)
        st = stage_timings if stage_timings is not None else {}
        # Degenerate clusters can still throw; swallow and log so one bad scan
        # does not take down the node.
        try:
            res = self._detector.detect(
                point_cloud,
                debug_counters=debug_counters,
                compare_logger=compare_logger,
                stage_timings=st,
                keep_scan=viz_full_cloud,
            )
            cones = [
                ConeObservation(
                    x=float(entry[0]),
                    y=float(entry[1]),
                    height_m=float(entry[2]) if len(entry) >= 3 else 0.0,
                    sigma_xy=float(entry[3]) if len(entry) >= 4 else -1.0,
                )
                for entry in res.cones
            ]
            above_ground_xyz = res.segmentation.above_ground_xyz
            labels = res.segmentation.labels
            if viz_full_cloud:
                rotated_xyz = res.segmentation.ground_frame_scan()
        except Exception:
            self._log.error(traceback.format_exc())

        return DetectionResult(
            cones=cones,
            debug_counters=debug_counters,
            rotated_xyz=rotated_xyz,
            above_ground_xyz=above_ground_xyz,
            labels=labels,
            stage_timings=st,
        )
