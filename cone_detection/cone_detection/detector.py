"""LiDAR cone detection: one scan -> per-cone ``(x, y, height_m, sigma_xy)``.

:meth:`RealtimeConeDetector.detect` runs the stages in order; each lives in its
own package and has one entry point:

    preprocessing   crop_scan         raw scan -> cropped scan               (range crop)
    ground          separate_ground   cropped scan -> above-ground points    (ground removal, ground frame)
    clustering      cluster_scan      above-ground points -> DBSCAN labels   (pre-filters, DBSCAN)
    fitting         extract_cones     labelled points -> cones               (gates, cone fit, sigma_xy)

The ground stage already puts the above-ground points in the ground frame (z =
height above the floor) because pre-filters and DBSCAN read z that way; there is
no separate rotation stage. Rotating the *whole* scan is only for visualisation
and is done on demand (:meth:`Segmentation.ground_frame_scan`).

Tools that want the clusters rather than the cones read
``ScanResult.segmentation``, so they see exactly what the cones were fitted on.
The ROS node uses the detector through
:class:`~cone_detection.strategies.BaseConeDetection`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from cone_detection.clustering.stage import cluster_scan
from cone_detection.config import ConeDetectionConfig
from cone_detection.fitting.extraction import Cone, extract_cones
from cone_detection.fitting.fitters import cluster_fitter
from cone_detection.ground.alignment import rotate_xyz_to_ground
from cone_detection.ground.methods import make_ground_remover
from cone_detection.ground.separation import separate_ground
from cone_detection.instrumentation import ScanObserver, StageTimings
from cone_detection.preprocessing.range_crop import crop_scan


def _empty_xyz() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


@dataclass(frozen=True)
class Segmentation:
    """Everything up to clustering for one scan.

    Returned in :class:`ScanResult` so callers (strategy, benchmarks, tests) read
    the clusters the cones were fitted on instead of re-running the stages.
    Arrays are the ones the pipeline worked on, not copies: do not modify them.
    """

    # The cropped scan, sensor frame.
    scan: np.ndarray = field(default_factory=_empty_xyz)
    # The filtered above-ground returns DBSCAN saw, ground frame (z = height above floor).
    above_ground_xyz: np.ndarray = field(default_factory=_empty_xyz)
    # DBSCAN label per ``above_ground_xyz`` row (-1 = noise).
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Ground plane ``[bias, n_x, n_y, n_z]`` in the sensor frame; None when the
    # crop left nothing to fit.
    plane: np.ndarray | None = None

    def ground_frame_scan(self) -> np.ndarray:
        """The whole cropped scan in the ground frame, ground included (computed on call)."""
        if self.plane is None:
            return _empty_xyz()
        return rotate_xyz_to_ground(self.scan, self.plane)


@dataclass(frozen=True)
class ScanResult:
    """What :meth:`RealtimeConeDetector.detect` found in one scan."""

    cones: list[Cone] = field(default_factory=list)
    segmentation: Segmentation = field(default_factory=Segmentation)


class RealtimeConeDetector:
    """Configurable single-scan cone detector (see :class:`ConeDetectionConfig`).

    The only state it keeps between scans is the ground method's (see
    :class:`~cone_detection.ground.methods.GroundRemover`); every per-scan output
    is in the returned :class:`ScanResult`.
    """

    __slots__ = ("config", "_fit_cluster", "_ground")

    def __init__(self, config: ConeDetectionConfig | None = None) -> None:
        self.config = config or ConeDetectionConfig()
        self._fit_cluster = cluster_fitter(self.config.fit_backend)
        self._ground = make_ground_remover(self.config)

    def reset(self) -> None:
        """Forget state from previous scans (bag loop / seek, new run)."""
        self._ground.reset()

    def detect(
        self,
        data,
        *,
        debug_counters: dict | None = None,
        compare_logger=None,
        clustering_class: type = DBSCAN,
        stage_timings: dict[str, float] | None = None,
        observer: ScanObserver | None = None,
    ) -> ScanResult:
        """Detect cones in a single LiDAR scan.

        Args:
            data: ``(N, 3)`` array of LiDAR returns in the sensor frame.
            debug_counters: Optional dict filled with per-stage cluster counts.
            compare_logger: Optional object with ``info(str)``; gets a fit summary line.
            clustering_class: sklearn-style clustering class (see ``cluster_points``).
            stage_timings: Optional dict filled with per-stage timings (ms) and counts.
            observer: Optional :class:`ScanObserver` told every stage's result.

        Returns:
            The accepted cones (``(x, y, height_m, sigma_xy)`` each) and the
            segmentation they came from.
        """
        cfg = self.config
        # Counters and timings always go somewhere (a throwaway dict when the
        # caller passed none): the guard counters below read the timings back.
        counters = debug_counters if debug_counters is not None else {}
        timings = StageTimings(stage_timings)

        # 1. Preprocessing. Nothing left -> no plane to fit, and the ground
        # method's state (RANSAC's warm start) is left untouched.
        cropped = crop_scan(data, cfg) if len(data) else data
        if len(cropped) == 0:
            return ScanResult(segmentation=Segmentation(scan=cropped))
        counters["n_input_points"] = len(cropped)

        # 2. Ground.
        above, plane = separate_ground(cropped, self._ground, timings, observer=observer)

        # 3. Clustering.
        points, labels = cluster_scan(
            above, cfg, timings, observer=observer, clustering_class=clustering_class
        )
        # Same data as ScanResult.segmentation, for observers recording every stage.
        # No observer (the live node): this check is all it costs.
        if observer is not None:
            observer.clustered(labels, points, plane)
        dropped = int(timings.values.get("n_dbscan_guard_dropped", 0))
        counters["dbscan_guard_dropped"] = dropped
        counters["dbscan_guard_scans"] = int(dropped > 0)
        seg = Segmentation(scan=cropped, above_ground_xyz=points, labels=labels, plane=plane)
        if len(labels) == 0:
            return ScanResult(segmentation=seg)

        # 4. Fitting.
        cones = extract_cones(
            points,
            labels,
            cfg,
            self._fit_cluster,
            counters=counters,
            timings=timings,
            observer=observer,
            logger=compare_logger,
        )
        return ScanResult(cones=cones, segmentation=seg)
