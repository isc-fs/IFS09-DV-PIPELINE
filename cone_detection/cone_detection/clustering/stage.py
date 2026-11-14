"""Clustering stage entry point: above-ground points -> ``(points, labels)``."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN

from cone_detection.clustering.dbscan import cluster_points
from cone_detection.clustering.prefilters import apply_prefilters
from cone_detection.config import ConeDetectionConfig
from cone_detection.instrumentation import ScanObserver, StageTimings


def cluster_scan(
    above: np.ndarray,
    cfg: ConeDetectionConfig,
    timings: StageTimings,
    *,
    observer: ScanObserver | None = None,
    clustering_class: type = DBSCAN,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-filters, DBSCAN size guard, then DBSCAN on the ground-frame ``above`` points.

    Returns ``(points, labels)``: ``points`` is what DBSCAN saw (a subset of
    ``above``) and ``labels`` its label per row (-1 = noise). ``clustering_class``
    is passed to :func:`~cone_detection.clustering.dbscan.cluster_points`.
    """
    points = apply_prefilters(above, cfg, timings, observer)
    return points, cluster_points(points, cfg, timings, clustering_class)
