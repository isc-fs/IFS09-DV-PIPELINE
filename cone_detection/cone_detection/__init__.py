"""LiDAR cone detection for the DV pipeline.

Layout, one package per pipeline stage, in pipeline order:

    preprocessing/   raw scan -> cropped scan                 (range crop)
    ground/          cropped scan -> above-ground points       (ground methods,
                                                               rotation into the ground frame)
    clustering/      above-ground points -> DBSCAN labels      (pre-filters, DBSCAN backends)
    fitting/         labelled points -> cones                  (gates, cone fits, sigma_xy)

and, around them:

    config.py           ConeDetectionConfig: every tunable, one ROS parameter each
    detector.py         RealtimeConeDetector: runs the stages in order (swappable:
                        the node only sees it through a strategy)
    instrumentation.py  stage timings and the ScanObserver hooks
    warmup.py           configure-time Numba warmup
    strategies/         per-behavior detection strategies the node selects; the seam
                        between ROS and the algorithm (DetectionResult)

and the ROS 2 side, at the root because it is the package's entry point and stays
whatever detector runs behind the strategies:

    node.py             the lifecycle node (``cone_detection_node`` executable)
    conversions.py      ROS message <-> numpy conversions the node uses

Nothing heavy is imported here, so ``cone_detection.config`` stays cheap to import.
"""
