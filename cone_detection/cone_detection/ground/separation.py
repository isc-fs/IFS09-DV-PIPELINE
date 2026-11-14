"""Ground stage: cropped sensor-frame scan -> above-ground points in the ground frame.

The :class:`~cone_detection.ground.methods.GroundRemover` labels the ground (see
``ground.methods``); the remaining points are rotated so the fitted plane normal is
+z and shifted so the plane is z = 0, so every later stage reads z as height above
the floor.
"""

from __future__ import annotations

import numpy as np

from cone_detection.ground.alignment import rotate_non_ground_to_ground
from cone_detection.ground.methods import GroundRemover
from cone_detection.instrumentation import ScanObserver, StageTimings


def separate_ground(
    scan: np.ndarray,
    ground: GroundRemover,
    timings: StageTimings,
    *,
    observer: ScanObserver | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(above_ground_points, plane)``; ``plane`` is ``[bias, n_x, n_y, n_z]``."""
    with timings.time("ground_ms"):
        non_ground, plane = ground.remove(scan)
    if observer is not None:
        observer.ground_removed(scan, non_ground, plane)
    with timings.time("rotate_ms"):
        above = rotate_non_ground_to_ground(np.asarray(scan), non_ground, plane)
    timings.record("n_above_ground", len(above))
    return above, plane
