"""What the pipeline reports about a scan, apart from the cones themselves.

* :class:`StageTimings` -- per-stage durations and point counts, written into the
  caller's ``stage_timings`` dict (the ROS node publishes some of them).
* :class:`ScanObserver` -- hooks every stage calls with its intermediate result, so a
  tool can see what each stage did without re-implementing or patching it. The
  detector does no observer work at all when none is passed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cone_detection.fitting.clusters import Cluster
    from cone_detection.fitting.fitters import ConeFit


class StageTimings:
    """Writes stage timings (ms) and counts into ``target``, or a private dict if None."""

    __slots__ = ("values",)

    def __init__(self, target: dict[str, float] | None = None) -> None:
        self.values = target if target is not None else {}

    def time(self, key: str) -> _Timer:
        """Record how long the ``with`` body took under ``key`` (ms)."""
        return _Timer(self.values, key)

    def record(self, key: str, value: float) -> None:
        self.values[key] = float(value)


class _Timer:
    """``StageTimings.time``'s context manager; a slotted class, not ``@contextmanager``,
    because it runs several times per scan and the generator version costs ~1 us each."""

    __slots__ = ("_values", "_key", "_t0")

    def __init__(self, values: dict[str, float], key: str) -> None:
        self._values = values
        self._key = key

    def __enter__(self) -> None:
        self._t0 = time.perf_counter()

    def __exit__(self, *exc) -> None:
        self._values[self._key] = (time.perf_counter() - self._t0) * 1000.0


class ScanObserver:
    """Receives each stage's intermediate result. Every hook is a no-op; override some.

    Point clouds are passed as the arrays the pipeline holds: do not modify them.
    Above-ground points are in the ground frame (plane normal = +z, floor at z=0).
    """

    def ground_removed(self, scan: np.ndarray, non_ground_mask: np.ndarray, plane: np.ndarray) -> None:
        """Ground removal ran on the cropped sensor-frame ``scan``; ``non_ground_mask`` is
        True for non-ground returns, ``plane`` is ``[bias, n_x, n_y, n_z]``."""

    def points_filtered(self, name: str, points: np.ndarray, keep_mask: np.ndarray | None) -> None:
        """Pre-filter ``name`` kept ``points[keep_mask]`` (``None``: kept every point)."""

    def points_capped(self, before: np.ndarray, after: np.ndarray) -> None:
        """The DBSCAN size guard reduced ``before`` to ``after`` (often the same array)."""

    def clustered(self, labels: np.ndarray, points: np.ndarray, plane: np.ndarray) -> None:
        """DBSCAN labelled ``points`` (-1 = noise).

        Same data as ``ScanResult.segmentation``; prefer that unless recording every stage.
        """

    def cluster_judged(
        self,
        cluster: Cluster,
        rejected_by: str | None,
        reason: str,
        fit: ConeFit | None,
        fit_points: np.ndarray | None,
    ) -> None:
        """One cluster went through the gates.

        ``rejected_by`` is the gate that dropped it (None: accepted), ``reason`` a
        readable verdict, ``fit`` the cone fit if it got that far and ``fit_points``
        the returns the fit used (the cluster minus its floor-margin returns).
        """
