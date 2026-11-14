"""DBSCAN labels -> one :class:`Cluster` per label, with the scalars the gates need.

Grouping and the per-cluster statistics are Numba kernels over the whole labelled
cloud, so a cluster the gates reject (most of them, and the largest) is never
masked or copied.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np


class Cluster(NamedTuple):
    """One DBSCAN cluster in the ground frame (z = height above the floor).

    A NamedTuple rather than a frozen dataclass: one is built per cluster per scan,
    and a frozen dataclass costs about twice as much to construct.
    """

    label: int
    points: np.ndarray  # every return of the cluster (a view, not a copy)
    n_clean: int  # returns above the floor margin
    top_m: float  # highest return above the floor margin (0 if none)
    bottom_m: float  # lowest return above the floor margin (0 if none)
    mean_x: float  # centroid of the returns above the floor margin
    mean_y: float

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def height_m(self) -> float:
        return self.top_m - self.bottom_m

    @property
    def range_m(self) -> float:
        return math.hypot(self.mean_x, self.mean_y)

    def points_above(self, floor_margin_m: float) -> np.ndarray:
        """The returns the cone fit uses."""
        return self.points[self.points[:, 2] > floor_margin_m]


class ClusterSet:
    """Every cluster of one scan, grouped by ascending label (noise, -1, first)."""

    def __init__(self, points: np.ndarray, labels: np.ndarray, floor_margin_m: float) -> None:
        self._points, starts, group_labels = _group_by_label(points, np.asarray(labels))
        self._starts = starts.tolist()
        self._labels = group_labels.tolist()
        n_clean, top, bottom, mean_x, mean_y = cluster_stats(
            self._points, starts, float(floor_margin_m)
        )
        self._stats = (
            n_clean.tolist(), top.tolist(), bottom.tolist(), mean_x.tolist(), mean_y.tolist()
        )

    def __len__(self) -> int:
        return len(self._starts) - 1

    def __iter__(self):
        n_clean, top, bottom, mean_x, mean_y = self._stats
        starts = self._starts
        for g in range(len(self)):
            yield Cluster(
                label=self._labels[g],
                points=self._points[starts[g] : starts[g + 1]],
                n_clean=n_clean[g],
                top_m=top[g],
                bottom_m=bottom[g],
                mean_x=mean_x[g],
                mean_y=mean_y[g],
            )


def _group_by_label(points: np.ndarray, labels: np.ndarray):
    """``(sorted_points, starts, group_labels)``: group g, with label ``group_labels[g]``,
    at ``sorted_points[starts[g]:starts[g + 1]]``.

    Ascending label, original order inside a group -- what a stable argsort gives.
    """
    sorted_points, starts, group_labels = group_by_label(
        points, np.asarray(labels, dtype=np.int64)
    )
    if len(starts) == 0:  # label range too wide to count: sort instead
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        bounds = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
        sorted_points = points[order]
        starts = np.concatenate(([0], bounds, [len(order)])).astype(np.int64)
        group_labels = sorted_labels[starts[:-1]]
    return sorted_points, starts, group_labels


@numba.njit(cache=True)
def group_by_label(data, labels):
    """Rows of ``data`` grouped by ascending label, original order kept inside a group.

    Same result as a stable argsort on ``labels`` plus a gather, by counting sort
    (labels are small consecutive ints). Returns ``(sorted_data, starts,
    group_labels)`` with group ``g`` (label ``group_labels[g]``) at
    ``sorted_data[starts[g]:starts[g + 1]]``; ``starts`` is empty when the label
    range is too wide to count (caller falls back to argsort).
    """
    n = labels.shape[0]
    lo = labels[0]
    hi = labels[0]
    for i in range(1, n):
        v = labels[i]
        if v < lo:
            lo = v
        elif v > hi:
            hi = v
    span = hi - lo + 1
    if span > 4 * n + 64:
        return data[:0].copy(), np.empty(0, np.int64), np.empty(0, np.int64)
    count = np.zeros(span + 1, np.int64)
    for i in range(n):
        count[labels[i] - lo + 1] += 1
    n_groups = 0
    for s in range(1, span + 1):
        if count[s] > 0:
            n_groups += 1
    starts = np.empty(n_groups + 1, np.int64)
    group_labels = np.empty(n_groups, np.int64)
    g = 0
    pos = 0
    for s in range(1, span + 1):
        c = count[s]
        count[s] = pos  # becomes the write cursor of label lo + s - 1
        if c > 0:
            starts[g] = pos
            group_labels[g] = lo + s - 1
            g += 1
            pos += c
    starts[n_groups] = n
    ncol = data.shape[1]
    out = np.empty((n, ncol), data.dtype)
    for i in range(n):
        s = labels[i] - lo + 1
        k = count[s]
        count[s] = k + 1
        for j in range(ncol):
            out[k, j] = data[i, j]
    return out, starts, group_labels


@numba.njit(cache=True)
def cluster_stats(pts, starts, floor_margin_m):
    """Per-group stats of the returns above ``floor_margin_m``.

    Returns ``(n_clean, z_top, z_min, mean_x, mean_y)``, one entry per group. Lets
    the shape gates run on scalars, so a rejected cluster (most of them, and the
    largest ones) never has its points masked or copied.
    """
    m = starts.shape[0] - 1
    n_clean = np.zeros(m, np.int64)
    z_top = np.zeros(m)
    z_min = np.zeros(m)
    mean_x = np.zeros(m)
    mean_y = np.zeros(m)
    for g in range(m):
        cnt = 0
        top = -np.inf
        low = np.inf
        sx = 0.0
        sy = 0.0
        for i in range(starts[g], starts[g + 1]):
            z = pts[i, 2]
            if z > floor_margin_m:
                cnt += 1
                if z > top:
                    top = z
                if z < low:
                    low = z
                sx += pts[i, 0]
                sy += pts[i, 1]
        n_clean[g] = cnt
        if cnt > 0:
            z_top[g] = top
            z_min[g] = low
            mean_x[g] = sx / cnt
            mean_y[g] = sy / cnt
    return n_clean, z_top, z_min, mean_x, mean_y
