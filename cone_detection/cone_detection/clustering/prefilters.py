"""Filters on the above-ground cloud before DBSCAN (z = height above the floor).

:func:`apply_prefilters` runs, in order, every enabled :class:`PointFilter` in
``PREFILTERS`` (flat-cell filter, tall-column veto) and then the DBSCAN size /
density guard (:func:`bound_dbscan_input`). Add a filter by adding a ``PointFilter``;
the pipeline does not change. See the matching fields of
:class:`~cone_detection.config.ConeDetectionConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.instrumentation import ScanObserver, StageTimings


# Grid cells are keyed as i * stride + j, injective while |j| < stride / 2.
# With the 25 m input crop and the 0.3 m default cell, |i|, |j| <= ~90.
_VETO_KEY_STRIDE = 1 << 20
# Largest occupancy grid the veto kernel may allocate (1 byte per cell); past it
# the sort-based path runs. The 25 m crop at the 0.3 m default cell is ~170 x 170.
_VETO_GRID_MAX_CELLS = 4_000_000


def _cell_keys(xy: np.ndarray, cell_m: float) -> np.ndarray:
    """One int64 key per point for its ``cell_m`` xy grid cell (see ``_VETO_KEY_STRIDE``)."""
    ij = np.floor(xy / cell_m).astype(np.int64)
    return ij[:, 0] * _VETO_KEY_STRIDE + ij[:, 1]


def column_veto_mask(
    xy: np.ndarray, height_above_ground: np.ndarray, cfg: ConeDetectionConfig
) -> np.ndarray | None:
    """Keep-mask over points, vetoing xy columns that contain a tall return.

    Returns ``None`` when no point exceeds ``tall_column_veto_height_m`` (the
    common case) so the caller can skip masking entirely.
    """
    # Dense-grid kernel: three passes, no int64 key arrays, sort or isin. The grid
    # only spans the tall returns, so it is small unless they are km apart.
    status, keep = tall_column_keep(
        xy,
        height_above_ground,
        float(cfg.tall_column_veto_height_m),
        float(cfg.tall_column_veto_cell_m),
        _VETO_GRID_MAX_CELLS,
    )
    if status == 0:
        return None
    if status == 1:
        return keep
    tall = height_above_ground > cfg.tall_column_veto_height_m
    keys = _cell_keys(xy, cfg.tall_column_veto_cell_m)
    tall_keys = np.unique(keys[tall])
    # 3x3 dilation in key space: cell (i+di, j+dj) has key + di*stride + dj.
    offsets = np.array(
        [di * _VETO_KEY_STRIDE + dj for di in (-1, 0, 1) for dj in (-1, 0, 1)],
        dtype=np.int64,
    )
    veto_keys = np.unique((tall_keys[:, None] + offsets).ravel())
    return ~np.isin(keys, veto_keys)


def flat_cell_mask(xy: np.ndarray, z: np.ndarray, cfg: ConeDetectionConfig) -> np.ndarray | None:
    """Keep-mask dropping points of flat xy cells (see ``flat_cell_filter``).

    Returns ``None`` when no cell is flat so the caller can skip masking.
    """
    keys = _cell_keys(xy, cfg.flat_cell_size_m)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    zs = z[order]
    span = np.maximum.reduceat(zs, starts) - np.minimum.reduceat(zs, starts)
    count = np.diff(np.r_[starts, len(sk)])
    flat_cell = (count >= cfg.flat_cell_min_points) & (span < cfg.flat_cell_range_m)
    if not flat_cell.any():
        return None
    cell_of_sorted = np.repeat(np.arange(len(starts)), count)
    keep = np.empty(len(keys), dtype=bool)
    keep[order] = ~flat_cell[cell_of_sorted]
    return keep


# Voxel keys for the DBSCAN guard: per-axis index offset into [0, 2^21) and
# packed into 63 bits. Injective for |index| < 2^20, i.e. any coordinate
# within ±20 km at the 0.02 m default cell — far beyond LiDAR range.
_GUARD_KEY_OFFSET = 1 << 20
_GUARD_KEY_BITS = 21


def _voxel_first_points(data: np.ndarray, voxel_m: float) -> np.ndarray:
    """Keep the first point of every ``voxel_m`` cell (row order preserved)."""
    ijk = np.floor(data[:, :3] / voxel_m).astype(np.int64) + _GUARD_KEY_OFFSET
    keys = (
        (ijk[:, 0] << (2 * _GUARD_KEY_BITS))
        | (ijk[:, 1] << _GUARD_KEY_BITS)
        | ijk[:, 2]
    )
    _, first = np.unique(keys, return_index=True)
    return data[np.sort(first)]


def bound_dbscan_input(
    data: np.ndarray, cfg: ConeDetectionConfig
) -> np.ndarray:
    """Cap the density and count of the cloud handed to DBSCAN.

    No-op (returns ``data`` itself) while ``len(data) <= dbscan_max_points``
    or the guard is disabled. See ``ConeDetectionConfig.dbscan_max_points``
    for the rationale. Row order is preserved so downstream label handling
    is unaffected.
    """
    cap = int(cfg.dbscan_max_points)
    if cap <= 0 or len(data) <= cap:
        return data
    voxel = float(cfg.dbscan_guard_voxel_m)
    while (
        voxel > 0.0
        and voxel <= cfg.dbscan_guard_voxel_max_m + 1e-9
        and len(data) > cap
    ):
        data = _voxel_first_points(data, voxel)
        voxel *= 2.0
    if len(data) > cap:
        # Seeded per scan size so a given input is reproducible; the choice
        # itself is uniform, so each cone keeps a proportional share.
        rng = np.random.default_rng(len(data))
        keep = rng.choice(len(data), cap, replace=False)
        data = data[np.sort(keep)]
    return data

# A keep-mask over the above-ground points from their xy and height (None: keep all).
KeepMask = Callable[[np.ndarray, np.ndarray, ConeDetectionConfig], "np.ndarray | None"]


@dataclass(frozen=True)
class PointFilter:
    """One pre-DBSCAN filter and how it is reported."""

    name: str  # passed to ScanObserver.points_filtered
    enabled: Callable[[ConeDetectionConfig], bool]
    keep_mask: KeepMask
    time_key: str  # stage_timings key of its duration (ms)
    dropped_key: str  # stage_timings key of how many points it dropped


PREFILTERS = (
    PointFilter(
        "flat_cell", lambda cfg: cfg.flat_cell_filter, flat_cell_mask,
        "flat_filter_ms", "n_flat_dropped",
    ),
    PointFilter(
        "column_veto", lambda cfg: cfg.tall_column_veto, column_veto_mask,
        "column_veto_ms", "n_vetoed",
    ),
)


def apply_prefilters(
    points: np.ndarray,
    cfg: ConeDetectionConfig,
    timings: StageTimings,
    observer: ScanObserver | None = None,
) -> np.ndarray:
    """``points`` minus what every enabled filter drops, capped by the DBSCAN guard."""
    for point_filter in PREFILTERS:
        if len(points) and point_filter.enabled(cfg):
            points = _apply(point_filter, points, cfg, timings, observer)
    if len(points) == 0:
        timings.record("n_dbscan_guard_dropped", 0)
        return points
    capped = bound_dbscan_input(points, cfg)
    timings.record("n_dbscan_guard_dropped", len(points) - len(capped))
    if observer is not None:
        observer.points_capped(points, capped)
    return capped


def _apply(point_filter, points, cfg, timings, observer):
    with timings.time(point_filter.time_key):
        keep = point_filter.keep_mask(points[:, :2], points[:, 2], cfg)
        kept = points if keep is None else points[keep]
    timings.record(point_filter.dropped_key, len(points) - len(kept))
    if observer is not None:
        observer.points_filtered(point_filter.name, points, keep)
    return kept


@numba.njit(cache=True)
def tall_column_keep(xy, height, tall_m, cell_m, max_cells):
    """Tall-column veto on a dense occupancy grid over the tall returns' bounding box.

    Returns ``(status, keep)``: status 0 = nothing tall (``keep`` is empty), 1 =
    ``keep`` is the mask, 2 = the grid would exceed ``max_cells`` (caller falls back
    to the sort-based version).
    """
    n = xy.shape[0]
    empty = np.empty(0, np.bool_)
    first = True
    i_lo = i_hi = j_lo = j_hi = 0
    for p in range(n):
        if height[p] > tall_m and math.isfinite(xy[p, 0]) and math.isfinite(xy[p, 1]):
            i = int(math.floor(xy[p, 0] / cell_m))
            j = int(math.floor(xy[p, 1] / cell_m))
            if first:
                i_lo = i_hi = i
                j_lo = j_hi = j
                first = False
            else:
                if i < i_lo:
                    i_lo = i
                elif i > i_hi:
                    i_hi = i
                if j < j_lo:
                    j_lo = j
                elif j > j_hi:
                    j_hi = j
    if first:
        return 0, empty
    # One cell of padding each side holds the 3x3 dilation.
    ni = i_hi - i_lo + 3
    nj = j_hi - j_lo + 3
    if ni * nj > max_cells:
        return 2, empty
    veto = np.zeros((ni, nj), np.bool_)
    for p in range(n):
        if height[p] > tall_m and math.isfinite(xy[p, 0]) and math.isfinite(xy[p, 1]):
            i = int(math.floor(xy[p, 0] / cell_m)) - i_lo
            j = int(math.floor(xy[p, 1] / cell_m)) - j_lo
            for di in range(3):
                for dj in range(3):
                    veto[i + di, j + dj] = True
    keep = np.ones(n, np.bool_)
    for p in range(n):
        fi = math.floor(xy[p, 0] / cell_m)
        fj = math.floor(xy[p, 1] / cell_m)
        # Cells outside the padded box hold no veto (also skips non-finite xy).
        if fi >= i_lo - 1 and fi <= i_hi + 1 and fj >= j_lo - 1 and fj <= j_hi + 1:
            if veto[int(fi) - i_lo + 1, int(fj) - j_lo + 1]:
                keep[p] = False
    return 1, keep
