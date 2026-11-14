"""Height-grid methods: ``height_jump``, ``groundgrid`` and ``pmf``."""

from __future__ import annotations

import warnings

import numpy as np

from .height_grid import _cell_stats, _fill_nearest, _grey_opening, _window_extreme
from .scan_geometry import _level_cells
from .plane_fitting import _refit_on_ground
from .registry import register


# Height-jump grid: a cell whose returns span less than this is flat (floor,
# sidewalk, road bump); a cone's footprint spans >= ~0.15 m once a few rows hit.
_HJ_CELL_M = 0.30


_HJ_JUMP_M = 0.10


_HJ_MIN_POINTS = 3


# A flat cell counts as ground only if it is at most this far above the local
# floor (a curb / sidewalk), and the local floor itself must be near the plane:
# a few same-height returns on a far cone's flank are flat but sit higher.
_HJ_STEP_M = 0.18


_HJ_MAX_FLOOR_M = 0.30


_HJ_FLOOR_RADIUS = 2  # cells -> 5x5 window, 1.5 m


@register("height_jump")
def ground_height_jump(data, cfg):
    """Min/max height map: flat cells near the floor and the bottom of every cell are ground."""
    thr = cfg.ground_threshold_m
    cell, nx, ny, h, coarse = _level_cells(data, thr, _HJ_CELL_M)
    cnt, mn, mx, _, _ = _cell_stats(cell, h, nx * ny)
    # Local floor: min over a (2r+1)^2 window; empty cells are +inf, so never win.
    floor = _window_extreme(mn.reshape(nx, ny), _HJ_FLOOR_RADIUS, False).ravel()
    with np.errstate(invalid="ignore"):  # empty cells: inf - inf, compares False
        flat = (cnt >= _HJ_MIN_POINTS) & (mx - mn < _HJ_JUMP_M) & (mn - floor < _HJ_STEP_M)
    f = floor[cell]
    ground = ((h - f < thr) | flat[cell]) & (f < _HJ_MAX_FLOOR_M)
    non_ground = ~ground
    return non_ground, _refit_on_ground(data, non_ground, coarse)


# GroundGrid (Steinke et al., RA-L 2023), single scan: the paper accumulates the
# grid over time with odometry; here it is rebuilt every scan.
_GG_CELL_M = 0.33


_GG_MIN_POINTS = 4


_GG_MAX_STD_M = 0.03


_GG_MAX_DEV_M = 0.20  # a confident cell this far off its neighbours' median is a roof/table


@register("groundgrid")
def ground_groundgrid(data, cfg):
    """Elevation grid from low-variance cells, interpolated into the rest."""
    from scipy import ndimage

    thr = cfg.ground_threshold_m
    cell, nx, ny, h, coarse = _level_cells(data, thr, _GG_CELL_M)
    cnt, mn, _, mean, var = _cell_stats(cell, h, nx * ny)
    conf = (cnt >= _GG_MIN_POINTS) & (var < _GG_MAX_STD_M ** 2) & (np.abs(mean) < _HJ_MAX_FLOOR_M)
    elev = np.where(conf, mean, np.nan).reshape(nx, ny)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = ndimage.generic_filter(elev, np.nanmedian, size=7, mode="constant", cval=np.nan)
    conf2 = conf.reshape(nx, ny) & ~(np.abs(elev - med) > _GG_MAX_DEV_M)
    grid = _fill_nearest(np.nan_to_num(elev), conf2).ravel()
    non_ground = h - grid[cell] >= thr
    return non_ground, _refit_on_ground(data, non_ground, coarse)


# Progressive morphological filter (Zhang et al., TGRS 2003).
_PMF_CELL_M = 0.25


_PMF_WINDOWS = (3, 5, 9, 17)  # cells, 2 * 2^k + 1


_PMF_SLOPE = 0.05


_PMF_MAX_DH_M = 0.30


@register("pmf")
def ground_pmf(data, cfg):
    """Openings of growing size on the min-height surface; points far above it are objects."""
    thr = cfg.ground_threshold_m
    cell, nx, ny, h, coarse = _level_cells(data, thr, _PMF_CELL_M)
    cnt, mn, _, _, _ = _cell_stats(cell, h, nx * ny)
    surf = _fill_nearest(np.where(cnt > 0, mn, 0.0).reshape(nx, ny), (cnt > 0).reshape(nx, ny))
    non_ground = np.zeros(len(data), dtype=bool)
    prev_w = 1
    for w in _PMF_WINDOWS:
        opened = _grey_opening(surf, w)
        dh = thr if prev_w == 1 else min(thr + _PMF_SLOPE * (w - prev_w) * _PMF_CELL_M, _PMF_MAX_DH_M)
        non_ground |= h - opened.ravel()[cell] > dh
        surf = opened
        prev_w = w
    # Points under the final surface by more than the band are noise, not floor.
    non_ground |= h - surf.ravel()[cell] < -_PMF_MAX_DH_M
    return non_ground, _refit_on_ground(data, non_ground, coarse)
