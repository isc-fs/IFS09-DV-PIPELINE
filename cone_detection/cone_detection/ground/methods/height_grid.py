"""Per-cell statistics and image-style filters over the height grid."""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def _cell_stats(cell, h, n_cells):
    """Per-cell count, min, max, mean and variance of ``h`` (empty: min=inf)."""
    cnt = np.zeros(n_cells, dtype=np.int64)
    mn = np.full(n_cells, np.inf)
    mx = np.full(n_cells, -np.inf)
    s = np.zeros(n_cells)
    ss = np.zeros(n_cells)
    for k in range(len(cell)):
        c = cell[k]
        v = h[k]
        cnt[c] += 1
        s[c] += v
        ss[c] += v * v
        if v < mn[c]:
            mn[c] = v
        if v > mx[c]:
            mx[c] = v
    mean = np.zeros(n_cells)
    var = np.zeros(n_cells)
    for c in range(n_cells):
        if cnt[c] > 0:
            mean[c] = s[c] / cnt[c]
            var[c] = max(ss[c] / cnt[c] - mean[c] * mean[c], 0.0)
    return cnt, mn, mx, mean, var


@numba.njit(cache=True)
def _window_extreme(grid, radius, take_max):
    """Min (or max) over a (2r+1)^2 window clipped at the border, one axis at a time."""
    nx, ny = grid.shape
    tmp = np.empty((nx, ny))
    out = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            best = grid[i, j]
            for b in range(max(0, j - radius), min(ny, j + radius + 1)):
                v = grid[i, b]
                if (v > best) if take_max else (v < best):
                    best = v
            tmp[i, j] = best
    for i in range(nx):
        a_lo = max(0, i - radius)
        a_hi = min(nx, i + radius + 1)
        for j in range(ny):
            best = tmp[i, j]
            for a in range(a_lo, a_hi):
                v = tmp[a, j]
                if (v > best) if take_max else (v < best):
                    best = v
            out[i, j] = best
    return out


def _grey_opening(grid: np.ndarray, size: int) -> np.ndarray:
    """``scipy.ndimage.grey_opening(grid, size=(size, size))`` for odd ``size``.

    Erosion then dilation with a flat square window. scipy's default ``reflect``
    border only repeats values already inside the window, so a window clipped at
    the border gives the same result. Compiled: on these few-thousand-cell grids
    the scipy call was mostly overhead.
    """
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    radius = int(size) // 2
    return _window_extreme(_window_extreme(grid, radius, False), radius, True)


def _fill_nearest(grid: np.ndarray, known: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Copy every unknown cell from its nearest known cell (``default`` if none)."""
    from scipy import ndimage

    if not known.any():
        return np.full(grid.shape, default)
    idx = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return grid[tuple(idx)]
