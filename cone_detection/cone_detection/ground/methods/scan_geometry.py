"""Scan geometry the grid / ray / range-image methods share.

They work in a frame levelled on a coarse GPF plane: ``uv`` are coordinates in that
plane, ``h`` the height above it. The real sensor is ~9 deg off the floor, so cells
and slopes on raw xyz would put the whole floor on a ramp. Also: azimuth sectors and
laser channels recovered from the elevation angle.
"""

from __future__ import annotations

import numba
import numpy as np

from .plane_fitting import _coarse_plane


# ---------------------------------------------------------------------------
# Grid / ray / range-image methods. They work in a frame levelled on a coarse
# GPF plane: ``uv`` are coordinates in that plane, ``h`` the height above it.
# The real sensor is ~9 deg off the floor, so cells and slopes on raw xyz would
# put the whole floor on a ramp.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _project_point(data, k, e1, e2, n, d):
    """Row ``k`` of ``data`` in the levelled frame: ``(u, v, h)`` in float64."""
    x = np.float64(data[k, 0])
    y = np.float64(data[k, 1])
    z = np.float64(data[k, 2])
    u = x * e1[0] + y * e1[1] + z * e1[2]
    v = x * e2[0] + y * e2[1] + z * e2[2]
    return u, v, x * n[0] + y * n[1] + z * n[2] + d


@numba.njit(cache=True)
def _project_uvh(data, e1, e2, n, d):
    """One pass: in-plane coords ``uv`` and height ``h = n.p + d`` (float64).

    Reads float32 or float64 rows directly, so the cloud is not upcast to a
    float64 copy (twice) as ``data @ M`` and ``plane_residual`` would do.
    """
    m = data.shape[0]
    uv = np.empty((m, 2))
    h = np.empty(m)
    for k in range(m):
        uv[k, 0], uv[k, 1], h[k] = _project_point(data, k, e1, e2, n, d)
    return uv, h


def _level_frame(data: np.ndarray, thr: float):
    """Coarse plane and the in-plane axes ``e1``, ``e2`` of the levelled frame."""
    plane = _coarse_plane(data, thr)
    n = plane[1:]
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return plane, e1, e2, np.ascontiguousarray(n)


def _level(data: np.ndarray, thr: float):
    """Coarse plane -> ``(uv, h, plane)`` for ``data`` in the levelled frame."""
    plane, e1, e2, n = _level_frame(data, thr)
    uv, h = _project_uvh(np.ascontiguousarray(data), e1, e2, n, float(plane[0]))
    return uv, h, plane


@numba.njit(cache=True)
def _level_cells_nb(data, e1, e2, n, d, size):
    """``_project_uvh`` fused with binning into ``size`` cells: ``(cell, h, nx, ny)``.

    For the grid methods, which only use ``uv`` to bin the points: ``uv`` is never
    stored, and each point's cell coordinates are computed once (the grid origin is
    only known after the first pass, so they are kept and offset in the second).
    """
    m = data.shape[0]
    h = np.empty(m)
    cell = np.empty(m, dtype=np.int64)
    if m == 0:
        return cell, h, 1, 1
    cj = np.empty(m, dtype=np.int64)
    imin = jmin = np.iinfo(np.int64).max
    imax = jmax = np.iinfo(np.int64).min
    for k in range(m):
        u, v, h[k] = _project_point(data, k, e1, e2, n, d)
        i = np.int64(np.floor(u / size))
        j = np.int64(np.floor(v / size))
        cell[k] = i
        cj[k] = j
        imin = min(imin, i)
        imax = max(imax, i)
        jmin = min(jmin, j)
        jmax = max(jmax, j)
    ny = jmax - jmin + 1
    for k in range(m):
        cell[k] = (cell[k] - imin) * ny + (cj[k] - jmin)
    return cell, h, imax - imin + 1, ny


def _level_cells(data: np.ndarray, thr: float, size: float):
    """Coarse plane -> ``(cell, nx, ny, h, plane)``: flat cell id and height per point."""
    plane, e1, e2, n = _level_frame(data, thr)
    cell, h, nx, ny = _level_cells_nb(
        np.ascontiguousarray(data), e1, e2, n, float(plane[0]), float(size)
    )
    return cell, int(nx), int(ny), h, plane


def _sectors(uv: np.ndarray, n_sectors: int):
    """Range in the levelled plane, plus points sorted by (sector, range)."""
    rng = np.hypot(uv[:, 0], uv[:, 1])
    az = np.arctan2(uv[:, 1], uv[:, 0])
    sector = ((az + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64) % n_sectors
    order = np.lexsort((rng, sector)).astype(np.int64)
    bounds = np.searchsorted(sector[order], np.arange(n_sectors + 1))
    return rng, order, bounds[:-1].astype(np.int64), bounds[1:].astype(np.int64)


# Laser channels are recovered from the elevation angle (the xyz-only cloud has
# no ring field): returns of one channel sit in a thin elevation band, and a gap
# wider than this starts a new one. The ATX's channels are not flat: elevation
# drifts linearly with azimuth (~0.0023 deg/deg, +-0.14 deg over its 120 deg FoV,
# the same for all channels), which smears neighbouring channels (0.085 deg apart)
# together. The slope is estimated per scan and removed first; on the real bag
# that brings the in-channel spread from 0.08 to 0.008 deg.
_RING_GAP_DEG = 0.03


_RING_MAX_CHANNELS = 400


_RING_SLOPES = np.linspace(-0.01, 0.01, 41)  # deg of elevation per deg of azimuth


def _ring_slope(el: np.ndarray, az: np.ndarray) -> float:
    """Azimuth slope of the channel elevations: the one that splits the most channels."""
    sub = slice(None, None, max(1, len(el) // 20_000))
    e, a = el[sub], az[sub]

    def n_gaps(k):
        return int((np.diff(np.sort(e - k * a)) > _RING_GAP_DEG).sum())

    best = max(_RING_SLOPES, key=n_gaps)
    step = _RING_SLOPES[1] - _RING_SLOPES[0]
    return float(max(np.linspace(best - step, best + step, 11), key=n_gaps))


def _recover_rings(data: np.ndarray) -> np.ndarray:
    """Laser channel per point from the sensor-frame elevation (see ``_RING_GAP_DEG``)."""
    el = np.degrees(np.arctan2(data[:, 2], np.hypot(data[:, 0], data[:, 1])))
    if len(el) > 1:
        az = np.degrees(np.arctan2(data[:, 1], data[:, 0]))
        el = el - _ring_slope(el, az) * az
    order = np.argsort(el, kind="stable")
    gaps = np.diff(el[order]) > _RING_GAP_DEG
    if gaps.sum() + 1 > _RING_MAX_CHANNELS:
        gaps = np.diff(np.floor(el[order] / 0.15)) > 0
    ring = np.empty(len(data), dtype=np.int64)
    ring[order] = np.r_[0, np.cumsum(gaps)]
    return ring
