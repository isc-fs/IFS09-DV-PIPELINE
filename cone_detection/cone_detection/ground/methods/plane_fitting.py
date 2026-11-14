"""Plane fits shared by every ground method: total least squares, GPF and the final refit.

Planes are ``[bias, n_x, n_y, n_z]`` with a unit normal and ``n_z > 0``.
"""

from __future__ import annotations

import numba
import numpy as np


# Max points used for plane *fits* (labelling always covers the full cloud).
_FIT_MAX_POINTS = 20_000


def _subsample(data: np.ndarray, cap: int = _FIT_MAX_POINTS) -> np.ndarray:
    if len(data) <= cap:
        return data
    return data[:: int(np.ceil(len(data) / cap))]


@numba.njit(cache=True)
def _plane_from_moments(n, sx, sy, sz, cxx, cxy, cxz, cyy, cyz, czz):
    """Plane through the centroid along the smallest-eigenvalue direction of the covariance."""
    cx, cy, cz = sx / n, sy / n, sz / n
    cov = np.empty((3, 3))
    cov[0, 0], cov[0, 1], cov[0, 2] = cxx, cxy, cxz
    cov[1, 0], cov[1, 1], cov[1, 2] = cxy, cyy, cyz
    cov[2, 0], cov[2, 1], cov[2, 2] = cxz, cyz, czz
    _, vecs = np.linalg.eigh(cov)
    nx, ny, nz = vecs[0, 0], vecs[1, 0], vecs[2, 0]
    if nz < 0:
        nx, ny, nz = -nx, -ny, -nz
    return np.array([-(nx * cx + ny * cy + nz * cz), nx, ny, nz])


@numba.njit(cache=True)
def _plane_of_selected(p, sel, n, sx, sy, sz):
    """Plane through the ``n`` rows of ``p`` with ``sel`` set, given their coordinate sums.

    Centring happens per point inside the loop, so the cloud is neither copied nor
    upcast (``points - centroid`` used to allocate a float64 (N, 3) array). Summing
    raw products instead would cancel catastrophically: the cm-level height spread
    sits next to metres of x / y range.
    """
    cx, cy, cz = sx / n, sy / n, sz / n
    cxx = cxy = cxz = cyy = cyz = czz = 0.0
    for i in range(p.shape[0]):
        if sel[i]:
            x = p[i, 0] - cx
            y = p[i, 1] - cy
            z = p[i, 2] - cz
            cxx += x * x
            cxy += x * y
            cxz += x * z
            cyy += y * y
            cyz += y * z
            czz += z * z
    return _plane_from_moments(n, sx, sy, sz, cxx, cxy, cxz, cyy, cyz, czz)


@numba.njit(cache=True)
def _fit_plane_nb(p):
    """Two passes over ``p`` (any float dtype / layout), float64 accumulation."""
    n = p.shape[0]
    sx = sy = sz = 0.0
    for i in range(n):
        sx += p[i, 0]
        sy += p[i, 1]
        sz += p[i, 2]
    return _plane_of_selected(p, np.ones(n, np.bool_), n, sx, sy, sz)


def fit_plane(points: np.ndarray) -> np.ndarray:
    """Total-least-squares plane through ``points`` -> ``[bias, n_x, n_y, n_z]``."""
    return _fit_plane_nb(points)


def plane_residual(data: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Signed point-to-plane distance (positive = above the plane)."""
    return data @ coefs[1:] + coefs[0]


_DEFAULT_PLANE = np.array([0.0, 0.0, 0.0, 1.0])


def _gpf_plane(
    pts: np.ndarray,
    thr: float,
    n_iter: int = 10,
    seed_band_m: float = 0.12,
    seed_pct: float = 2.0,
) -> np.ndarray:
    """Zermas ground-plane fitting on ``pts`` (already subsampled)."""
    if len(pts) < 3:
        return _DEFAULT_PLANE.copy()
    # Lowest-point representative: a low percentile, not the minimum, so a few
    # multipath returns under the floor do not drag the seed band below it.
    # Linear-interpolated like ``np.percentile`` but from two partitioned ranks
    # (no full sort).
    z = pts[:, 2].astype(np.float64)
    rank = (len(z) - 1) * seed_pct / 100.0
    lo = int(np.floor(rank))
    hi = min(lo + 1, len(z) - 1)
    part = np.partition(z, (lo, hi))
    lpr = part[lo] + (part[hi] - part[lo]) * (rank - lo)
    return _gpf_grow(pts, float(thr), int(n_iter), float(lpr + seed_band_m))


@numba.njit(cache=True)
def _gpf_grow(pts, thr, n_iter, z_cut):
    """Seed on ``z < z_cut``, then refit / regrow the inlier set up to ``n_iter`` times.

    Same iteration as the gather-based version (fit the seeds, keep every point
    within ``thr`` of the plane, stop when the set stops changing), but the set
    is a boolean mask over ``pts``: no ``pts[mask]`` copies and no centred array.
    """
    n_pts = pts.shape[0]
    mask = np.empty(n_pts, np.bool_)
    # The pass that (re)builds the mask also sums its points, so an iteration is
    # two passes (centred moments, regrow) instead of three. Same sums in the same
    # order as a separate pass.
    n = 0
    sx = sy = sz = 0.0
    for i in range(n_pts):
        mask[i] = pts[i, 2] < z_cut
        if mask[i]:
            n += 1
            sx += pts[i, 0]
            sy += pts[i, 1]
            sz += pts[i, 2]
    plane = np.array([0.0, 0.0, 0.0, 1.0])
    for _ in range(n_iter):
        if n < 3:
            break
        plane = _plane_of_selected(pts, mask, n, sx, sy, sz)
        grown = 0
        sx = sy = sz = 0.0
        for i in range(n_pts):
            r = pts[i, 0] * plane[1] + pts[i, 1] * plane[2] + pts[i, 2] * plane[3] + plane[0]
            mask[i] = abs(r) < thr
            if mask[i]:
                grown += 1
                sx += pts[i, 0]
                sy += pts[i, 1]
                sz += pts[i, 2]
        if grown == n:
            break
        n = grown
    return plane


def _coarse_plane(data: np.ndarray, thr: float) -> np.ndarray:
    """GPF plane of the (subsampled) whole cloud: the prior most methods start from."""
    return _gpf_plane(_subsample(data), thr)


def _all_but_ground(n: int, ground_idx) -> np.ndarray:
    """Outlier mask over ``n`` points with only ``ground_idx`` marked ground."""
    non_ground = np.ones(n, dtype=bool)
    non_ground[ground_idx] = False
    return non_ground


def _refit_on_ground(data: np.ndarray, non_ground: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    # Subsample the ground *indices*, then gather only those rows: same points as
    # ``_subsample(data[~non_ground])`` without copying the whole ground set first.
    # (now inside one kernel: no index array, no gathered copy of the ground rows)
    if (
        isinstance(non_ground, np.ndarray)
        and non_ground.dtype == np.bool_
        and isinstance(data, np.ndarray)
        and data.ndim == 2
    ):
        ok, plane = _fit_plane_ground(data, non_ground, _FIT_MAX_POINTS)
        return plane if ok else fallback
    idx = np.flatnonzero(~non_ground)
    if len(idx) < 3:
        return fallback
    if len(idx) > _FIT_MAX_POINTS:
        idx = idx[:: int(np.ceil(len(idx) / _FIT_MAX_POINTS))]
    return fit_plane(data[idx])


@numba.njit(cache=True)
def _fit_plane_ground(data, non_ground, cap):
    """``_fit_plane_nb`` over every ``step``-th ground row of ``data``, read in place.

    ``step`` thins the ground set to at most ``cap`` points. Returns ``(ok, plane)``;
    not ok with fewer than 3 ground points.
    """
    m = data.shape[0]
    n_ground = 0
    for i in range(m):
        if not non_ground[i]:
            n_ground += 1
    plane = np.array([0.0, 0.0, 0.0, 1.0])
    if n_ground < 3:
        return False, plane
    step = 1
    if n_ground > cap:
        step = (n_ground + cap - 1) // cap
    # ``skip`` counts down to the next kept ground row (no modulo per point).
    sel = np.zeros(m, np.bool_)
    n = 0
    skip = 0
    sx = sy = sz = 0.0
    for i in range(m):
        if not non_ground[i]:
            if skip == 0:
                sel[i] = True
                n += 1
                sx += data[i, 0]
                sy += data[i, 1]
                sz += data[i, 2]
                skip = step
            skip -= 1
    return True, _plane_of_selected(data, sel, n, sx, sy, sz)
