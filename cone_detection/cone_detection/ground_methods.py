"""Pluggable LiDAR ground-removal methods (``ConeDetectionConfig.ground_method``).

Every method has the same contract::

    method(data, cfg, initial_plane, ransac_iter_subsample_max) -> (outlier_mask, plane_coefs)

``data`` is the (N, 3) cloud in the sensor frame, ``outlier_mask`` is True for
non-ground returns and ``plane_coefs`` is ``[bias, n_x, n_y, n_z]`` with a unit
normal and ``n_z > 0`` (``bias + n . p = 0`` on the plane). The plane is always a
single global plane even for the non-planar methods, because the rest of the
pipeline uses it to rotate the outliers upright and to cull the floor margin. For
those methods it is least-squares refit on the points the method labelled ground.

Register a new method with ``@register("name")``; ``GROUND_METHODS`` is what the
benchmark's ``--ground-method`` and the ROS parameter validate against.

Methods:
    ransac       baseline, ``ransac.ransac2`` (single plane, warm-started).
    gpf          Ground Plane Fitting (Zermas et al., ICRA 2017): lowest-point-
                 representative seeds, iterated SVD plane fit.
    grid_resid   GPF plane + a per-cell ground-height correction from the cell's
                 lowest returns; follows bumps and slow slopes a plane cannot.
    ray_slope    per-azimuth ray walk (Autoware ray_ground_classifier style):
                 a return is ground while the slope from the last ground return
                 stays below a limit. Planarity-free.
    czm          concentric zone model (Patchwork-lite): one GPF plane per
                 range-ring x azimuth-sector patch, global plane as fallback.
    ransac_zones RANSAC per range-ring x azimuth-sector subregion.
    ransac_rings RANSAC per laser channel (arc), channels recovered from the
                 elevation angle.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numba
import numpy as np

from cone_detection.ransac import ransac2

# Max points used for plane *fits* (labelling always covers the full cloud).
_FIT_MAX_POINTS = 20_000


def _subsample(data: np.ndarray, cap: int = _FIT_MAX_POINTS) -> np.ndarray:
    if len(data) <= cap:
        return data
    return data[:: int(np.ceil(len(data) / cap))]


def fit_plane(points: np.ndarray) -> np.ndarray:
    """Total-least-squares plane through ``points`` -> ``[bias, n_x, n_y, n_z]``."""
    c = points.mean(axis=0)
    u = points - c
    _, vecs = np.linalg.eigh(u.T @ u)
    n = vecs[:, 0]
    if n[2] < 0:
        n = -n
    return np.array([-float(n @ c), n[0], n[1], n[2]], dtype=np.float64)


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
    lpr = np.percentile(pts[:, 2], seed_pct)
    seeds = pts[pts[:, 2] < lpr + seed_band_m]
    plane = _DEFAULT_PLANE.copy()
    for _ in range(n_iter):
        if len(seeds) < 3:
            break
        plane = fit_plane(seeds)
        grown = pts[np.abs(plane_residual(pts, plane)) < thr]
        converged = len(grown) == len(seeds)
        seeds = grown
        if converged:
            break
    return plane


def _refit_on_ground(data: np.ndarray, outlier: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    ground = data[~outlier]
    if len(ground) < 3:
        return fallback
    return fit_plane(_subsample(ground))


METHODS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        METHODS[name] = fn
        return fn

    return deco


@register("ransac")
def ground_ransac(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    A = np.c_[np.ones(data.shape[0]), data]
    warm = np.asarray(initial_plane, dtype=np.float64) if initial_plane is not None else np.zeros(0)
    inliers, coefs = ransac2(
        A,
        prob=cfg.ransac_prob,
        threshold=cfg.ransac_threshold,
        iter_subsample_max=ransac_iter_subsample_max,
        initial_coefs=warm,
    )
    outlier = np.ones(data.shape[0], dtype=bool)
    outlier[inliers] = False
    return outlier, coefs


@register("gpf")
def ground_gpf(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    thr = cfg.ransac_threshold
    plane = _gpf_plane(_subsample(data), thr)
    outlier = np.abs(plane_residual(data, plane)) >= thr
    return outlier, plane


@register("grid_resid")
def ground_grid_resid(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    thr = cfg.ransac_threshold
    cell = 0.5
    max_offset = 0.15
    plane = _gpf_plane(_subsample(data), thr)
    resid = plane_residual(data, plane)

    ij = np.floor(data[:, :2] / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    nx, ny = ij.max(axis=0) + 1
    flat = ij[:, 0] * ny + ij[:, 1]
    # A cell's ground height = its lowest return, but only trusted when the cell
    # has a few returns; a lone return may be the tip of a cone.
    cell_min = np.full(nx * ny, np.inf)
    np.minimum.at(cell_min, flat, resid)
    cell_n = np.bincount(flat, minlength=nx * ny)
    cell_min[cell_n < 4] = np.inf
    grid = cell_min.reshape(nx, ny)
    # 3x3 median over trusted neighbours rejects cells whose lowest return is a
    # cone sitting alone in the cell (occluded floor), without smearing bumps.
    padded = np.full((nx + 2, ny + 2), np.nan)
    padded[1:-1, 1:-1] = np.where(np.isfinite(grid), grid, np.nan)
    stack = np.stack(
        [padded[a : a + nx, b : b + ny] for a in range(3) for b in range(3)], axis=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods
        offset = np.nanmedian(stack, axis=0)
    offset = np.clip(np.nan_to_num(offset, nan=0.0), -max_offset, max_offset)
    outlier = np.abs(resid - offset.ravel()[flat]) >= thr
    return outlier, _refit_on_ground(data, outlier, plane)


# ray_slope runs as one Numba kernel so the python, fused and cpp detector paths
# share it: perception_cpp/cone_detection_core ports it 1:1 (same loop and
# expression order), and test_cpp_detect.py checks the two agree bit for bit.
# The plane fits use a fixed-sweep 3x3 Jacobi eigen solve instead of LAPACK eigh
# so both sides do the same arithmetic.
RAY_SLOPE_SECTORS = 360
RAY_SLOPE_MAX_SLOPE = float(np.tan(np.radians(6.0)))  # passed in, not recomputed per side
RAY_SLOPE_MAX_GAP_M = 3.0
RAY_SLOPE_MAX_HEIGHT_M = 0.10
_GPF_ITERS = 10
_GPF_SEED_BAND_M = 0.12
_GPF_SEED_PCT = 2.0


@numba.njit(cache=True)
def _sym3_min_eigvec(a00, a01, a02, a11, a12, a22):
    """Unit eigenvector of the smallest eigenvalue of a symmetric 3x3 (cyclic Jacobi)."""
    a = np.empty((3, 3))
    a[0, 0] = a00
    a[0, 1] = a01
    a[0, 2] = a02
    a[1, 0] = a01
    a[1, 1] = a11
    a[1, 2] = a12
    a[2, 0] = a02
    a[2, 1] = a12
    a[2, 2] = a22
    v = np.zeros((3, 3))
    v[0, 0] = 1.0
    v[1, 1] = 1.0
    v[2, 2] = 1.0
    for _sweep in range(50):
        if a[0, 1] == 0.0 and a[0, 2] == 0.0 and a[1, 2] == 0.0:
            break
        for pq in range(3):
            p = 0 if pq < 2 else 1
            q = 1 if pq == 0 else 2
            apq = a[p, q]
            if apq == 0.0:
                continue
            theta = (a[q, q] - a[p, p]) / (2.0 * apq)
            t = 1.0 / (abs(theta) + np.sqrt(theta * theta + 1.0))
            if theta < 0.0:
                t = -t
            c = 1.0 / np.sqrt(t * t + 1.0)
            s = t * c
            for k in range(3):
                akp = a[k, p]
                akq = a[k, q]
                a[k, p] = c * akp - s * akq
                a[k, q] = s * akp + c * akq
            for k in range(3):
                apk = a[p, k]
                aqk = a[q, k]
                a[p, k] = c * apk - s * aqk
                a[q, k] = s * apk + c * aqk
            a[p, q] = 0.0
            a[q, p] = 0.0
            for k in range(3):
                vkp = v[k, p]
                vkq = v[k, q]
                v[k, p] = c * vkp - s * vkq
                v[k, q] = s * vkp + c * vkq
    j = 0
    if a[1, 1] < a[j, j]:
        j = 1
    if a[2, 2] < a[j, j]:
        j = 2
    return v[0, j], v[1, j], v[2, j]


@numba.njit(cache=True)
def _fit_plane_idx(pts, idx, m):
    """``fit_plane(pts[idx[:m]])``: total least squares, ``[bias, n]`` with n_z >= 0."""
    sx = 0.0
    sy = 0.0
    sz = 0.0
    for k in range(m):
        i = idx[k]
        sx += pts[i, 0]
        sy += pts[i, 1]
        sz += pts[i, 2]
    cx = sx / m
    cy = sy / m
    cz = sz / m
    xx = 0.0
    xy = 0.0
    xz = 0.0
    yy = 0.0
    yz = 0.0
    zz = 0.0
    for k in range(m):
        i = idx[k]
        ux = pts[i, 0] - cx
        uy = pts[i, 1] - cy
        uz = pts[i, 2] - cz
        xx += ux * ux
        xy += ux * uy
        xz += ux * uz
        yy += uy * uy
        yz += uy * uz
        zz += uz * uz
    nx, ny, nz = _sym3_min_eigvec(xx, xy, xz, yy, yz, zz)
    if nz < 0.0:
        nx = -nx
        ny = -ny
        nz = -nz
    out = np.empty(4)
    out[0] = -(nx * cx + ny * cy + nz * cz)
    out[1] = nx
    out[2] = ny
    out[3] = nz
    return out


@numba.njit(cache=True)
def _stride_idx(src, m, cap):
    """``_subsample``: every ceil(m / cap)-th of ``src[:m]`` (all of it when m <= cap)."""
    step = 1 if m <= cap else (m + cap - 1) // cap
    k = (m + step - 1) // step
    out = np.empty(k, dtype=np.int64)
    for j in range(k):
        out[j] = src[j * step]
    return out


@numba.njit(cache=True)
def _height(pts, i, c):
    return ((pts[i, 0] * c[1] + pts[i, 1] * c[2]) + pts[i, 2] * c[3]) + c[0]


@numba.njit(cache=True)
def _gpf_plane_idx(pts, sub, thr):
    """Zermas GPF on ``pts[sub]`` (= ``_gpf_plane``)."""
    n = len(sub)
    plane = np.zeros(4)
    plane[3] = 1.0
    if n < 3:
        return plane
    # np.percentile(z, 2.0), method "linear" (numpy's _lerp, both branches).
    zs = np.empty(n)
    for k in range(n):
        zs[k] = pts[sub[k], 2]
    zs.sort()
    h = (n - 1) * (_GPF_SEED_PCT / 100.0)
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    g = h - lo
    d = zs[hi] - zs[lo]
    lpr = zs[hi] - d * (1.0 - g) if g >= 0.5 else zs[lo] + d * g
    seeds = np.empty(n, dtype=np.int64)
    m = 0
    for k in range(n):
        if pts[sub[k], 2] < lpr + _GPF_SEED_BAND_M:
            seeds[m] = sub[k]
            m += 1
    for _ in range(_GPF_ITERS):
        if m < 3:
            break
        plane = _fit_plane_idx(pts, seeds, m)
        grown = 0
        for k in range(n):
            if abs(_height(pts, sub[k], plane)) < thr:
                seeds[grown] = sub[k]
                grown += 1
        converged = grown == m
        m = grown
        if converged:
            break
    return plane


@numba.njit(cache=True)
def ray_slope_kernel(pts, thr, max_slope, max_gap, max_height, n_sectors, fit_cap):
    """ray_slope ground removal on the (N, 3) float64 cloud -> (outlier, plane).

    Walks each azimuth ray nearest first on height above a coarse GPF plane (the
    sensor is not level with the floor: ~9 deg on the real car). A return is
    ground while the slope from the last ground return stays under ``max_slope``
    (+ ``thr``) and it is within ``max_height`` of the prior; a gap over
    ``max_gap`` (occlusion) restarts the walk from the prior. The plane is then
    refit on the ground returns.
    """
    n = len(pts)
    all_idx = np.arange(n)
    prior = _gpf_plane_idx(pts, _stride_idx(all_idx, n, fit_cap), thr)
    height = np.empty(n)
    rng = np.empty(n)
    sector = np.empty(n, dtype=np.int64)
    two_pi = 2.0 * np.pi
    for i in range(n):
        height[i] = _height(pts, i, prior)
        rng[i] = np.hypot(pts[i, 0], pts[i, 1])
        az = np.arctan2(pts[i, 1], pts[i, 0])
        sector[i] = np.int64((az + np.pi) / two_pi * n_sectors) % n_sectors
    # = np.lexsort((rng, sector)): by sector, then range, then index.
    order = np.argsort(rng, kind="mergesort")
    order = order[np.argsort(sector[order], kind="mergesort")]
    out = np.ones(n, dtype=np.bool_)
    k = 0
    while k < n:
        s = sector[order[k]]
        prev_r = 0.0
        prev_z = 0.0
        while k < n and sector[order[k]] == s:
            i = order[k]
            dr = rng[i] - prev_r
            if dr > max_gap:
                # Too long without a ground return (occlusion): restart from the
                # prior instead of extrapolating a slope across it.
                prev_r = rng[i]
                prev_z = 0.0
                dr = 0.0
            dz = height[i] - prev_z
            # The height cap stops a steep object (cone) being climbed one small
            # tolerated step at a time: the slope test is only local.
            if abs(dz) <= dr * max_slope + thr and abs(height[i]) <= max_height:
                out[i] = False
                prev_r = rng[i]
                prev_z = height[i]
            k += 1
    m = 0
    for i in range(n):
        if not out[i]:
            all_idx[m] = i
            m += 1
    plane = np.zeros(4)
    plane[3] = 1.0
    if m >= 3:
        sub = _stride_idx(all_idx, m, fit_cap)
        plane = _fit_plane_idx(pts, sub, len(sub))
    return out, plane


@register("ray_slope")
def ground_ray_slope(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    return ray_slope_kernel(
        np.ascontiguousarray(data, dtype=np.float64),
        float(cfg.ransac_threshold),
        RAY_SLOPE_MAX_SLOPE,
        RAY_SLOPE_MAX_GAP_M,
        RAY_SLOPE_MAX_HEIGHT_M,
        RAY_SLOPE_SECTORS,
        _FIT_MAX_POINTS,
    )


# Patchwork-style zones: ring edges (m) and azimuth sectors per ring. The inner
# rings are small and densely sampled; the outer ones are wide and sparse.
_CZM_EDGES = np.array([0.0, 4.0, 8.0, 14.0, 1e9])
_CZM_SECTORS = (8, 16, 24, 16)
_CZM_MIN_POINTS = 12
_CZM_MIN_NZ = 0.9


@register("czm")
def ground_czm(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    thr = cfg.ransac_threshold
    glob = _gpf_plane(_subsample(data), thr)
    outlier = np.abs(plane_residual(data, glob)) >= thr

    rng = np.hypot(data[:, 0], data[:, 1])
    ring = np.searchsorted(_CZM_EDGES, rng, side="right") - 1
    az = np.arctan2(data[:, 1], data[:, 0])
    sec_n = np.array(_CZM_SECTORS)[ring]
    sec = (((az + np.pi) / (2 * np.pi)) * sec_n).astype(np.int64) % sec_n
    patch = ring * 64 + sec
    order = np.argsort(patch, kind="stable")
    sorted_patch = patch[order]
    uniq, starts = np.unique(sorted_patch, return_index=True)
    ends = np.append(starts[1:], len(order))
    for start, end in zip(starts, ends):
        idx = order[start:end]
        if len(idx) < _CZM_MIN_POINTS:
            continue  # keep the global-plane labels for sparse patches
        pts = data[idx]
        plane = _gpf_plane(pts, thr)
        # A patch plane that is not roughly horizontal fitted a wall or a
        # cone cluster, not the floor: keep the global labels there.
        if plane[3] < _CZM_MIN_NZ:
            continue
        outlier[idx] = np.abs(plane_residual(pts, plane)) >= thr
    return outlier, _refit_on_ground(data, outlier, glob)


def _ransac_patch(pts: np.ndarray, cfg, subsample_max: int):
    """RANSAC plane on one patch -> (outlier_mask, coefs)."""
    inliers, coefs = ransac2(
        np.c_[np.ones(len(pts)), pts],
        prob=cfg.ransac_prob,
        threshold=cfg.ransac_threshold,
        iter_subsample_max=subsample_max,
    )
    outlier = np.ones(len(pts), dtype=bool)
    outlier[inliers] = False
    return outlier, coefs


# A patch/ring plane is accepted only if its normal is within this angle of the
# global plane's. Otherwise RANSAC fitted a wall, a tyre stack or a sky-only
# ring (no floor at all), and the global labels are kept for those points.
_MAX_NORMAL_DEV_COS = np.cos(np.radians(10.0))
_PATCH_MIN_POINTS = 30


def _local_ransac(data, cfg, groups, ransac_iter_subsample_max, initial_plane):
    """Global RANSAC labels, overridden by per-group RANSAC where it is trustworthy."""
    outlier, glob = ground_ransac(data, cfg, initial_plane, ransac_iter_subsample_max)
    for idx in groups:
        if len(idx) < _PATCH_MIN_POINTS:
            continue
        local_out, plane = _ransac_patch(data[idx], cfg, ransac_iter_subsample_max)
        if float(plane[1:] @ glob[1:]) < _MAX_NORMAL_DEV_COS:
            continue
        outlier[idx] = local_out
    return outlier, _refit_on_ground(data, outlier, glob)


@register("ransac_zones")
def ground_ransac_zones(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    """RANSAC per concentric-ring x azimuth-sector subregion."""
    rng = np.hypot(data[:, 0], data[:, 1])
    ring = np.searchsorted(_CZM_EDGES, rng, side="right") - 1
    az = np.arctan2(data[:, 1], data[:, 0])
    sec_n = np.array(_CZM_SECTORS)[ring]
    sec = (((az + np.pi) / (2 * np.pi)) * sec_n).astype(np.int64) % sec_n
    patch = ring * 64 + sec
    order = np.argsort(patch, kind="stable")
    _, starts = np.unique(patch[order], return_index=True)
    groups = np.split(order, starts[1:])
    return _local_ransac(data, cfg, groups, ransac_iter_subsample_max, initial_plane)


# Laser channels are recovered from the elevation angle (the xyz-only cloud has
# no ring field): one channel keeps its elevation at every range and azimuth, so
# returns of a channel sit in a thin band. A gap wider than this starts a new one.
_RING_GAP_DEG = 0.05
_RING_MAX_CHANNELS = 400


@register("ransac_rings")
def ground_ransac_rings(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    """RANSAC per laser channel (the arc of ground one laser draws)."""
    el = np.degrees(np.arctan2(data[:, 2], np.hypot(data[:, 0], data[:, 1])))
    order = np.argsort(el, kind="stable")
    gaps = np.diff(el[order]) > _RING_GAP_DEG
    if gaps.sum() + 1 > _RING_MAX_CHANNELS:
        # Elevation is not channel-clean (pitched/rolling car, noisy sensor):
        # fall back to fixed-width elevation bands.
        gaps = np.diff(np.floor(el[order] / 0.15)) > 0
    groups = np.split(order, np.flatnonzero(gaps) + 1)
    return _local_ransac(data, cfg, groups, ransac_iter_subsample_max, initial_plane)


GROUND_METHODS = tuple(METHODS)


def remove_ground(
    method: str,
    data: np.ndarray,
    cfg,
    initial_plane: np.ndarray | None = None,
    ransac_iter_subsample_max: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to ``method``; see the module docstring for the contract."""
    try:
        fn = METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown ground_method {method!r}; choose from {sorted(METHODS)}"
        ) from None
    if len(data) < 3:  # no plane to fit; callers treat an empty/tiny cloud as all-outlier
        return np.ones(len(data), dtype=bool), _DEFAULT_PLANE.copy()
    return fn(data, cfg, initial_plane, ransac_iter_subsample_max)
