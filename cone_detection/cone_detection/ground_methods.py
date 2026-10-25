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


@numba.njit(cache=True)
def _ray_walk(
    order, rng, z, resid, sector_start, sector_end, z0, max_slope, tol, max_gap, max_height
):
    """Classify points along each azimuth ray, nearest first. True = outlier."""
    out = np.ones(len(order), dtype=np.bool_)
    for s in range(len(sector_start)):
        prev_r = 0.0
        prev_z = z0
        for k in range(sector_start[s], sector_end[s]):
            i = order[k]
            dr = rng[i] - prev_r
            if dr > max_gap:
                # Too long without a ground return (occlusion): restart from the
                # sensor-height prior instead of extrapolating a slope across it.
                prev_r = rng[i]
                prev_z = z0
                dr = 0.0
            dz = z[i] - prev_z
            # The height cap stops a steep object (cone) being climbed one
            # small tolerated step at a time: the slope test is only local.
            if abs(dz) <= dr * max_slope + tol and abs(resid[i]) <= max_height:
                out[i] = False
                prev_r = rng[i]
                prev_z = z[i]
    return out


@register("ray_slope")
def ground_ray_slope(data, cfg, initial_plane=None, ransac_iter_subsample_max=5000):
    n_sectors = 360
    rng = np.hypot(data[:, 0], data[:, 1])
    az = np.arctan2(data[:, 1], data[:, 0])
    sector = ((az + np.pi) / (2 * np.pi) * n_sectors).astype(np.int64) % n_sectors
    order = np.lexsort((rng, sector)).astype(np.int64)
    sorted_sector = sector[order]
    bounds = np.searchsorted(sorted_sector, np.arange(n_sectors + 1))
    # Walk on height above a coarse GPF plane, not on raw z: the sensor is not
    # level with the floor (the real car's is ~9 deg off), and a slope limit on
    # raw z would reject all of that tilted ground.
    prior = _gpf_plane(_subsample(data), cfg.ransac_threshold)
    height = np.ascontiguousarray(plane_residual(data, prior))
    outlier = _ray_walk(
        order,
        rng,
        height,
        height,
        bounds[:-1].astype(np.int64),
        bounds[1:].astype(np.int64),
        0.0,
        np.tan(np.radians(6.0)),
        cfg.ransac_threshold,
        3.0,
        0.10,
    )
    return outlier, _refit_on_ground(data, outlier, _DEFAULT_PLANE.copy())


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
