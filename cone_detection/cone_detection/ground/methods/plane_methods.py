"""Plane-based methods: ``ransac`` (production), ``gpf``, ``grid_resid``, ``czm``,
``ransac_zones`` and ``ransac_rings``.
"""

from __future__ import annotations

import warnings

import numpy as np

from cone_detection.ground.ransac import ransac2, ransac_ground_mask

from .scan_geometry import _recover_rings
from .plane_fitting import (
    _coarse_plane,
    _gpf_plane,
    _all_but_ground,
    _refit_on_ground,
    plane_residual,
)
from .registry import register


def ground_ransac(data, cfg, warm_plane=None):
    """One RANSAC plane fit -> ``(non_ground_mask, plane)``; ``warm_plane`` is tried first.

    Pure function; :class:`RansacGround` is the registered method that keeps the
    previous scan's plane as ``warm_plane``.
    """
    warm = np.asarray(warm_plane, dtype=np.float64) if warm_plane is not None else np.zeros(0)
    iter_max = int(cfg.ransac_iter_subsample_max)
    if not cfg.ground_threshold_m or data.ndim != 2 or data.shape[1] != 3:
        return _ground_ransac_legacy(data, cfg, warm, iter_max)
    # The kernel reads the decoded cloud in place: no bias-augmented float64 copy,
    # and it returns the non-ground mask itself.
    return ransac_ground_mask(
        data,
        float(cfg.ransac_prob),
        float(cfg.ground_threshold_m),
        200,
        iter_max,
        warm,
    )


def _ground_ransac_legacy(data, cfg, warm, ransac_iter_subsample_max):
    """``ransac2`` on the augmented matrix -> ``(non_ground_mask, coefs)``.

    For a MAD threshold (``ground_threshold_m`` 0), non-3-D data, and the per-patch
    fits (``warm`` None: no warm start).
    """
    A = np.empty((data.shape[0], data.shape[1] + 1))
    A[:, 0] = 1.0
    A[:, 1:] = data
    inliers, coefs = ransac2(
        A,
        prob=cfg.ransac_prob,
        threshold=cfg.ground_threshold_m,
        iter_subsample_max=ransac_iter_subsample_max,
        initial_coefs=warm,
    )
    return _all_but_ground(data.shape[0], inliers), coefs


@register("ransac")
class RansacGround:
    """Single-plane RANSAC, warm-started from the previous scan's plane.

    At 10 Hz the plane barely moves frame to frame, so the warm candidate usually
    wins immediately and collapses the iteration budget; a stale one just loses
    the support comparison.
    """

    __slots__ = ("_cfg", "_prev_plane")

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._prev_plane: np.ndarray | None = None

    def _fit_global(self, data):
        non_ground, plane = ground_ransac(data, self._cfg, self._prev_plane)
        self._prev_plane = plane
        return non_ground, plane

    def remove(self, data):
        return self._fit_global(data)

    def reset(self) -> None:
        self._prev_plane = None


@register("gpf")
def ground_gpf(data, cfg):
    thr = cfg.ground_threshold_m
    plane = _coarse_plane(data, thr)
    non_ground = np.abs(plane_residual(data, plane)) >= thr
    return non_ground, plane


@register("grid_resid")
def ground_grid_resid(data, cfg):
    thr = cfg.ground_threshold_m
    cell = 0.5
    max_offset = 0.15
    plane = _coarse_plane(data, thr)
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
    non_ground = np.abs(resid - offset.ravel()[flat]) >= thr
    return non_ground, _refit_on_ground(data, non_ground, plane)


# Patchwork-style zones: ring edges (m) and azimuth sectors per ring. The inner
# rings are small and densely sampled; the outer ones are wide and sparse.
_CZM_EDGES = np.array([0.0, 4.0, 8.0, 14.0, 1e9])


_CZM_SECTORS = (8, 16, 24, 16)


_CZM_MIN_POINTS = 12


_CZM_MIN_NZ = 0.9


def _czm_patches(data: np.ndarray) -> list[np.ndarray]:
    """Point indices of every non-empty range-ring x azimuth-sector patch."""
    rng = np.hypot(data[:, 0], data[:, 1])
    ring = np.searchsorted(_CZM_EDGES, rng, side="right") - 1
    az = np.arctan2(data[:, 1], data[:, 0])
    sec_n = np.array(_CZM_SECTORS)[ring]
    sec = (((az + np.pi) / (2 * np.pi)) * sec_n).astype(np.int64) % sec_n
    patch = ring * 64 + sec
    order = np.argsort(patch, kind="stable")
    _, starts = np.unique(patch[order], return_index=True)
    return np.split(order, starts[1:])


@register("czm")
def ground_czm(data, cfg):
    thr = cfg.ground_threshold_m
    glob = _coarse_plane(data, thr)
    non_ground = np.abs(plane_residual(data, glob)) >= thr
    for idx in _czm_patches(data):
        if len(idx) < _CZM_MIN_POINTS:
            continue  # keep the global-plane labels for sparse patches
        pts = data[idx]
        plane = _gpf_plane(pts, thr)
        # A patch plane that is not roughly horizontal fitted a wall or a
        # cone cluster, not the floor: keep the global labels there.
        if plane[3] < _CZM_MIN_NZ:
            continue
        non_ground[idx] = np.abs(plane_residual(pts, plane)) >= thr
    return non_ground, _refit_on_ground(data, non_ground, glob)


# A patch/ring plane is accepted only if its normal is within this angle of the
# global plane's. Otherwise RANSAC fitted a wall, a tyre stack or a sky-only
# ring (no floor at all), and the global labels are kept for those points.
_MAX_NORMAL_DEV_COS = np.cos(np.radians(10.0))


_PATCH_MIN_POINTS = 30


class _LocalRansacGround(RansacGround):
    """Global (warm-started) RANSAC labels, overridden by per-group RANSAC where
    it is trustworthy. Subclasses say how the scan is split into groups."""

    __slots__ = ()

    @staticmethod
    def _groups(data) -> list[np.ndarray]:
        raise NotImplementedError

    def remove(self, data):
        cfg = self._cfg
        non_ground, glob = self._fit_global(data)
        for idx in self._groups(data):
            if len(idx) < _PATCH_MIN_POINTS:
                continue
            local_out, plane = _ground_ransac_legacy(
                data[idx], cfg, None, cfg.ransac_iter_subsample_max
            )
            if float(plane[1:] @ glob[1:]) < _MAX_NORMAL_DEV_COS:
                continue
            non_ground[idx] = local_out
        return non_ground, _refit_on_ground(data, non_ground, glob)


@register("ransac_zones")
class RansacZonesGround(_LocalRansacGround):
    """RANSAC per concentric-ring x azimuth-sector subregion."""

    __slots__ = ()
    _groups = staticmethod(_czm_patches)


def _ring_groups(data) -> list[np.ndarray]:
    ring = _recover_rings(data)
    order = np.argsort(ring, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(ring[order])) + 1)


@register("ransac_rings")
class RansacRingsGround(_LocalRansacGround):
    """RANSAC per laser channel (the arc of ground one laser draws)."""

    __slots__ = ()
    _groups = staticmethod(_ring_groups)
