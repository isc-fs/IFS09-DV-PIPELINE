"""Pluggable LiDAR ground-removal methods (``ConeDetectionConfig.ground_method``).

The detector owns one :class:`GroundRemover` built by :func:`make_ground_remover`::

    remover = make_ground_remover(cfg)
    non_ground_mask, plane_coefs = remover.remove(data)   # once per scan
    remover.reset()                                     # forget previous scans

A method that keeps nothing between scans is registered as a plain function
``fn(data, cfg) -> (non_ground_mask, plane_coefs)``; one that does (RANSAC's warm-start
plane, Patchwork++'s estimator) is registered as a class taking ``cfg`` and holding
that state itself. Nothing outside a method knows what, if anything, it remembers.

``data`` is the (N, 3) cloud in the sensor frame, ``non_ground_mask`` is True for
non-ground returns and ``plane_coefs`` is ``[bias, n_x, n_y, n_z]`` with a unit
normal and ``n_z > 0`` (``bias + n . p = 0`` on the plane). The plane is always a
single global plane even for the non-planar methods, because the rest of the
pipeline uses it to rotate the non-ground points upright and to cull the floor margin. For
those methods it is least-squares refit on the points the method labelled ground.

Register a new method with ``@register("name")`` (``registry``) on the function or class in one of the method
modules below and import that module here; ``GROUND_METHODS`` is what the benchmark's
``--ground-method`` and the ROS parameter validate against.

Layout:
    registry          ``METHODS`` / ``register`` / ``GroundRemover``
    plane_fitting     plane fits every method shares (TLS, GPF, the final refit)
    scan_geometry     levelled frame, grid cells, azimuth sectors, laser channels
    height_grid       per-cell height stats and window filters over the grid
    plane_methods     ransac, gpf, grid_resid, czm, ransac_zones, ransac_rings
    ray_methods       ray_slope, linefit, gp_insac, range_image
    grid_methods      height_jump, groundgrid, pmf
    mrf_methods       mrf (graph cut)
    external_methods  csf, patchworkpp (optional packages)

Methods:
    ransac       baseline, ``ransac.ransac2`` (single plane, warm-started from the
                 previous scan's plane).
    gpf          Ground Plane Fitting (Zermas et al., ICRA 2017): lowest-point-
                 representative seeds, iterated SVD plane fit.
    grid_resid   GPF plane + a per-cell ground-height correction from the cell's
                 lowest returns; follows bumps and slow slopes a plane cannot.
    ray_slope    per-azimuth ray walk (Autoware ray_ground_classifier style):
                 a return is ground while the slope from the last ground return
                 stays below a limit. Planarity-free.
    czm          concentric zone model (Patchwork-lite): one GPF plane per
                 range-ring x azimuth-sector patch, global plane as fallback.
    ransac_zones RANSAC per range-ring x azimuth-sector subregion (global fit warm-started).
    ransac_rings RANSAC per laser channel (arc), channels recovered from the
                 elevation angle.
    height_jump  min/max height map: flat cells near the floor (incl. sidewalks)
                 and the bottom of every cell are ground.
    linefit      line segments per azimuth sector (Himmelsbach et al., IV 2010).
    range_image  angle labelling on a range image (Bogoslavskyi & Stachniss,
                 IROS 2016); rows are channels recovered from the elevation.
    groundgrid   elevation grid from low-variance cells, interpolated (Steinke
                 et al., RA-L 2023), single scan (no temporal accumulation).
    pmf          progressive morphological filter (Zhang et al., TGRS 2003).
    gp_insac     Gaussian-process INSAC per sector (Douillard et al., ICRA 2011).
    mrf          binary MRF over cells solved exactly by graph cut (scipy max-flow;
                 PyMaxflow optional).
    csf          cloth simulation filter (Zhang et al., 2016; needs
                 cloth-simulation-filter).
    patchworkpp  Patchwork++ (Lee et al., RA-L 2022; needs pypatchworkpp).

csf and patchworkpp use optional packages that are imported on first use, so the
production image does not need them unless one of them is selected.
"""

from __future__ import annotations

import numpy as np

# Importing a method module registers its methods (registry order = import order).
from . import (  # noqa: F401
    external_methods,
    grid_methods,
    mrf_methods,
    plane_methods,
    ray_methods,
)
from .mrf_methods import warm_mrf_graph
from .plane_fitting import _DEFAULT_PLANE, fit_plane, plane_residual
from .registry import METHODS, GroundRemover, build_remover, register

__all__ = [
    "GROUND_METHODS",
    "METHODS",
    "GroundRemover",
    "fit_plane",
    "make_ground_remover",
    "plane_residual",
    "register",
    "remove_ground",
    "warm_mrf_graph",
]


GROUND_METHODS = tuple(METHODS)


class _TinyCloudGuard:
    """Short-circuits clouds with no plane to fit (< 3 points) before the method sees
    them, so a degenerate scan cannot overwrite a method's state either."""

    __slots__ = ("_inner",)

    def __init__(self, inner: GroundRemover) -> None:
        self._inner = inner

    def remove(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(data) < 3:  # callers treat an empty/tiny cloud as all non-ground
            return np.ones(len(data), dtype=bool), _DEFAULT_PLANE.copy()
        return self._inner.remove(data)

    def reset(self) -> None:
        self._inner.reset()


def make_ground_remover(cfg) -> GroundRemover:
    """The remover for ``cfg.ground_method``, bound to ``cfg``; raises on an unknown name."""
    return _TinyCloudGuard(build_remover(cfg.ground_method, cfg))


def remove_ground(method: str, data: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """One-shot ``method`` on one scan with a fresh remover (no state from earlier scans)."""
    return _TinyCloudGuard(build_remover(method, cfg)).remove(data)
