"""Cluster -> :class:`ConeFit`, one function per ``ConeDetectionConfig.fit_backend``.

Register a backend in ``FIT_BACKENDS``; :func:`cluster_fitter` resolves the
configured one once, when the detector is built, so an unknown name fails at
configure time instead of on every scan.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.fitting.alternative_fits import cone_fit_2params
from cone_detection.fitting.closed_form_fits import (
    LINEARITY_THRESHOLD,
    cluster_linearity,
    collinear_apex,
)
from cone_detection.fitting.cone_classes import CONE_BIG_D
from cone_detection.fitting.template_fit import cone_fit_template_dispatch


class ConeFit(NamedTuple):
    """One cluster's fit of ``z = d - c * ||(x, y) - (a, b)||`` (ground frame).

    ``res_other`` is the residual of the template that lost: ``inf`` when only one
    template was tried (height early exit), ``nan`` when no template ran
    (collinear closed form, two_param).
    """

    a: float
    b: float
    c: float
    d: float
    res_min: float
    res_other: float
    is_big: bool


ClusterFitter = Callable[[np.ndarray, ConeDetectionConfig, float], ConeFit]


def _cone_mse(a: float, b: float, c: float, d: float, xyz: np.ndarray) -> float:
    """Mean squared error in z of the cone ``(a, b, c, d)`` over ``xyz``."""
    x = xyz[:, 0].astype(np.float64, copy=False)
    y = xyz[:, 1].astype(np.float64, copy=False)
    z = xyz[:, 2].astype(np.float64, copy=False)
    gam = np.sqrt((x - a) ** 2 + (y - b) ** 2)
    z_pred = d - c * gam
    return float(np.mean((z - z_pred) ** 2))


def fit_two_param(points: np.ndarray, cfg: ConeDetectionConfig, height_m: float) -> ConeFit:
    """Free apex (a, b) at a nominal cone, then (c, d) by least squares."""
    a, b, c, d = cone_fit_2params(points, solver=cfg.cone_fit_solver)
    res_min = _cone_mse(a, b, c, d, points)
    is_big = bool(d > cfg.big_orange_d_threshold_m)
    return ConeFit(float(a), float(b), float(c), float(d), res_min, float("nan"), is_big)


def _fit_collinear(points: np.ndarray) -> ConeFit | None:
    """Closed-form fit of a (near-)collinear cluster; None if it cannot classify it."""
    # Not cone_fit_collinear: its geometric fallback would be computed and thrown
    # away, since the caller falls back to the template fit instead.
    fit = collinear_apex(np.asarray(points, dtype=np.float64, order="C"), (0.0, 0.0), 0.0)
    if fit is None:
        return None
    a, b, c, d_class = fit
    res_min = _cone_mse(a, b, c, d_class, points)
    is_big = bool(d_class >= CONE_BIG_D - 1e-6)
    return ConeFit(float(a), float(b), float(c), float(d_class), res_min, float("nan"), is_big)


def fit_template_dispatch(points: np.ndarray, cfg: ConeDetectionConfig, height_m: float) -> ConeFit:
    """Production fit: collinear closed form if it applies, else the two templates."""
    if cfg.use_collinear_fit and cluster_linearity(points) < LINEARITY_THRESHOLD:
        fit = _fit_collinear(points)
        if fit is not None:
            return fit
    only_small = only_big = None
    if cfg.template_fit_early_exit:
        # A clearly short / tall cluster only needs the one template.
        if height_m < cfg.template_early_exit_height_small_m:
            only_small = True
        elif height_m > cfg.template_early_exit_height_big_m:
            only_big = True
    return ConeFit(
        *cone_fit_template_dispatch(
            points,
            lidar_xy=(0.0, 0.0),
            solver=cfg.cone_fit_solver,
            maxiter=cfg.template_fit_maxiter,
            only_small=only_small,
            only_big=only_big,
        )
    )


FIT_BACKENDS: dict[str, ClusterFitter] = {
    "template_dispatch": fit_template_dispatch,
    "two_param": fit_two_param,
}


def cluster_fitter(backend: str) -> ClusterFitter:
    try:
        return FIT_BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown fit_backend {backend!r}; choose from {sorted(FIT_BACKENDS)}"
        ) from None
