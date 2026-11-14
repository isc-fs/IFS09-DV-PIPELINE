"""Production fit: (a, b) against the fixed small / big-orange cone templates.

Cones are ``z ≈ d - c * ||(x, y) - (a, b)||``. :func:`cone_fit_template_dispatch`
fits each cluster twice with a fixed-shape template, once for the FSAE small cone
(``c=CONE_SMALL_C, d=CONE_SMALL_D``) and once for the big orange marker
(``c=CONE_BIG_C, d=CONE_BIG_D``). The lower-residual fit wins, giving ``(a, b)`` and
a class label in one shot. ``solver="gn"`` is the in-process Numba Gauss-Newton/LM
path; any scipy ``minimize`` method selects the scipy path.

Why templates instead of free 4-parameter VARPRO:

* On a vertical-beam LiDAR most clusters are essentially a line of points along the
  line of sight (zero perpendicular spread). A 4-parameter cone fit has a flat
  valley along that direction and either lands in the mirror-image basin (``c<0``)
  or returns the unmoved warm-start.
* The two FSAE cone classes are well separated empirically (``c≈5.16, d≈0.43`` vs
  ``c≈6.45, d≈0.65``) and the rulebook guarantees only those two classes are on
  track, so freely fitting ``(c, d)`` adds noise without information.
* Type comes for free from the residual comparison: no separate downstream
  classifier needed.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from scipy.optimize import minimize

from .closed_form_fits import (
    NEAR_FACE_CENTROID_BIAS_FRAC,
    VARPRO_AB_BOUND_MARGIN_M,
    warm_start_and_bounds,
)
from .cone_classes import (
    CONE_BASE_RADIUS_M,
    CONE_BIG_C,
    CONE_BIG_D,
    CONE_SMALL_C,
    CONE_SMALL_D,
)


def template_loss_and_grad(ab, x, y, z, c, d):
    """MSE loss and ``d/d(a,b)`` gradient for the fixed-shape cone fit.

    Model: ``z_pred = d - c * sqrt((x-a)^2 + (y-b)^2)`` with ``(c, d)``
    constants. Loss is mean squared error, scaled by ``1/N``. Closed-form
    gradient lets L-BFGS-B converge in 5–10 iterations even from a poor
    warm-start.
    """
    a, b = ab
    n = z.shape[0]
    dx = x - a
    dy = y - b
    r = np.sqrt(dx * dx + dy * dy)
    r_safe = np.maximum(r, 1e-12)
    z_pred = d - c * r
    res = z - z_pred
    loss = float(np.dot(res, res)) / n
    inv_n = 2.0 * c / n
    grad_a = -inv_n * float(np.sum(res * dx / r_safe))
    grad_b = -inv_n * float(np.sum(res * dy / r_safe))
    return loss, np.array([grad_a, grad_b])


@numba.njit(cache=True)
def _template_sse(x, y, z, c, d, a, b):
    """Sum of squared z residuals of the cone ``z = d - c*||(x,y)-(a,b)||``."""
    sse = 0.0
    for i in range(x.shape[0]):
        dx = x[i] - a
        dy = y[i] - b
        r = math.sqrt(dx * dx + dy * dy)
        e = z[i] - (d - c * r)
        sse += e * e
    return sse


@numba.njit(cache=True)
def _template_fit_gn(x, y, z, c, d, a0, b0, a_lo, a_hi, b_lo, b_hi, max_iter):
    """Gauss-Newton with Levenberg-Marquardt damping for the fixed-(c, d) fit.

    Same objective as :func:`template_loss_and_grad` (MSE of
    ``z = d - c*||(x,y)-(a,b)||``), but solved as the nonlinear least-squares
    problem it is: exact 2x2 normal equations per step instead of L-BFGS-B's
    quasi-Newton line search through scipy's Python-level machinery. Steps are
    projected into the ``[a_lo, a_hi] x [b_lo, b_hi]`` box (same bounds the
    scipy path uses). LM damping covers near-singular JtJ on degenerate
    clusters — those normally never get here (collinear dispatch runs first).

    Returns ``(a, b, mse)`` with ``mse`` matching scipy's ``result.fun``.
    """
    n = x.shape[0]
    a = a0
    b = b0
    loss = _template_sse(x, y, z, c, d, a, b)
    lam = 1e-8
    for _ in range(max_iter):
        h00 = 0.0
        h01 = 0.0
        h11 = 0.0
        g0 = 0.0
        g1 = 0.0
        for i in range(n):
            dx = x[i] - a
            dy = y[i] - b
            r = math.sqrt(dx * dx + dy * dy)
            if r < 1e-12:
                r = 1e-12
            e = z[i] - (d - c * r)
            ja = -c * dx / r
            jb = -c * dy / r
            h00 += ja * ja
            h01 += ja * jb
            h11 += jb * jb
            g0 += ja * e
            g1 += jb * e
        step_a = 0.0
        step_b = 0.0
        improved = False
        for _try in range(8):
            d00 = h00 * (1.0 + lam)
            d11 = h11 * (1.0 + lam)
            det = d00 * d11 - h01 * h01
            if det <= 1e-300:
                break
            da = -(d11 * g0 - h01 * g1) / det
            db = -(d00 * g1 - h01 * g0) / det
            a_new = min(max(a + da, a_lo), a_hi)
            b_new = min(max(b + db, b_lo), b_hi)
            loss_new = _template_sse(x, y, z, c, d, a_new, b_new)
            if loss_new < loss:
                step_a = a_new - a
                step_b = b_new - b
                a = a_new
                b = b_new
                loss = loss_new
                lam = max(lam / 10.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved:
            break
        if abs(step_a) < 1e-7 and abs(step_b) < 1e-7:
            break
    return a, b, loss / n


@numba.njit(cache=True)
def _template_fit_gn_cluster(data, c, d, lidar_x, lidar_y, max_iter):
    """Warm start, bounds and :func:`_template_fit_gn` for one cluster in one call.

    Same warm start (:func:`cone_pos_geometric`) and bounds (xy bounding box plus
    :data:`VARPRO_AB_BOUND_MARGIN_M`) that :func:`cone_fit_template` builds for
    the scipy path, computed in a single pass over the cluster instead of two
    means and four min / max calls.
    """
    n = data.shape[0]
    sx = 0.0
    sy = 0.0
    x_lo = np.inf
    x_hi = -np.inf
    y_lo = np.inf
    y_hi = -np.inf
    for i in range(n):
        x = data[i, 0]
        y = data[i, 1]
        sx += x
        sy += y
        if x < x_lo:
            x_lo = x
        if x > x_hi:
            x_hi = x
        if y < y_lo:
            y_lo = y
        if y > y_hi:
            y_hi = y
    a0 = sx / n
    b0 = sy / n
    rx = a0 - lidar_x
    ry = b0 - lidar_y
    norm = math.hypot(rx, ry)
    if norm >= 1e-6:
        offset = NEAR_FACE_CENTROID_BIAS_FRAC * CONE_BASE_RADIUS_M
        a0 += offset * rx / norm
        b0 += offset * ry / norm
    a_lo = x_lo - VARPRO_AB_BOUND_MARGIN_M
    a_hi = x_hi + VARPRO_AB_BOUND_MARGIN_M
    b_lo = y_lo - VARPRO_AB_BOUND_MARGIN_M
    b_hi = y_hi + VARPRO_AB_BOUND_MARGIN_M
    a0 = min(max(a0, a_lo), a_hi)
    b0 = min(max(b0, b_lo), b_hi)
    return _template_fit_gn(
        data[:, 0], data[:, 1], data[:, 2], c, d, a0, b0, a_lo, a_hi, b_lo, b_hi, max_iter
    )


def cone_fit_template(
    data,
    c_fix,
    d_fix,
    lidar_xy=(0.0, 0.0),
    solver="L-BFGS-B",
    maxiter: int = 12,
):
    """Fit (a, b) only, with cone slope ``c`` and apex height ``d`` held fixed.

    Replaces the free 4-parameter VARPRO on the production hot path: with
    ``(c, d)`` fixed to known cone-class values the (a, b) objective is well
    posed even on the (very common) collinear-along-the-LiDAR-ray clusters,
    so we don't need a separate dispatcher for degenerate geometries.

    Warm-start is :func:`cone_pos_geometric` (closed-form centroid + ray
    correction), then a single L-BFGS-B step on the analytical
    :func:`template_loss_and_grad` objective. ``(a, b)`` is constrained to
    the cluster xy bounding box plus :data:`VARPRO_AB_BOUND_MARGIN_M` as a
    defense-in-depth against the optimizer wandering off on degenerate
    inputs.

    Args:
        data: ``(N, 3)`` array ``[x, y, z]``.
        c_fix: Cone slope to hold fixed (e.g. :data:`CONE_SMALL_C`).
        d_fix: Apex height to hold fixed (e.g. :data:`CONE_SMALL_D`).
        lidar_xy: Sensor xy origin in the same frame; defaults to ``(0, 0)``.
        solver: ``"gn"`` for the in-process Numba Gauss-Newton/LM solver
            (production default via ``ConeDetectionConfig``), or any
            ``minimize`` method supporting ``jac=True`` and ``bounds=``
            (e.g. the previous default ``"L-BFGS-B"``) for the scipy path.

    Returns:
        Tuple ``(a, b, c_fix, d_fix, residual_mse)``. ``residual_mse`` is the
        final objective value (mean squared error in m²); use it to compare
        against the other template (see
        :func:`cone_fit_template_dispatch`).
    """
    data = np.asarray(data, dtype=np.float64, order="C")
    if solver == "gn":
        a, b, residual_mse = _template_fit_gn_cluster(
            data,
            float(c_fix),
            float(d_fix),
            float(lidar_xy[0]),
            float(lidar_xy[1]),
            int(maxiter),
        )
        return float(a), float(b), float(c_fix), float(d_fix), float(residual_mse)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    a0, b0, bounds = warm_start_and_bounds(data, lidar_xy)
    try:
        result = minimize(
            template_loss_and_grad,
            [a0, b0],
            args=(x, y, z, c_fix, d_fix),
            method=solver,
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        a = float(result.x[0])
        b = float(result.x[1])
        residual_mse = float(result.fun)
        return a, b, float(c_fix), float(d_fix), residual_mse
    except Exception as exc:
        print(exc)
        return 0.0, 0.0, float(c_fix), float(d_fix), float("inf")


def cone_fit_template_dispatch(
    data,
    lidar_xy=(0.0, 0.0),
    solver="L-BFGS-B",
    maxiter: int = 12,
    *,
    only_small: bool | None = None,
    only_big: bool | None = None,
):
    """Fit a cluster against both FSAE templates; return the lower-residual one.

    Runs :func:`cone_fit_template` twice — once with the small-cone constants
    and once with the big-orange constants — and picks whichever fits the
    data better in mean squared error. The chosen template's ``(c, d)`` is
    returned alongside the fitted ``(a, b)``, and the unchosen template's
    residual is returned too so the caller can implement an ambiguity gate
    (e.g. require ``min_res / max_res < 0.5`` for a confident class call).

    Args:
        data: ``(N, 3)`` array ``[x, y, z]``.
        lidar_xy: Sensor xy origin in the same frame; defaults to ``(0, 0)``.
        solver: ``minimize`` method (default L-BFGS-B).

    Returns:
        Tuple ``(a, b, c, d, residual_min, residual_other, is_big)`` where
        ``(c, d)`` is the chosen template's constants, ``residual_min`` is
        the chosen fit's MSE (m²), ``residual_other`` is the rejected fit's
        MSE, and ``is_big`` is True iff the big-orange template won.
    """
    def fit(c_fix, d_fix):
        return cone_fit_template(
            data, c_fix, d_fix, lidar_xy=lidar_xy, solver=solver, maxiter=maxiter
        )

    if only_big:
        return (*fit(CONE_BIG_C, CONE_BIG_D), float("inf"), True)
    small = fit(CONE_SMALL_C, CONE_SMALL_D)
    if only_small:
        return (*small, float("inf"), False)
    big = fit(CONE_BIG_C, CONE_BIG_D)
    if small[4] <= big[4]:
        return (*small, big[4], False)
    return (*big, small[4], True)
