"""Fitters kept for benchmarks and the ``two_param`` backend, not on the default path.

* :func:`cone_fit_2params` -- (a, b) at the nominal ``c=5.5, d=0.35``, then (c, d)
  by linear least squares (``fit_backend="two_param"``).
* :func:`cone_fit_varpro` -- free 4-parameter VARPRO fit (a, b, c, d).
"""

from __future__ import annotations

import numba
import numpy as np

from scipy.optimize import minimize

from .closed_form_fits import warm_start_and_bounds


@numba.njit(cache=True)
def cone_model(params, x, y):
    """Cone surface: z = d - c * sqrt((x-a)^2 + (y-b)^2)."""
    a, b, c, d = params
    return d - c * np.sqrt((x - a) ** 2 + (y - b) ** 2)


@numba.njit(cache=True)
def objective_function_v2(params, x, y, z):
    """MSE for optimizing (a, b) with fixed nominal (c, d) during the nonlinear stage."""
    a, b = params
    c = 5.5
    d = 0.35
    z_pred = cone_model(np.array([a, b, c, d]), x, y)
    return np.mean((z - z_pred) ** 2)


@numba.njit(cache=True)
def get_gammas(x, y, a, b):
    return np.sqrt((x - a) ** 2 + (y - b) ** 2)


@numba.njit(cache=True)
def lst_sqrs_fit(X, y_vec):
    # Closed-form normal equations for the (N, 2) design matrix used by
    # ``cone_fit_2params``. Equivalent to ``pinv(X) @ y_vec`` for full-rank
    # X, written as an explicit loop to keep numba happy regardless of the
    # caller's array layout (LAPACK's pinv returns F-contig; ``data[:, 2]``
    # is a strided view -- both confused the previous @-based version).
    n = X.shape[0]
    s00 = 0.0
    s01 = 0.0
    s11 = 0.0
    sy0 = 0.0
    sy1 = 0.0
    for i in range(n):
        a = X[i, 0]
        b = X[i, 1]
        y = y_vec[i]
        s00 += a * a
        s01 += a * b
        s11 += b * b
        sy0 += a * y
        sy1 += b * y
    det = s00 * s11 - s01 * s01
    theta = np.empty(2)
    theta[0] = (s11 * sy0 - s01 * sy1) / det
    theta[1] = (s00 * sy1 - s01 * sy0) / det
    return theta


def cone_fit_2params(data, solver="L-BFGS-B"):
    """
    Two-stage fit: optimize (a, b), then recover (c, d) by linear least squares.

    Args:
        data: (N, 3) array [x, y, z].
        solver: ``scipy.optimize.minimize`` method for the (a, b) step.

    Returns:
        Tuple (a, b, c, d).
    """
    # The numba helpers (lst_sqrs_fit, get_gammas) require a single dtype per
    # signature. Live clusters come in as float64 post ground-plane rotation,
    # but benchmarks / warmup may feed float32; promote consistently.
    data = np.asarray(data, dtype=np.float64, order="C")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    a = np.mean(x)
    b = np.mean(y)
    try:
        result = minimize(
            objective_function_v2,
            [a, b],
            args=(x, y, z),
            method=solver,
        )
        a, b = result.x
        gamma = get_gammas(x, y, a, b)
        m_design = np.ascontiguousarray(np.column_stack((np.ones(len(data)), gamma)))
        d_lin, c_lin = lst_sqrs_fit(m_design, z)
        return a, b, -c_lin, d_lin
    except Exception as exc:
        print(exc)
        return 0.0, 0.0, 0.0, 0.0


def _nnls_linear_step(r, z):
    """Solve ``min ||[1, -r] @ [d, c] - z||^2`` subject to ``d, c >= 0``.

    For our 2-column design we can do active-set NNLS in closed form: solve the
    unconstrained 2x2 normal equations; if c < 0, snap to (d=mean(z), c=0).
    z is bounded below by the floor cutoff so d=mean(z) >= 0 always, hence we
    never need to consider the d=0 / c free case.

    Returns ``(d, c, active)`` where ``active`` is True iff the c>=0 constraint
    is active (c was clipped from a negative unconstrained value).
    """
    theta = _line_normal_equations(r, z)
    if theta is not None and theta[0] >= 0.0 and theta[1] >= 0.0:
        return theta[0], theta[1], False
    return float(np.mean(z)), 0.0, True


def _line_normal_equations(r, z):
    """``(d, c)`` minimising ``||[1, -r] @ [d, c] - z||``; None when singular."""
    N = r.shape[0]
    sum_r = float(np.sum(r))
    sum_rr = float(np.dot(r, r))
    sum_z = float(np.sum(z))
    sum_rz = float(np.dot(r, z))
    # Design [1, -r] -> A^T A = [[N, -sum_r], [-sum_r, sum_rr]]
    #                  A^T z = [sum_z, -sum_rz]
    det = N * sum_rr - sum_r * sum_r
    if det <= 0.0:
        return None
    d_unc = (sum_rr * sum_z - sum_r * sum_rz) / det
    c_unc = (sum_r * sum_z - N * sum_rz) / det
    return d_unc, c_unc


def _unconstrained_theta(r, z):
    """Closed-form (d, c) for the 2-column unconstrained lstsq on [1, -r]."""
    theta = _line_normal_equations(r, z)
    return (0.0, 0.0) if theta is None else theta


def varpro_cone_loss_and_grad(ab, x, y, z):
    """
    Variable-projection objective for z ≈ d - c * r with r = ||(x,y)-(a,b)||.

    Inner step is the smooth unconstrained 2x2 lstsq: keeping it unconstrained
    gives L-BFGS-B a well-behaved gradient everywhere. The c>=0 / d>=0 NNLS
    projection is applied only as a post-processing step inside
    ``cone_fit_varpro``, after L-BFGS-B has converged on (a, b). The (a, b)
    bounds in ``cone_fit_varpro`` keep the optimizer away from the inverted
    "bowl" basin (mirror image of the apex across the line of points) so the
    inner-loop c rarely goes negative; when it does the NNLS at the end clips
    it to c=0, triggering the existing centroid fallback in
    ``RealtimeConeDetector.detect``.

    Returns:
        (loss, gradient w.r.t. (a, b)).
    """
    a, b = ab
    r = np.sqrt((x - a) ** 2 + (y - b) ** 2)
    d_unc, c_unc = _unconstrained_theta(r, z)
    theta = np.array([d_unc, c_unc])
    a_design = np.column_stack([np.ones_like(r), -r])
    residual = z - a_design @ theta
    loss = float(np.dot(residual, residual))

    r_safe = np.maximum(r, 1e-12)
    d_da = np.column_stack([np.zeros_like(r), (x - a) / r_safe])
    d_db = np.column_stack([np.zeros_like(r), (y - b) / r_safe])
    grad_a = -2.0 * float(np.dot(d_da @ theta, residual))
    grad_b = -2.0 * float(np.dot(d_db @ theta, residual))
    return loss, np.array([grad_a, grad_b])


def cone_fit_varpro(data, solver="L-BFGS-B"):
    """
    Fit z = d - c * sqrt((x-a)^2 + (y-b)^2) by optimizing (a, b) with VARPRO.

    Warm-starts (a, b) from :func:`cone_pos_geometric` — the closed-form
    centroid-plus-LiDAR-ray estimator. That puts the optimizer inside the
    correct c>0 basin from step one without any hardcoded ``(c, d)`` fit
    nominals: the only prior used is the cone base radius
    (:data:`CONE_BASE_RADIUS_M`), which is a physical property of the FSAE
    cone we're detecting, not a fit-shape parameter.

    Production calls reach this function only after the PCA-linearity
    dispatch in ``RealtimeConeDetector.detect`` has filtered out
    near-collinear clusters (those go to ``cone_pos_geometric`` directly),
    so the cluster always carries genuine perpendicular-to-line-of-sight
    information and a 4-parameter fit is well-posed.

    The inner step uses the smooth unconstrained 2x2 lstsq; the c>=0 / d>=0
    NNLS projection is applied only as a post-processing step on the final
    converged (a, b). The (a, b) bounds (cluster bbox plus
    :data:`VARPRO_AB_BOUND_MARGIN_M`) remain as a safety net so the
    optimizer cannot wander into the inverted-cone "bowl" basin.

    Args:
        data: (N, 3) array [x, y, z].
        solver: ``minimize`` method supporting ``jac=True`` and ``bounds=``
            (default L-BFGS-B).

    Returns:
        Tuple (a, b, c, d).
    """
    # PointCloud2 clusters are usually float32; lstsq / L-BFGS match synthetic
    # benchmarks much better in float64.
    data = np.asarray(data, dtype=np.float64, order="C")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    a0, b0, bounds = warm_start_and_bounds(data)
    try:
        result = minimize(
            varpro_cone_loss_and_grad,
            [a0, b0],
            args=(x, y, z),
            method=solver,
            jac=True,
            bounds=bounds,
        )
        a, b = float(result.x[0]), float(result.x[1])
        r = np.sqrt((x - a) ** 2 + (y - b) ** 2)
        d_val, c_val, _ = _nnls_linear_step(r, z)
        return a, b, c_val, d_val
    except Exception as exc:
        print(exc)
        return 0.0, 0.0, 0.0, 0.0
