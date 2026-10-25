"""
Cone fitting for traffic-cone-shaped clusters: z ≈ d - c * ||(x,y) - (a,b)||.

Production perception (see ``cone_detection.final_cone_result_rt``) uses
:func:`cone_fit_template_dispatch`: each cluster is fit twice with a
fixed-shape template — once for the FSAE small cone
(``c=_CONE_SMALL_C, d=_CONE_SMALL_D``) and once for the big orange marker
(``c=_CONE_BIG_C, d=_CONE_BIG_D``). The lower-residual fit wins, giving
``(a, b)`` and a class label in one shot. The template constants are
empirical, extracted from rosbag fits (see
``debug_tools/extract_cone_truth_params.py``).

Why templates instead of free 4-parameter VARPRO:
* On a vertical-beam LiDAR most clusters are essentially a line of points
  along the line of sight (zero perpendicular spread). A 4-parameter cone
  fit has a flat valley along that direction and either lands in the
  mirror-image basin (``c<0``) or returns the unmoved warm-start.
* The two FSAE cone classes are well separated empirically (``c≈5.16, d≈0.43``
  vs ``c≈6.45, d≈0.65``) and the rulebook guarantees only those two classes
  are on track, so freely fitting ``(c, d)`` adds noise without information.
* Type comes for free from the residual comparison — no separate downstream
  classifier needed.

Legacy fitters (kept for benchmarks, not on the hot path):
* :func:`cone_fit_varpro` — 4-parameter VARPRO (a, b, c, d).
* :func:`cone_fit_2params` — 2-parameter (a, b) with hardcoded
  ``c=5.5, d=0.35`` nominal.
* :func:`cone_fit_collinear` — type-aware closed-form fit for collinear
  clusters using the same template constants as the dispatch.
* :func:`cone_pos_geometric` — closed-form centroid + LiDAR-ray correction;
  still used as the warm-start for the template fits.
"""

import math

import numba
import numpy as np
from scipy.optimize import minimize


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


def apex_init(x, y, z=None):
    """Initial guess for cone apex (a, b); ``z`` is unused (call compatibility)."""
    return float(np.mean(x)), float(np.mean(y))


# FSAE traffic cones (small / "regular") have base diameter ~228 mm. The big
# orange marker is ~285 mm. We default to the small-cone radius for the
# centroid correction; the per-class size mismatch is absorbed by the empirical
# tuning of ``_NEAR_FACE_CENTROID_BIAS_FRAC`` below.
_CONE_BASE_RADIUS_M = 0.114

# Fraction of the cone base radius to push the cluster centroid back along the
# LiDAR ray. Physical origin: for a one-sided cluster on a cone of slope c and
# apex height d, the visible points (z >= z_min) have ``E[gamma] ~ (d-z_min)/(2c)``
# along the near-face radius, so the centroid lies that far in front of the
# axis. Expressed as a fraction of the FSAE small-cone base radius (0.114 m):
#   small cone (c=2.85, d=0.325): k = 0.40
#   big orange (c=3.54, d=0.505): k = 0.54
#   skinny synthetic (c=5.5, d=0.35): k = 0.23
# Calibration sweep (1500 cases each, /tmp/calibrate_geometric2.py) at k=0.4
# gives strict improvement over VARPRO on both FSAE scenarios (B p95 61→30 mm,
# C max 130→61 mm) while only mildly overcorrecting on the synthetic baseline.
# 0.4 picked as the compromise; raise to ~0.5 if logs show consistent
# undershoot toward the LiDAR on big-orange markers.
_NEAR_FACE_CENTROID_BIAS_FRAC = 0.4

# Eigenvalue ratio (small / large) of the xy covariance below which the cluster
# is treated as collinear and the geometric estimator is used instead of
# VARPRO. 0 = perfectly collinear, 1 = isotropic. At this threshold ~27% of
# synthetic clusters go geometric. Lower values miss too many degenerate cases
# (VARPRO long tail not cut); higher values route well-spread clusters where
# VARPRO is more accurate. Calibration table (FSAE small, combined p95 |dxy|):
#   thr   frac_geom   p95_combined
#   0.02     4%        61 mm  (VARPRO baseline, dispatch barely active)
#   0.05    13%        57 mm
#   0.10    27%        50 mm  <- chosen; strict median + p95 improvement vs VARPRO alone
#   0.20    47%        43 mm  (better p95 but median starts to degrade on well-spread clusters)
_LINEARITY_THRESHOLD = 0.10

# Below this many points VARPRO has too few degrees of freedom against 4
# unknowns; force the geometric path regardless of linearity.
_MIN_POINTS_FOR_VARPRO = 5


def cluster_linearity(xy):
    """Smaller / larger principal-axis variance ratio for the cluster's xy points.

    Returns a value in [0, 1]: 0 for a perfectly collinear cluster, 1 for an
    isotropic disk. Below :data:`_LINEARITY_THRESHOLD` the cluster carries
    essentially no perpendicular-to-LiDAR-ray information, and any 4-parameter
    cone fit has a flat valley in the residual along that direction.

    Args:
        xy: (N, 2) or (N, 3) array; only the first two columns are used.

    Returns:
        float in [0, 1]. ``0.0`` for fewer than 2 points.
    """
    pts = np.asarray(xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    pts = pts[:, :2]
    centered = pts - pts.mean(axis=0)
    cov = centered.T @ centered / pts.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    return float(eigvals[0] / (eigvals[1] + 1e-12))


def cone_pos_geometric(data, lidar_xy=(0.0, 0.0)):
    """Closed-form (a, b) estimate from cluster centroid + LiDAR ray geometry.

    A vertical-beam LiDAR scans the near-side face of the cone, so the xy
    centroid of returns is biased toward the sensor relative to the cone's true
    ground center. The cluster carries no information about position
    perpendicular to the LiDAR ray (that's the degenerate VARPRO direction), but
    along the ray the centroid is offset by a roughly constant fraction of the
    cone base radius. Pushing the centroid back along the ray by that fraction
    recovers a position estimate that's robust even for purely collinear
    clusters.

    Args:
        data: (N, 3) array [x, y, z] in the sensor (ground-corrected) frame.
        lidar_xy: Sensor xy origin in the same frame; defaults to (0, 0).

    Returns:
        Tuple (a, b) — estimated cone xy position.
    """
    pts = np.asarray(data, dtype=np.float64)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    rx = cx - float(lidar_xy[0])
    ry = cy - float(lidar_xy[1])
    norm = math.hypot(rx, ry)
    if norm < 1e-6:
        return cx, cy
    offset = _NEAR_FACE_CENTROID_BIAS_FRAC * _CONE_BASE_RADIUS_M
    return cx + offset * rx / norm, cy + offset * ry / norm


# Known FSAE cone classes: (apparent slope c, apex height d). These are the
# *empirical* values measured from free 4-parameter VARPRO fits of simulator
# clusters, with z measured above the RANSAC ground plane (IFSSIM
# tools/sim_benchmark/diagnose_cone_shape.py;
# median over 612 small-cone fits on a 1.74 M pts/s bag and 1858 big-cone fits
# on a 95 s bag that faces the start gate; fit RMSE ~7 mm):
#   small cones   : c ≈ 5.16, d ≈ 0.43 m
#   big orange    : c ≈ 6.45, d ≈ 0.65 m
# d is where the fitted sides meet, not the cone's height: the sim cones have
# flat tops (highest returns ~0.30 m / ~0.48 m), and z = d - c * r only
# describes the sides. The previous 5.0 / 0.35 and 5.5 / 0.55 sat 8-10 cm
# low, masked while z was measured from the sensor instead of the ground.
_CONE_SMALL_C = 5.16
_CONE_SMALL_D = 0.43
_CONE_BIG_C = 6.45
_CONE_BIG_D = 0.65

# Tolerance for matching the estimated slope to one of the known cone classes.
# Sources of noise on the slope estimate: LiDAR z noise (~1 cm) divided by the
# cluster's along-ray span (typically 5–15 cm), giving σ_c ~ 0.1–0.3. With the
# c values 5.16 and 6.45 about 1.3 apart, 0.5 keeps the two acceptance bands
# [4.66, 5.66] and [5.95, 6.95] disjoint while still absorbing slope noise;
# blatantly off slopes (e.g. ground residuals, walls) fall through to the
# geometric fallback.
_CONE_CLASS_TOLERANCE = 0.5

# Minimum cluster size for the type-aware collinear fit. The (t, z) regression
# needs at least 3 points to be meaningful; below that we fall back to the
# pure geometric estimator.
_MIN_PTS_COLLINEAR_FIT = 3


def cone_fit_collinear(data, lidar_xy=(0.0, 0.0), ground_z=0.0):
    """Type-aware closed-form fit for (near-)collinear clusters.

    Strategy:
      1. Project the points onto the LiDAR ray to get a 1-D ``(t, z)`` cloud.
         For a near-face cluster on a cone the geometry collapses to
         ``z = (d_apex − c·T_apex) + c·t`` — a straight line with slope ``c``.
      2. Linear regression of ``z`` on ``t`` recovers the cone slope directly.
      3. Snap that slope to whichever FSAE class (small/big) it's closest to.
         If neither class is within :data:`_CONE_CLASS_TOLERANCE`, give up
         and fall back to :func:`cone_pos_geometric`.
      4. With the class's known apex height above ground, compute the apex's
         absolute z (``ground_z + d_class``) and solve in closed form:
         ``T_apex = t_mean + (d_apex_abs − z_mean) / c``. Project back into
         xy to get ``(a, b)``.

    The classification is what removes the "infinitely many solutions" problem
    of fitting a free 4-parameter cone to collinear points: the FSAE rulebook
    constrains us to two discrete cone types, and we use the data only to
    decide which one it is.

    Args:
        data: (N, 3) array [x, y, z] in the sensor (ground-corrected) frame.
            Note that z here is *absolute* in that frame (e.g. ground at
            ``ground_z``, cone apex at ``ground_z + d_class``).
        lidar_xy: Sensor xy origin in the same frame; defaults to (0, 0).
        ground_z: z value of the ground plane in the same frame. After the
            RANSAC ground-removal step the ground sits at the LiDAR-to-floor
            distance (typically negative for downward-mounted sensors); pass
            it here so the classifier can convert the class's height-above-
            ground ``d`` into an absolute apex z. Default 0 is appropriate
            for tests / benchmarks where the data is pre-shifted to ground.

    Returns:
        Tuple ``(a, b, c, d_class)``. When classification succeeds ``c`` is
        the chosen class's slope and ``d_class`` its *height above ground*
        (not absolute z). On fallback both are ``nan`` and the ``(a, b)`` is
        from :func:`cone_pos_geometric`. The caller can check
        ``np.isfinite(c)`` to tell the two paths apart.
    """
    arr = np.asarray(data, dtype=np.float64, order="C")
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    n = arr.shape[0]
    cx = float(x.mean())
    cy = float(y.mean())
    rx = cx - float(lidar_xy[0])
    ry = cy - float(lidar_xy[1])
    norm = math.hypot(rx, ry)
    if norm < 1e-6 or n < _MIN_PTS_COLLINEAR_FIT:
        ax, by = cone_pos_geometric(arr, lidar_xy)
        return ax, by, float("nan"), float("nan")
    ux = rx / norm
    uy = ry / norm
    t = (x - float(lidar_xy[0])) * ux + (y - float(lidar_xy[1])) * uy
    t_mean = float(t.mean())
    z_mean = float(z.mean())
    dt = t - t_mean
    dz = z - z_mean
    s_tt = float(np.dot(dt, dt))
    s_tz = float(np.dot(dt, dz))
    # Regression is ill-conditioned if the cluster has essentially no along-ray
    # spread (e.g. all points at the same range slice). Fall back rather than
    # produce a garbage slope.
    if s_tt < 1e-6:
        ax, by = cone_pos_geometric(arr, lidar_xy)
        return ax, by, float("nan"), float("nan")
    c_est = s_tz / s_tt
    # Near-face geometry gives ``dz/dt = +c``. A negative slope means we're
    # looking at the far face (LiDAR somehow above the cone) or pure noise;
    # either way the closed-form below would put the apex behind the cluster.
    if c_est <= 0.0:
        ax, by = cone_pos_geometric(arr, lidar_xy)
        return ax, by, float("nan"), float("nan")
    # Classify against the two known FSAE cone types.
    err_small = abs(c_est - _CONE_SMALL_C)
    err_big = abs(c_est - _CONE_BIG_C)
    if err_small <= err_big:
        if err_small > _CONE_CLASS_TOLERANCE:
            ax, by = cone_pos_geometric(arr, lidar_xy)
            return ax, by, float("nan"), float("nan")
        c_use = _CONE_SMALL_C
        d_class = _CONE_SMALL_D
    else:
        if err_big > _CONE_CLASS_TOLERANCE:
            ax, by = cone_pos_geometric(arr, lidar_xy)
            return ax, by, float("nan"), float("nan")
        c_use = _CONE_BIG_C
        d_class = _CONE_BIG_D
    # Closed-form optimum of T_apex given (c_use, d_apex_abs) and the data:
    #   minimise Σ (z_i − d_apex_abs + c·(T_apex − t_i))²
    #     ⇒  T_apex = t_mean + (d_apex_abs − z_mean)/c.
    # d_apex_abs is the class's *height above ground* shifted into the
    # LiDAR-rotated frame where the cluster z values live.
    d_apex_abs = ground_z + d_class
    t_apex = t_mean + (d_apex_abs - z_mean) / c_use
    # Sanity check: T_apex should land just past the cluster centroid along
    # the ray (within roughly one cone radius). Far outside that range means
    # the slope/intercept fit was bad regardless of how it classified.
    offset_along_ray = t_apex - t_mean
    if offset_along_ray < 0.0 or offset_along_ray > 2.0 * _CONE_BASE_RADIUS_M + 0.05:
        ax, by = cone_pos_geometric(arr, lidar_xy)
        return ax, by, float("nan"), float("nan")
    a = float(lidar_xy[0]) + t_apex * ux
    b = float(lidar_xy[1]) + t_apex * uy
    return a, b, c_use, d_class


def _nnls_linear_step(r, z):
    """Solve ``min ||[1, -r] @ [d, c] - z||^2`` subject to ``d, c >= 0``.

    For our 2-column design we can do active-set NNLS in closed form: solve the
    unconstrained 2x2 normal equations; if c < 0, snap to (d=mean(z), c=0).
    z is bounded below by the floor cutoff so d=mean(z) >= 0 always, hence we
    never need to consider the d=0 / c free case.

    Returns ``(d, c, active)`` where ``active`` is True iff the c>=0 constraint
    is active (c was clipped from a negative unconstrained value).
    """
    N = r.shape[0]
    sum_r = float(np.sum(r))
    sum_rr = float(np.dot(r, r))
    sum_z = float(np.sum(z))
    sum_rz = float(np.dot(r, z))
    # Design [1, -r] -> A^T A = [[N, -sum_r], [-sum_r, sum_rr]]
    #                  A^T z = [sum_z, -sum_rz]
    det = N * sum_rr - sum_r * sum_r
    if det <= 0.0:
        return float(np.mean(z)), 0.0, True
    d_unc = (sum_rr * sum_z - sum_r * sum_rz) / det
    c_unc = (sum_r * sum_z - N * sum_rz) / det
    if c_unc >= 0.0 and d_unc >= 0.0:
        return d_unc, c_unc, False
    return float(np.mean(z)), 0.0, True


def _unconstrained_theta(r, z):
    """Closed-form (d, c) for the 2-column unconstrained lstsq on [1, -r]."""
    N = r.shape[0]
    sum_r = float(np.sum(r))
    sum_rr = float(np.dot(r, r))
    sum_z = float(np.sum(z))
    sum_rz = float(np.dot(r, z))
    det = N * sum_rr - sum_r * sum_r
    if det <= 0.0:
        return 0.0, 0.0
    d_unc = (sum_rr * sum_z - sum_r * sum_rz) / det
    c_unc = (sum_r * sum_z - N * sum_rz) / det
    return d_unc, c_unc


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
    ``cone_detection.final_cone_result_rt``.

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


# Bound the VARPRO (a, b) search to the cluster's xy bounding box plus this
# margin (m). With a vertical-beam LiDAR most clusters are essentially a few
# points along the near-side line of sight, leaving (a, b) unconstrained in the
# perpendicular direction. Without bounds the optimizer walks ~1 m to the
# mirror image of the apex across the line of points, producing the inverted
# cone (c<0, d<0) basin we've observed in live logs.
#
# Margin choice rationale (1500 realistic clusters, /tmp/sweep_margin.py):
#   margin   gross_miss(>0.15m)   p95 |dxy|
#   0.02 m         0%             0.030 m
#   0.05 m         0%             0.049 m   <- chosen; matches one-sided p95
#   0.10 m         0%             0.100 m
#   0.20 m         9%             0.201 m   (mirror basin reachable)
# 5 cm gives ample slack for the apex to sit slightly outside one-sided cluster
# bboxes (~1-2 cm) without leaving room for the mirror image. With the
# PCA-dispatch filter in ``cone_detection.final_cone_result_rt`` collinear
# clusters never reach VARPRO, so this bound is almost never active; it
# remains as defense-in-depth.
_VARPRO_AB_BOUND_MARGIN_M = 0.05


def cone_fit_varpro(data, solver="L-BFGS-B"):
    """
    Fit z = d - c * sqrt((x-a)^2 + (y-b)^2) by optimizing (a, b) with VARPRO.

    Warm-starts (a, b) from :func:`cone_pos_geometric` — the closed-form
    centroid-plus-LiDAR-ray estimator. That puts the optimizer inside the
    correct c>0 basin from step one without any hardcoded ``(c, d)`` fit
    nominals: the only prior used is the cone base radius
    (:data:`_CONE_BASE_RADIUS_M`), which is a physical property of the FSAE
    cone we're detecting, not a fit-shape parameter.

    Production calls reach this function only after the PCA-linearity
    dispatch in ``cone_detection.final_cone_result_rt`` has filtered out
    near-collinear clusters (those go to ``cone_pos_geometric`` directly),
    so the cluster always carries genuine perpendicular-to-line-of-sight
    information and a 4-parameter fit is well-posed.

    The inner step uses the smooth unconstrained 2x2 lstsq; the c>=0 / d>=0
    NNLS projection is applied only as a post-processing step on the final
    converged (a, b). The (a, b) bounds (cluster bbox plus
    :data:`_VARPRO_AB_BOUND_MARGIN_M`) remain as a safety net so the
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
    a0, b0 = cone_pos_geometric(data)
    if not (np.isfinite(a0) and np.isfinite(b0)):
        a0, b0 = apex_init(x, y, z)
    margin = _VARPRO_AB_BOUND_MARGIN_M
    bounds = (
        (float(x.min()) - margin, float(x.max()) + margin),
        (float(y.min()) - margin, float(y.max()) + margin),
    )
    # Clamp the warm-start into the bound box: cone_pos_geometric pushes the
    # cluster centroid back along the LiDAR ray by ~5 cm and can land just
    # outside the cluster bbox; L-BFGS-B requires the initial point feasible.
    a0 = min(max(a0, bounds[0][0]), bounds[0][1])
    b0 = min(max(b0, bounds[1][0]), bounds[1][1])
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
    loss = 0.0
    for i in range(n):
        dx = x[i] - a
        dy = y[i] - b
        r = math.sqrt(dx * dx + dy * dy)
        e = z[i] - (d - c * r)
        loss += e * e
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
            loss_new = 0.0
            for i in range(n):
                dx = x[i] - a_new
                dy = y[i] - b_new
                r = math.sqrt(dx * dx + dy * dy)
                e = z[i] - (d - c * r)
                loss_new += e * e
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
    the cluster xy bounding box plus :data:`_VARPRO_AB_BOUND_MARGIN_M` as a
    defense-in-depth against the optimizer wandering off on degenerate
    inputs.

    Args:
        data: ``(N, 3)`` array ``[x, y, z]``.
        c_fix: Cone slope to hold fixed (e.g. :data:`_CONE_SMALL_C`).
        d_fix: Apex height to hold fixed (e.g. :data:`_CONE_SMALL_D`).
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
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    a0, b0 = cone_pos_geometric(data, lidar_xy)
    if not (np.isfinite(a0) and np.isfinite(b0)):
        a0, b0 = apex_init(x, y, z)
    margin = _VARPRO_AB_BOUND_MARGIN_M
    bounds = (
        (float(x.min()) - margin, float(x.max()) + margin),
        (float(y.min()) - margin, float(y.max()) + margin),
    )
    a0 = min(max(a0, bounds[0][0]), bounds[0][1])
    b0 = min(max(b0, bounds[1][0]), bounds[1][1])
    if solver == "gn":
        a, b, residual_mse = _template_fit_gn(
            x,
            y,
            z,
            float(c_fix),
            float(d_fix),
            float(a0),
            float(b0),
            bounds[0][0],
            bounds[0][1],
            bounds[1][0],
            bounds[1][1],
            maxiter,
        )
        return float(a), float(b), float(c_fix), float(d_fix), float(residual_mse)
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
    if only_big:
        a_b, b_b, c_b, d_b, res_b = cone_fit_template(
            data,
            _CONE_BIG_C,
            _CONE_BIG_D,
            lidar_xy=lidar_xy,
            solver=solver,
            maxiter=maxiter,
        )
        return a_b, b_b, c_b, d_b, res_b, float("inf"), True

    a_s, b_s, c_s, d_s, res_s = cone_fit_template(
        data,
        _CONE_SMALL_C,
        _CONE_SMALL_D,
        lidar_xy=lidar_xy,
        solver=solver,
        maxiter=maxiter,
    )
    if only_small:
        return a_s, b_s, c_s, d_s, res_s, float("inf"), False

    a_b, b_b, c_b, d_b, res_b = cone_fit_template(
        data,
        _CONE_BIG_C,
        _CONE_BIG_D,
        lidar_xy=lidar_xy,
        solver=solver,
        maxiter=maxiter,
    )
    if res_s <= res_b:
        return a_s, b_s, c_s, d_s, res_s, res_b, False
    return a_b, b_b, c_b, d_b, res_b, res_s, True


def cone_params_valid_for_plot(cone_params):
    """Heuristic band for traffic-cone-like parameters (plot filter)."""
    _, _, c, d = cone_params
    return bool((c < 8 and c > 2 and d < 1 and d > 0.1))


def generate_cone_plot_points(a, b, c, d, num_slices=50):
    """Sample points on a cone mesh for Matplotlib."""
    theta = np.linspace(0, 2 * np.pi, num_slices)
    z = np.linspace(-0.1, d, num_slices)
    if abs(c) < 1e-9:
        return 0, 0, 0
    r = d / c
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = (r * (d - z_grid) / d * np.cos(theta_grid)) + a
    y = (r * (d - z_grid) / d * np.sin(theta_grid)) + b
    return x, y, z_grid


def plot_cone_vs_data(cone_params, data, fig=None, ax=None, only_valid=False):
    """Plot fitted cone vs Lidar points (optional dev visualization)."""
    import matplotlib.pyplot as plt

    a, b, c, d = cone_params
    x, y, z = generate_cone_plot_points(a, b, c, d)
    created_fig = fig is None
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(projection="3d")
    cone_val = d - c * np.sqrt(((data[:, 0] - a) ** 2 + (data[:, 1] - b) ** 2))
    if len(data) < 5:
        return
    if only_valid and not cone_params_valid_for_plot(cone_params):
        return
    ax.scatter(x, y, z, color="red", label="Cone surface")
    ax.scatter(data[:, 0], data[:, 1], cone_val, color="green", label="Predicted z")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="blue", label="Points")
    if created_fig:
        plt.show()
