"""Closed-form cone estimators, and the (a, b) warm start every iterative fit uses.

* :func:`cone_pos_geometric` -- centroid + LiDAR-ray correction; warm start for
  every iterative fit.
* :func:`cone_fit_collinear` -- type-aware closed-form fit for (near-)collinear
  clusters, using the same class constants as the template dispatch;
  :func:`collinear_apex` is the same fit without the geometric fallback.
* :func:`cluster_linearity` -- decides which clusters count as collinear.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from .cone_classes import (
    CONE_BASE_RADIUS_M,
    CONE_BIG_C,
    CONE_BIG_D,
    CONE_CLASS_TOLERANCE,
    CONE_SMALL_C,
    CONE_SMALL_D,
)


def apex_init(x, y):
    """Initial guess for the cone apex (a, b): the cluster centroid."""
    return float(np.mean(x)), float(np.mean(y))


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
NEAR_FACE_CENTROID_BIAS_FRAC = 0.4

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
LINEARITY_THRESHOLD = 0.10

def cluster_linearity(xy):
    """Smaller / larger principal-axis variance ratio for the cluster's xy points.

    Returns a value in [0, 1]: 0 for a perfectly collinear cluster, 1 for an
    isotropic disk. Below :data:`LINEARITY_THRESHOLD` the cluster carries
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
    return float(_linearity_nb(pts))


@numba.njit(cache=True)
def _linearity_nb(pts):
    """Eigenvalue ratio of the 2x2 xy covariance, in closed form.

    One compiled call per cluster instead of mean / centre / matmul / LAPACK
    ``eigvalsh`` on a handful of points, where the NumPy call overhead was the cost.
    """
    n = pts.shape[0]
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += pts[i, 0]
        my += pts[i, 1]
    mx /= n
    my /= n
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for i in range(n):
        dx = pts[i, 0] - mx
        dy = pts[i, 1] - my
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    sxx /= n
    sxy /= n
    syy /= n
    mid = 0.5 * (sxx + syy)
    half = math.sqrt((0.5 * (sxx - syy)) ** 2 + sxy * sxy)
    return (mid - half) / (mid + half + 1e-12)


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
    offset = NEAR_FACE_CENTROID_BIAS_FRAC * CONE_BASE_RADIUS_M
    return cx + offset * rx / norm, cy + offset * ry / norm


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
         If neither class is within :data:`CONE_CLASS_TOLERANCE`, give up
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
            ground-removal step the ground sits at the LiDAR-to-floor
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
    fit = collinear_apex(arr, lidar_xy, ground_z)
    if fit is None:
        ax, by = cone_pos_geometric(arr, lidar_xy)
        return ax, by, float("nan"), float("nan")
    return fit


def _nearest_cone_class(c_est):
    """``(c, d)`` of the FSAE class whose slope is nearest ``c_est``; None if neither is close."""
    err_small = abs(c_est - CONE_SMALL_C)
    err_big = abs(c_est - CONE_BIG_C)
    if err_small <= err_big:
        err, c_cls, d_cls = err_small, CONE_SMALL_C, CONE_SMALL_D
    else:
        err, c_cls, d_cls = err_big, CONE_BIG_C, CONE_BIG_D
    return None if err > CONE_CLASS_TOLERANCE else (c_cls, d_cls)


def collinear_apex(arr, lidar_xy, ground_z):
    """Steps 1-4 of :func:`cone_fit_collinear` on a float64 ``arr``: ``(a, b, c, d_class)``,
    or None where that function falls back to the geometric fit."""
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    rx = float(x.mean()) - float(lidar_xy[0])
    ry = float(y.mean()) - float(lidar_xy[1])
    norm = math.hypot(rx, ry)
    if norm < 1e-6 or arr.shape[0] < _MIN_PTS_COLLINEAR_FIT:
        return None
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
        return None
    c_est = s_tz / s_tt
    # Near-face geometry gives ``dz/dt = +c``. A negative slope means we're
    # looking at the far face (LiDAR somehow above the cone) or pure noise;
    # either way the closed-form below would put the apex behind the cluster.
    cone_class = _nearest_cone_class(c_est) if c_est > 0.0 else None
    if cone_class is None:
        return None
    c_use, d_class = cone_class
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
    if offset_along_ray < 0.0 or offset_along_ray > 2.0 * CONE_BASE_RADIUS_M + 0.05:
        return None
    a = float(lidar_xy[0]) + t_apex * ux
    b = float(lidar_xy[1]) + t_apex * uy
    return a, b, c_use, d_class


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
# PCA-dispatch filter in ``RealtimeConeDetector.detect`` collinear
# clusters never reach VARPRO, so this bound is almost never active; it
# remains as defense-in-depth.
VARPRO_AB_BOUND_MARGIN_M = 0.05


def warm_start_and_bounds(data, lidar_xy=(0.0, 0.0)):
    """:func:`cone_pos_geometric` warm start clamped into the (a, b) search box.

    The box is the cluster xy bounding box plus :data:`VARPRO_AB_BOUND_MARGIN_M`.
    Clamping matters because cone_pos_geometric pushes the centroid back along
    the LiDAR ray by ~5 cm and can land just outside it; L-BFGS-B requires a
    feasible start. Returns ``(a0, b0, bounds)`` in scipy's ``bounds=`` form.
    """
    x = data[:, 0]
    y = data[:, 1]
    a0, b0 = cone_pos_geometric(data, lidar_xy)
    if not (np.isfinite(a0) and np.isfinite(b0)):
        a0, b0 = apex_init(x, y)
    margin = VARPRO_AB_BOUND_MARGIN_M
    bounds = (
        (float(x.min()) - margin, float(x.max()) + margin),
        (float(y.min()) - margin, float(y.max()) + margin),
    )
    a0 = min(max(a0, bounds[0][0]), bounds[0][1])
    b0 = min(max(b0, bounds[1][0]), bounds[1][1])
    return a0, b0, bounds
