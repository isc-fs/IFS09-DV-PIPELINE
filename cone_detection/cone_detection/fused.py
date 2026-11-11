"""Fused Numba detection backend (``ConeDetectionConfig.detector_backend = "fused"``).

Runs the same scan as :meth:`RealtimeConeDetector.detect` in about ten compiled
calls instead of a few hundred numpy calls. On the Python path most of the time is
spent on glue, temporary arrays and dtype changes, not on arithmetic. Measured
over 350 recorded Hesai scans with the production config, with cones and counters
identical scan by scan:

* Apple M1: detect p50/p95/p99 went from 23.4/33.4/46.6 ms to 9.7/14.9/20.0 ms.
* RK3588, older revision: 52/72/95 ms became 32/42/57 ms.

The output is the Python path's output, not an approximation of it:

* The RANSAC sampler has the same random number sequence as ``ransac.ransac2``.
  Both draw from the Numba RNG, so seeding it reproduces both.
* The rotation and the ground shift are the same.
* The tall-column veto uses the same cells. A dense bitmap replaces sort/unique/isin.
* The clustering-input guard is the same. A hash set replaces ``np.unique``.
* Clustering uses ``fast_dbscan`` and ``cluster_backend``, called as on the
  Python path.
* Fitting uses the same per-cluster gates and the same fits. The Gauss-Newton
  solver is ``cone_fit._template_fit_gn`` itself. The label -1 noise group is
  fitted like any other cluster, as on the Python path.

The remaining differences are float rounding at the 1e-16 level. These are:

* explicit 3-term dot products instead of BLAS;
* Rodrigues' formula instead of scipy's ``Rotation``;
* sequential instead of pairwise sums.

The equivalence tests in ``test/test_fused_detect.py`` guard all of this.

Only configs it reproduces exactly run fused. :func:`fused_unsupported_reason`
names the first difference, and the detector then falls back to the Python path.
"""

from __future__ import annotations

import math
import time

import numba as nb
import numpy as np

from cone_detection.clustering import fast_dbscan
from cone_detection.cone_fit import (
    _CONE_BASE_RADIUS_M,
    _CONE_BIG_C,
    _CONE_BIG_D,
    _CONE_CLASS_TOLERANCE,
    _CONE_SMALL_C,
    _CONE_SMALL_D,
    _LINEARITY_THRESHOLD,
    _NEAR_FACE_CENTROID_BIAS_FRAC,
    _VARPRO_AB_BOUND_MARGIN_M,
    _template_fit_gn,
)

# Voxel keys of the clustering-input guard; same packing as
# cone_detection._GUARD_KEY_OFFSET / _GUARD_KEY_BITS.
_GUARD_KEY_OFFSET = 1 << 20
_GUARD_KEY_BITS = 21

# RANSAC hard iteration cap, ransac.ransac2's default (ground_ransac keeps it).
_RANSAC_MAX_ITER = 200
_DEFAULT_PLANE = np.array([0.0, 0.0, 0.0, 1.0])

# fit_clusters counter slots.
N_CLUSTERS, AFTER_MIN_PTS, AFTER_SHAPE_GATE, ACCEPTED, FAR_DROPPED = 0, 1, 2, 3, 4
SMALL, BIG, AMBIGUOUS, SHAPE_REJECTED, RESIDUAL_REJECTED = 5, 6, 7, 8, 9


def fused_unsupported_reason(cfg, data, clustering_class) -> str | None:
    """``None`` when the fused path reproduces ``cfg`` exactly, else why not."""
    if cfg.ground_method != "ransac":
        return f"ground_method={cfg.ground_method!r} (fused: 'ransac' only)"
    if cfg.fit_backend != "template_dispatch":
        return f"fit_backend={cfg.fit_backend!r} (fused: 'template_dispatch' only)"
    if cfg.cone_fit_solver != "gn":
        return f"cone_fit_solver={cfg.cone_fit_solver!r} (fused: 'gn' only)"
    if cfg.flat_cell_filter:
        return "flat_cell_filter=True (not in the fused path)"
    shape = getattr(data, "shape", ())
    if len(shape) != 2 or shape[1] != 3:
        return f"input shape {shape} (fused: (N, 3) only)"
    if getattr(data, "dtype", None) not in (np.float32, np.float64):
        return f"input dtype {getattr(data, 'dtype', None)} (fused: float32/float64 only)"
    return None


# --------------------------------------------------------------------------
# front end: crop, RANSAC, rotate, veto, guard
# --------------------------------------------------------------------------
@nb.njit(cache=True)
def crop_promote(xyz, crop2, min2):
    """Horizontal range crop and min 3-D range drop, then float64 promotion.

    Tests run in the input dtype, like the numpy reference, so a float32 scan
    keeps exactly the same points. ``crop2`` / ``min2`` <= 0 disable a test.
    """
    n = xyz.shape[0]
    out = np.empty((n, 3), dtype=np.float64)
    m = 0
    for i in range(n):
        x = xyz[i, 0]
        y = xyz[i, 1]
        z = xyz[i, 2]
        r2 = x * x + y * y
        if crop2 > 0 and not (r2 <= crop2):
            continue
        if min2 > 0 and not (r2 + z * z >= min2):
            continue
        out[m, 0] = x
        out[m, 1] = y
        out[m, 2] = z
        m += 1
    return out[:m]


@nb.njit(cache=True, error_model="numpy")  # IEEE nan/inf like the numpy reference
def ransac_plane(xyz, prob, threshold, max_iter, n_sub_max, warm):
    """``ransac.ransac2`` (m=3) on xyz without the augmented copy.

    Same algorithm and the same Numba RNG call sequence, so for the same seed
    it picks the same plane. Returns (outlier mask, coefs [bias, nx, ny, nz]).
    """
    m = 3
    n = xyz.shape[0]
    if n > n_sub_max:
        sub = np.random.choice(n, n_sub_max, replace=False)
        ns = n_sub_max
    else:
        sub = np.arange(n)
        ns = n
    sx = np.empty(ns)
    sy = np.empty(ns)
    sz = np.empty(ns)
    for i in range(ns):
        j = sub[i]
        sx[i] = xyz[j, 0]
        sy[i] = xyz[j, 1]
        sz[i] = xyz[j, 2]
    k = float(max_iter)
    support = 0
    best = np.array([0.0, 0.0, 0.0, 1.0])
    if warm.shape[0] == 4 and abs(warm[3]) > 1e-12:
        c0 = warm[0] / -warm[3]
        c1 = warm[1] / -warm[3]
        c2 = warm[2] / -warm[3]
        s = 0
        for i in range(ns):
            if abs(c0 + c1 * sx[i] + c2 * sy[i] - sz[i]) < threshold:
                s += 1
        s -= m
        if s > support:
            support = s
            best = warm.copy()
            r = support / ns
            if 0.0 < r < 1.0:
                kn = math.log(1 - prob) / math.log(1 - r**m)
                if kn < k:
                    k = kn
    idx = np.empty(m, dtype=np.int64)
    it = 0
    while it < k and it < max_iter:
        for jj in range(m):
            while True:
                cand = np.random.randint(0, n)
                dup = False
                for kk in range(jj):
                    if idx[kk] == cand:
                        dup = True
                        break
                if not dup:
                    idx[jj] = cand
                    break
        p0x = xyz[idx[0], 0]
        p0y = xyz[idx[0], 1]
        p0z = xyz[idx[0], 2]
        v1x = xyz[idx[1], 0] - p0x
        v1y = xyz[idx[1], 1] - p0y
        v1z = xyz[idx[1], 2] - p0z
        v2x = xyz[idx[2], 0] - p0x
        v2y = xyz[idx[2], 1] - p0y
        v2z = xyz[idx[2], 2] - p0z
        nx = v1y * v2z - v1z * v2y
        ny = v1z * v2x - v1x * v2z
        nz = v1x * v2y - v1y * v2x
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= nn
        ny /= nn
        nz /= nn
        bias = -(nx * p0x + ny * p0y + nz * p0z)
        c0 = bias / -nz
        c1 = nx / -nz
        c2 = ny / -nz
        s = 0
        for i in range(ns):
            if abs(c0 + c1 * sx[i] + c2 * sy[i] - sz[i]) < threshold:
                s += 1
        s -= m
        if s > support:
            support = s
            best = np.array([bias, nx, ny, nz])
            r = support / ns
            if 0.0 < r < 1.0:
                kn = math.log(1 - prob) / math.log(1 - r**m)
                if kn < k:
                    k = kn
        it += 1
    if best[3] < 0:
        best = -best
    c0 = best[0] / -best[3]
    c1 = best[1] / -best[3]
    c2 = best[2] / -best[3]
    out = np.empty(n, dtype=np.bool_)
    for i in range(n):
        out[i] = not (abs(c0 + c1 * xyz[i, 0] + c2 * xyz[i, 1] - xyz[i, 2]) < threshold)
    return out, best


@nb.njit(cache=True, error_model="numpy")
def rot_to_normal(coefs):
    """Rotation mapping +z onto the plane normal (= ground_rotation_matrix)."""
    w = coefs[1:]
    nn = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
    if nn < 1e-12:
        return np.eye(3)
    vx = w[0] / nn
    vy = w[1] / nn
    vz = w[2] / nn
    ax, ay = -vy, vx  # cross((0, 0, 1), v)
    s = math.sqrt(ax * ax + ay * ay)
    R = np.eye(3)
    if s == 0.0:
        if vz < 0:
            R = -R
        return R
    ang = math.acos(min(1.0, max(-1.0, vz)))
    ax /= s
    ay /= s
    c = math.cos(ang)
    sn = math.sin(ang)
    t = 1 - c
    R[0, 0] = c + ax * ax * t
    R[0, 1] = ax * ay * t
    R[0, 2] = ay * sn
    R[1, 0] = ax * ay * t
    R[1, 1] = c + ay * ay * t
    R[1, 2] = -ax * sn
    R[2, 0] = -ay * sn
    R[2, 1] = ax * sn
    R[2, 2] = c
    return R


@nb.njit(cache=True)
def select_rotate(xyz, outl, R, zshift):
    """``xyz[outl] @ R``, then ``z -= zshift`` (ground to z = 0)."""
    m = 0
    for i in range(outl.shape[0]):
        if outl[i]:
            m += 1
    out = np.empty((m, 3))
    j = 0
    for i in range(outl.shape[0]):
        if outl[i]:
            x = xyz[i, 0]
            y = xyz[i, 1]
            z = xyz[i, 2]
            out[j, 0] = x * R[0, 0] + y * R[1, 0] + z * R[2, 0]
            out[j, 1] = x * R[0, 1] + y * R[1, 1] + z * R[2, 1]
            out[j, 2] = (x * R[0, 2] + y * R[1, 2] + z * R[2, 2]) - zshift
            j += 1
    return out


@nb.njit(cache=True)
def plane_outliers(xyz, coefs, threshold):
    """Outlier mask of a given plane (for callers that force the plane)."""
    c0 = coefs[0] / -coefs[3]
    c1 = coefs[1] / -coefs[3]
    c2 = coefs[2] / -coefs[3]
    out = np.empty(xyz.shape[0], dtype=np.bool_)
    for i in range(xyz.shape[0]):
        out[i] = not (abs(c0 + c1 * xyz[i, 0] + c2 * xyz[i, 1] - xyz[i, 2]) < threshold)
    return out


@nb.njit(cache=True, error_model="numpy")
def column_veto(pts, h_veto, cell):
    """Tall-column veto (= ``_tall_column_veto_mask``) on z = height above ground.

    Drops every point whose 3x3-dilated xy cell holds a return higher than
    ``h_veto``. A dense bitmap over the cloud's cell range replaces the key
    sort: O(n), two passes, the same cells (floor(x / cell)). Returns
    (pts unchanged, 0) when nothing is tall.
    """
    n = pts.shape[0]
    ci = np.empty(n, dtype=np.int64)
    cj = np.empty(n, dtype=np.int64)
    ntall = 0
    imin = jmin = np.int64(1) << 62
    imax = jmax = -(np.int64(1) << 62)
    for k in range(n):
        i = np.int64(math.floor(pts[k, 0] / cell))
        j = np.int64(math.floor(pts[k, 1] / cell))
        ci[k] = i
        cj[k] = j
        imin = min(imin, i)
        imax = max(imax, i)
        jmin = min(jmin, j)
        jmax = max(jmax, j)
        if pts[k, 2] > h_veto:
            ntall += 1
    if ntall == 0:
        return pts, 0
    veto = np.zeros((imax - imin + 3, jmax - jmin + 3), dtype=np.uint8)  # 1-cell pad
    for k in range(n):
        if pts[k, 2] > h_veto:
            a = ci[k] - imin + 1
            b = cj[k] - jmin + 1
            for da in range(-1, 2):
                for db in range(-1, 2):
                    veto[a + da, b + db] = 1
    m = 0
    keep = np.empty(n, dtype=np.bool_)
    for k in range(n):
        keep[k] = veto[ci[k] - imin + 1, cj[k] - jmin + 1] == 0
        if keep[k]:
            m += 1
    out = np.empty((m, 3))
    j = 0
    for k in range(n):
        if keep[k]:
            out[j, 0] = pts[k, 0]
            out[j, 1] = pts[k, 1]
            out[j, 2] = pts[k, 2]
            j += 1
    return out, n - m


@nb.njit(cache=True)
def voxel_first(pts, voxel, offset, bits):
    """Ascending indices of the first point of every voxel (= ``_voxel_first_points``).

    One pass in row order with an open-addressing hash set, O(n).
    """
    n = pts.shape[0]
    cap = 1
    lg = 0
    while cap < 2 * n:
        cap <<= 1
        lg += 1
    sh = np.uint64(64 - max(lg, 1))
    table = np.full(cap, -1, dtype=np.int64)  # stores keys; every key is >= 0
    keep = np.empty(n, dtype=np.int64)
    m = 0
    for k in range(n):
        a = np.int64(math.floor(pts[k, 0] / voxel)) + offset
        b = np.int64(math.floor(pts[k, 1] / voxel)) + offset
        c = np.int64(math.floor(pts[k, 2] / voxel)) + offset
        key = (a << (2 * bits)) | (b << bits) | c
        # Fibonacci hashing: the HIGH bits of key * 2^64/phi depend on every key bit
        # (the low bits would only see z, and the table would degenerate).
        slot = np.int64((np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)) >> sh) & (cap - 1)
        while True:
            if table[slot] == -1:
                table[slot] = key
                keep[m] = k
                m += 1
                break
            if table[slot] == key:
                break
            slot = (slot + 1) & (cap - 1)
    return keep[:m]


def bound_cluster_input(pts: np.ndarray, cfg) -> np.ndarray:
    """``cone_detection._bound_dbscan_input``: same loop, same seeded fallback."""
    cap = int(cfg.dbscan_max_points)
    if cap <= 0 or len(pts) <= cap:
        return pts
    voxel = float(cfg.dbscan_guard_voxel_m)
    while voxel > 0.0 and voxel <= cfg.dbscan_guard_voxel_max_m + 1e-9 and len(pts) > cap:
        pts = pts[voxel_first(pts, voxel, _GUARD_KEY_OFFSET, _GUARD_KEY_BITS)]
        voxel *= 2.0
    if len(pts) > cap:
        rng = np.random.default_rng(len(pts))
        keep = rng.choice(len(pts), cap, replace=False)
        pts = pts[np.sort(keep)]
    return pts


# --------------------------------------------------------------------------
# back end: split by label, gate and fit every cluster
# --------------------------------------------------------------------------
@nb.njit(cache=True)
def _geometric(x, y):
    """``cone_fit.cone_pos_geometric`` with the LiDAR at the origin."""
    n = x.shape[0]
    cx = 0.0
    cy = 0.0
    for i in range(n):
        cx += x[i]
        cy += y[i]
    cx /= n
    cy /= n
    nr = math.hypot(cx, cy)
    if nr < 1e-6:
        return cx, cy
    off = _NEAR_FACE_CENTROID_BIAS_FRAC * _CONE_BASE_RADIUS_M
    return cx + off * cx / nr, cy + off * cy / nr


@nb.njit(cache=True)
def _linearity(x, y):
    """``cone_fit.cluster_linearity``: closed-form 2x2 eigenvalues."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    mx = x.mean()
    my = y.mean()
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    sxx /= n
    sxy /= n
    syy /= n
    tr = 0.5 * (sxx + syy)
    disc = math.sqrt(max(0.0, 0.25 * (sxx - syy) ** 2 + sxy * sxy))
    return (tr - disc) / (tr + disc + 1e-12)


@nb.njit(cache=True)
def _mse(a, b, c, d, x, y, z):
    s = 0.0
    for i in range(x.shape[0]):
        e = z[i] - (d - c * math.sqrt((x[i] - a) ** 2 + (y[i] - b) ** 2))
        s += e * e
    return s / x.shape[0]


@nb.njit(cache=True)
def _collinear(x, y, z):
    """``cone_fit.cone_fit_collinear`` (LiDAR at the origin, ground_z = 0)."""
    nan = np.nan
    n = x.shape[0]
    cx = x.mean()
    cy = y.mean()
    nr = math.hypot(cx, cy)
    if nr < 1e-6 or n < 3:
        a, b = _geometric(x, y)
        return a, b, nan, nan
    ux = cx / nr
    uy = cy / nr
    t = x * ux + y * uy
    tm = t.mean()
    zm = z.mean()
    stt = 0.0
    stz = 0.0
    for i in range(n):
        stt += (t[i] - tm) * (t[i] - tm)
        stz += (t[i] - tm) * (z[i] - zm)
    if stt < 1e-6:
        a, b = _geometric(x, y)
        return a, b, nan, nan
    ce = stz / stt
    if ce <= 0.0:
        a, b = _geometric(x, y)
        return a, b, nan, nan
    es = abs(ce - _CONE_SMALL_C)
    eb = abs(ce - _CONE_BIG_C)
    if es <= eb:
        if es > _CONE_CLASS_TOLERANCE:
            a, b = _geometric(x, y)
            return a, b, nan, nan
        cu, dc = _CONE_SMALL_C, _CONE_SMALL_D
    else:
        if eb > _CONE_CLASS_TOLERANCE:
            a, b = _geometric(x, y)
            return a, b, nan, nan
        cu, dc = _CONE_BIG_C, _CONE_BIG_D
    ta = tm + (dc - zm) / cu
    off = ta - tm
    if off < 0.0 or off > 2.0 * _CONE_BASE_RADIUS_M + 0.05:
        a, b = _geometric(x, y)
        return a, b, nan, nan
    return ta * ux, ta * uy, cu, dc


@nb.njit(cache=True)
def _template(x, y, z, c, d, maxiter):
    """``cone_fit.cone_fit_template(solver="gn")``: same init, bounds and solver."""
    a0, b0 = _geometric(x, y)
    if not (np.isfinite(a0) and np.isfinite(b0)):
        a0, b0 = x.mean(), y.mean()
    alo = x.min() - _VARPRO_AB_BOUND_MARGIN_M
    ahi = x.max() + _VARPRO_AB_BOUND_MARGIN_M
    blo = y.min() - _VARPRO_AB_BOUND_MARGIN_M
    bhi = y.max() + _VARPRO_AB_BOUND_MARGIN_M
    a0 = min(max(a0, alo), ahi)
    b0 = min(max(b0, blo), bhi)
    return _template_fit_gn(x, y, z, c, d, a0, b0, alo, ahi, blo, bhi, maxiter)


@nb.njit(cache=True)
def fit_clusters(pts, labels, p):
    """Split by label and gate and fit every cluster, like the detect() loop.

    Clusters go in ascending label order (= ``np.unique``), each in row order
    (stable sort). z is height above ground.
    p = [min_pts, floor_margin, hmin, hmax, rmax, res_gate, amb_ratio,
         h_small, h_big, maxiter, use_collinear]
    (h_small = -inf and h_big = +inf disable the early exit.)
    Returns rows (K, 5) [a, b, d, sigma_xy, residual] and counters (see the
    module's slot names).
    """
    n = pts.shape[0]
    order = np.argsort(labels, kind="mergesort")
    cnt = np.zeros(10, dtype=np.int64)
    out = np.empty((n, 5))
    k = 0
    s = 0
    while s < n:
        lab = labels[order[s]]
        e = s
        while e < n and labels[order[e]] == lab:
            e += 1
        cnt[N_CLUSTERS] += 1
        size = e - s
        st = s
        s = e
        if size < p[0]:
            continue
        cnt[AFTER_MIN_PTS] += 1
        thr = p[1]
        m = 0
        for q in range(st, st + size):
            if pts[order[q], 2] > thr:
                m += 1
        if m == 0:
            continue
        x = np.empty(m)
        y = np.empty(m)
        z = np.empty(m)
        j = 0
        for q in range(st, st + size):
            r = order[q]
            if pts[r, 2] > thr:
                x[j] = pts[r, 0]
                y[j] = pts[r, 1]
                z[j] = pts[r, 2]
                j += 1
        h = z.max() - z.min()
        rng = math.hypot(x.mean(), y.mean())
        if not (p[2] <= h <= p[3] and rng <= p[4]):
            cnt[SHAPE_REJECTED] += 1
            if rng > p[4]:
                cnt[FAR_DROPPED] += 1
            continue
        cnt[AFTER_SHAPE_GATE] += 1
        done = False
        a = b = d = res = 0.0
        other = np.inf
        big = False
        if p[10] != 0.0 and _linearity(x, y) < _LINEARITY_THRESHOLD:
            a, b, c, dc = _collinear(x, y, z)
            if np.isfinite(c):
                res = _mse(a, b, c, dc, x, y, z)
                other = np.nan
                big = dc >= _CONE_BIG_D - 1e-6
                d = dc
                done = True
        if not done:
            mi = int(p[9])
            if h < p[7]:
                a, b, res = _template(x, y, z, _CONE_SMALL_C, _CONE_SMALL_D, mi)
                d, big, other = _CONE_SMALL_D, False, np.inf
            elif h > p[8]:
                a, b, res = _template(x, y, z, _CONE_BIG_C, _CONE_BIG_D, mi)
                d, big, other = _CONE_BIG_D, True, np.inf
            else:
                a_s, b_s, r_s = _template(x, y, z, _CONE_SMALL_C, _CONE_SMALL_D, mi)
                a_b, b_b, r_b = _template(x, y, z, _CONE_BIG_C, _CONE_BIG_D, mi)
                if r_s <= r_b:
                    a, b, res, d, big, other = a_s, b_s, r_s, _CONE_SMALL_D, False, r_b
                else:
                    a, b, res, d, big, other = a_b, b_b, r_b, _CONE_BIG_D, True, r_s
        if not (np.isfinite(res) and res <= p[5]):
            cnt[RESIDUAL_REJECTED] += 1
            continue
        amb = np.isfinite(other) and other / max(res, 1e-12) < p[6]
        if amb:
            cnt[AMBIGUOUS] += 1
        if big:
            cnt[BIG] += 1
        else:
            cnt[SMALL] += 1
        cnt[ACCEPTED] += 1
        sig = 1.5 * (0.05 + 0.005 * math.hypot(a, b)) / math.sqrt(max(1, m) / 10.0)
        if amb:
            sig *= 1.5
        out[k, 0] = a
        out[k, 1] = b
        out[k, 2] = d
        out[k, 3] = sig
        out[k, 4] = res
        k += 1
    return out[:k].copy(), cnt


def fit_params(cfg) -> np.ndarray:
    """``fit_clusters``' parameter vector for ``cfg``."""
    early = bool(cfg.template_fit_early_exit)
    return np.array(
        [
            cfg.min_cluster_points,
            cfg.floor_margin_m,
            cfg.cluster_height_min_m,
            cfg.cluster_height_max_m,
            cfg.range_gate_max_m,
            cfg.residual_gate_mse,
            cfg.ambiguous_margin_ratio,
            cfg.template_early_exit_height_small_m if early else -np.inf,
            cfg.template_early_exit_height_big_m if early else np.inf,
            cfg.template_fit_maxiter,
            1.0 if cfg.use_collinear_fit else 0.0,
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
class FusedScan:
    """One fused scan's results, in the shapes ``RealtimeConeDetector`` publishes."""

    __slots__ = ("cropped", "coefs", "clean", "labels", "rows", "counters", "guard_dropped")

    def __init__(self):
        self.cropped = np.zeros((0, 3))
        self.coefs = _DEFAULT_PLANE.copy()
        self.clean = np.zeros((0, 3))
        self.labels = np.zeros(0, dtype=np.int64)
        self.rows = np.zeros((0, 5))
        self.counters = np.zeros(10, dtype=np.int64)
        self.guard_dropped = 0


def run_scan(
    data: np.ndarray,
    cfg,
    *,
    initial_plane: np.ndarray | None,
    ransac_iter_subsample_max: int,
    clustering_class: type,
    fast_clustering: bool,
    stage_timings: dict,
) -> FusedScan:
    """Crop, ground, rotate, veto, guard, cluster, fit. ``stage_timings`` gets the
    Python path's keys. ``scan.cropped`` is empty when the crop left nothing.
    ``fast_clustering`` = the caller kept the default ``clustering_class`` (the
    Python path then uses ``cluster_backend``; so does this one)."""
    st = stage_timings
    out = FusedScan()
    dt = data.dtype.type
    crop2 = dt(cfg.input_range_crop_m**2) if cfg.input_range_crop_m > 0 else dt(0)
    min2 = dt(cfg.input_min_range_m**2) if cfg.input_min_range_m > 0 else dt(0)
    pts = crop_promote(np.ascontiguousarray(data), crop2, min2)
    out.cropped = pts
    if len(pts) == 0:
        return out

    t0 = time.perf_counter()
    if len(pts) < 3:  # remove_ground: no plane to fit, everything is an outlier
        outl = np.ones(len(pts), dtype=np.bool_)
        coefs = _DEFAULT_PLANE.copy()
    else:
        warm = (
            np.asarray(initial_plane, dtype=np.float64)
            if initial_plane is not None
            else np.zeros(0)
        )
        outl, coefs = ransac_plane(
            pts,
            float(cfg.ransac_prob),
            float(cfg.ransac_threshold),
            _RANSAC_MAX_ITER,
            int(ransac_iter_subsample_max),
            warm,
        )
    out.coefs = coefs
    t1 = time.perf_counter()
    st["ransac_ms"] = (t1 - t0) * 1000.0

    w = coefs[1:]
    nrm = math.sqrt(float(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]))
    floor = 0.0 if nrm < 1e-12 else float(np.dot(np.array([0.0, 0.0, -float(coefs[0])]), w) / nrm)
    sel = select_rotate(pts, outl, rot_to_normal(coefs), floor)
    t2 = time.perf_counter()
    st["rotate_ms"] = (t2 - t1) * 1000.0
    st["n_outliers"] = float(len(sel))
    out.clean = sel
    if len(sel) == 0:
        st["dbscan_ms"] = 0.0
        st["n_dbscan_guard_dropped"] = 0.0
        return out

    if cfg.tall_column_veto:
        sel, n_vetoed = column_veto(
            sel, float(cfg.tall_column_veto_height_m), float(cfg.tall_column_veto_cell_m)
        )
        t3 = time.perf_counter()
        st["column_veto_ms"] = (t3 - t2) * 1000.0
        st["n_vetoed"] = float(n_vetoed)
        out.clean = sel
        if len(sel) == 0:
            st["dbscan_ms"] = 0.0
            st["n_dbscan_guard_dropped"] = 0.0
            return out

    n_before = len(sel)
    sel = bound_cluster_input(sel, cfg)
    out.guard_dropped = n_before - len(sel)
    st["n_dbscan_guard_dropped"] = float(out.guard_dropped)
    out.clean = sel

    t4 = time.perf_counter()
    labels = None
    if cfg.cluster_backend != "sklearn" and fast_clustering:
        labels = fast_dbscan(sel, cfg.dbscan_eps, cfg.dbscan_min_samples, cfg.cluster_backend)
    if labels is None:
        labels = clustering_class(
            eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples
        ).fit_predict(sel)
    out.labels = np.ascontiguousarray(labels, dtype=np.int64)
    t5 = time.perf_counter()
    st["dbscan_ms"] = (t5 - t4) * 1000.0

    out.rows, out.counters = fit_clusters(sel, out.labels, fit_params(cfg))
    st["cluster_prep_ms"] = 0.0  # the split is part of fit_clusters
    st["n_clusters"] = float(out.counters[N_CLUSTERS])
    st["fit_ms"] = (time.perf_counter() - t5) * 1000.0
    return out


def warmup() -> None:
    """Compile what a normal warm-up scan does not reach (the guard's hash set)."""
    pts = np.random.default_rng(0).uniform(0.0, 1.0, (64, 3))
    voxel_first(pts, 0.02, _GUARD_KEY_OFFSET, _GUARD_KEY_BITS)
    plane_outliers(pts, _DEFAULT_PLANE.copy(), 0.05)
