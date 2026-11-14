"""ray_slope ground removal as one Numba kernel, for the fused and cpp detectors.

perception_cpp/cone_detection_core ports it 1:1 (same loop and expression order),
and test_cpp_detect.py checks the two agree bit for bit. The plane fits use a
fixed-sweep 3x3 Jacobi eigen solve instead of LAPACK eigh so both sides do the
same arithmetic.

This is the ray_slope from before the ground-method rewrite on dev (#44); the
python path now runs ``ground/methods/ray_methods.py``, which is not the same
arithmetic, so ``fused.fused_unsupported_reason`` sends ray_slope configs to the
python path. Production uses ransac.
"""
from __future__ import annotations

import numba
import numpy as np

# Cap on points used for a plane fit (ground_methods._FIT_MAX_POINTS before #44).
_FIT_MAX_POINTS = 20_000
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
