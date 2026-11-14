"""Methods that walk the scan outward from the sensor: ``ray_slope``, ``linefit``,
``gp_insac`` (per azimuth sector) and ``range_image`` (per laser channel).
"""

from __future__ import annotations

import numba
import numpy as np

from .scan_geometry import _level, _recover_rings, _sectors
from .plane_fitting import _DEFAULT_PLANE, _coarse_plane, _refit_on_ground, plane_residual
from .registry import register


@numba.njit(cache=True)
def _ray_walk(
    order, rng, z, resid, sector_start, sector_end, z0, max_slope, tol, max_gap, max_height
):
    """Classify points along each azimuth ray, nearest first. True = non-ground."""
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
def ground_ray_slope(data, cfg):
    rng, order, s0, s1 = _sectors(data, 360)
    # Walk on height above a coarse GPF plane, not on raw z: the sensor is not
    # level with the floor (the real car's is ~9 deg off), and a slope limit on
    # raw z would reject all of that tilted ground.
    prior = _coarse_plane(data, cfg.ground_threshold_m)
    height = np.ascontiguousarray(plane_residual(data, prior))
    non_ground = _ray_walk(
        order,
        rng,
        height,
        height,
        s0,
        s1,
        0.0,
        np.tan(np.radians(6.0)),
        cfg.ground_threshold_m,
        3.0,
        0.10,
    )
    return non_ground, _refit_on_ground(data, non_ground, _DEFAULT_PLANE.copy())


@numba.njit(cache=True)
def _linefit_walk(order, sec_start, sec_end, r, h, bin_m, max_slope, max_rmse,
                  max_start_h, max_gap, thr):
    """Himmelsbach line extraction per sector over lowest-per-bin prototypes."""
    out = np.ones(len(r), dtype=np.bool_)
    for s in range(len(sec_start)):
        a = sec_start[s]
        b = sec_end[s]
        if b <= a:
            continue
        n_pts = b - a
        pr = np.empty(n_pts)
        ph = np.empty(n_pts)
        proto_of = np.empty(n_pts, dtype=np.int64)
        n_proto = 0
        k = a
        while k < b:
            bi = np.floor(r[order[k]] / bin_m)
            best = np.inf
            best_r = 0.0
            while k < b and np.floor(r[order[k]] / bin_m) == bi:
                i = order[k]
                if h[i] < best:
                    best = h[i]
                    best_r = r[i]
                proto_of[k - a] = n_proto
                k += 1
            pr[n_proto] = best_r
            ph[n_proto] = best
            n_proto += 1

        # Greedy segments. A prototype that would bend the line too much is
        # skipped, not a line break: a lone cone bin must not end the floor.
        seg_of = np.full(n_proto, -1, dtype=np.int64)
        seg_m = np.zeros(n_proto)
        seg_c = np.zeros(n_proto)
        n_seg = 0
        cur = -1
        n = 0
        sx = sy = sxx = sxy = syy = 0.0
        last_r = 0.0
        for j in range(n_proto):
            x = pr[j]
            y = ph[j]
            if cur >= 0 and x - last_r > max_gap:
                cur = -1
                n = 0
            if cur < 0:
                if abs(y) <= max_start_h:
                    cur = n_seg
                    n_seg += 1
                    n = 1
                    sx = x
                    sy = y
                    sxx = x * x
                    sxy = x * y
                    syy = y * y
                    seg_m[cur] = 0.0
                    seg_c[cur] = y
                    seg_of[j] = cur
                    last_r = x
                continue
            tn = n + 1
            tsx = sx + x
            tsy = sy + y
            tsxx = sxx + x * x
            tsxy = sxy + x * y
            tsyy = syy + y * y
            den = tn * tsxx - tsx * tsx
            if abs(den) < 1e-9:
                m = 0.0
                c = tsy / tn
            else:
                m = (tn * tsxy - tsx * tsy) / den
                c = (tsy - m * tsx) / tn
            sse = tsyy - 2 * m * tsxy - 2 * c * tsy + m * m * tsxx + 2 * m * c * tsx + tn * c * c
            rmse = np.sqrt(max(sse, 0.0) / tn)
            # The new prototype must also sit on the current line, not only keep
            # the fit's RMSE low (a long line absorbs one bad point).
            pred = seg_m[cur] * x + seg_c[cur]
            if abs(m) <= max_slope and rmse <= max_rmse and abs(y - pred) <= 2 * max_rmse + thr:
                n = tn
                sx = tsx
                sy = tsy
                sxx = tsxx
                sxy = tsxy
                syy = tsyy
                seg_m[cur] = m
                seg_c[cur] = c
                seg_of[j] = cur
                last_r = x
            else:
                seg_of[j] = cur  # label against the line it failed to join

        for k2 in range(a, b):
            i = order[k2]
            sg = seg_of[proto_of[k2 - a]]
            if sg >= 0 and abs(h[i] - (seg_m[sg] * r[i] + seg_c[sg])) < thr:
                out[i] = False
    return out


@register("linefit")
def ground_linefit(data, cfg):
    """Himmelsbach et al. (IV 2010): line segments over each azimuth sector."""
    thr = cfg.ground_threshold_m
    uv, h, coarse = _level(data, thr)
    rng, order, s0, s1 = _sectors(uv, 360)
    non_ground = _linefit_walk(
        order, s0, s1, rng, np.ascontiguousarray(h),
        0.25, np.tan(np.radians(8.0)), 0.03, 0.20, 3.0, thr,
    )
    return non_ground, _refit_on_ground(data, non_ground, coarse)


# GP-INSAC (Douillard et al., ICRA 2011): per-sector 1-D Gaussian process of
# height over range, grown from near-sensor seeds.
_GP_SECTORS = 180


_GP_TRAIN_MAX = 200


_GP_LENGTH_M = 4.0


_GP_SIGNAL_M = 0.20


_GP_NOISE_M = 0.03


_GP_MAX_STD_M = 0.10


_GP_ITERS = 8


def _gp_predict(xt, yt, xq):
    k = lambda a, b: _GP_SIGNAL_M ** 2 * np.exp(-0.5 * ((a[:, None] - b[None, :]) / _GP_LENGTH_M) ** 2)
    K = k(xt, xt) + _GP_NOISE_M ** 2 * np.eye(len(xt))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, yt))
    Ks = k(xq, xt)
    v = np.linalg.solve(L, Ks.T)
    return Ks @ alpha, np.maximum(_GP_SIGNAL_M ** 2 - (v * v).sum(axis=0), 0.0)


@register("gp_insac")
def ground_gp_insac(data, cfg):
    """Gaussian-process INSAC: grow ground where the GP predicts it confidently."""
    thr = cfg.ground_threshold_m
    uv, h, coarse = _level(data, thr)
    rng, order, s0, s1 = _sectors(uv, _GP_SECTORS)
    ground = np.zeros(len(data), dtype=bool)
    for a, b in zip(s0, s1):
        idx = order[a:b]
        if len(idx) < 5:
            continue
        r, z = rng[idx], h[idx]
        # Seeds: the near returns that sit on the coarse plane.
        inl = (r < np.percentile(r, 20)) & (np.abs(z) < 2 * thr)
        for _ in range(_GP_ITERS):
            if inl.sum() < 3:
                break
            tr = np.flatnonzero(inl)
            if len(tr) > _GP_TRAIN_MAX:
                tr = tr[np.linspace(0, len(tr) - 1, _GP_TRAIN_MAX).astype(np.int64)]
            mu, var = _gp_predict(r[tr], z[tr], r)
            new = (np.abs(z - mu) < thr) & (np.sqrt(var) < _GP_MAX_STD_M)
            if np.array_equal(new, inl):
                break
            inl = new
        ground[idx] = inl
    non_ground = ~ground
    return non_ground, _refit_on_ground(data, non_ground, coarse)


@numba.njit(cache=True)
def _range_image_bfs(pix, n_rows, n_cols, r, h, start_deg, step_deg, max_h):
    """Bogoslavskyi & Stachniss (IROS 2016) angle labelling on a range image.

    ``pix`` holds one point index per (row, col), -1 for empty; row 0 is the
    lowest beam. ``alpha`` of a pixel is the inclination of the line to the next
    valid pixel below it in the same column, smoothed over 3 rows.
    """
    alpha = np.full((n_rows, n_cols), np.nan)
    for c in range(n_cols):
        prev = -1
        for row in range(n_rows):
            i = pix[row, c]
            if i < 0:
                continue
            if prev >= 0:
                dr = abs(r[i] - r[prev])
                dz = abs(h[i] - h[prev])
                alpha[row, c] = np.degrees(np.arctan2(dz, dr))
            prev = i
        # The lowest valid pixel takes its upper neighbour's angle.
        first = -1
        for row in range(n_rows):
            if pix[row, c] >= 0:
                if first < 0:
                    first = row
                elif np.isnan(alpha[first, c]):
                    alpha[first, c] = alpha[row, c]
                    break
    sm = alpha.copy()
    for c in range(n_cols):
        for row in range(n_rows):
            if np.isnan(alpha[row, c]):
                continue
            acc = 0.0
            w = 0
            for d in (-1, 0, 1):
                rr = row + d
                if 0 <= rr < n_rows and not np.isnan(alpha[rr, c]):
                    acc += alpha[rr, c]
                    w += 1
            sm[row, c] = acc / w

    label = np.zeros((n_rows, n_cols), dtype=np.bool_)
    stack_r = np.empty(n_rows * n_cols, dtype=np.int64)
    stack_c = np.empty(n_rows * n_cols, dtype=np.int64)
    top = 0
    for c in range(n_cols):
        for row in range(n_rows):
            i = pix[row, c]
            if i < 0:
                continue
            # Seed: the lowest valid pixel of the column, if it is flat and low.
            if not np.isnan(sm[row, c]) and sm[row, c] < start_deg and abs(h[i]) < max_h:
                label[row, c] = True
                stack_r[top] = row
                stack_c[top] = c
                top += 1
            break
    while top > 0:
        top -= 1
        row = stack_r[top]
        c = stack_c[top]
        for dr_, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = row + dr_
            cc = (c + dc) % n_cols
            # Step over empty pixels along the column (dropped returns).
            while 0 <= rr < n_rows and dc == 0 and pix[rr, cc] < 0:
                rr += dr_
            if not (0 <= rr < n_rows) or pix[rr, cc] < 0 or label[rr, cc]:
                continue
            if np.isnan(sm[rr, cc]) or abs(sm[rr, cc] - sm[row, c]) >= step_deg:
                continue
            if abs(h[pix[rr, cc]]) >= max_h:
                continue
            label[rr, cc] = True
            stack_r[top] = rr
            stack_c[top] = cc
            top += 1
    return label


_RI_COL_DEG = 0.1


@register("range_image")
def ground_range_image(data, cfg):
    """Range-image angle labelling (Bogoslavskyi & Stachniss, IROS 2016)."""
    thr = cfg.ground_threshold_m
    uv, h, coarse = _level(data, thr)
    ring = _recover_rings(data)
    n_rows = int(ring.max()) + 1
    n_cols = int(round(360.0 / _RI_COL_DEG))
    az = np.degrees(np.arctan2(data[:, 1], data[:, 0])) + 180.0
    col = np.minimum((az / _RI_COL_DEG).astype(np.int64), n_cols - 1)
    rng = np.hypot(uv[:, 0], uv[:, 1])
    # One point per pixel: the nearest (the first surface the beam hit).
    pix = np.full((n_rows, n_cols), -1, dtype=np.int64)
    order = np.argsort(-rng, kind="stable")
    pix[ring[order], col[order]] = order
    label = _range_image_bfs(pix, n_rows, n_cols, rng, np.ascontiguousarray(h), 5.0, 5.0, 0.30)
    # Every point takes its pixel's label, then must be near the pixel's surface.
    rep = pix[ring, col]
    ground = label[ring, col] & (np.abs(h - h[rep]) < thr)
    non_ground = ~ground
    return non_ground, _refit_on_ground(data, non_ground, coarse)
