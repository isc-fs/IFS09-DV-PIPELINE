"""Numba-compiled RANSAC plane fit, used for ground-plane removal.

Called by the ``ransac`` family of ground methods (``ground/methods/plane_methods.py``)
on every scan. ``ransac_ground_mask`` is the 3-D production kernel; ``ransac2`` also
handles the MAD threshold and 2-D data (extending it to higher dimensions would need
changes to the cross-product step in the iteration loop).
"""
import math

import numpy as np
import numba


@numba.njit(cache=True)
def _iteration_budget(k, support, n_scored, prob, m):
    """Standard adaptive RANSAC bound after a new best ``support``, never above ``k``.

    Guards the perfect-inlier case (log(0) = -inf) and never lets the budget
    climb: degenerate scans must exit cleanly instead of running for thousands
    of iterations.
    """
    inlier_ratio = support / n_scored
    if 0.0 < inlier_ratio < 1.0:
        k_new = np.log(1 - prob) / np.log(1 - inlier_ratio ** m)
        if k_new < k:
            return k_new
    return k


@numba.njit(cache=True)
def _support_on_slope_form(A_xy, A_z, coefs, threshold, m):
    """Points within ``threshold`` of ``z = A_xy . coefs[:-1] / -coefs[-1]``, minus ``m``."""
    support = 0
    for i in A_xy.dot(coefs[:-1] / (-1 * coefs[-1])) - A_z:
        if abs(i) < threshold:
            support += 1
    return support - m


@numba.njit(cache=True)
def _support_3d(sx, sy, sz, c0, c1, c2, c3, threshold, m):
    """Subsample points within vertical distance ``threshold`` of the plane, minus ``m``.

    ``|bias + n . p| < threshold * |n_z|  <=>  |z_plane(x, y) - z| < threshold``.
    """
    lim = threshold * abs(c3)
    cnt = 0
    for s in range(sx.shape[0]):
        if abs(c0 + c1 * sx[s] + c2 * sy[s] + c3 * sz[s]) < lim:
            cnt += 1
    return cnt - m


# cache=True persists compiled machine code in __pycache__ (or
# NUMBA_CACHE_DIR) so restarts skip the multi-second JIT. Requires every
# default value to be picklable — hence initial_coefs defaults to None,
# not an empty array.
@numba.jit(nopython=True, cache=True)
def ransac2(
    data,
    m=3,
    prob=0.999,
    threshold=None,
    max_iter=200,
    iter_subsample_max=5000,
    initial_coefs=None,
):
    """RANSAC plane fit. Returns (inlier_mask, plane coefficients).

    Args:
        data: (N, dim+1) — already augmented with a leading column of 1s
            for the bias term; numba doesn't play nicely with the
            np.c_/np.stack we'd otherwise use to add it on the fly.
        m: number of points that define a candidate plane.
        prob: target probability that the chosen subset is outlier-free
            (refines the iteration budget after each improving sample).
        threshold: inlier residual threshold. If None, set to the
            median absolute deviation of the last column — same heuristic
            sklearn's RANSACRegressor uses, with absolute (not squared)
            error so it's robust when the last dim isn't the largest-
            variance one.
        max_iter: hard upper bound on iterations. Degenerate scans
            with no consensus fall through cleanly at this cap.
        iter_subsample_max: cap on points used in the RANSAC iteration
            loop (#247). Consensus scoring runs on at most this many
            points; the final inlier mask always uses the full cloud.
            Set to 0 or negative to disable subsampling (profiling only).
        initial_coefs: optional warm-start plane ``[bias, n_x, ..., n_z]``
            (e.g. the previous scan's result). Scored as candidate zero
            before the random loop: if it still explains the ground, the
            adaptive iteration bound collapses to a handful of iterations;
            if the scene changed, it loses the support comparison and the
            full random search runs as before. None or an empty array = no
            warm start.

    Returns:
        np.where-style index of inlier points in the full input cloud,
        and the plane coefficients [bias, n_x, n_y, ...] (last component
        is always positive — flipped at the end if necessary).
    """
    if not threshold:
        threshold = np.median(np.abs(data[:, -1] - np.median(data[:, -1])))
    A = data
    # Hoist the per-iteration loop-invariant slices into contiguous
    # arrays. `A[:, :-1]` is a strided (Fortran-flavoured) view of a
    # C-contiguous A; passing it directly to `.dot(...)` inside Numba
    # falls back to a slow path with a NumbaPerformanceWarning.
    # At 174 k pts/scan the slow path costs ~30-60 ms, enough to push
    # Cone_Detection CPU-bound. Materialising the slice once removes
    # the per-iteration overhead.
    A_xy_full = np.ascontiguousarray(A[:, :-1])
    A_z_full = np.ascontiguousarray(A[:, -1])
    data_size = len(data)
    # Subsample for the iteration loop (#247). RANSAC's consensus
    # probability depends on the inlier *ratio*, not absolute count —
    # running max_iter iterations against 5 000 random points finds
    # the same ground plane as the full ~95 k LiDAR cloud at ~19× the
    # per-iteration cost. Each iteration's `A_xy.dot(...)` is BLAS-
    # parallel and on the full cloud was sustaining 8-10 cores at 90%.
    # The final inlier set is computed against the FULL cloud in the
    # return statement so callers see identical outlier indices.
    n_sub = int(iter_subsample_max)
    if n_sub <= 0:
        n_sub = data_size
    if data_size > n_sub:
        sub_idx = np.random.choice(data_size, n_sub, replace=False)
        A_xy = np.ascontiguousarray(A_xy_full[sub_idx])
        A_z = np.ascontiguousarray(A_z_full[sub_idx])
        iter_data_size = n_sub
    else:
        A_xy = A_xy_full
        A_z = A_z_full
        iter_data_size = data_size
    k = float(max_iter)
    support = 0
    iters = 0

    # Sentinel z-up plane so def_coefs is never None even on a
    # degenerate cloud where no random sample beats support=0. The
    # previous version initialised def_coefs to None and bailed at
    # iters>50 to avoid hanging — but if the loop exited that way, the
    # subsequent `def_coefs[-1] < 0` check crashed.
    def_coefs = np.array([0.0, 0.0, 0.0, 1.0])

    # Warm start: score the caller-provided plane exactly like a random
    # candidate (same subsample, same threshold, same -m correction) so the
    # support comparison against loop candidates stays fair.
    if (
        initial_coefs is not None
        and initial_coefs.shape[0] == data.shape[1]
        and abs(initial_coefs[-1]) > 1e-12
    ):
        warm = initial_coefs.astype(np.float64)
        support_aux = _support_on_slope_form(A_xy, A_z, warm, threshold, m)
        if support_aux > support:
            support = support_aux
            def_coefs = warm.copy()
            k = _iteration_budget(k, support, iter_data_size, prob, m)

    points_ind = np.empty(m, dtype=np.int64)
    while iters < k and iters < max_iter:
        # Sample m anchors from the FULL cloud — the iteration subsample
        # is only used to score consensus cheaply, not to constrain
        # which points can define a candidate plane. Rejection-sampled
        # randints, NOT np.random.choice(..., replace=False): numba
        # implements the latter as a full O(n) permutation per call, which
        # dominated the whole fit on low-inlier-ratio (tire-wall) scans
        # where the iteration budget pins at max_iter (~400 ms -> ~4 ms).
        for jj in range(m):
            while True:
                cand = np.random.randint(0, data_size)
                dup = False
                for kk in range(jj):
                    if points_ind[kk] == cand:
                        dup = True
                        break
                if not dup:
                    points_ind[jj] = cand
                    break
        points = A[points_ind]
        # Build the candidate plane via cross product (numba doesn't
        # JIT np.linalg.svd cleanly).
        vectors = points[1:][:, 1:] - points[0][1:]
        if m > 2:
            normal_vector = np.cross(vectors[0], vectors[1])
        else:
            normal_vector = np.array([-1 * vectors[0][1], vectors[0][0]])
        normal_vector /= np.linalg.norm(normal_vector)
        bias = -np.dot(normal_vector, points[0][1:])
        coefs = np.array([bias] + list(normal_vector))

        support_aux = _support_on_slope_form(A_xy, A_z, coefs, threshold, m)
        if support_aux > support:
            support = support_aux
            def_coefs = coefs
            k = _iteration_budget(k, support, iter_data_size, prob, m)
        iters += 1
    if def_coefs[-1] < 0:
        def_coefs = -1 * def_coefs
    # Final inlier mask over the FULL cloud — callers expect indices
    # into `data`, not the iteration subsample.
    return (
        np.where(
            np.abs(A_xy_full.dot(def_coefs[:-1] / (-1 * def_coefs[-1])) - A_z_full)
            < threshold
        ),
        def_coefs,
    )


@numba.njit(cache=True)
def ransac_ground_mask(
    data,
    prob,
    threshold,
    max_iter,
    iter_subsample_max,
    initial_coefs,
):
    """``ransac2`` for the 3-D ground plane, without its temporaries.

    Same algorithm and the same plane test (vertical distance to the plane below
    ``threshold``, warm start scored as candidate zero, adaptive iteration bound),
    restructured around what dominated its run time on a 100-170 k point scan:

    * reads the raw ``(N, 3)`` cloud (float32 as decoded, any layout) instead of a
      float64 bias-augmented copy, and never materialises ``A[:, :-1]`` / ``A[:, -1]``;
    * the scoring subsample is ``iter_subsample_max`` independent random draws
      (O(n_sub)); ``np.random.choice(..., replace=False)`` is a full O(N)
      permutation in Numba. Scoring estimates an inlier *ratio*, so draws with
      replacement serve as well;
    * candidates are scored in a scalar loop over the subsample: no per-iteration
      ``dot`` result array and no BLAS threads waking up for 5000 points;
    * the final pass writes the outlier mask directly (no index array to invert).

    ``threshold`` must be positive (``ransac2`` keeps the MAD fallback).

    Returns ``(non_ground_mask, plane coefficients [bias, n_x, n_y, n_z])`` with
    ``n_z >= 0``. Non-finite points are outliers.
    """
    n = data.shape[0]
    n_sub = int(iter_subsample_max)
    if n_sub <= 0 or n_sub > n:
        n_sub = n
    sx = np.empty(n_sub)
    sy = np.empty(n_sub)
    sz = np.empty(n_sub)
    if n_sub < n:
        for s in range(n_sub):
            i = np.random.randint(0, n)
            sx[s] = data[i, 0]
            sy[s] = data[i, 1]
            sz[s] = data[i, 2]
    else:
        for s in range(n):
            sx[s] = data[s, 0]
            sy[s] = data[s, 1]
            sz[s] = data[s, 2]

    m = 3
    k = float(max_iter)
    support = 0
    best = np.array([0.0, 0.0, 0.0, 1.0])

    if initial_coefs.shape[0] == 4 and abs(initial_coefs[3]) > 1e-12:
        c0 = float(initial_coefs[0])
        c1 = float(initial_coefs[1])
        c2 = float(initial_coefs[2])
        c3 = float(initial_coefs[3])
        cnt = _support_3d(sx, sy, sz, c0, c1, c2, c3, threshold, m)
        if cnt > support:
            support = cnt
            best[0] = c0
            best[1] = c1
            best[2] = c2
            best[3] = c3
            k = _iteration_budget(k, support, n_sub, prob, m)

    iters = 0
    while iters < k and iters < max_iter:
        iters += 1
        # Anchors come from the FULL cloud; the subsample only scores them.
        i0 = np.random.randint(0, n)
        i1 = np.random.randint(0, n)
        while i1 == i0:
            i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)
        while i2 == i0 or i2 == i1:
            i2 = np.random.randint(0, n)
        px = np.float64(data[i0, 0])
        py = np.float64(data[i0, 1])
        pz = np.float64(data[i0, 2])
        ax = np.float64(data[i1, 0]) - px
        ay = np.float64(data[i1, 1]) - py
        az = np.float64(data[i1, 2]) - pz
        bx = np.float64(data[i2, 0]) - px
        by = np.float64(data[i2, 1]) - py
        bz = np.float64(data[i2, 2]) - pz
        nx = ay * bz - az * by
        ny = az * bx - ax * bz
        nz = ax * by - ay * bx
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        # Collinear / duplicate / non-finite anchors, or a vertical plane (no
        # z = f(x, y) form): no support, like the NaN / inf scores they used to get.
        if not (norm > 0.0) or not math.isfinite(norm) or nz == 0.0:
            continue
        nx /= norm
        ny /= norm
        nz /= norm
        c0 = -(nx * px + ny * py + nz * pz)
        cnt = _support_3d(sx, sy, sz, c0, nx, ny, nz, threshold, m)
        if cnt > support:
            support = cnt
            best[0] = c0
            best[1] = nx
            best[2] = ny
            best[3] = nz
            k = _iteration_budget(k, support, n_sub, prob, m)

    if best[3] < 0:
        best = -1 * best
    c0 = best[0]
    c1 = best[1]
    c2 = best[2]
    c3 = best[3]
    lim = threshold * abs(c3)
    outlier = np.empty(n, np.bool_)
    for i in range(n):
        outlier[i] = not (abs(c0 + c1 * data[i, 0] + c2 * data[i, 1] + c3 * data[i, 2]) < lim)
    return outlier, best
