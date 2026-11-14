"""``mrf``: binary ground/object MRF over height-grid cells, solved exactly by graph cut.

See ``docs/mrf_ground_removal.md``.
"""

from __future__ import annotations

import numba
import numpy as np

from .height_grid import _cell_stats, _fill_nearest, _grey_opening
from .scan_geometry import _level_cells
from .plane_fitting import _refit_on_ground
from .registry import _optional, register


# MRF over the height grid, solved exactly with a graph cut. The unary cost of
# calling a cell ground grows with its height spread and with how far its lowest
# return sits above the local floor; calling it an object costs 1. Neighbouring
# cells at similar heights pay _MRF_SMOOTH to disagree, so the label only changes
# across a height step.
_MRF_CELL_M = 0.30


_MRF_SPAN_M = 0.10          # height spread that costs as much as calling the cell an object


_MRF_STEP_M = 0.10          # height above the local floor that costs the same


_MRF_MAX_COST = 10.0


_MRF_FLOOR_WINDOW = 9       # cells; the local floor ignores objects narrower than this


_MRF_SMOOTH = 0.5


_MRF_SMOOTH_DH_M = 0.05


_MRF_GROUND_BAND_M = 0.10   # points this close to a ground cell's floor are ground


# Graph-cut backend: "scipy" (scipy.sparse.csgraph.maximum_flow, BSD, always
# available) or "pymaxflow" (Boykov-Kolmogorov via the optional, GPL PyMaxflow).
# Both solve the same cut exactly.
_MRF_SOLVER = "scipy"


# scipy's max-flow needs int32 capacities. Costs are scaled by up to this factor;
# rounding then shifts the cut's energy by < 1e-7 of its total (ties aside, the
# same cut as PyMaxflow), and the scale is lowered so that the total flow fits.
_MRF_INT_SCALE = 100_000


# Which graph the scipy solver cuts:
#   "reduced"  occupied cells only (an empty cell has no points and no unary cost; it
#              only chained its neighbours together), minus the cells proven below.
#              Not bit-identical to "full": a few labels at the edge of an empty gap
#              can differ (about 0.4 of ~3300 occupied cells per scan on the sim bag).
#   "exact"    every cell, minus the proven ones. Same labels as "full".
#   "full"     every cell, no reduction (the original formulation; reference).
# PyMaxflow always cuts the full grid.
_MRF_GRAPH = "reduced"


# A cell is fixed only when its proof holds with this much slack (the scipy solver
# quantizes capacities to 1e-5, so a tie closer than that is not worth deciding).
_MRF_PERSIST_MARGIN = 1e-4


@register("mrf")
def ground_mrf(data, cfg):
    """Binary ground/object MRF on cells: unary from height and spread, Potts smoothness."""
    thr = cfg.ground_threshold_m
    cell, nx, ny, h, coarse = _level_cells(data, thr, _MRF_CELL_M)
    _, mn, mx, _, _ = _cell_stats(cell, h, nx * ny)
    has = (mn != np.inf).reshape(nx, ny)
    mn0 = np.where(has, mn.reshape(nx, ny), 0.0)
    span = np.where(has, (mx - mn).reshape(nx, ny), 0.0)
    mn_g = _fill_nearest(mn0, has)
    ref = _grey_opening(mn_g, _MRF_FLOOR_WINDOW)
    # Unary costs, and neighbour weights: cells at similar heights want the same label.
    cost_ground, cost_object, w_down, w_right = _mrf_costs(
        span, mn_g, ref, has,
        _MRF_SPAN_M, _MRF_STEP_M, _MRF_MAX_COST, _MRF_SMOOTH, _MRF_SMOOTH_DH_M,
    )
    if _MRF_GRAPH not in ("reduced", "exact", "full"):
        raise ValueError(f"unknown _MRF_GRAPH {_MRF_GRAPH!r}")
    if _MRF_GRAPH == "full" or _MRF_SOLVER != "scipy":
        is_ground = _grid_mincut(cost_ground, cost_object, w_down, w_right) & has
    else:
        active = has if _MRF_GRAPH == "reduced" else np.ones_like(has)
        is_ground = _grid_mincut_reduced(cost_ground, cost_object, w_down, w_right, active) & has
    floor = _fill_nearest(mn_g, is_ground).ravel()
    non_ground = _mrf_point_non_ground(h, cell, floor, is_ground.ravel(), _MRF_GROUND_BAND_M, thr)
    return non_ground, _refit_on_ground(data, non_ground, coarse)


@numba.njit(cache=True)
def _mrf_costs(span, mn_g, ref, has, span_m, step_m, max_cost, smooth, dh_m):
    """Unary costs and 4-neighbour weights of the MRF in one pass.

    ``w_down[i, j]`` couples (i, j)-(i+1, j), ``w_right[i, j]`` couples
    (i, j)-(i, j+1); the last row / column pairs with nothing (dh = 0).
    """
    nx, ny = span.shape
    cost_ground = np.empty((nx, ny))
    cost_object = np.empty((nx, ny))
    w_down = np.empty((nx, ny))
    w_right = np.empty((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if has[i, j]:
                a = span[i, j] / span_m
                b = (mn_g[i, j] - ref[i, j]) / step_m
                cost_ground[i, j] = min(a * a + b * b, max_cost)
                cost_object[i, j] = 1.0
            else:
                cost_ground[i, j] = 0.0
                cost_object[i, j] = 0.0
            dd = abs(mn_g[i + 1, j] - mn_g[i, j]) / dh_m if i < nx - 1 else 0.0
            dr = abs(mn_g[i, j + 1] - mn_g[i, j]) / dh_m if j < ny - 1 else 0.0
            w_down[i, j] = smooth * np.exp(-(dd * dd))
            w_right[i, j] = smooth * np.exp(-(dr * dr))
    return cost_ground, cost_object, w_down, w_right


@numba.njit(cache=True)
def _mrf_point_non_ground(h, cell, floor, is_ground, band_ground, band_object):
    """``h - floor[cell] >= band``; band depends on the cell's label (one pass)."""
    out = np.empty(h.shape[0], dtype=np.bool_)
    for k in range(h.shape[0]):
        c = cell[k]
        out[k] = h[k] - floor[c] >= (band_ground if is_ground[c] else band_object)
    return out


def _grid_mincut(cost_ground, cost_object, w_down, w_right, solver=None):
    """Exact binary min-cut on a 4-connected grid -> True where a cell is ground.

    ``w_down[i, j]`` couples cell (i, j) with (i + 1, j), ``w_right[i, j]`` with
    (i, j + 1). A cell labelled ground pays ``cost_ground``, otherwise ``cost_object``.
    """
    solver = solver or _MRF_SOLVER
    nx, ny = cost_ground.shape
    if solver == "pymaxflow":
        maxflow = _optional("maxflow", "PyMaxflow")
        g = maxflow.Graph[float]()
        nodes = g.add_grid_nodes((nx, ny))
        g.add_grid_edges(nodes, weights=w_down, structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), symmetric=True)
        g.add_grid_edges(nodes, weights=w_right, structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), symmetric=True)
        # Sink side = ground: a node pays its source capacity for being ground.
        g.add_grid_tedges(nodes, cost_ground, cost_object)
        g.maxflow()
        return g.get_grid_segments(nodes)
    if solver != "scipy":
        raise ValueError(f"unknown MRF solver {solver!r}")

    # Every cell is a graph node, numbered row-major.
    node = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
    is_source = _scipy_cut_source_side(
        cost_ground, cost_object, w_down, w_right, node, nx * ny, float(cost_ground.sum())
    )
    return ~is_source.reshape(nx, ny)


def _scipy_cut_source_side(cg, co, wd, wr, node, m, total_cg):
    """Source side of the exact min cut over the ``m`` cells with ``node >= 0`` (scipy).

    Costs are quantized to int32 (see ``_MRF_INT_SCALE``); ``total_cg`` bounds the
    max flow, which is what keeps it inside int32. Returns a bool per node id.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    scale = min(_MRF_INT_SCALE, (2**31 - 1) / max(total_cg + 1.0, 1.0))
    q = lambda a: np.rint(np.asarray(a, dtype=np.float64) * scale).astype(np.int32)
    indptr, indices, caps = _sub_csr(q(cg), q(co), q(wd), q(wr), node, m)
    cap = csr_matrix((caps, indices, indptr), shape=(m + 2, m + 2))
    cap.has_sorted_indices = True  # built row by row in column order
    flow = maximum_flow(cap, m, m + 1, method="dinic").flow
    flow.sort_indices()  # no-op when scipy already returns it sorted
    # Source side of the cut = nodes still reachable from the source through
    # edges with residual capacity cap(u, v) - flow(u, v) > 0.
    is_source = _residual_reach(
        indptr, indices, caps, flow.indptr, flow.indices, flow.data, m, m + 2
    )
    return is_source[:m]


@numba.njit(cache=True)
def _residual_reach(indptr, indices, cap, f_indptr, f_indices, f_data, src, n_nodes):
    """BFS from ``src`` over edges with ``cap - flow > 0``.

    ``flow`` (scipy's result) holds every capacity edge plus its reverse, both
    with sorted columns, so each row is a two-pointer merge. Reverse-only edges
    lead back to the source or out of the sink (unreachable at max flow), so
    they never change the reachable set.
    """
    seen = np.zeros(n_nodes, dtype=np.bool_)
    queue = np.empty(n_nodes, dtype=np.int64)
    seen[src] = True
    queue[0] = src
    head, tail = 0, 1
    while head < tail:
        u = queue[head]
        head += 1
        p = f_indptr[u]
        p_end = f_indptr[u + 1]
        for e in range(indptr[u], indptr[u + 1]):
            v = indices[e]
            while p < p_end and f_indices[p] < v:
                p += 1
            f = f_data[p] if p < p_end and f_indices[p] == v else 0
            if not seen[v] and cap[e] - f > 0:
                seen[v] = True
                queue[tail] = v
                tail += 1
    return seen


@numba.njit(cache=True)
def _persistent_states(cg, co, wd, wr, active, margin):
    """Cells whose label is the same in every optimal labelling -> ``int8`` grid.

    1 = ground, 0 = object, -1 = undecided, -2 = not in the graph (``active`` False).

    A set G of cells is ground in every optimal labelling when, for each c in G,
    ``cg[c] - co[c] + sum(w[c, n] for neighbours n not in G) < 0``. Proof: take an
    optimal labelling and let T be the cells of G it calls object. Flipping T to
    ground changes the unaries by ``sum(cg - co)`` over T, leaves edges inside T
    alone, removes the cut edges between T and the rest of G (they were cut, now
    both ground), and adds at most ``w`` for each edge from T to a cell outside G.
    So the energy changes by at most ``sum_T (cg - co + sum_{n notin G} w) < 0``:
    a contradiction unless T is empty. The largest such G is found by starting from
    every cell with ``cg < co`` and dropping violators until none is left (dropping
    only ever raises the others' boundary sums, so the result does not depend on
    order). Object cells are the same with the roles swapped. Needs ``w >= 0``.

    On a flat floor almost every occupied cell has ``cg = 0 < co = 1``, so most of
    them qualify and only the cells next to objects or gaps stay in the graph.
    """
    nx, ny = cg.shape
    state = np.full((nx, ny), -2, np.int8)
    for i in range(nx):
        for j in range(ny):
            if active[i, j]:
                state[i, j] = -1
    for side in range(2):
        inset = np.zeros((nx, ny), np.bool_)
        for i in range(nx):
            for j in range(ny):
                if active[i, j]:
                    d = cg[i, j] - co[i, j] if side == 0 else co[i, j] - cg[i, j]
                    inset[i, j] = d < -margin
        changed = True
        while changed:
            changed = False
            for i in range(nx):
                for j in range(ny):
                    if not inset[i, j]:
                        continue
                    d = cg[i, j] - co[i, j] if side == 0 else co[i, j] - cg[i, j]
                    b = 0.0
                    if i > 0 and active[i - 1, j] and not inset[i - 1, j]:
                        b += wd[i - 1, j]
                    if i < nx - 1 and active[i + 1, j] and not inset[i + 1, j]:
                        b += wd[i, j]
                    if j > 0 and active[i, j - 1] and not inset[i, j - 1]:
                        b += wr[i, j - 1]
                    if j < ny - 1 and active[i, j + 1] and not inset[i, j + 1]:
                        b += wr[i, j]
                    if d + b >= -margin:
                        inset[i, j] = False
                        changed = True
        for i in range(nx):
            for j in range(ny):
                if inset[i, j]:
                    state[i, j] = 1 if side == 0 else 0
    return state


@numba.njit(cache=True)
def _reduce_graph(cg, co, wd, wr, state):
    """Compact ids for the undecided cells, and their unaries with fixed neighbours folded in.

    An edge to a fixed-ground neighbour costs ``w`` if the cell is object (added to
    ``co``); to a fixed-object neighbour it costs ``w`` if the cell is ground (added
    to ``cg``). Returns ``(node, m, cg', co', sum of cg' over undecided cells)``.
    """
    nx, ny = cg.shape
    node = np.full((nx, ny), -1, np.int64)
    m = 0
    for i in range(nx):
        for j in range(ny):
            if state[i, j] == -1:
                node[i, j] = m
                m += 1
    acg = cg.copy()
    aco = co.copy()
    total = 0.0
    for i in range(nx):
        for j in range(ny):
            if state[i, j] != -1:
                continue
            for k in range(4):
                if k == 0:
                    ni, nj = i - 1, j
                elif k == 1:
                    ni, nj = i + 1, j
                elif k == 2:
                    ni, nj = i, j - 1
                else:
                    ni, nj = i, j + 1
                if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                    continue
                s = state[ni, nj]
                if s != 0 and s != 1:
                    continue
                if k == 0:
                    w = wd[ni, nj]
                elif k == 1:
                    w = wd[i, j]
                elif k == 2:
                    w = wr[ni, nj]
                else:
                    w = wr[i, j]
                if s == 1:
                    aco[i, j] += w
                else:
                    acg[i, j] += w
            total += acg[i, j]
    return node, m, acg, aco, total


@numba.njit(cache=True)
def _sub_csr(cg, co, wd, wr, node, m):
    """CSR capacity matrix of the grid cut over the ``m`` cells with ``node >= 0``.

    ``node`` maps a cell to its node id (row-major order, -1 = not in the graph);
    ``m`` is the source and ``m + 1`` the sink. Built directly in sorted order:
    a cell row lists its up, left, right, down neighbours (pairwise weight, only
    neighbours in the graph), then the sink (``co``); the source row lists every
    cell (``cg``). Zero capacities are left out.
    """
    nx, ny = cg.shape
    nnz_max = 5 * m + m
    indptr = np.empty(m + 3, dtype=np.int32)
    indices = np.empty(nnz_max, dtype=np.int32)
    data = np.empty(nnz_max, dtype=np.int32)
    e = 0
    for i in range(nx):
        for j in range(ny):
            r = node[i, j]
            if r < 0:
                continue
            indptr[r] = e
            if i > 0 and node[i - 1, j] >= 0 and wd[i - 1, j] > 0:
                indices[e] = node[i - 1, j]
                data[e] = wd[i - 1, j]
                e += 1
            if j > 0 and node[i, j - 1] >= 0 and wr[i, j - 1] > 0:
                indices[e] = node[i, j - 1]
                data[e] = wr[i, j - 1]
                e += 1
            if j < ny - 1 and node[i, j + 1] >= 0 and wr[i, j] > 0:
                indices[e] = node[i, j + 1]
                data[e] = wr[i, j]
                e += 1
            if i < nx - 1 and node[i + 1, j] >= 0 and wd[i, j] > 0:
                indices[e] = node[i + 1, j]
                data[e] = wd[i, j]
                e += 1
            if co[i, j] > 0:
                indices[e] = m + 1
                data[e] = co[i, j]
                e += 1
    indptr[m] = e
    for i in range(nx):
        for j in range(ny):
            r = node[i, j]
            if r >= 0 and cg[i, j] > 0:
                indices[e] = r
                data[e] = cg[i, j]
                e += 1
    indptr[m + 1] = e
    indptr[m + 2] = e
    return indptr, indices[:e], data[:e]


def _grid_mincut_reduced(cost_ground, cost_object, w_down, w_right, active):
    """Min-cut over the ``active`` cells with provably fixed cells taken out (scipy solver).

    Same contract as ``_grid_mincut`` (True where a cell is ground); inactive cells
    come back False. With every cell active the labels are those of the full cut
    (see ``_persistent_states``); with a subset, the graph is that subset's induced
    grid.
    """
    state = _persistent_states(
        cost_ground, cost_object, w_down, w_right, active, _MRF_PERSIST_MARGIN
    )
    node, m, acg, aco, total = _reduce_graph(cost_ground, cost_object, w_down, w_right, state)
    ground = state == 1
    if m == 0:
        return ground
    is_source = _scipy_cut_source_side(acg, aco, w_down, w_right, node, m, total)
    free = state == -1
    ground[free] = ~is_source[node[free]]
    return ground


def warm_mrf_graph() -> None:
    """Compile the graph-cut kernels with an input that leaves undecided cells.

    A tidy warm-up scene (flat floor, a few cones) is fully decided by
    ``_persistent_states``, so the flow kernels would never run and would JIT on
    the first real scan instead (~1 s). This grid has object cells and gaps next to
    ground, which always leaves cells to cut.
    """
    rng = np.random.default_rng(0)
    shape = (24, 30)
    cg = np.zeros(shape)
    co = np.ones(shape)
    cg[8:12, 10:14] = 9.0                      # an object
    cg[rng.random(shape) < 0.15] = 0.6         # ambiguous cells
    empty = rng.random(shape) < 0.25
    cg[empty] = co[empty] = 0.0
    w = np.full(shape, 0.5)
    for active in (~empty, np.ones(shape, bool)):
        _grid_mincut_reduced(cg, co, w, w.copy(), active)
