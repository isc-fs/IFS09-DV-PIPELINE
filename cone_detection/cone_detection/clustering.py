"""Exact, fast replacement for sklearn ``DBSCAN(eps, min_samples=2)`` on 3-D points.

With ``min_samples=2`` (counting the point itself) every point that has any other
point within ``eps`` is a core point, so there are no border points: the clusters
are exactly the connected components of the "within eps" graph and isolated points
are noise (-1). That needs no neighbour lists, which is what makes sklearn's DBSCAN
quadratic in point density.

The kernel bins points into cubes whose diagonal is <= eps, so any two points in
one cube are within eps and a cube with 2+ points is already a cluster. Only
cube pairs within reach (+-2 cubes) are tested, point by point, and a pair of cubes
stops at its first connecting point pair. Union-find merges the cubes.

Same partition as sklearn, not the same label numbers (sklearn numbers clusters in
point order, this numbers them in cube order); downstream only uses ``np.unique``.
"""

from __future__ import annotations

import logging
import math
import os

import numba
import numpy as np

_SQRT3 = math.sqrt(3.0)


@numba.njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@numba.njit(cache=True)
def _grid_components(pts, eps):
    n = pts.shape[0]
    side = eps / 1.7320508075688772 * (1.0 - 1e-9)
    cell = np.empty((n, 3), dtype=np.int64)
    for i in range(n):
        for d in range(3):
            cell[i, d] = np.floor(pts[i, d] / side)
    lo = np.empty(3, dtype=np.int64)
    dim = np.empty(3, dtype=np.int64)
    for d in range(3):
        lo[d] = cell[:, d].min()
        dim[d] = cell[:, d].max() - lo[d] + 5  # 2 cubes of padding each side
    keys = np.empty(n, dtype=np.int64)
    for i in range(n):
        keys[i] = (
            ((cell[i, 0] - lo[0] + 2) * dim[1] + (cell[i, 1] - lo[1] + 2)) * dim[2]
            + (cell[i, 2] - lo[2] + 2)
        )
    order = np.argsort(keys)
    # unique cubes: start offset of each, in sorted order
    m = 0
    starts = np.empty(n + 1, dtype=np.int64)
    ckeys = np.empty(n, dtype=np.int64)
    for k in range(n):
        if k == 0 or keys[order[k]] != keys[order[k - 1]]:
            starts[m] = k
            ckeys[m] = keys[order[k]]
            m += 1
    starts[m] = n
    # key deltas to the 62 "forward" neighbour cubes within +-2
    deltas = np.empty(124, dtype=np.int64)
    nd = 0
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                delta = (dx * dim[1] + dy) * dim[2] + dz
                if delta > 0:
                    deltas[nd] = delta
                    nd += 1
    parent = np.arange(m)
    has_nbr = np.zeros(m, dtype=np.bool_)
    eps2 = eps * eps
    for a in range(m):
        for t in range(nd):
            want = ckeys[a] + deltas[t]
            b = np.searchsorted(ckeys[:m], want)
            if b >= m or ckeys[b] != want:
                continue
            ra = _find(parent, a)
            rb = _find(parent, b)
            if ra == rb:
                continue
            found = False
            for ia in range(starts[a], starts[a + 1]):
                p = order[ia]
                for ib in range(starts[b], starts[b + 1]):
                    q = order[ib]
                    dx = pts[p, 0] - pts[q, 0]
                    dy = pts[p, 1] - pts[q, 1]
                    dz = pts[p, 2] - pts[q, 2]
                    if dx * dx + dy * dy + dz * dz <= eps2:
                        found = True
                        break
                if found:
                    break
            if found:
                parent[rb] = ra
                has_nbr[a] = True
                has_nbr[b] = True
    labels = np.full(n, -1, dtype=np.int64)
    cid = np.full(m, -1, dtype=np.int64)
    nxt = 0
    for a in range(m):
        if starts[a + 1] - starts[a] >= 2 or has_nbr[a]:
            r = _find(parent, a)
            if cid[r] < 0:
                cid[r] = nxt
                nxt += 1
            for k in range(starts[a], starts[a + 1]):
                labels[order[k]] = cid[r]
    return labels


def grid_dbscan(points: np.ndarray, eps: float) -> np.ndarray:
    """Labels equal (as a partition) to ``DBSCAN(eps, min_samples=2).fit_predict``."""
    pts = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)
    return _grid_components(pts, float(eps))


_log = logging.getLogger(__name__)
_warned: set[str] = set()
_pypi_dbscan = None  # the `dbscan` package's DBSCAN, imported on first use


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        _log.warning(msg)


def pypi_dbscan(points: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    """Labels from the PyPI ``dbscan`` package (parallel grid DBSCAN, C++).

    The package starts one worker thread per hardware thread unless told
    otherwise; on a loaded or core-limited machine that oversubscribes badly (about
    20x slower on one core). ``PARLAY_NUM_THREADS`` is read when the pool starts, so
    it is defaulted to 1 here before the package is first used; set it in the
    environment to override. Not guaranteed bit-identical to sklearn on points at
    almost exactly ``eps``.

    Raises ImportError if the package is not installed.
    """
    global _pypi_dbscan
    if _pypi_dbscan is None:
        os.environ.setdefault("PARLAY_NUM_THREADS", "1")
        from dbscan import DBSCAN as _impl  # noqa: PLC0415

        _pypi_dbscan = _impl
    pts = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)
    labels, _core = _pypi_dbscan(pts, eps=float(eps), min_samples=int(min_samples))
    return np.asarray(labels, dtype=np.int64)


def fast_dbscan(
    points: np.ndarray, eps: float, min_samples: int, backend: str
) -> np.ndarray | None:
    """Cluster with ``backend`` ("pypi" or "grid"); None means "use sklearn".

    ``pypi`` falls back to ``grid`` (then to sklearn) when the package is missing or
    fails, with one warning per reason, so a missing optional dependency never takes
    perception down. ``grid`` only handles ``min_samples == 2``.
    """
    if backend == "pypi":
        try:
            return pypi_dbscan(points, eps, min_samples)
        except Exception as exc:  # ImportError, or anything the C++ side raises
            _warn_once(
                "pypi",
                f"cluster_backend=pypi unavailable ({exc!r}); falling back to grid/sklearn",
            )
            backend = "grid"
    if backend == "grid" and min_samples == 2:
        return grid_dbscan(points, eps)
    return None
