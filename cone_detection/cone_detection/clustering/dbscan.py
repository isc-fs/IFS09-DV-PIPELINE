"""Clustering stage: above-ground points -> DBSCAN labels (-1 = noise).

``ConeDetectionConfig.cluster_backend`` picks the implementation: ``sklearn``,
``grid`` (:func:`grid_dbscan`, exact for ``min_samples == 2``) or ``pypi`` (the
``dbscan`` package, falling back to grid, then sklearn). :func:`cluster_points` is
the stage entry point.
"""

from __future__ import annotations

import logging
import math
import os

import numba
import numpy as np
from sklearn.cluster import DBSCAN

from cone_detection.config import ConeDetectionConfig
from cone_detection.instrumentation import StageTimings

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
    side = eps / _SQRT3 * (1.0 - 1e-9)
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
    # Key deltas to the 62 "forward" neighbour cubes within +-2, the 13 that touch
    # this cube first. Touching cubes are merged for every cube before any distant
    # pair is looked at (two sweeps): by then most distant pairs are already in one
    # component and skip their point-by-point test, which is the expensive part on
    # dense structure. Components do not depend on the merge order.
    deltas = np.empty(62, dtype=np.int64)
    nd = 0
    for ring in range(2):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                for dz in range(-2, 3):
                    delta = (dx * dim[1] + dy) * dim[2] + dz
                    touching = max(abs(dx), abs(dy), abs(dz)) <= 1
                    if delta > 0 and touching == (ring == 0):
                        deltas[nd] = delta
                        nd += 1
        if ring == 0:
            n_touch = nd
    parent = np.arange(m)
    has_nbr = np.zeros(m, dtype=np.bool_)
    eps2 = eps * eps
    # ckeys is sorted, so for a fixed delta the wanted key grows with ``a``: one
    # cursor per delta walks ckeys once per sweep instead of a binary search per
    # (cube, delta).
    cursor = np.zeros(62, dtype=np.int64)
    for ring in range(2):
        t_lo = 0 if ring == 0 else n_touch
        t_hi = n_touch if ring == 0 else nd
        for a in range(m):
            for t in range(t_lo, t_hi):
                want = ckeys[a] + deltas[t]
                b = cursor[t]
                while b < m and ckeys[b] < want:
                    b += 1
                cursor[t] = b
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


def _xyz_f64(points: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(points[:, :3], dtype=np.float64)


def grid_dbscan(points: np.ndarray, eps: float) -> np.ndarray:
    """Labels equal (as a partition) to ``DBSCAN(eps, min_samples=2).fit_predict``.

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
    pts = _xyz_f64(points)
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
    pts = _xyz_f64(points)
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


def cluster_points(
    points: np.ndarray,
    cfg: ConeDetectionConfig,
    timings: StageTimings,
    clustering_class: type = DBSCAN,
) -> np.ndarray:
    """DBSCAN labels of ``points`` with the configured backend (empty in, empty out).

    ``clustering_class`` is any sklearn-style class (``fit_predict``); passing one
    other than sklearn's ``DBSCAN`` always uses it instead of the fast backends.
    """
    if len(points) == 0:
        timings.record("dbscan_ms", 0.0)
        return np.array([])
    with timings.time("dbscan_ms"):
        labels = None
        if cfg.cluster_backend != "sklearn" and clustering_class is DBSCAN:
            labels = fast_dbscan(points, cfg.dbscan_eps, cfg.dbscan_min_samples, cfg.cluster_backend)
        if labels is None:
            labels = clustering_class(
                eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples
            ).fit_predict(points)
    return labels
