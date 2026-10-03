"""grid_dbscan is the same partition as sklearn DBSCAN(eps, min_samples=2)."""

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from cone_detection.clustering import grid_dbscan


def _same_partition(a, b):
    assert np.array_equal(a == -1, b == -1), "noise sets differ"
    m = a != -1
    pairs = set(zip(a[m].tolist(), b[m].tolist()))
    # a bijection between cluster ids
    assert len({p[0] for p in pairs}) == len(pairs) == len({p[1] for p in pairs})


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("eps", [0.1, 0.3])
def test_matches_sklearn_on_mixed_scenes(seed, eps):
    rng = np.random.default_rng(seed)
    blobs = [rng.normal(c, s, (n, 3)) for c, s, n in
             [((2, 1, 0.1), 0.05, 80), ((6, -2, 0.2), 0.15, 200), ((9, 3, 0.1), 0.3, 150)]]
    sheet = np.c_[rng.uniform(0, 8, 3000), rng.uniform(-3, 3, 3000), rng.normal(0.12, 0.01, 3000)]
    noise = rng.uniform([-5, -5, 0], [15, 5, 1], (300, 3))
    pts = np.vstack(blobs + [sheet, noise])
    _same_partition(grid_dbscan(pts, eps), DBSCAN(eps=eps, min_samples=2).fit_predict(pts))


def test_distance_exactly_eps_is_connected():
    pts = np.array([[0.0, 0, 0], [0.3, 0, 0], [5.0, 5, 5]])
    labels = grid_dbscan(pts, 0.3)
    assert labels[0] == labels[1] != -1 and labels[2] == -1


def test_empty_and_single():
    assert len(grid_dbscan(np.zeros((0, 3)), 0.3)) == 0
    assert grid_dbscan(np.zeros((1, 3)), 0.3).tolist() == [-1]


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    blobs = [rng.normal(c, s, (n, 3)) for c, s, n in
             [((2, 1, 0.1), 0.05, 80), ((6, -2, 0.2), 0.15, 200)]]
    return np.vstack(blobs + [rng.uniform([-5, -5, 0], [15, 5, 1], (300, 3))])


def test_pypi_backend_matches_sklearn_when_installed():
    pytest.importorskip("dbscan")
    from cone_detection.clustering import fast_dbscan

    pts = _scene()
    got = fast_dbscan(pts, 0.3, 2, "pypi")
    _same_partition(got, DBSCAN(eps=0.3, min_samples=2).fit_predict(pts))


def test_pypi_backend_falls_back_to_grid_when_missing(monkeypatch, caplog):
    import builtins

    import cone_detection.clustering as cl

    real_import = builtins.__import__

    def no_dbscan(name, *a, **k):
        if name == "dbscan":
            raise ImportError("No module named 'dbscan'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_dbscan)
    monkeypatch.setattr(cl, "_pypi_dbscan", None)
    monkeypatch.setattr(cl, "_warned", set())
    pts = _scene(1)
    with caplog.at_level("WARNING"):
        got = cl.fast_dbscan(pts, 0.3, 2, "pypi")
        cl.fast_dbscan(pts, 0.3, 2, "pypi")
    _same_partition(got, DBSCAN(eps=0.3, min_samples=2).fit_predict(pts))
    assert sum("falling back" in r.message for r in caplog.records) == 1  # warned once


def test_unsupported_min_samples_defers_to_sklearn():
    from cone_detection.clustering import fast_dbscan

    assert fast_dbscan(_scene(), 0.3, 5, "grid") is None
