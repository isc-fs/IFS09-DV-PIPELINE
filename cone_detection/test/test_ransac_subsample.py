"""Floyd's sampler that replaces np.random.choice in both RANSAC paths."""
import numpy as np

from cone_detection.ransac import sample_without_replacement, seed_numba_rng


def test_distinct_and_in_range():
    seed_numba_rng(1)
    for n, k in ((10, 10), (10, 1), (171_000, 5000), (5001, 5000)):
        idx = sample_without_replacement(n, k)
        assert idx.shape == (k,)
        assert idx.min() >= 0 and idx.max() < n
        assert len(np.unique(idx)) == k


def test_deterministic_for_a_seed():
    seed_numba_rng(42)
    a = sample_without_replacement(100_000, 5000)
    seed_numba_rng(42)
    b = sample_without_replacement(100_000, 5000)
    np.testing.assert_array_equal(a, b)


def test_uniform():
    # Each index of 50 should be picked k/n = 20 % of the time.
    seed_numba_rng(7)
    n, k, trials = 50, 10, 20_000
    counts = np.zeros(n)
    for _ in range(trials):
        counts[sample_without_replacement(n, k)] += 1
    freq = counts / trials
    assert np.all(np.abs(freq - k / n) < 0.015), freq
