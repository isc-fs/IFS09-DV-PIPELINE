"""ScanClock: per-scan CPU time and GC pauses for SLAM_PROF."""
import gc

from cone_slam.scan_prof import ScanClock


def test_counts_full_collections_and_cpu():
    clock = ScanClock()
    try:
        clock.start()
        junk = [[i] for i in range(200_000)]
        gc.collect()  # a full (generation-2) collection
        del junk
        r = clock.read()
        assert r["gc2"] >= 1
        assert r["gc_ms"] > 0.0
        assert r["cpu_ms"] > 0.0
        clock.start()
        assert clock.read()["gc2"] == 0
    finally:
        clock.close()
    assert clock._on_gc not in gc.callbacks
