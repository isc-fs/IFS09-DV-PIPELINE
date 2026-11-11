"""freeze_heap(): startup objects leave the reach of full GC passes (PERF-02)."""
import gc
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "gc_tuning",
    os.path.join(os.path.dirname(__file__), os.pardir, "node_base", "gc_tuning.py"),
)
gc_tuning = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gc_tuning)


def test_freeze_moves_survivors_out_of_full_collections():
    keep = [[i] for i in range(50_000)]  # long-lived startup state
    try:
        frozen, ms = gc_tuning.freeze_heap()
        assert frozen >= len(keep)
        assert ms >= 0.0
        # A full collection now only sees objects created after the freeze.
        assert sum(len(g) for g in (gc.get_objects(i) for i in range(3))) < frozen
    finally:
        gc.unfreeze()
