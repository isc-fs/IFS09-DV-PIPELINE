"""Keep full garbage collections off the hot path (PERF-02).

A full (generation-2) collection of Python's cyclic GC walks every tracked
object in the process. After bring-up that heap is mostly static: imported
modules, Numba dispatchers and compiled code, ROS entities, warm-up buffers.
None of it ever becomes garbage, but every full collection walks it again.

Measured on the RK3588 (dv_fitbench, --contention 3, 1749 scans): the 4 worst
SLAM scans were exactly the 4 that ran a gen-2 collection, each taking about
128 ms of the 143-149 ms scan (about 304 ms when bring-up had JIT-compiled,
which breaks PERF-02's 300 ms max). Planning showed the same pattern (max 217
ms against a p99 of 16 ms).

``freeze_heap()`` runs one full collection and then ``gc.freeze()``s every
survivor into the permanent generation, which later collections skip. Call it
once the node is fully set up and before the data plane starts. Collections
then only walk objects created while running.
"""
from __future__ import annotations

import gc
import time


def freeze_heap() -> tuple[int, float]:
    """Collect, then freeze all surviving objects. Returns (objects now frozen, ms)."""
    t0 = time.perf_counter()
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count(), (time.perf_counter() - t0) * 1e3
