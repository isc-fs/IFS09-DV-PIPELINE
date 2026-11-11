"""Per-scan CPU time and garbage-collector pauses, for the SLAM_PROF breakdown.

Under CPU contention on the RK3588, SLAM shows rare isolated spikes: one scan
per run at 127-369 ms, a different scan each run, while p99 stays at about
28 ms. Wall-clock stage marks cannot tell what a spike is. Two counters can:

* ``cpu_ms``: this thread's CPU time over the scan. ``total - cpu`` is time
  the thread was runnable or blocked but not running (preempted, waiting on a
  lock). A large gap with normal CPU means scheduling, not work.
* ``gc_ms`` / ``gc2``: time inside the cyclic garbage collector and the number
  of full (generation-2) collections during the scan. A full collection walks
  every tracked Python object in the process, so it grows with the map and the
  history buffers, and it fires on allocation counts, not on scans.
"""
from __future__ import annotations

import gc
import time


class ScanClock:
    """Accumulates GC pauses process-wide; ``start()``/``read()`` bracket a scan."""

    def __init__(self) -> None:
        self._gc_t0 = 0.0
        self._gc_ms = 0.0
        self._gc2 = 0
        self._cpu0 = 0.0
        gc.callbacks.append(self._on_gc)

    def close(self) -> None:
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)

    def _on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._gc_t0 = time.perf_counter()
            return
        self._gc_ms += (time.perf_counter() - self._gc_t0) * 1e3
        if info.get("generation") == 2:
            self._gc2 += 1

    def start(self) -> None:
        self._gc_ms = 0.0
        self._gc2 = 0
        self._cpu0 = time.thread_time()

    def read(self) -> dict:
        return {
            "cpu_ms": (time.thread_time() - self._cpu0) * 1e3,
            "gc_ms": self._gc_ms,
            "gc2": float(self._gc2),
        }
