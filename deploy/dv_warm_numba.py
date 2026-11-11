#!/usr/bin/env python3
"""Bake the Numba JIT cache the autonomy nodes load at bring-up (SW-08).

Called by dv_warm_numba.sh (see there for when and as which user). It warms
every signature a race bring-up compiles, then checks that the result is warm.

* cone_detection: the config cone_detection_node will run. That is the
  strategy's config with params.yaml applied on top, the same merge
  ``ConeDetectionNode._config_from_params`` does. The bake runs it with both
  ``detector_backend`` values and with the grid clusterer as well, so
  flipping either parameter never means a cold boot.
* path_planning: ``warmup_numba_planner`` for every mission type a planner
  strategy uses (trackdrive, acceleration, skidpad), not only trackdrive.
* Then a verification pass in a fresh interpreter repeats the same warm-ups.
  It must take seconds; a cold cache takes 20 s for cone_detection and about
  130 s for the planner on the RK3588. If it is slow, the bake did not stick,
  for example because the cache dir is unwritable or the user is wrong. The
  verdict is printed so the update log shows it.

Exit status: 0 when the cache is warm, 1 when it is not or the bake failed
(the shell wrapper treats both as non-fatal).
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time

# Verification budget per node for a warm cache, seconds. Warm bring-up
# measures about 1-2 s per node on the RK3588; mode_manager allows 120 s.
WARM_BUDGET_S = 30.0


def params_file() -> str | None:
    """bringup's installed params.yaml, else the source tree's."""
    try:
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(get_package_share_directory("bringup"), "config", "params.yaml")
        if os.path.isfile(path):
            return path
    except Exception:  # noqa: BLE001 - no ament index outside a sourced workspace
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, os.pardir, "bringup", "config", "params.yaml")
    return path if os.path.isfile(path) else None


def node_params(node: str, path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return dict((data.get(node) or {}).get("ros__parameters") or {})


def cone_detection_configs(path: str | None = None) -> list:
    """The configs a cone_detection bring-up can compile: params.yaml over the
    strategy config, once per detector backend."""
    from cone_detection.config import ConeDetectionConfig

    base = None
    try:
        from cone_detection.strategies.base_cone_detection import BaseConeDetection

        base = BaseConeDetection.CONE_DETECTION_CONFIG
    except Exception:  # noqa: BLE001 - strategy import pulls ROS; defaults suffice
        pass
    defaults = ConeDetectionConfig()
    fields = {f.name for f in dataclasses.fields(ConeDetectionConfig)}
    params = node_params("cone_detection_node", path if path is not None else params_file())
    changed = {
        k: v for k, v in params.items() if k in fields and v != getattr(defaults, k)
    }
    cfg = dataclasses.replace(base or defaults, **changed)
    # The grid clusterer is Numba, the pypi one is not. Warm grid as well, so
    # a switch to it, or pypi falling back to it, costs no JIT at bring-up.
    return [
        dataclasses.replace(cfg, detector_backend=b, cluster_backend=c)
        for b in dict.fromkeys([cfg.detector_backend, "python", "fused"])
        for c in dict.fromkeys([cfg.cluster_backend, "grid"])
    ]


def planner_missions() -> list:
    from path_planning.planner_strategies import PATH_PLANNING_STRATEGY_MAP

    missions = []
    for cls in PATH_PLANNING_STRATEGY_MAP.values():
        m = cls.__new__(cls).get_mission_type()  # get_mission_type reads no state
        if m not in missions:
            missions.append(m)
    return missions


def warm_all() -> dict[str, float]:
    """Run every warm-up; returns seconds per node."""
    out = {}
    t0 = time.perf_counter()
    from cone_detection.cone_detection import warmup_numba_functions

    for cfg in cone_detection_configs():
        warmup_numba_functions(cfg)
    out["cone_detection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from path_planning.fasttube_adapter import warmup_numba_planner

    for mission in planner_missions():
        warmup_numba_planner(mission)
    out["path_planning"] = time.perf_counter() - t0
    return out


def main(argv: list[str]) -> int:
    if argv[1:] == ["--warm-only"]:
        for node, s in warm_all().items():
            print(f"[warm-numba] {node}: {s:.1f}s", flush=True)
        return 0
    print(f"[warm-numba] NUMBA_CACHE_DIR={os.environ.get('NUMBA_CACHE_DIR', '(unset)')}", flush=True)
    print(f"[warm-numba] params: {params_file()}", flush=True)
    try:
        for node, s in warm_all().items():
            print(f"[warm-numba] bake {node}: {s:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warm-numba] bake FAILED: {exc!r}", flush=True)
        return 1
    # Verify from a fresh interpreter: only the on-disk cache carries over.
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--warm-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    times = {}
    for line in proc.stdout.splitlines():
        if line.startswith("[warm-numba] ") and ": " in line and line.endswith("s"):
            node, val = line[len("[warm-numba] "):].rsplit(": ", 1)
            times[node] = float(val[:-1])
    if proc.returncode != 0 or len(times) != 2:
        print(f"[warm-numba] verify FAILED (rc={proc.returncode}): {proc.stderr[-2000:]}", flush=True)
        return 1
    slow = {n: s for n, s in times.items() if s > WARM_BUDGET_S}
    for node, s in times.items():
        print(f"[warm-numba] verify {node}: {s:.1f}s", flush=True)
    if slow:
        print(
            f"[warm-numba] CACHE NOT WARM: {slow} over {WARM_BUDGET_S:.0f}s. "
            "Bring-up will JIT inside mode_manager's 120 s budget. Check that "
            "NUMBA_CACHE_DIR matches dv-pipeline.service and is writable by isc.",
            flush=True,
        )
        return 1
    print(f"[warm-numba] cache warm (verify {time.perf_counter() - t0:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
