#!/usr/bin/env python3
"""Time cone detection per scan and per stage for one detector_backend.

Replays the LiDAR scans of an MCAP bag (or the golden scans) through the node's
decode and the production detector (bringup/config/params.yaml), the way
cone_detection_node does, and writes per-scan timings plus a summary as JSON.
Run once per backend and compare:

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
    taskset -c 4-7 python3 tools/detect_backend_bench.py --backend fused \
        --bag manual_20260920_154527_0.mcap --contention 3 --out fused.json
    taskset -c 4-7 python3 tools/detect_backend_bench.py --backend cpp \
        --bag manual_20260920_154527_0.mcap --contention 3 --out cpp.json

Without --bag it uses cone_detection/test/golden/scans.npz (no ROS needed; decode
is then not timed). The first --warmup scans are run but not counted.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "cone_detection"))
sys.path.insert(0, str(REPO / "cone_detection" / "test" / "golden"))

STAGES = ("crop_ms", "ransac_ms", "rotate_ms", "column_veto_ms", "guard_ms", "dbscan_ms", "fit_ms")


def _spin():
    while True:
        pass


class Contention:
    """N busy-loop processes competing for the same cores (as dv_fitbench)."""

    def __init__(self, n: int):
        self.n = n
        self.procs: list[mp.Process] = []

    def __enter__(self):
        for _ in range(self.n):
            p = mp.Process(target=_spin, daemon=True)
            p.start()
            self.procs.append(p)
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()


def bag_scans(path: str, limit: int):
    """Yield (deserialize_ms, decode_ms, xyz) per /lidar_points message."""
    from mcap.reader import make_reader
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2

    from cone_detection.cone_detection_node import ConeDetectionNode

    yield_decode = bag_scans.decode  # set by main
    with open(path, "rb") as fh:
        for k, (_s, _c, message) in enumerate(
            make_reader(fh).iter_messages(topics=["/lidar_points"], log_time_order=True)
        ):
            if limit and k >= limit:
                return
            t0 = time.perf_counter()
            cloud = deserialize_message(message.data, PointCloud2)
            t1 = time.perf_counter()
            xyz = ConeDetectionNode.pointcloud2_to_xyz(cloud, yield_decode)
            t2 = time.perf_counter()
            yield (t1 - t0) * 1e3, (t2 - t1) * 1e3, xyz


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", required=True, choices=("python", "fused", "cpp"))
    ap.add_argument("--bag", default=None, help="MCAP bag; default: the golden scans")
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="passes over the golden scans")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--contention", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import golden

    from cone_detection.cone_detection import RealtimeConeDetector, warmup_numba_functions

    cfg = golden.production_config(args.backend)
    t = time.perf_counter()
    warmup_numba_functions(cfg)
    bringup_ms = (time.perf_counter() - t) * 1e3
    det = RealtimeConeDetector(cfg)

    if args.bag:
        decode = None
        if args.backend == "cpp":
            from cone_detection_core import decode_xyz as decode
        bag_scans.decode = decode
        source = bag_scans(args.bag, args.max_scans)
    else:
        scans = golden.load_scans() * args.repeat
        source = ((0.0, 0.0, s) for s in scans)

    rows = []
    with Contention(args.contention):
        for deser_ms, decode_ms, xyz in source:
            st: dict = {}
            t0 = time.perf_counter()
            cones = det.detect(xyz, stage_timings=st)
            detect_ms = (time.perf_counter() - t0) * 1e3
            rows.append(
                dict(
                    n_points=len(xyz),
                    n_cones=len(cones),
                    deser_ms=deser_ms,
                    decode_ms=decode_ms,
                    detect_ms=detect_ms,
                    stages={k: float(st.get(k, 0.0)) for k in STAGES},
                )
            )
    used = rows[args.warmup:]

    def summary(vals):
        return {"p50": pct(vals, 50), "p95": pct(vals, 95), "p99": pct(vals, 99), "max": max(vals, default=0.0)}

    summ = {
        "detect_ms": summary([r["detect_ms"] for r in used]),
        "decode_ms": summary([r["decode_ms"] for r in used]),
        "deser_ms": summary([r["deser_ms"] for r in used]),
        "decode_detect_ms": summary([r["decode_ms"] + r["detect_ms"] for r in used]),
        "stages": {k: summary([r["stages"][k] for r in used]) for k in STAGES},
    }
    meta = {
        "backend": args.backend,
        "source": args.bag or "golden scans",
        "scans": len(used),
        "warmup": args.warmup,
        "contention": args.contention,
        "bringup_ms": bringup_ms,
        "machine": platform.machine(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    print(json.dumps({"meta": meta, "summary": summ}, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "summary": summ, "rows": rows}, fh)


if __name__ == "__main__":
    main()
