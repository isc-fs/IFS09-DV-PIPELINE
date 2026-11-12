#!/usr/bin/env python3
"""Regenerate reference.json from scans.npz (see golden.py).

Runs the production config with both detector backends, requires them to agree,
and writes the result with the commit and library versions it came from.

    OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        python3 cone_detection/test/golden/make_golden.py
"""
import json
import os
import platform
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden  # noqa: E402


def main() -> int:
    if not golden.scans_available():
        print("scans.npz missing or an LFS pointer: run `git lfs pull` first", file=sys.stderr)
        return 1
    scans = golden.load_scans()
    py = golden.run(golden.production_config("python"), scans)
    fu = golden.run(golden.production_config("fused"), scans)
    for k, (a, b) in enumerate(zip(py, fu)):
        if a["counters"] != b["counters"]:
            print(f"scan {k}: counters differ\n python {a['counters']}\n fused  {b['counters']}")
            return 1
        np.testing.assert_allclose(b["plane"], a["plane"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(b["cones"], a["cones"], rtol=0, atol=1e-6)
    import numba

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    meta = {
        "commit": commit,
        "bag": "manual_20260920_154527, every 58th scan",
        "machine": platform.machine(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "config": "bringup/config/params.yaml over ConeDetectionConfig()",
    }
    with open(golden.REFERENCE, "w") as fh:
        json.dump({"meta": meta, "scans": py}, fh, indent=1)
        fh.write("\n")
    n = sum(len(s["cones"]) for s in py)
    print(f"wrote {golden.REFERENCE}: {len(py)} scans, {n} cones, python == fused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
