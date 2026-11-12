"""Golden fixtures for cone detection: real car scans and their frozen outputs.

``scans.npz`` (Git LFS) holds 30 LiDAR scans from the DVPC bag
``manual_20260920_154527`` (every 58th of 1,749), as float32 xyz, exactly as
``pointcloud2_to_xyz`` returns them. ``reference.json`` holds what the detector
produced for each scan at the commit named in its ``meta``: cones, the RANSAC
plane and every debug counter.

The detector runs with the production config: ``bringup/config/params.yaml``
over ``ConeDetectionConfig``. RANSAC is seeded per scan from the scan's content
(``ransac_seed_per_scan``), so the outputs are deterministic. One detector
processes the scans in order, as the node does, so the warm-start plane carries
over between scans.

Regenerate with ``python3 cone_detection/test/golden/make_golden.py`` after a
deliberate output change (new RNG sequence, new algorithm). Say why in the
commit.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCANS = os.path.join(HERE, "scans.npz")
REFERENCE = os.path.join(HERE, "reference.json")
PARAMS = os.path.join(HERE, os.pardir, os.pardir, os.pardir, "bringup", "config", "params.yaml")

# Counters that are timings or depend on the run, not on the output.
_SKIP_COUNTERS = ()


def scans_available() -> bool:
    """False when the file is missing or still a Git LFS pointer (no ``git lfs pull``)."""
    if not os.path.isfile(SCANS):
        return False
    with open(SCANS, "rb") as fh:
        return not fh.read(64).startswith(b"version https://git-lfs")


def load_scans() -> list[np.ndarray]:
    d = np.load(SCANS)
    xyz, lens = d["xyz"], d["lens"]
    ends = np.cumsum(lens)
    return [xyz[e - n : e] for n, e in zip(lens, ends)]


def production_config(detector_backend: str | None = None):
    """``params.yaml``'s cone_detection_node parameters over the config defaults."""
    import yaml

    from cone_detection.config import ConeDetectionConfig

    with open(PARAMS) as fh:
        params = (yaml.safe_load(fh) or {})["cone_detection_node"]["ros__parameters"]
    fields = {f.name for f in dataclasses.fields(ConeDetectionConfig)}
    cfg = dataclasses.replace(
        ConeDetectionConfig(), **{k: v for k, v in params.items() if k in fields}
    )
    if detector_backend is not None:
        cfg = dataclasses.replace(cfg, detector_backend=detector_backend)
    return cfg


def run(cfg, scans) -> list[dict]:
    """Detect every scan in order with one detector, as the node does."""
    from cone_detection.cone_detection import RealtimeConeDetector

    det = RealtimeConeDetector(cfg)
    out = []
    for scan in scans:
        counters: dict = {}
        cones = det.detect(scan, debug_counters=counters)
        out.append(
            {
                "cones": np.asarray(cones, dtype=float).reshape(-1, 4).tolist(),
                "plane": np.asarray(det._prev_plane, dtype=float).tolist(),
                "counters": {
                    k: (int(v) if float(v).is_integer() else float(v))
                    for k, v in sorted(counters.items())
                    if k not in _SKIP_COUNTERS
                },
            }
        )
    return out
