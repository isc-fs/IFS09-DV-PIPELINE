"""cluster_top_min_m: drop low fragments (curbs, wall feet) that pass the height-extent gate."""

import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.detector import RealtimeConeDetector
from cone_detection.fitting.cone_classes import CONE_SMALL_C, CONE_SMALL_D

SENSOR_H = 1.07


def _scene():
    rng = np.random.default_rng(5)
    n = 6000
    ground = np.column_stack(
        [rng.uniform(0.5, 15.0, n), rng.uniform(-5.0, 5.0, n), rng.normal(-SENSOR_H, 0.005, n)]
    )
    # Near face of a small cone at (7, 1.5), top return 0.30 m above ground.
    cx, cy = 7.0, 1.5
    h = rng.uniform(0.06, 0.30, 60)
    r = (CONE_SMALL_D - h) / CONE_SMALL_C
    ux, uy = np.array([cx, cy]) / np.hypot(cx, cy)
    psi = rng.uniform(-1.0, 1.0, 60)
    cone = np.column_stack(
        [
            cx - r * (np.cos(psi) * ux - np.sin(psi) * uy),
            cy - r * (np.cos(psi) * uy + np.sin(psi) * ux),
            h - SENSOR_H,
        ]
    )
    # Curb fragment at (5, -2): 0.25 x 0.14 m, 0.05-0.11 m above ground.
    m = 40
    curb = np.column_stack(
        [
            5.0 + rng.uniform(-0.125, 0.125, m),
            -2.0 + rng.uniform(-0.07, 0.07, m),
            rng.uniform(0.05, 0.11, m) - SENSOR_H,
        ]
    )
    return np.vstack([ground, cone, curb]).astype(np.float32), (cx, cy)


def _detect(top_min):
    scan, _ = _scene()
    cfg = ConeDetectionConfig(ground_method="gpf", cluster_top_min_m=top_min)
    return RealtimeConeDetector(cfg).detect(scan).cones


def test_disabled_gate_keeps_low_fragment():
    cones = _detect(0.0)
    assert len(cones) == 2


def test_top_gate_drops_fragment_and_keeps_cone():
    _, (cx, cy) = _scene()
    cones = _detect(0.15)
    assert len(cones) == 1
    assert np.hypot(cones[0][0] - cx, cones[0][1] - cy) < 0.1
