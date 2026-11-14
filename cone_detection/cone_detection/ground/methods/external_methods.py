"""Methods backed by optional third-party packages, imported on first use: ``csf``
(cloth-simulation-filter) and ``patchworkpp`` (pypatchworkpp).
"""

from __future__ import annotations

import contextlib

import numpy as np

from .scan_geometry import _level
from .plane_fitting import _all_but_ground, _refit_on_ground
from .registry import _optional, register


# Cloth Simulation Filter (Zhang et al., Remote Sensing 2016), reference C++.
_CSF_CELL_M = 0.30


_CSF_CLASS_THR_M = 0.08


@register("csf")
def ground_csf(data, cfg):
    """Drop a cloth onto the upside-down cloud; points near the cloth are ground."""
    CSF = _optional("CSF", "cloth-simulation-filter")
    thr = cfg.ground_threshold_m
    uv, h, coarse = _level(data, thr)
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = _CSF_CELL_M
    csf.params.rigidness = 3
    csf.params.time_step = 0.65
    csf.params.class_threshold = _CSF_CLASS_THR_M
    csf.params.interations = 500
    csf.setPointCloud(np.c_[uv, h].astype(np.float64))
    ground_idx, off_idx = CSF.VecInt(), CSF.VecInt()
    with _quiet_stdout():  # the library prints its progress with cout
        csf.do_filtering(ground_idx, off_idx, exportCloth=False)
    non_ground = _all_but_ground(len(data), np.asarray(ground_idx, dtype=np.int64))
    return non_ground, _refit_on_ground(data, non_ground, coarse)


@contextlib.contextmanager
def _quiet_stdout():
    import os
    import sys

    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


@register("patchworkpp")
class PatchworkGround:
    """Patchwork++ (Lee et al., RA-L 2022), reference C++ via pypatchworkpp.

    The estimator is stateful (TGR / elevation storage learn across scans), so the
    instance keeps one, rebuilt when the sensor height above the coarse plane
    changes (it is a construction parameter).
    """

    __slots__ = ("_cfg", "_est", "_est_key")

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._est = None
        self._est_key: float | None = None

    def _estimator(self, sensor_h: float):
        key = round(sensor_h, 2)
        if self._est is None or key != self._est_key:
            ppp = _optional("pypatchworkpp", "pypatchworkpp")
            p = ppp.Parameters()
            p.sensor_height = sensor_h
            p.min_range = 1.0
            p.max_range = 80.0
            p.th_dist = self._cfg.ground_threshold_m
            p.enable_RNR = False  # reflectivity-based noise removal; this cloud has no intensity
            p.intensity_thr = 0.0
            p.verbose = False
            self._est, self._est_key = ppp.patchworkpp(p), key
        return self._est

    def remove(self, data):
        uv, h, coarse = _level(data, self._cfg.ground_threshold_m)
        sensor_h = float(coarse[0])  # height of the sensor origin above the coarse plane
        est = self._estimator(sensor_h)
        # Levelled frame with the sensor at the origin: ground sits at z = -sensor_h.
        pts = np.c_[uv, h - sensor_h, np.zeros(len(h))].astype(np.float32)
        est.estimateGround(pts)
        non_ground = _all_but_ground(len(data), np.asarray(est.getGroundIndices(), dtype=np.int64))
        return non_ground, _refit_on_ground(data, non_ground, coarse)

    def reset(self) -> None:
        self._est = None
        self._est_key = None
