"""Preprocessing stage: raw sensor-frame scan -> the points worth processing.

Drops returns beyond ``input_range_crop_m`` (horizontal) and closer than
``input_min_range_m`` (3D: the driver's no-return (0, 0, 0) points) before ground
removal, in one compiled pass.
"""

from __future__ import annotations

import numba
import numpy as np

from cone_detection.config import ConeDetectionConfig


def crop_scan(scan: np.ndarray, cfg: ConeDetectionConfig) -> np.ndarray:
    """``scan`` without the returns the range limits drop (both off: ``scan`` itself)."""
    if cfg.input_range_crop_m > 0 or cfg.input_min_range_m > 0:
        return range_crop(
            np.asarray(scan), float(cfg.input_range_crop_m), float(cfg.input_min_range_m)
        )
    return scan


@numba.njit(cache=True)
def range_crop(data, crop_m, min_range_m):
    """Rows of ``data`` within ``crop_m`` horizontally and at least ``min_range_m`` away in 3D.

    A bound <= 0 is off. Non-finite rows fail the comparisons and are dropped, like
    the masked version. One pass, one allocation: the result is the leading rows of
    a buffer sized for the whole scan.
    """
    n = data.shape[0]
    ncol = data.shape[1]
    crop2 = crop_m * crop_m
    min2 = min_range_m * min_range_m
    out = np.empty((n, ncol), data.dtype)
    k = 0
    for i in range(n):
        x = np.float64(data[i, 0])
        y = np.float64(data[i, 1])
        r2 = x * x + y * y
        if crop_m > 0.0 and not (r2 <= crop2):
            continue
        if min_range_m > 0.0:
            z = np.float64(data[i, 2])
            if not (r2 + z * z >= min2):
                continue
        for j in range(ncol):
            out[k, j] = data[i, j]
        k += 1
    return out[:k]
