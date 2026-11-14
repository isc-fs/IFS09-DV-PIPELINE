"""Which clusters become cones: the acceptance gates, cheapest first.

Before the fit, :func:`first_failed_pre_fit_gate` checks the scalars of
:class:`~cone_detection.fitting.clusters.Cluster`; after it,
:func:`passes_residual_gate` checks the fit. Gates report *which* gate failed; the
human-readable wording (:func:`describe_rejection`) is only built when something is
listening, so the hot path does no string work.
"""

from __future__ import annotations

import math

from cone_detection.config import ConeDetectionConfig
from cone_detection.fitting.clusters import Cluster

# Pre-fit gates, in the order they are checked.
MIN_POINTS = "min_points"  # too few returns to be a cone
ABOVE_FLOOR = "above_floor"  # nothing left above the floor margin
HEIGHT = "height"  # height extent outside the cone range
TOP = "top"  # highest return too close to the floor (curb / wall foot)
RANGE = "range"  # too far away to trust
# The gates that make up the "shape" rejection count in the fit summary.
SHAPE_GATES = frozenset({HEIGHT, TOP, RANGE})
RESIDUAL = "residual"  # the fit does not look like a cone


def first_failed_pre_fit_gate(cluster: Cluster, cfg: ConeDetectionConfig) -> str | None:
    """The first pre-fit gate ``cluster`` fails, or None if it should be fitted."""
    if cluster.n_points < cfg.min_cluster_points:
        return MIN_POINTS
    if cluster.n_clean == 0:
        return ABOVE_FLOOR
    if not cfg.cluster_height_min_m <= cluster.height_m <= cfg.cluster_height_max_m:
        return HEIGHT
    if not cluster.top_m >= cfg.cluster_top_min_m:
        return TOP
    if not cluster.range_m <= cfg.range_gate_max_m:
        return RANGE
    return None


def passes_residual_gate(res_min: float, cfg: ConeDetectionConfig) -> bool:
    return bool(math.isfinite(res_min) and res_min <= cfg.residual_gate_mse)


def is_ambiguous(res_min: float, res_other: float, cfg: ConeDetectionConfig) -> bool:
    """Both templates fit about equally well (the class call is a guess)."""
    return bool(
        math.isfinite(res_other)
        and res_other / max(res_min, 1e-12) < cfg.ambiguous_margin_ratio
    )


def _rmse_mm(mse: float) -> float:
    return 1000.0 * math.sqrt(mse) if math.isfinite(mse) else float("inf")


def describe_rejection(
    gate: str, cluster: Cluster, cfg: ConeDetectionConfig, res_min: float = math.nan
) -> str:
    """Readable verdict for a cluster ``gate`` dropped."""
    if gate == MIN_POINTS:
        return f"dropped: {cluster.n_points} pts < min {cfg.min_cluster_points}"
    if gate == ABOVE_FLOOR:
        return "dropped: all points below floor margin"
    if gate == HEIGHT:
        return (
            f"dropped: height {cluster.height_m:.2f} m outside "
            f"[{cfg.cluster_height_min_m}, {cfg.cluster_height_max_m}]"
        )
    if gate == TOP:
        return f"dropped: top {cluster.top_m:.2f} m < min {cfg.cluster_top_min_m}"
    if gate == RANGE:
        return f"dropped: range {cluster.range_m:.1f} m > {cfg.range_gate_max_m}"
    if gate == RESIDUAL:
        return (
            f"dropped: residual RMSE {_rmse_mm(res_min):.0f} mm > gate "
            f"{1000 * cfg.residual_gate_mse ** 0.5:.0f} mm"
        )
    raise ValueError(f"unknown gate {gate!r}")


def describe_acceptance(is_big: bool, res_min: float) -> str:
    return f"accepted ({'big' if is_big else 'small'}, RMSE {_rmse_mm(res_min):.0f} mm)"
