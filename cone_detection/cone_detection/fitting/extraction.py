"""Fitting stage: labelled above-ground points -> accepted cones.

Each cluster goes through the pre-fit gates, the configured cluster fitter and the
residual gate (see ``gates`` and ``fitters``); this module only runs them in order
and keeps the books: ``debug_counters``, stage timings, the optional observer and the
per-scan fit summary log line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.fitting import gates
from cone_detection.fitting.clusters import Cluster, ClusterSet
from cone_detection.fitting.fitters import ClusterFitter, ConeFit
from cone_detection.fitting.uncertainty import sigma_xy
from cone_detection.instrumentation import ScanObserver, StageTimings

# (x, y, height_m, sigma_xy) of one accepted cone, ground frame.
Cone = tuple[float, float, float, float]

# Per-cluster counters, zeroed every scan that reaches the gates.
GATE_COUNTERS = (
    "after_min_pts",
    "after_shape_gate",
    "after_residual_gate",
    "accepted",
    "far_dropped",
    "small_count",
    "big_count",
    "ambiguous_count",
)


@dataclass
class _FitSummary:
    """What the per-scan fit log line reports."""

    n_shape_rejected: int = 0
    n_residual_rejected: int = 0
    residuals: list[float] = field(default_factory=list)


def extract_cones(
    points: np.ndarray,
    labels: np.ndarray,
    cfg: ConeDetectionConfig,
    fit_cluster: ClusterFitter,
    *,
    counters: dict,
    timings: StageTimings,
    observer: ScanObserver | None = None,
    logger=None,
) -> list[Cone]:
    """Gate and fit every cluster of ``labels`` over ``points``; return the cones."""
    with timings.time("cluster_prep_ms"):
        clusters = ClusterSet(points, labels, cfg.floor_margin_m)
    timings.record("n_clusters", len(clusters))
    counters["n_clusters"] = len(clusters)
    counters.update(dict.fromkeys(GATE_COUNTERS, 0))

    summary = _FitSummary()
    cones: list[Cone] = []
    min_points = cfg.min_cluster_points
    with timings.time("fit_ms"):
        for cluster in clusters:
            # Most clusters are too small; with no observer, skip _judge for them
            # (it counts nothing for this gate).
            if observer is None and len(cluster.points) < min_points:
                continue
            cone = _judge(cluster, cfg, fit_cluster, counters, summary, observer)
            if cone is not None:
                cones.append(cone)

    if logger is not None and (cones or summary.n_residual_rejected or summary.n_shape_rejected):
        _log_fit_summary(logger, cfg, len(cones), counters, summary)
    return cones


def _judge(
    cluster: Cluster,
    cfg: ConeDetectionConfig,
    fit_cluster: ClusterFitter,
    counters: dict,
    summary: _FitSummary,
    observer: ScanObserver | None,
) -> Cone | None:
    """Run one cluster through the gates and the fit; the cone if it is accepted."""
    gate = gates.first_failed_pre_fit_gate(cluster, cfg)
    _count_pre_fit(gate, cluster, cfg, counters, summary)
    if gate is not None:
        if observer is not None:
            observer.cluster_judged(
                cluster, gate, gates.describe_rejection(gate, cluster, cfg), None, None
            )
        return None

    fit_points = cluster.points_above(cfg.floor_margin_m)
    fit = fit_cluster(fit_points, cfg, cluster.height_m)
    if not gates.passes_residual_gate(fit.res_min, cfg):
        summary.n_residual_rejected += 1
        if observer is not None:
            reason = gates.describe_rejection(gates.RESIDUAL, cluster, cfg, fit.res_min)
            observer.cluster_judged(cluster, gates.RESIDUAL, reason, fit, fit_points)
        return None

    ambiguous = gates.is_ambiguous(fit.res_min, fit.res_other, cfg)
    _count_accepted(fit, ambiguous, counters)
    summary.residuals.append(fit.res_min)
    if observer is not None:
        reason = gates.describe_acceptance(fit.is_big, fit.res_min)
        observer.cluster_judged(cluster, None, reason, fit, fit_points)
    return (
        float(fit.a),
        float(fit.b),
        float(fit.d),
        sigma_xy(fit.a, fit.b, cluster.n_clean, ambiguous),
    )


def _count_pre_fit(
    gate: str | None,
    cluster: Cluster,
    cfg: ConeDetectionConfig,
    counters: dict,
    summary: _FitSummary,
) -> None:
    if gate == gates.MIN_POINTS:
        return
    counters["after_min_pts"] += 1
    if gate in gates.SHAPE_GATES:
        summary.n_shape_rejected += 1
        # Counted for any shape rejection of a far cluster, whichever gate fired.
        if cluster.range_m > cfg.range_gate_max_m:
            counters["far_dropped"] += 1
    elif gate is None:
        counters["after_shape_gate"] += 1


def _count_accepted(fit: ConeFit, ambiguous: bool, counters: dict) -> None:
    counters["after_residual_gate"] += 1
    if ambiguous:
        counters["ambiguous_count"] += 1
    counters["big_count" if fit.is_big else "small_count"] += 1
    counters["accepted"] += 1


def _log_fit_summary(logger, cfg: ConeDetectionConfig, n_kept: int, counters: dict,
                     summary: _FitSummary) -> None:
    if summary.residuals:
        res_arr = np.asarray(summary.residuals, dtype=float)
        rmse_med = float(np.sqrt(np.median(res_arr)))
        rmse_max = float(np.sqrt(res_arr.max()))
    else:
        rmse_med = 0.0
        rmse_max = 0.0
    logger.info(
        f"[cone_fit backend={cfg.fit_backend}] kept={n_kept} "
        f"(small={counters['small_count']} big={counters['big_count']} "
        f"amb={counters['ambiguous_count']}) "
        f"rejected: shape={summary.n_shape_rejected} residual={summary.n_residual_rejected} "
        f"| RMSE med={rmse_med * 1000:.1f}mm max={rmse_max * 1000:.1f}mm"
    )
