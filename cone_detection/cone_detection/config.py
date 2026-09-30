"""The perception pipeline's tunable settings (``ConeDetectionConfig``).

Kept apart from ``cone_detection.py`` so it imports nothing heavy: the ROS node declares one
parameter per field at construction, before ``~/setup``, and must not pull in sklearn / numba
then (mode_manager's service wait times out on a cold start). ``cone_detection.cone_detection``
re-exports both names, so existing imports keep working.

Each field is a ROS parameter of ``cone_detection_node`` with the same name, type and default;
the values that run are in ``bringup/config/params.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ConeFitBackend = Literal["template_dispatch", "two_param"]


@dataclass
class ConeDetectionConfig:
    """Tunable perception pipeline (clustering, gates, cone fit)."""

    # Drop a cluster if the chosen fit's MSE residual exceeds this (RMSE = √mse:
    # 0.2 → 447 mm). Loosened from the sim-tuned 0.002 (RMSE 45 mm), which
    # rejected 100% of shape-valid candidates on the real Hesai ATX track
    # (CONE_FILTER: ~20/scan pass shape, 0 pass residual → empty /Conos_raw →
    # no map/path → car won't move). Deliberately permissive to get the car
    # driving; expect more false positives. RE-TIGHTEN toward 0.02-0.03 once a
    # real-cloud residual histogram is measured (bag /lidar_points via #68).
    residual_gate_mse: float = 0.2

    # Coarse cluster-shape pre-filter (meters).
    cluster_height_min_m: float = 0.02
    cluster_height_max_m: float = 0.55

    # Range cutoff (m): beyond this, drop as too sparse / unreliable.
    range_gate_max_m: float = 20.0

    # Input range pre-crop (m): drop points beyond this horizontal radius
    # BEFORE RANSAC + DBSCAN. Everything past range_gate_max_m is discarded
    # downstream anyway (far_dropped), so cropping here is behavior-preserving
    # (radius > range_gate_max_m + cone extent) and removes most of the ~174k
    # ATX returns (far ground) before the O(n) clustering — the dominant cost.
    # Set to 0 to disable.
    input_range_crop_m: float = 25.0

    # Drop returns closer than this 3D range (m) before anything else. The
    # Hesai ATX driver keeps every firing slot in the cloud and encodes "no
    # return" as (0, 0, 0): ~14% of a real scan (24.6k of 174k) sat at the
    # sensor origin. After the ground shift those land ~1 m above the floor,
    # where the tall-column veto silently discarded them, and they still cost
    # RANSAC time. 0.1 m is far inside the sensor's blind zone, so no real
    # return is lost. Set to 0 to disable.
    input_min_range_m: float = 0.1

    # Confidence margin (residual_other / residual_min) for template_dispatch
    # ambiguity. Ignored when ``res_other`` is not finite (e.g. two_param path).
    ambiguous_margin_ratio: float = 1.5

    # For ``fit_backend == "two_param"``: classify big orange when fitted apex
    # height ``d`` exceeds this (template path uses fixed 0.35 / 0.55 instead).
    big_orange_d_threshold_m: float = 0.45

    # Ground strip: drop returns closer than this above the estimated floor (m).
    floor_margin_m: float = 0.04

    # Minimum number of returns in a DBSCAN cluster before floor culling.
    min_cluster_points: int = 3

    # RANSAC plane inliers
    ransac_prob: float = 0.9999
    ransac_threshold: float = 0.05

    # DBSCAN on rotated above-ground cloud
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2

    # DBSCAN memory/time guard. sklearn's DBSCAN materialises EVERY point's
    # full neighbour list (radius_neighbors), so its cost is
    # O(N × neighbours-within-eps), i.e. quadratic in point density. A scan
    # whose above-ground cloud is both large and dense — tire walls, car
    # off-track facing terrain / an object at close range, or a mis-fitted
    # ground plane that turns the ground itself into "outliers" — blows
    # straight through the container: measured 90k pts @ 20k pts/m² → 4.7 GB
    # peak, 11 s, ~2.6 GB never returned to the OS. Two such scans OOM-kill
    # the node (8 GB limit) after stalling SLAM for seconds. A cone scene is
    # nowhere near: ~100 cones × ≤200 pts ≈ 20k worst case, typically 1-2k.
    #
    # Above dbscan_max_points the cloud is voxel-deduplicated, starting at
    # dbscan_guard_voxel_m and doubling the cell up to dbscan_guard_voxel_max_m
    # until the count fits, then uniformly subsampled to the cap as a last
    # resort. Normal scans never trigger it. Set dbscan_max_points to 0 to
    # disable.
    dbscan_max_points: int = 20_000
    dbscan_guard_voxel_m: float = 0.02
    dbscan_guard_voxel_max_m: float = 0.10

    # Tall-column veto (tire walls, fences, people): before DBSCAN, drop EVERY
    # point whose xy grid cell (3x3-dilated) contains a return more than
    # tall_column_veto_height_m above the fitted ground plane. No cone exceeds
    # cluster_height_max_m, so such a return proves non-cone structure in that
    # column; vetoing the whole column also removes the structure's cone-height
    # low band, which a plain z-crop would leave behind as truncated stumps.
    # Tire-wall scans (~40-90k DBSCAN points, seconds) collapse back to normal
    # (~ms) at zero measured cone change; scans with nothing tall skip the
    # masking entirely.
    tall_column_veto: bool = True
    tall_column_veto_height_m: float = 0.75
    tall_column_veto_cell_m: float = 0.30

    # Template (a, b) solver: "gn" = in-process Numba Gauss-Newton/LM
    # (production default); any scipy.optimize.minimize method name
    # (e.g. "L-BFGS-B", the previous default) selects the scipy path for A/B.
    cone_fit_solver: str = "gn"
    template_fit_maxiter: int = 12

    # Near-collinear clusters: closed-form collinear fit (no scipy).
    use_collinear_fit: bool = True

    # Skip the second template L-BFGS-B when cluster height strongly
    # indicates small vs big-orange (saves ~half of fit time).
    template_fit_early_exit: bool = True
    template_early_exit_height_small_m: float = 0.40
    template_early_exit_height_big_m: float = 0.50

    fit_backend: ConeFitBackend = "template_dispatch"
