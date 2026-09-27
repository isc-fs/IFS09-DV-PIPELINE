"""Cone-graph SLAM node — PR B (cone observation factors + DA).

Subscribes:
    /imu               (sensor_msgs/Imu, BEST_EFFORT, ~400 Hz)
    /Conos_raw         (visualization_msgs/MarkerArray, base_link
                       frame, 10 Hz from Cone_Detection)
    /odom              (nav_msgs/Odometry from sim_supervisor, 100 Hz)
                       Phase 2 (#382): used to compute map→odom drift
                       correction. slam_node's pose is absolute (map
                       frame); supervisor's /odom is dead-reckoning
                       (odom frame); the difference is the drift to
                       absorb into the map→odom transform.

Publishes:
    /tf                (map -> odom, dynamic drift correction
                       computed as slam_pose ⊖ latest /odom; replaces
                       the pre-#382 odom→base_link broadcast which
                       supervisor now owns)
    /slam/pose         (nav_msgs/Odometry — absolute pose+velocity in
                       map frame; renamed from /cone_slam/state in
                       #382 so the topic name doesn't lock us into
                       "cone_slam" when we eventually swap iSAM2
                       backends)
    /Conos             (visualization_msgs/MarkerArray, MAP-frame
                       cone map with persistent landmark IDs encoded
                       as marker.id; was odom-frame pre-#382)

State machine:
    INIT_WAITING_IMU  → INIT_CALIBRATING (3 s) → SLAM_RUNNING

Lifecycle:
    INIT_CALIBRATING expects the car to be stationary. We accumulate
    IMU samples, then estimate accel/gyro biases as the mean (with
    gravity assumption (0,0,-9.81) for accel). Anchor x_0 at world
    origin and start the factor graph.

    SLAM_RUNNING: every LiDAR header.stamp triggers an IMU
    preintegration window since the last trigger, an ImuFactor between
    the prev pose and a new pose, and an iSAM2 update. PR B adds cone
    factors. PR C adds GPS.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import gtsam
import numpy as np
import rclpy

# color_classifier deleted: SLAM is position-only, no per-cone colour
# anywhere. Visualization renders all landmarks with the same colour.
from cone_slam.data_association import DISTANCE_GATE_M, Observation, associate
from cone_slam.factor_graph import FactorGraph, ScanResult
from cone_slam.imu_preintegrator import ImuPreintegrator, ImuSample
from cone_slam.landmark_db import LandmarkDb
from cone_slam.lap_counter import LapCounter, LapCounterConfig
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from node_base.base_lifecycle_node import BaseLifecycleNode
from rclpy.lifecycle import State as LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _odom_to_pose3(msg: Odometry) -> "gtsam.Pose3":
    """Convert nav_msgs/Odometry pose into gtsam.Pose3."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z),
        np.array([p.x, p.y, p.z]),
    )


# compute_map_to_odom lives in cone_slam.tf_math (pure-Python helper,
# no rclpy import, unit-testable). Re-imported here for the call site.
from cone_slam.tf_math import compute_map_to_odom  # noqa: E402, F401

CALIBRATION_SECONDS = 3.0

# Observations beyond this body-frame range are dropped before reaching
# data association. Universal practice across FSD teams (EUFS 20 m,
# QUTMS 25 m, MUR ~25 m, KIT19d 42 m): far cones have low point counts
# (Hesai ATX gives 3-5 rays at 25 m, ~30+ at 5 m), so centroid noise
# dominates and the bearing-range factor stops being informative. With
# the constant 0.20 m sigma we previously reported, a noisy 30 m cone
# was actively dragging the optimizer toward a wrong rotation — that's
# the late-drive yaw snap we kept hitting on 2026-04-27.
MAX_OBSERVATION_RANGE_M = 25.0


def cascade_spike_triggered(
    n_new: int,
    total: int,
    step: int,
    *,
    pct_min_obs: int = 5,
    pct_threshold: float = 0.60,
    count_threshold: int = 3,
    discovery_step_floor: int = 30,
) -> tuple[bool, list[str]]:
    """Pure-function expression of the two DA-failure spike gates.

    Returns (triggered, reasons). Both gates require the graph to be
    past the early-discovery phase (``step > discovery_step_floor``)
    so the local cone map is mature and almost every observation
    *should* be a re-association of a known landmark.

      * **Percentage gate**: ``n_new / total > pct_threshold`` once
        ``total >= pct_min_obs``. Catches "everything looks new" bursts
        — the historical cascade signature, see #441.
      * **Count gate**: ``n_new >= count_threshold`` AND
        ``n_associated == 0`` (i.e. ``total == n_new``). The zero-
        association requirement is what discriminates a true cascade
        (pose has drifted off the map → nothing matches) from
        legitimate cornering discovery (new cones swing into the LiDAR
        FoV but several previously-mapped ones still associate). The
        2026-05-12 live test of the initial assoc-agnostic count gate
        showed it firing on `obs=12 new=7 assoc=5` during the first
        turn and starving SLAM into IMU-only drift → off-track. Set
        ``count_threshold = 0`` to disable.

    Both gates are AND'd with the discovery-step floor.
    Extracted from cone_graph_slam_node._on_cones for unit-testability
    (test_cascade_spike_detector.py).
    """
    if step <= discovery_step_floor:
        return False, []
    reasons: list[str] = []
    if total >= pct_min_obs and n_new > int(pct_threshold * total):
        reasons.append(f"pct>{int(pct_threshold * 100)}%")
    if count_threshold > 0 and n_new >= count_threshold and n_new == total:
        reasons.append(f"n_new≥{count_threshold}&assoc=0")
    return bool(reasons), reasons


# Motor-RPM → body-frame longitudinal velocity.
#
# 2026-07-12 (Option B): WheelRadius corrected 0.228 → 0.202 m to match the
# real IFS-08 tyre. Must stay in sync with odometry_filter's kRpmToMs — both
# scale the same /motor_rpm, and a mismatch splits the odom and SLAM velocity
# estimates. The sim generates rpm from ground speed with the same radius, so
# /odom.vx = GT for any consistent radius (GT-neutral in sim); on the real car
# (true rotor rpm) 0.202 is the correct geometry.
#     RPM_TO_MS = (2π × 0.202 / 2.909) / 60 = 0.00727
#
# History (issue #380): the prior 0.00821 (= 0.00898 × 0.9140, matching the
# sim's self-consistent 0.228 geometry) was validated only against sim GT, not
# the real tyre. Superseded by the geometric 0.202 above.
RPM_TO_MS = 0.00727

# Drop /motor_rpm samples this old (seconds). Sustained RPM staleness
# means the bridge stopped publishing; fall back to no velocity prior
# rather than constraining the optimizer with last-good-but-stale data.
RPM_STALE_S = 0.5

# Wheel speed (m/s, already scaled by RPM_TO_MS) below which the car counts
# as stationary for INIT bias calibration. A non-stationary calibration window
# soaks real rotation into the gyro bias and drifts SLAM's heading.
STATIONARY_SPEED_MS = 0.1


def _env_truthy(name: str) -> bool:
    """Read a boolean toggle from the environment (1/true/yes/on)."""
    import os

    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class State(Enum):
    INIT_WAITING_IMU = 1
    INIT_CALIBRATING = 2
    SLAM_RUNNING = 3


class ConeGraphSlamNode(BaseLifecycleNode):
    """Cone-graph SLAM as a managed LifecycleNode.

    Lifecycle layout:

      on_configure  declare parameters (re-readable on every
                    re-configure), instantiate components (preint /
                    graph / db), open the optional landmark capture
                    file, create lifecycle publishers + TF
                    broadcasters, emit the map→odom static identity.
      on_activate   reset state machine to INIT_WAITING_IMU, create
                    the four subscriptions (/imu, /Conos_raw,
                    /motor_rpm, /testing_only/odom), super().on_activate
                    flips lifecycle pubs to emitting state.
      on_deactivate destroy subscriptions; pubs go quiet via super().
      on_cleanup    destroy publishers + broadcasters, close the
                    capture file, drop component refs.
    """

    NODE_NAME = "slam_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # --- Parameter declarations live in __init__ so they exist
        # before configure (mode_manager will eventually pre-set the
        # mission strategy flag via parameters).
        # Frames (REP-105). Post-#382 the SLAM-owned absolute frame
        # is `map`, not `odom` — the `odom` frame is supervisor's
        # dead-reckoning, slam computes `map → odom` to absorb the
        # difference. Backward-compat parameter names preserved.
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        # Motion model between LiDAR scans (issue: "use the EKF instead
        # of IMU preintegration").
        #   "odom" (default) — EKF-as-motion-model. The 9-state EKF's
        #          /odom per-scan delta-pose is the PRIMARY pose-to-pose
        #          constraint (stage_odom_motion_step). No IMU
        #          preintegration factor, no separate RPM velocity prior
        #          or steering BetweenFactor — the EKF already fused IMU
        #          + RPM + steering, so re-adding them double-counts. The
        #          EKF prediction also drives data association and the
        #          pose-jump sanity check. Cone BearingRange factors keep
        #          doing what only the graph can: correcting accumulated
        #          drift via landmark consistency.
        #   "imu"  — legacy path. ImuFactor motion model + RPM velocity
        #          prior + steering BetweenFactor + a soft /odom backstop,
        #          pim.predict() for DA. Kept for A/B benchmarking and
        #          rollback. Defaults can flip back here without code
        #          changes.
        # An invalid value falls back to "odom" with a warning.
        self.declare_parameter("motion_model", "odom")
        # Pose-jump sanity check thresholds (#273 follow-up).
        # Maximum allowed deviation between iSAM2's optimized pose and
        # the IMU-predicted pose at each scan. When a wrong cone match
        # passes the DA gate, iSAM2 snaps pose to fit the bad factor;
        # the sanity check catches that and re-anchors at the IMU
        # prediction.
        # 0.8 m: bigger than any legitimate single-scan iSAM2 correction
        #         (typical IMU drift over 100 ms is sub-decimeter, the
        #         correction iSAM2 applies via cone factors is at most
        #         a few centimeters per scan once the map has matured).
        # 0.3 rad (~17°): ditto for yaw — much more than any legitimate
        #         single-scan refinement.
        self.declare_parameter("pose_jump_max_pos_m", 0.8)
        self.declare_parameter("pose_jump_max_yaw_rad", 0.3)

        # Proximity veto for new-landmark *creation* (separate from the
        # DA match gate). When pose drift creeps to ~DA_GATE_M (1.0 m),
        # re-observations of existing cones fall just past the gate,
        # DA flags them as new, and they get spawned as ghost landmarks
        # anchored at the drifted pose — the cascade trigger documented
        # in bag lap_attempt_20260511_142317 (5 of 6 "new" obs were
        # 1.02-1.20 m from an existing landmark; pose had drifted ~1 m
        # off truth). The veto refuses to create a new landmark whose
        # world position is closer than this to any existing landmark.
        # Genuine new cones (track-min spacing ≥ 3 m, gate-min ≥ 1.5 m
        # in FSD rules) still get spawned; cascade ghosts don't.
        # Set to 0 to disable the veto.
        self.declare_parameter("new_landmark_proximity_veto_m", 1.5)

        # IMU bias-calibration window at init (INIT_CALIBRATING assumes the car
        # is stationary). Default = the module constant; exposed as a param so a
        # rosbag REPLAY on the car — where the bag may not present a clean
        # stationary window at the moment the node activates — can shorten it
        # without a code change. Default behaviour is unchanged.
        self.declare_parameter("imu_calibration_seconds", CALIBRATION_SECONDS)

        # Count-based DA-spike threshold (issue #447, post-Phase-2 finding).
        # The original spike detector compared n_new_pre against
        # 60 % of total observations — but a 5-of-14 burst (36 %)
        # slipped through it in lap_postveto_20260511_183638 and
        # lap_ekf_inloop_20260511_235510, spawning the ghosts that
        # detonated the cascade in both runs. Three or more new
        # landmark candidates from a single mature-phase scan is
        # *always* anomalous in FSD: legitimate new-cone arrivals
        # come at vehicle-speed × scan-period spacing (typically 1
        # cone every 3–4 scans), never bursting. Setting this to 0
        # disables the count-based gate; the percentage gate stays
        # active either way.
        self.declare_parameter("cascade_spike_new_count_threshold", 3)

        # Minimum re-observation count before a landmark is published
        # on /Conos for downstream consumers. Single-shot detections
        # are noisy (one-frame perception artifacts, phantom DA spawns
        # from /odom yaw drift) and pollute the cone map the planner
        # sees. Counting how many factors point at a landmark before
        # exposing it filters those out without affecting the SLAM
        # solve itself (the landmark stays in the factor graph, just
        # not on /Conos). 3 is the smallest count that survives
        # genuine perception noise (cone seen → missed → seen again);
        # 0 disables the gate (legacy behaviour).
        self.declare_parameter("min_observations_for_publish", 3)

        # Radius (m) around the car within which landmarks are PUBLISHED
        # to /Conos. The full map stays in the iSAM2 graph (loop closure
        # still uses every landmark) — this only crops the per-scan
        # MarkerArray we serialize out. Without it, /Conos publishing is
        # O(map): SLAM_PROF showed the publish stage growing to 50-84 ms
        # at map=212 (the dominant per-scan cost, > the 100 ms / 10 Hz
        # budget — the real "path falls behind in corners" mechanism).
        # The only /Conos consumer is path_planning, which is event-
        # driven on the topic and internally sorts cones outward from the
        # car, so it only needs the local track ahead; far/behind cones
        # are pure overhead. Cropping makes the publish cost O(local) ≈
        # constant regardless of laps driven, at full 10 Hz cadence.
        # Default 25 m: ~2-3× a path planner's lookahead, covers cones
        # ahead and a margin behind. Set to 0 to disable (publish all).
        self.declare_parameter("cone_map_publish_radius_m", 25.0)

        # KEYFRAME interval (in scans) for the full cone map on
        # /Conos_full. /Conos_full is published every scan as a delta
        # (only new/moved cones — see _publish_full_cone_map), so the
        # per-scan cost is O(changed), not O(map). Every N scans we instead
        # send a keyframe (DELETEALL + the whole map) to resync a late-
        # joining viewer (VOLATILE QoS, no history) and recover any dropped
        # frames — that one is O(map). Keep it brisk enough that a freshly-
        # connected Foxglove fills in quickly but rare enough that the
        # O(map) build doesn't land on the loop-closure relinearization
        # burst. Default 50 (~5 s at 10 Hz). 0 disables keyframes (deltas
        # only; a viewer that misses the initial spawn never recovers).
        self.declare_parameter("cone_map_full_publish_every_n_scans", 50)

        # Cascade-skip recovery. After this many consecutive scans
        # rejected by the cascade-spike gate, force-accept the next
        # scan instead of skipping it. The theory: 10 scans in a row
        # all flagged as "all-new" is far longer than any real cascade
        # burst — it's almost certainly the car driving into a section
        # of track with legitimately-new cones that haven't been
        # mapped yet. Refusing to spawn them indefinitely strands
        # SLAM (cones lose alignment with the IMU-only pose dead-
        # reckon, /slam/pose freezes, controller drives blind).
        #
        # The proximity veto (`new_landmark_proximity_veto_m`) still
        # gates individual cones inside the force-accepted scan, so
        # cascade ghosts that are spatially adjacent to existing
        # landmarks are still rejected — we only relax the "skip the
        # whole scan" decision. Set to 0 to disable recovery; the
        # legacy "stay stuck forever" behaviour returns.
        #
        # Tuning: 5 scans @ 10 Hz = 0.5 s of stuck. Originally 10 but
        # bag autocross_track_20260404_013721_20260518_095215 had a
        # cascade that lasted only 9 scans before the obs count dropped
        # to 1 — recovery never fired despite SLAM being structurally
        # in trouble. 5 catches shorter cascades without false-firing
        # on transient ~0.2-0.3 s DA bursts.
        self.declare_parameter("cascade_skip_recovery_threshold", 5)

        # --- Loop-closure mapping freeze -------------------------------
        # Once the car completes a lap, the track is fully mapped: every
        # subsequent observation should re-associate to an existing
        # landmark, never spawn a new one. Freezing landmark creation at
        # loop closure kills the duplicate-spawn amplifier — when the
        # loop-closure relinearization burst leaves /slam/pose briefly
        # stale, a transient bad DA can no longer bloat the map with
        # phantom cones (the failure that ran the map 239→359 and drove
        # the car off track). Matched cone factors still apply, so the
        # known map keeps correcting pose; unmatched observations are
        # simply dropped instead of spawned.
        self.declare_parameter("freeze_mapping_on_loop_close", True)
        # Loop closure is declared when the pose returns within
        # loop_close_radius_m of the origin (SLAM frame origin == car
        # spawn ≈ start/finish) AFTER having been at least
        # loop_close_min_radius_m away (proof a real lap happened, not
        # just start-line jitter).
        self.declare_parameter("loop_close_radius_m", 6.0)
        self.declare_parameter("loop_close_min_radius_m", 15.0)

        # --- Localization-only after loop closure ----------------------
        # Once mapping is frozen the map is final, so we stop growing the
        # smoothed iSAM2 graph and switch each scan to a fixed-size
        # pose-only solve against the frozen landmarks (FactorGraph.
        # localize). This removes the loop-closure relinearization burst
        # (the 150–250 ms scans that stalled the pose feed and let the
        # car drift off track): there is no graph growth and no
        # relinearization, so per-scan cost stays flat and constant for
        # the rest of the run. The pose prior anchors the solve at the
        # EKF-odom motion prediction; cone BearingRange factors correct
        # accumulated drift on top of it. Set False to keep the legacy
        # full-SLAM update after closure (still benefits from the
        # mapping freeze, but the burst returns).
        self.declare_parameter("localization_only_after_loop_close", True)
        # Pose-prior sigmas for the localization solve. Loose enough that
        # cones can pull the pose to correct drift, tight enough to dead-
        # reckon on the prediction when cones are sparse/rejected. Mirror
        # the EKF-odom motion confidence (xy ~5 cm/scan, yaw ~1°), widened
        # to absolute-pose scale since this anchors the absolute pose, not
        # a between-step delta.
        self.declare_parameter("loc_prior_sigma_xy_m", 0.30)
        self.declare_parameter("loc_prior_sigma_yaw_rad", 0.05)

        # --- Mission completion → /slam/finished (#384) ----------------
        # Lap/distance completion detector (see lap_counter.LapCounter).
        # Origin ≈ spawn ≈ start/finish, so lap counting reuses the
        # loop_close_* geometry above (arm >= min_radius, close <= radius).
        # Finish is gated on standstill: entering AS Finished OPENS THE SDC,
        # which cuts the EBS supply path and thereby ACTIVATES the EBS
        # (FS-Rules T14.8.1/T15.2.2). The firmware does NOT check standstill on
        # that byte (uDV#177), so signalling at speed would full-pressure stop
        # the car with the TS cut. FS rules also require standstill for
        # AS Finished. Per-mission defaults live in _resolve_finish_config
        # (autocross=1 lap, trackdrive=10, accel=75 m distance). -1 => use the
        # mission default; >= 0 overrides. Trackdrive additionally needs
        # control_node to hold its stop-anchor to the closing lap — that is
        # what /slam/final_lap drives.
        self.declare_parameter("laps_to_finish", -1)
        self.declare_parameter("finish_distance_m", -1.0)
        self.declare_parameter("finish_standstill_speed_mps", 0.5)

        # --- GT-cone debug mode (sim only) -----------------------------
        # When `debug_gt_cones` is true, SLAM ignores the *perceived*
        # cone positions on /Conos_raw and instead synthesizes body-frame
        # cone observations from the latched sim ground-truth track
        # (/testing_only/track) projected through the GT pose
        # (/testing_only/odom). The real /Conos_raw message is still used
        # as the scan trigger and timestamp, so IMU preintegration, the
        # /odom BetweenFactor cadence and everything else are byte-for-byte
        # identical to a normal run — ONLY the cone xy positions change
        # from "what perception saw" to "perfect". This is the isolation
        # test for "does SLAM lose lock because of cone quality, or in
        # spite of perfect cones?": if it still diverges here, the fault
        # is in the pose/odom fusion (IMU bias, /odom drift), not
        # perception. Requires the sim GT topics, so it only works in
        # simulation (they're hidden in competition mode). Default off,
        # but DV_SLAM_GT_CONES=1 in the environment flips the default so
        # docker-compose can carry it across container restarts (a bare
        # `ros2 param set` is lost on relaunch). An explicit launch/CLI
        # parameter still overrides the env default.
        self.declare_parameter("debug_gt_cones", _env_truthy("DV_SLAM_GT_CONES"))
        # FOV/range gate applied to the GT cones so SLAM sees roughly the
        # same cone *set* a real LiDAR scan would (not the whole track).
        # Defaults mirror the sim LiDAR (settings.json: ±60° H-FOV,
        # 0.5 m min range) and the node's MAX_OBSERVATION_RANGE_M reach.
        self.declare_parameter("debug_gt_range_m", MAX_OBSERVATION_RANGE_M)
        self.declare_parameter("debug_gt_min_range_m", 0.5)
        self.declare_parameter("debug_gt_hfov_deg", 60.0)

        # I/O references — populated in on_configure / on_activate.
        self.map_frame: str = ""
        self.odom_frame: str = ""
        self.base_frame: str = ""
        self._motion_model: str = "odom"
        self._preint: Optional[ImuPreintegrator] = None
        self._graph: Optional[FactorGraph] = None
        self._db: Optional[LandmarkDb] = None
        self._latest_result: Optional[ScanResult] = None
        self._state = State.INIT_WAITING_IMU
        self._calib_started_t: Optional[float] = None
        # Count of consecutive scans rejected by the cascade-spike
        # gate. Reset to 0 on any scan that passes through normally;
        # crossing the recovery threshold force-accepts the next scan.
        # See `cascade_skip_recovery_threshold` param.
        self._consecutive_cascade_skips: int = 0

        self._lm_capture_path: str = ""
        self._lm_capture_fh = None

        self._obs_diag: dict = {}
        self._obs_n_scans = 0
        self._obs_last_log_ns = 0

        import time as _time

        self._time = _time
        self._latest_rpm: Optional[float] = None
        self._latest_rpm_t: Optional[float] = None
        self._latest_steering_rad: Optional[float] = None
        self._latest_steering_t: Optional[float] = None
        self._latest_gt: Optional[Odometry] = None
        self._gt_init_pose: Optional[gtsam.Pose3] = None

        # GT-cone debug mode (see debug_gt_cones parameter). Read in
        # on_configure; the latched sim track is cached here when active.
        self._debug_gt_cones: bool = False
        self._latest_track: Optional[list[tuple[float, float, int]]] = None

        # Subscription handles (created in on_activate)
        self._sub_imu = None
        self._sub_cones = None
        self._sub_rpm = None
        self._sub_steering = None
        self._sub_gt = None
        self._sub_track = None

        # Subscription for sim_supervisor's /odom (Phase 2 #382 —
        # needed to compute map→odom drift correction). Cached
        # latest sample is consumed in _publish_map_to_odom on each
        # scan tick.
        self._sub_supervisor_odom = None
        self._latest_supervisor_odom: Optional[Odometry] = None
        # /odom pose at the previous SCAN tick (not the previous /odom
        # message). Used to compute the EKF's per-scan delta-pose and
        # stage it as a BetweenFactor on X(k-1)→X(k). Reset on activate.
        self._prev_scan_odom_pose: Optional[gtsam.Pose3] = None
        # Last published map→odom correction (dx, dy, dyaw). On a scan
        # with no cone correction (e.g. zero cones), we hold this
        # constant and recompose it with the live supervisor odom so
        # map→base keeps moving on the EKF instead of freezing. See
        # _ekf_holdover_result. Reset on activate/cleanup.
        self._last_correction: Optional[tuple] = None

        # Publisher / broadcaster handles (created in on_configure)
        self._tf_broadcaster = None
        self._state_pub = None
        self._cones_pub = None
        self._cones_full_pub = None
        # /Conos_full delta state: last published position per landmark id.
        # We publish only new/moved cones each scan and let the viewer hold
        # the rest (markers persist by id until replaced/deleted), so the
        # cost is O(changed) not O(map). Reset on activate so the first
        # publish of a run is a full keyframe. See _publish_full_cone_map.
        self._full_pub_last: dict[int, tuple] = {}
        self._gt_aligned_pub = None
        self._gt_error_pub = None
        # /slam/finished publisher (#384). Always emits default-false
        # on activate; will go true on real mission-completion detection
        # in a follow-up. Stays latched so a late mission_control
        # subscriber inherits the current value.
        self._finished_pub = None
        self._final_lap_pub = None
        self._stop_request_pub = None
        self._pub_hz = None
        self._pub_latency_ms = None
        self._pub_age_ms = None
        self._pub_commit_ms = None
        self._pub_map_size = None
        self._pub_n_obs = None

    # ------------------------------------------------------------------
    # Run-memory reset
    # ------------------------------------------------------------------
    def _reset_run_memory(self) -> None:
        """Drop every piece of state that carries information from a run.

        This is the single source of truth for "forget the previous
        run": the factor graph, landmark DB and IMU preintegrator are
        rebuilt from scratch, the state machine returns to
        INIT_WAITING_IMU, and all cached sensor samples / diagnostics
        are cleared. Called from on_configure, on_activate (so a fresh
        run starts clean) and on_deactivate (so a *stopped* node retains
        nothing, regardless of when — or whether — it is re-activated).
        """
        self._state = State.INIT_WAITING_IMU
        self._calib_started_t = None
        self._latest_result = None
        self._preint = ImuPreintegrator()
        self._graph = FactorGraph()
        self._db = LandmarkDb()
        self._latest_rpm = None
        self._latest_rpm_t = None
        self._latest_steering_rad = None
        self._latest_steering_t = None
        self._latest_gt = None
        self._gt_init_pose = None
        self._latest_track = None
        self._latest_supervisor_odom = None
        self._prev_scan_odom_pose = None
        self._last_correction = None
        self._obs_diag = {}
        self._obs_n_scans = 0
        self._obs_last_log_ns = 0
        self._consecutive_cascade_skips = 0
        # Loop-closure mapping freeze (see params + _update_loop_closure).
        # _mapping_frozen latches true once a lap is detected; reset when a
        # fresh run anchors at origin. _loop_max_dist tracks how far the
        # pose has ever been from origin, the "a real lap happened" proof.
        self._mapping_frozen = False
        self._loop_max_dist = 0.0
        # Monotonic scan counter for the post-freeze localization path
        # (DA staleness gating / diagnostics). Continues from the frozen
        # graph step since iSAM2's own step stops advancing once frozen.
        self._loc_scan = 0
        # /slam/finished lap/distance detector (#384). Rebuilt each run
        # from the mission's resolved completion criteria (self._finish_cfg,
        # set in on_configure) so counts start clean; a fresh node with no
        # resolved config yet gets a disabled counter (never finishes).
        self._lap_counter = LapCounter(getattr(self, "_finish_cfg", None))
        # Last value published on /slam/final_lap; None = nothing sent yet, so
        # a fresh run always re-announces even if it matches the last run's.
        self._final_lap_published = None
        # Last /slam/stop_request value published; None = nothing sent yet.
        self._stop_request_published = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        ret = super().on_configure(state)
        if ret != TransitionCallbackReturn.SUCCESS:
            return ret
        self.get_logger().info("on_configure: components + publishers + static TF")

        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        motion_model = str(self.get_parameter("motion_model").value).lower()
        if motion_model not in ("odom", "imu"):
            self.get_logger().warn(
                f"motion_model='{motion_model}' invalid — falling back to "
                f"'odom' (EKF-as-motion-model)"
            )
            motion_model = "odom"
        self._motion_model = motion_model
        if self._motion_model == "odom":
            self.get_logger().info(
                "motion_model=odom — EKF /odom delta is the primary "
                "motion constraint; IMU preintegration / RPM-prior / "
                "steering factors are DISABLED (the EKF already fuses "
                "them)."
            )
        else:
            self.get_logger().info(
                "motion_model=imu — legacy IMU-preintegration motion "
                "model with soft /odom backstop."
            )

        self._debug_gt_cones = bool(self.get_parameter("debug_gt_cones").value)
        if self._debug_gt_cones:
            self.get_logger().warn(
                "DEBUG_GT_CONES ENABLED — feeding SLAM perfect cone "
                "positions from /testing_only/track projected through the "
                "GT pose. /Conos_raw is the scan trigger only; perceived "
                "cone xy are IGNORED. SIM-ONLY diagnostic; never run on car."
            )

        # Resolve this mission's /slam/finished criteria from the behavior
        # string (== mission name) BEFORE _reset_run_memory builds the lap
        # counter from it. See _resolve_finish_config.
        self._finish_cfg = self._resolve_finish_config()

        # Components + run-memory: start from a clean slate.
        self._reset_run_memory()

        # Optional per-landmark creation diagnostic. When DV_SLAM_LANDMARK_CAPTURE
        # is set to a writable path, every new landmark dumps one JSON line
        # (id, body_x, body_y, range, color, pred_yaw_deg, step) at creation.
        # Used to verify the colour-lock-on-first-observation hypothesis (#188):
        # if many landmarks created at long range or during a yaw rotation
        # have body_y near the threshold and get locked the wrong colour, the
        # hypothesis is confirmed.
        import os as _os

        self._lm_capture_path = _os.environ.get("DV_SLAM_LANDMARK_CAPTURE", "")
        if self._lm_capture_path:
            try:
                self._lm_capture_fh = open(self._lm_capture_path, "w")
                self.get_logger().info(
                    f"DV_SLAM_LANDMARK_CAPTURE → {self._lm_capture_path}"
                )
            except OSError as ex:
                self.get_logger().error(f"landmark capture open failed: {ex}")
                self._lm_capture_fh = None

        # Publishers (lifecycle — silent until on_activate).
        # Phase 2 (#382): /cone_slam/state renamed to /slam/pose so
        # the topic doesn't bind to "cone_slam" as the algorithm
        # implementation. Frame_id flipped from odom → map (pose is
        # SLAM-absolute, drift-corrected; map→odom→base_link chain
        # resolves to the same value via TF).
        self._state_pub = self.create_lifecycle_publisher(Odometry, "/slam/pose", 10)
        self._cones_pub = self.create_lifecycle_publisher(MarkerArray, "/Conos", 10)
        # Full (uncropped) cone map for visualization only. /Conos is
        # cropped to a local radius for the planner (per-scan O(local)
        # cost); this topic carries the WHOLE map so Foxglove/RViz can
        # show the complete track. Published at low rate (every N scans —
        # see _publish_full_cone_map). QoS deliberately matches /Conos
        # (RELIABLE + VOLATILE, depth 10): the Foxglove/Lichtblick bridge
        # renders /Conos fine, but TRANSIENT_LOCAL (latched) topics
        # frequently fail to subscribe/render through the bridge (it left
        # /Conos_full at Subscription count 0 and a blank panel). VOLATILE
        # means a late-joining viewer waits one publish interval (≤ ~1 s)
        # for the next full-map frame instead of getting it instantly —
        # a fine trade for actually showing up. Not a planner input.
        self._cones_full_pub = self.create_lifecycle_publisher(
            MarkerArray, "/Conos_full", 10
        )
        # GT-aligned diagnostic odometry. Same Odometry shape as
        # /slam/pose but containing the ground truth re-expressed in
        # SLAM's anchored body-frame world. SLAM-vs-GT divergence in
        # this frame is the actual SLAM drift; in the raw GT frame it
        # would be that drift plus the static frame mismatch. Topic
        # name stays under /cone_slam/* — that's the diagnostic-only
        # namespace, separate from /slam/* which is the production
        # consumer surface.
        self._gt_aligned_pub = self.create_lifecycle_publisher(
            Odometry, "/cone_slam/gt_aligned", 10
        )
        self._gt_error_pub = self.create_lifecycle_publisher(
            Float32, "/cone_slam/gt_error_m", 10
        )

        # /slam/finished — mission-completion signal consumed by
        # mission_control_node (#384). Latched + default-false so a
        # late-joining mission_control sees an unambiguous starting
        # state. Pre-#384 slam had no way to signal mission-end
        # directly (control_node's stop-latch handled it via braking
        # to zero); post-#384 mission_control needs an explicit
        # rising-edge signal to close the RuntimeControl action with
        # outcome="finished". Currently a stub — slam still uses its
        # internal stop-anchor logic in control_node for braking;
        # this publisher just emits the default false until a future
        # PR wires the lap-min-distance + big-orange detector to it.
        finished_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._finished_pub = self.create_lifecycle_publisher(
            Bool, "/slam/finished", finished_qos
        )
        # /slam/final_lap — "the next finish gate is the closing one".
        # control_node gates its stop-anchor latch on this so trackdrive
        # brakes at lap 10 instead of lap 1. Deliberately separate from
        # /slam/finished: that one requires standstill, and the car only
        # stops BECAUSE control braked — gating the anchor on it deadlocks.
        # Same latched QoS: control must inherit the current value whenever
        # it (re)activates mid-run, not wait for the next edge.
        self._final_lap_pub = self.create_lifecycle_publisher(
            Bool, "/slam/final_lap", finished_qos
        )
        # /slam/stop_request — "criterion met, car still rolling: stop it".
        # mission_control turns this into DV_STOPPING so the uDV can brake.
        # Distinct from /slam/finished, which requires standstill — the car
        # has no service brake, so it can never reach standstill unaided and
        # /slam/finished would never fire. Latched for the same reason as its
        # siblings: a late/re-activating consumer must inherit the request,
        # not miss the edge and leave the car rolling.
        self._stop_request_pub = self.create_lifecycle_publisher(
            Bool, "/slam/stop_request", finished_qos
        )
        self._pub_hz = self.create_lifecycle_publisher(
            Float32, "/cone_slam/hz", 10
        )
        self._pub_latency_ms = self.create_lifecycle_publisher(
            Float32, "/cone_slam/latency_ms", 10
        )
        self._pub_age_ms = self.create_lifecycle_publisher(
            Float32, "/cone_slam/age_ms", 10
        )
        self._pub_commit_ms = self.create_lifecycle_publisher(
            Float32, "/cone_slam/commit_ms", 10
        )
        self._pub_map_size = self.create_lifecycle_publisher(
            Float32, "/cone_slam/map_size", 10
        )
        self._pub_n_obs = self.create_lifecycle_publisher(
            Float32, "/cone_slam/n_obs", 10
        )

        # TF broadcaster — non-lifecycle (tf2 doesn't ship lifecycle
        # variants). Used to publish the dynamic `map → odom` drift
        # correction inside _publish_map_to_odom, called per scan
        # tick from _on_cones. Pre-#382 slam owned odom→base_link
        # AND a static map→odom identity; both are retired.
        # sim_supervisor now owns odom→base_link at 100 Hz; slam owns
        # map→odom at scan rate (~10 Hz).
        self._tf_broadcaster = TransformBroadcaster(self)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(
            "on_activate: subscriptions + state-machine reset "
            f"(map='{self.map_frame}', odom='{self.odom_frame}', "
            f"base='{self.base_frame}')"
        )

        # Reset state machine and runtime state. A deactivate→activate
        # cycle should look like a fresh run, not a resumed one — the
        # IMU preintegrator can't bridge the deactivated gap, and any
        # downstream consumer reading /tf during the gap will already
        # have stale data. (on_deactivate also calls this so a stopped
        # node holds no memory of the previous run even before the next
        # activate rebuilds it.)
        self._reset_run_memory()

        # Subscriptions.
        # Bigger IMU queue than qos_profile_sensor_data's default 10:
        # at 400 Hz we get ~40 samples per 100 ms scan window, and
        # rclpy's single-threaded executor cannot service the IMU
        # callback while the cone callback's iSAM2 update is running
        # (~tens of ms). With queue=10 we drop most of the window's
        # samples and the preintegrator sees ~3-5 instead of ~40.
        # 2000 leaves headroom for several scan windows of buffer.
        imu_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2000,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_imu = self.create_subscription(Imu, "/imu", self._on_imu, imu_qos)
        # /Conos_raw is the scan trigger AND the source of cone
        # observations. Cone_Detection publishes ~10 Hz, in base_link
        # frame, with cone height encoded on marker.scale.z.
        self._sub_cones = self.create_subscription(
            MarkerArray, "/Conos_raw", self._on_cones, 10
        )

        # /motor_rpm — bridge publishes at 100 Hz from getCarState().
        # We don't fire on every sample; we cache the latest and consume
        # it inside the scan callback as a velocity prior. BEST_EFFORT
        # is fine because the cache holds the last value through any
        # single dropped sample.
        rpm_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_rpm = self.create_subscription(
            Float32, "/motor_rpm", self._on_rpm, rpm_qos
        )

        # /steering_angle — bridge publishes the LWS-derived front-wheel
        # angle in radians. Same caching pattern as RPM: latest sample
        # consumed inside the scan callback. Drives the kinematic-bicycle
        # BetweenFactor (ω = (vx/L)·tan(δ)) which constrains yaw rate
        # during cone-poor windows, including cascade-spike skips.
        self._sub_steering = self.create_subscription(
            Float32, "/steering_angle", self._on_steering, rpm_qos
        )

        # /testing_only/odom — sim ground truth, used only for the
        # GT-aligned diagnostic. The bridge encodes this in ENU
        # (East-North-Up); SLAM internally anchors to the car's initial
        # pose, so direct comparison would mix two different world
        # frames. We snapshot the GT pose at calibration-end and publish
        # a pose-aligned residual on /cone_slam/gt_aligned. Bridge
        # publishes BEST_EFFORT (per `sensor_qos` in
        # ifssim_ros_wrapper.cpp:289); subscriber must match or messages
        # silently never arrive.
        gt_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_gt = self.create_subscription(
            Odometry, "/testing_only/odom", self._on_gt_odom, gt_qos
        )

        # /odom from sim_supervisor (Phase 2 — #382). Cached and
        # consumed in _publish_map_to_odom on every scan tick to
        # compute the drift-correction transform. RELIABLE since
        # supervisor publishes at 100 Hz and we only need fresh
        # samples every ~100 ms — dropped samples would cause
        # transient map→odom inconsistency.
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._sub_supervisor_odom = self.create_subscription(
            Odometry, "/odom", self._on_supervisor_odom, odom_qos
        )

        # /testing_only/track — latched full-track GT layout, only
        # subscribed in GT-cone debug mode. The bridge publishes it
        # TRANSIENT_LOCAL (latched) at ~0.2 Hz, so a late subscriber
        # still inherits the layout; match durability or it never
        # arrives. fs_msgs is imported lazily so production builds that
        # don't ship the debug path don't hard-depend on it here.
        if self._debug_gt_cones:
            from fs_msgs.msg import Track

            track_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            self._sub_track = self.create_subscription(
                Track, "/testing_only/track", self._on_track, track_qos
            )

        # Latch /slam/finished=false on activate (#384 stub). The
        # publisher exists from on_configure; this fires the default
        # value so a subscriber that joins between configure and
        # activate sees a defined latched state.
        if self._finished_pub is not None:
            self._finished_pub.publish(Bool(data=False))
        # Same for /slam/final_lap — publish the mission's STARTING value,
        # not a blanket false: for autocross/accel/skidpad final_lap is true
        # from the first tick, and control must see that immediately or it
        # would never arm its stop anchor for those missions.
        self._publish_final_lap(force=True)
        self._publish_stop_request(force=True)

        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate: dropping subscriptions")
        for sub in (
            self._sub_imu,
            self._sub_cones,
            self._sub_rpm,
            self._sub_steering,
            self._sub_gt,
            self._sub_supervisor_odom,
            self._sub_track,
        ):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_imu = None
        self._sub_cones = None
        self._sub_rpm = None
        self._sub_steering = None
        self._sub_gt = None
        self._sub_supervisor_odom = None
        self._sub_track = None

        # Wipe all memory of the run we just stopped. The components
        # (factor graph, landmark DB, IMU preintegrator) and the cached
        # sensor samples are recreated fresh here so a deactivated node
        # carries no landmarks/poses/bias from the previous run — even
        # if the next activate is skipped by an orchestrator that thinks
        # the stack is "already prepared". on_activate calls this again
        # (cheap) so the rebuild is idempotent.
        self._reset_run_memory()
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup: destroying publishers + components")
        for sub in (
            self._sub_imu,
            self._sub_cones,
            self._sub_rpm,
            self._sub_steering,
            self._sub_gt,
            self._sub_supervisor_odom,
            self._sub_track,
        ):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_imu = None
        self._sub_cones = None
        self._sub_rpm = None
        self._sub_steering = None
        self._sub_gt = None
        self._sub_supervisor_odom = None
        self._sub_track = None

        for pub in (
            self._state_pub,
            self._cones_pub,
            self._cones_full_pub,
            self._gt_aligned_pub,
            self._gt_error_pub,
            self._finished_pub,
            self._pub_hz,
            self._pub_latency_ms,
            self._pub_age_ms,
            self._pub_commit_ms,
            self._pub_map_size,
            self._pub_n_obs,
        ):
            if pub is not None:
                self.destroy_publisher(pub)
        self._state_pub = None
        self._cones_pub = None
        self._cones_full_pub = None
        self._full_pub_last = {}
        self._gt_aligned_pub = None
        self._gt_error_pub = None
        self._finished_pub = None
        self._final_lap_pub = None
        self._stop_request_pub = None
        self._pub_hz = None
        self._pub_latency_ms = None
        self._pub_age_ms = None
        self._pub_commit_ms = None
        self._pub_map_size = None
        self._pub_n_obs = None

        # tf2 broadcasters are not lifecycle-aware; drop the ref.
        # The static map→odom broadcaster was retired in #382 (map→odom
        # is now dynamic, computed from slam_pose ⊖ /odom).
        self._tf_broadcaster = None
        self._latest_supervisor_odom = None
        self._prev_scan_odom_pose = None
        self._last_correction = None

        # Components
        self._preint = None
        self._graph = None
        self._db = None
        self._latest_result = None

        if self._lm_capture_fh is not None:
            try:
                self._lm_capture_fh.close()
            except Exception:
                pass
            self._lm_capture_fh = None

        return super().on_cleanup(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        # Reuse cleanup logic — shutdown after cleanup is a no-op.
        if self._lm_capture_fh is not None:
            try:
                self._lm_capture_fh.close()
            except Exception:
                pass
            self._lm_capture_fh = None
        return TransitionCallbackReturn.SUCCESS

    # ----- RPM callback ------------------------------------------------------

    def _on_rpm(self, msg: Float32) -> None:
        # Convert motor RPM → body-frame longitudinal velocity (m/s).
        # We cache instead of firing optimizer steps off RPM because RPM
        # arrives at 100 Hz and scans arrive at 10 Hz; one prior per
        # scan is what the factor graph wants.
        self._latest_rpm = float(msg.data) * RPM_TO_MS
        self._latest_rpm_t = self._time.monotonic()

    def _on_steering(self, msg: Float32) -> None:
        # Same caching pattern as RPM. Drives the steering-kinematic
        # BetweenFactor inside _on_cones.
        self._latest_steering_rad = float(msg.data)
        self._latest_steering_t = self._time.monotonic()

    def _on_gt_odom(self, msg: Odometry) -> None:
        # Cache the latest /testing_only/odom sample. Used only to
        # publish the SLAM-vs-GT residual on /cone_slam/gt_aligned —
        # never feeds into the factor graph (that would defeat the
        # purpose of a SLAM diagnostic).
        self._latest_gt = msg
        # Lazy alignment anchor: if SLAM has already transitioned to
        # SLAM_RUNNING but no GT sample had arrived in time for
        # _finish_calibration to snapshot one, take the first sample
        # that does arrive as the anchor. Off by < 100 ms typically;
        # immaterial for a SLAM-drift visualisation.
        if self._state == State.SLAM_RUNNING and self._gt_init_pose is None:
            self._gt_init_pose = _odom_to_pose3(msg)
            self.get_logger().info(
                f"GT alignment anchor (late): "
                f"pos=({self._gt_init_pose.x():+.2f}, {self._gt_init_pose.y():+.2f}), "
                f"yaw={np.degrees(self._gt_init_pose.rotation().yaw()):+.1f}°"
            )

    def _on_track(self, msg) -> None:
        """Cache the latched sim GT track layout (GT-cone debug mode only).

        Stored as a flat ``(world_x, world_y, color)`` list in ENU world
        frame — the same frame /testing_only/odom lives in, so
        :meth:`_gt_observations` can project it straight into base_link.
        The layout is static for the session; we just keep the latest.
        """
        track = getattr(msg, "track", None)
        if not track:
            return
        self._latest_track = [
            (float(c.location.x), float(c.location.y), int(c.color)) for c in track
        ]

    def _gt_observations(self) -> list[Observation]:
        """Body-frame cone observations from GT track + GT pose.

        Projects every cached world cone into base_link using the latest
        /testing_only/odom pose, then applies the same forward-half-plane
        / min-range / max-range / horizontal-FOV gate a real LiDAR scan
        would, so SLAM sees roughly the cone *set* it normally would —
        just at perfect positions. Returns [] until both the track and a
        GT pose have arrived (that scan then runs IMU-only, exactly like
        a perceived empty scan). sigma_xy=-1 makes SLAM use its default
        range-only weighting, identical to a perceived cone that reports
        no sigma — so only the cone xy change, not the factor weights.
        """
        if self._latest_track is None or self._latest_gt is None:
            return []
        pos = self._latest_gt.pose.pose.position
        q = self._latest_gt.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny, cosy)
        c, s = np.cos(yaw), np.sin(yaw)
        rng = float(self.get_parameter("debug_gt_range_m").value)
        min_rng = float(self.get_parameter("debug_gt_min_range_m").value)
        hfov = np.radians(float(self.get_parameter("debug_gt_hfov_deg").value))
        out: list[Observation] = []
        for cx, cy, _color in self._latest_track:
            dx = cx - pos.x
            dy = cy - pos.y
            bx = c * dx + s * dy
            by = -s * dx + c * dy
            if bx <= 0.0:  # behind the car
                continue
            r = float(np.hypot(bx, by))
            if r < min_rng or r > rng:
                continue
            if abs(np.arctan2(by, bx)) > hfov + 1e-9:
                continue
            out.append(
                Observation(
                    body_x=float(bx), body_y=float(by), height=0.0, sigma_xy=-1.0
                )
            )
        return out

    # ----- IMU callback ------------------------------------------------------

    def _on_imu(self, msg: Imu) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        accel = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
        )
        gyro = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ]
        )
        # Tag whether the car is at a genuine standstill (wheel speed ~0) so
        # the INIT bias calibration only averages stationary samples. Before
        # the first /motor_rpm arrives (_latest_rpm is None) we can't confirm a
        # standstill, so treat the sample as moving — better to default the bias
        # to zero than to soak unknown motion into it (see estimate_bias).
        stationary = (
            self._latest_rpm is not None
            and abs(self._latest_rpm) <= STATIONARY_SPEED_MS
        )
        # Buffer raw samples only while someone will consume them: the
        # INIT bias calibration (both motion models) and the legacy IMU
        # preintegration factor (motion_model=imu). In the default
        # motion_model=odom the EKF's /odom delta is the motion source and
        # integrate_to() is never called, so an unconditional push would
        # grow the preintegrator buffer without bound for the whole run
        # (~400 samples/s → ~100 MB over a 10 min session).
        if self._preint is not None and (
            self._state != State.SLAM_RUNNING or self._motion_model == "imu"
        ):
            self._preint.push_sample(
                ImuSample(t=t, accel=accel, gyro=gyro, stationary=stationary)
            )

        if self._state == State.INIT_WAITING_IMU:
            self._calib_started_t = t
            self._state = State.INIT_CALIBRATING
            cal_s = float(self.get_parameter("imu_calibration_seconds").value)
            self.get_logger().info(
                "first IMU received — INIT_CALIBRATING (stationary, "
                f"{cal_s:.1f} s)"
            )

        elif self._state == State.INIT_CALIBRATING:
            cal_s = float(self.get_parameter("imu_calibration_seconds").value)
            if self._preint.has_enough_for_calibration(cal_s):
                self._finish_calibration()

    def _finish_calibration(self) -> None:
        accel_bias, gyro_bias, gravity_body = self._preint.estimate_bias()
        self.get_logger().info(
            "calibration done — "
            f"accel_bias=({accel_bias[0]:+.4f},{accel_bias[1]:+.4f},"
            f"{accel_bias[2]:+.4f}) m/s², "
            f"gyro_bias=({gyro_bias[0]:+.5f},{gyro_bias[1]:+.5f},"
            f"{gyro_bias[2]:+.5f}) rad/s, "
            f"gravity_body=({gravity_body[0]:+.3f},{gravity_body[1]:+.3f},"
            f"{gravity_body[2]:+.3f}) m/s²"
        )

        # Anchor x_0 at world origin, stationary.
        bias = gtsam.imuBias.ConstantBias(accel_bias, gyro_bias)
        self._graph.initialize_anchor(
            initial_pose=gtsam.Pose3(),
            initial_velocity=np.zeros(3),
            initial_bias=bias,
        )
        self._latest_result = self._graph.latest()
        self._state = State.SLAM_RUNNING
        # Fresh run: re-arm loop-closure detection (graph is anchored at
        # origin, so distance-from-origin starts at 0).
        self._mapping_frozen = False
        self._loop_max_dist = 0.0
        self._loc_scan = 0
        self.get_logger().info("SLAM_RUNNING — pose graph anchored at origin")

        # Snapshot the GT pose at this instant. SLAM's pose at t=0 is
        # identity (pose graph anchored at origin); GT is at whatever
        # (ENU) world coordinates UE5 spawned the car at. Future SLAM
        # poses are in a frame whose origin == car spawn, x = car-initial
        # forward, y = car-initial left. To compare to GT we have to
        # subtract this snapshot and rotate by -initial_yaw_enu.
        if self._latest_gt is not None:
            self._gt_init_pose = _odom_to_pose3(self._latest_gt)
            self.get_logger().info(
                f"GT alignment anchor: "
                f"pos=({self._gt_init_pose.x():+.2f}, {self._gt_init_pose.y():+.2f}), "
                f"yaw={np.degrees(self._gt_init_pose.rotation().yaw()):+.1f}°"
            )
        else:
            self.get_logger().warn(
                "no /testing_only/odom received before SLAM_RUNNING — "
                "GT-aligned diagnostic disabled this run"
            )

    # ----- Cone observation callback (scan trigger) -------------------------

    def _on_cones(self, msg: MarkerArray) -> None:
        if self._state != State.SLAM_RUNNING:
            return
        if self._latest_result is None:
            return

        # Wall-clock entry time for the per-scan latency diagnostic
        # (consumed in _publish_map_to_odom). Measures how long this scan's
        # processing takes — the signal that distinguishes a real-time
        # backlog (proc time climbs into a corner) from a clean solve.
        self._on_cones_t0 = self._time.monotonic()
        # Per-scan profiling: zero the iSAM2 timers and start a fresh set of
        # wall-clock marks for the SLAM_PROF breakdown (see _emit_slam_prof).
        self._graph.reset_prof()
        self._prof_marks = {"entry": self._on_cones_t0}

        # MarkerArray has no top-level header, but every Cone_Detection
        # marker carries the originating LiDAR scan's header.stamp.
        stamp = self._stamp_msg(msg)
        if stamp is None:
            # No cones this scan, but the EKF is still moving. Publish the
            # EKF-dead-reckoned pose (held map→odom correction recomposed
            # with the current supervisor odom) so map→base keeps
            # advancing and downstream pose lookups (path planning) don't
            # freeze on the last cone-corrected scan. Also republish the
            # held cone map so the planner keeps ticking against the fresh
            # pose. No-op until we have a correction + EKF sample.
            holdover = self._ekf_holdover_result()
            if holdover is not None:
                hstamp = self._latest_supervisor_odom.header.stamp
                self._publish_map_to_odom(hstamp, holdover)
                self._publish_state(hstamp, holdover)
                self._publish_cone_map(hstamp)
            return
        t_scan = stamp.sec + stamp.nanosec * 1e-9

        # --- Motion model: advance the graph by one scan ---------------
        # Both branches stage X(k)/V(k)/B(k) plus their motion factor(s)
        # and set `predicted_pose` — the pre-optimization pose at X(k)
        # that drives data association and the pose-jump sanity check.
        if self._motion_model == "odom":
            # EKF-as-motion-model. The 9-state EKF's /odom per-scan
            # delta-pose is the PRIMARY motion constraint
            # (stage_odom_motion_step). No IMU preintegration factor, no
            # RPM velocity prior, no steering BetweenFactor — the EKF
            # already fused IMU + RPM + steering with Coriolis correction
            # and online gyro-bias tracking, so re-adding any of them
            # would double-count those sensors and re-introduce the
            # degenerate-timestamp preintegration error the EKF is immune
            # to. Cone BearingRange factors still correct accumulated
            # global drift via landmark consistency.
            if self._latest_supervisor_odom is None:
                # No EKF estimate has arrived yet — no motion source this
                # scan. Hold pose and republish so /slam/pose and the TF
                # tree don't freeze.
                self._publish_map_to_odom(stamp, self._latest_result)
                self._publish_state(stamp, self._latest_result)
                self._publish_cone_map(stamp)
                return
            cur_odom_pose = _odom_to_pose3(self._latest_supervisor_odom)
            if self._prev_scan_odom_pose is None:
                # First scan after calibration: no previous /odom sample
                # to delta against. Record this one as the baseline and
                # hold pose; the next scan advances normally.
                self._prev_scan_odom_pose = cur_odom_pose
                self._publish_map_to_odom(stamp, self._latest_result)
                self._publish_state(stamp, self._latest_result)
                self._publish_cone_map(stamp)
                return
            between_pose = self._prev_scan_odom_pose.inverse().compose(cur_odom_pose)
            # Motion prediction at X(k): prev SLAM pose ∘ EKF delta. This is
            # the same value stage_odom_motion_step seeds X(k) with; computed
            # here too because the velocity below must be expressed in the
            # SLAM (map) frame — i.e. rotated through the *predicted map yaw*,
            # not the /odom-frame yaw. The two frames differ by the map→odom
            # drift correction, so rotating with the odom yaw would put V(k)
            # (and the /slam/pose twist derived from it) off by that dyaw.
            predicted_pose = self._latest_result.pose.compose(between_pose)
            v_world = self._odom_world_velocity(
                self._latest_supervisor_odom, predicted_pose.rotation().yaw()
            )
            # Localization-only fork: once the lap is closed and mapping is
            # frozen, stop growing the smoothed graph. Solve a fixed-size
            # pose-only problem against the frozen map instead — no iSAM2
            # relinearization, so the loop-closure latency burst is gone.
            # Branch BEFORE stage_odom_motion_step so X/V/B are never
            # appended to iSAM2 (the graph stays frozen at the closure step).
            if self._mapping_frozen and bool(self.get_parameter(
                    "localization_only_after_loop_close").value):
                self._prev_scan_odom_pose = cur_odom_pose
                # predicted_pose already computed above from the EKF delta.
                # Mission-completion (#384): keep counting laps after the
                # mapping-freeze switched us to the localization-only path.
                if self._lap_counter.update(
                        predicted_pose.x(), predicted_pose.y(),
                        self._current_speed()):
                    self._publish_finished()
                self._publish_final_lap()
                self._publish_stop_request()
                self._localize_scan(msg, stamp, predicted_pose, v_world)
                return
            predicted_pose = self._graph.stage_odom_motion_step(
                between_pose, self._latest_result, v_world
            )
            self._prev_scan_odom_pose = cur_odom_pose
        else:
            # Legacy IMU-preintegration motion model (motion_model=imu).
            # Stage IMU preintegration for this scan window.
            try:
                pim, _dt = self._preint.integrate_to(t_scan)
            except RuntimeError as e:
                self.get_logger().warn(f"skip scan: {e}")
                return
            self._graph.stage_imu_factor(pim, self._latest_result)

            # Stage the motor-RPM velocity prior on V(k). This is the
            # anchor that prevents the optimizer from rotating the global
            # frame to find a cheaper minimum during cone-poor windows
            # (the cascade root cause). Skipped if RPM is stale or hasn't
            # arrived yet — the prior would be misleading rather than
            # helpful in that regime.
            rpm_fresh = (
                self._latest_rpm is not None
                and self._latest_rpm_t is not None
                and (self._time.monotonic() - self._latest_rpm_t) <= RPM_STALE_S
            )
            if rpm_fresh:
                # Use the same predicted yaw we'll use for DA below.
                _nav_state = gtsam.NavState(
                    self._latest_result.pose, self._latest_result.velocity
                )
                _pred_state = pim.predict(_nav_state, self._latest_result.bias)
                _pred_yaw = _pred_state.pose().rotation().yaw()
                self._graph.stage_velocity_prior(
                    v_body_long=self._latest_rpm,
                    predicted_yaw=_pred_yaw,
                )

                # Stage the kinematic-bicycle yaw-rate BetweenFactor.
                # This is the steering-sensor channel into the graph
                # (PR C / fix for bag _095215's 94° pose jump during
                # cascade): even when cone factors are skipped, this
                # factor anchors yaw rate to the physical steering wheel
                # via ω = (vx/L)·tan(δ). Slip-gated inside the staging
                # method — if the IMU's measured Δyaw disagrees with the
                # steering prediction by > slip_threshold, the chassis is
                # sliding and the factor is dropped to avoid corruption.
                steering_fresh = (
                    self._latest_steering_rad is not None
                    and self._latest_steering_t is not None
                    and (self._time.monotonic() - self._latest_steering_t)
                    <= RPM_STALE_S
                )
                if steering_fresh:
                    _prev_yaw = self._latest_result.pose.rotation().yaw()
                    _imu_dyaw = (_pred_yaw - _prev_yaw + np.pi) % (2 * np.pi) - np.pi
                    _gate_state = self._graph.stage_steering_between(
                        dt=_dt,
                        v_body_long=self._latest_rpm,
                        steering_rad=self._latest_steering_rad,
                        prev_pose=self._latest_result.pose,
                        imu_predicted_dyaw_rad=_imu_dyaw,
                    )
                    # Periodic diagnostic — emit a sample every ~5 s so
                    # the operator can confirm the factor is being staged
                    # (not silently slipping itself into "always slip").
                    if not hasattr(self, "_steering_factor_log_counter"):
                        self._steering_factor_log_counter = 0
                    self._steering_factor_log_counter += 1
                    if self._steering_factor_log_counter % 50 == 0:
                        self.get_logger().info(
                            f"STEERING_FACTOR state={_gate_state} "
                            f"vx={self._latest_rpm:+.2f} "
                            f"delta={self._latest_steering_rad:+.3f}rad "
                            f"imu_dyaw={_imu_dyaw*180/np.pi:+.2f}deg "
                            f"dt={_dt:.3f}s"
                        )

            # Stage the /odom-derived BetweenFactor (the "trust the EKF"
            # channel into the graph) as a SOFT backstop alongside the
            # IMU factor. /odom is the post-#534/#539 EKF's fused IMU +
            # RPM + steering estimate with Coriolis correction. During
            # cone-poor windows (the cascade-spike root cause), this
            # factor anchors X(k) to a quality estimate that's drift-
            # tracked by all available sensors — much better than SLAM's
            # IMU-only internal fallback which produced 10 m / 94° pose
            # jumps during cornering on bag _095215. First scan after
            # activate has no previous cached /odom pose; we just record
            # this one and skip the factor — the IMU+RPM priors carry the
            # load that scan.
            if self._latest_supervisor_odom is not None:
                cur_odom_pose = _odom_to_pose3(self._latest_supervisor_odom)
                if self._prev_scan_odom_pose is not None:
                    between_pose = self._prev_scan_odom_pose.inverse().compose(
                        cur_odom_pose
                    )
                    self._graph.stage_odom_between(between_pose)
                self._prev_scan_odom_pose = cur_odom_pose

            # Predicted pose at X(k) from IMU preintegration, used for DA
            # and the pose-jump sanity check (we've staged X(k) but not
            # committed yet).
            nav_state = gtsam.NavState(
                self._latest_result.pose, self._latest_result.velocity
            )
            predicted_pose = pim.predict(nav_state, self._latest_result.bias).pose()

        # Parse cone observations from /Conos_raw markers. In GT-cone
        # debug mode the perceived positions are discarded and replaced
        # with perfect cones synthesized from the sim GT track + GT pose
        # — /Conos_raw still gated the scan above, so timing is unchanged.
        if self._debug_gt_cones:
            observations = self._gt_observations()
        else:
            observations = self._observations_from_markers(msg)

        # Per-scan obs tally for the SLAM_OBS diagnostic. No colour
        # breakdown anymore — the body_y classifier is gone, every
        # cone is just a position.
        per_scan = {"obs_total": len(observations)}

        # Data association runs against the predicted body-frame position
        # of every existing landmark, using the pre-commit predicted pose
        # at X(k) (EKF-delta in odom mode, pim.predict in imu mode).
        pred_x = predicted_pose.x()
        pred_y = predicted_pose.y()
        pred_yaw = predicted_pose.rotation().yaw()

        # Loop-closure check: once a lap is detected, freeze landmark
        # spawning for the rest of the run (track is fully mapped).
        self._update_loop_closure(pred_x, pred_y)

        # Mission-completion (#384): count laps / distance and raise
        # /slam/finished when the per-mission target is met AND the car is
        # stopped. Runs on both the mapping and localization-only paths so
        # it keeps counting after the first lap freezes mapping.
        if self._lap_counter.update(pred_x, pred_y, self._current_speed()):
            self._publish_finished()
        self._publish_final_lap()
        self._publish_stop_request()

        # Mahalanobis DA stays disabled. Three variants tested on
        # 2026-04-29:
        # (1) full pose-aware Mahalanobis (4×/16× covariance inflation
        #     + 0.49 m² floor): cascaded at t≈75s. iSAM2 marginal is
        #     internal certainty not actual error; even with inflation
        #     the gate is wrong during empty-scan-driven pose drift.
        # (2) Mahalanobis gate + Euclidean Hungarian cost: same.
        # (3) Landmark-cov-only Mahalanobis (no pose Jacobian): same.
        # In every variant, mid-drive tracking was comparable to
        # Euclidean (60 s ≈ 0.8 m) but the cascade still triggered
        # at the same lap position because the cascade root cause is
        # pose drift > gate during empty-scan windows — no DA
        # strategy can fix this because there's nothing to associate.
        # Real fix needs lost-track detection / scan rejection during
        # pose-prediction-confidence collapse, not gate widening.
        # APIs in factor_graph (pose_covariance, landmark_covariance)
        # and data_association (inflation constants) stay in place
        # for future revisits.
        # Pass current_step so associate() can expand per-landmark
        # gates for landmarks that haven't been associated recently —
        # the recovery mechanism for the rejection bursts triggered
        # by improvement A.
        self._prof_marks["assoc0"] = self._time.monotonic()
        matches = associate(
            observations,
            pred_x,
            pred_y,
            pred_yaw,
            self._db,
            current_step=self._graph.step,
        )
        self._prof_marks["assoc1"] = self._time.monotonic()

        # Pre-stage cascade-trigger detection. The cascade signature
        # observed on trackA_manual_001602 around t≈80 s is: a single
        # scan flips DA from "steady, mostly-associated" to "mostly
        # new" (e.g., obs=8 new=6 assoc=2). The optimizer then jumps
        # pose to accommodate the falsely-new landmarks and the graph
        # never recovers. Detection: if the new-rate suddenly spikes
        # when (a) we have ≥5 observations to be statistically
        # meaningful, (b) we're past the early-discovery phase
        # (step > 30, so most cones in the local map are mature),
        # (c) >60 % of obs are flagged new — the predicted pose is
        # likely wrong and committing the cone factors would corrupt
        # the graph.
        #
        # Recovery (#273): commit the IMU factor only, skip the cone
        # factors. Earlier behaviour discarded EVERYTHING (the IMU
        # factor too) and returned, freezing the graph at the prev
        # committed pose. While the car physically moved during the
        # skipped scans, the predicted pose for the next scan stayed
        # stale — so when DA recovered, observed cones were all far
        # from their landmarks (even further apart than the real drift
        # the IMU would have indicated), producing yet more "all-new"
        # scans, more skips, more drift. By committing the IMU we keep
        # pose dead-reckoning during the skipped window; drift over a
        # few hundred ms of IMU-only update is much smaller than over
        # the same window of pose-freeze.
        n_new_pre = sum(1 for m in matches if m.landmark_id == -1)
        total_pre = len(matches)
        # Two complementary triggers, both gated on step > 30
        # (post-discovery — the local cone map is mature, every
        # observation should be re-association of a known landmark
        # with high probability):
        #
        #   A. **Percentage trigger** (legacy): >60 % of the scan's
        #      observations flagged new. Catches "everything looks new"
        #      cascades — observed pre-#441 on bags with extreme drift.
        #      Requires ≥5 obs to be statistically meaningful.
        #
        #   B. **Count trigger** (post-#441 finding from in-loop runs):
        #      ≥ N new in a single scan, where N is small. Catches the
        #      5-of-14 (36 %) and 8-of-13 (62 %) bursts that slipped
        #      through the percentage gate in lap_postveto_20260511_142317
        #      and lap_ekf_inloop_20260511_235510. FSD cone spacing
        #      means legitimate new arrivals come at most 1 per ~3
        #      scans at typical speeds — anything ≥ N=3 in one scan
        #      is anomalous. Default 3, tunable via parameter, 0
        #      disables.
        n_count_thresh = int(
            self.get_parameter("cascade_spike_new_count_threshold").value
        )
        triggered, reasons = cascade_spike_triggered(
            n_new_pre, total_pre, self._graph.step, count_threshold=n_count_thresh
        )

        # Cascade-skip recovery: if we've been skipping continuously
        # for too long, force-accept this scan. Long-stretch skips
        # are almost certainly new-territory exploration (legitimate
        # cones not yet in the map), not an adversarial cascade.
        # Refusing forever strands SLAM in IMU-only dead reckoning
        # and the /slam/pose feed freezes for the controller.
        if triggered:
            self._consecutive_cascade_skips += 1
            recovery_thresh = int(
                self.get_parameter("cascade_skip_recovery_threshold").value
            )
            if (
                recovery_thresh > 0
                and self._consecutive_cascade_skips > recovery_thresh
            ):
                self.get_logger().warn(
                    f"CASCADE_SKIP_RECOVERY: force-accepting scan "
                    f"after {self._consecutive_cascade_skips} "
                    f"consecutive skips — assuming new-territory "
                    f"exploration (obs={total_pre} new={n_new_pre} "
                    f"assoc={total_pre - n_new_pre}); proximity veto "
                    f"still gates individual cones."
                )
                self._consecutive_cascade_skips = 0
                triggered = False  # fall through to the spawn path

        if triggered:
            motion_label = "EKF-odom" if self._motion_model == "odom" else "IMU"
            self.get_logger().warn(
                f"skip cone factors: DA-failure spike "
                f"[{', '.join(reasons)}] "
                f"(obs={total_pre} new={n_new_pre} "
                f"assoc={total_pre - n_new_pre}) — {motion_label}-only update"
            )
            # Commit the staged motion factor(s) (the only things staged
            # at this point — cone factors are staged only after this
            # check). iSAM2 advances pose by the motion model's
            # prediction; no cone constraints applied this scan.
            result = self._graph.commit()
            self._prof_marks["commit"] = self._time.monotonic()
            self._latest_result = result
            self._preint.update_bias(result.bias)
            self._db.update_from_estimate(self._graph.landmark_position)
            self._prof_marks["dbupd"] = self._time.monotonic()
            self._publish_map_to_odom(stamp, result)
            self._publish_state(stamp, result)
            self._publish_cone_map(stamp)
            self._emit_slam_prof("skip", len(observations))
            per_scan["skipped"] = 1
            self._accumulate_obs_diag(per_scan)
            return
        else:
            # Scan passed the gate cleanly — reset the consecutive-skip
            # counter so a fresh stretch of skips later can be detected.
            self._consecutive_cascade_skips = 0

        # For each matched obs → factor between current pose and the
        # known landmark. For unmatched → allocate a new landmark and
        # add a factor to it. Before allocation, apply the proximity
        # veto: refuse to spawn a landmark within
        # `new_landmark_proximity_veto_m` of any existing one (cascade
        # guard — see parameter docstring).
        veto_m = float(self.get_parameter("new_landmark_proximity_veto_m").value)
        n_new = 0
        n_assoc = 0
        n_vetoed = 0
        n_frozen_drop = 0
        self._prof_marks["spawn0"] = self._time.monotonic()
        for o, m in zip(observations, matches):
            if m.landmark_id == -1:
                # Mapping frozen post-loop-closure: the track is fully
                # mapped, so an unmatched observation is far more likely a
                # transient DA miss (stale pose during the relinearization
                # burst) than a genuinely new cone. Drop it instead of
                # spawning — no phantom landmark can bloat the map.
                if self._mapping_frozen:
                    n_frozen_drop += 1
                    continue
                world_xyz = self._body_to_world(
                    o.body_x, o.body_y, pred_x, pred_y, pred_yaw
                )
                if veto_m > 0.0:
                    d_near = self._db.nearest_xy_distance_m(world_xyz)
                    if d_near < veto_m:
                        n_vetoed += 1
                        continue
                lm = self._db.create(world_xyz, self._graph.step)
                self._graph.stage_new_landmark(lm.id, world_xyz)
                self._graph.stage_cone_observation(
                    lm.id, o.body_x, o.body_y, o.sigma_xy
                )
                n_new += 1
                if self._lm_capture_fh is not None:
                    try:
                        import json as _json
                        import math as _math

                        rng = _math.hypot(o.body_x, o.body_y)
                        bearing_deg = _math.degrees(_math.atan2(o.body_y, o.body_x))
                        self._lm_capture_fh.write(
                            _json.dumps(
                                {
                                    "id": lm.id,
                                    "step": self._graph.step,
                                    "body_x": o.body_x,
                                    "body_y": o.body_y,
                                    "range_m": rng,
                                    "bearing_deg": bearing_deg,
                                    "height": o.height,
                                    "pose_yaw_deg": _math.degrees(pred_yaw),
                                    "world_xyz": [
                                        float(world_xyz[0]),
                                        float(world_xyz[1]),
                                        float(world_xyz[2]),
                                    ],
                                }
                            )
                            + "\n"
                        )
                        self._lm_capture_fh.flush()
                    except Exception:
                        pass
            else:
                self._db.mark_observed(m.landmark_id, self._graph.step)
                self._graph.stage_cone_observation(
                    m.landmark_id, o.body_x, o.body_y, o.sigma_xy
                )
                n_assoc += 1
        self._prof_marks["spawn1"] = self._time.monotonic()
        per_scan["new"] = n_new
        per_scan["assoc"] = n_assoc
        per_scan["vetoed"] = n_vetoed
        per_scan["frozen_drop"] = n_frozen_drop
        if n_vetoed > 0:
            self.get_logger().info(
                f"proximity-veto: dropped {n_vetoed} would-be-new "
                f"landmark(s) within {veto_m:.2f} m of existing ones "
                f"(cascade guard; obs={len(observations)} "
                f"new={n_new} assoc={n_assoc})"
            )
        if n_frozen_drop > 0:
            self.get_logger().info(
                f"mapping-frozen: dropped {n_frozen_drop} unmatched "
                f"observation(s) post-loop-closure "
                f"(obs={len(observations)} assoc={n_assoc})"
            )

        self._accumulate_obs_diag(per_scan)

        # Commit IMU + cone factors with a post-commit pose-jump
        # sanity check (#273 follow-up). The cascade detector above
        # catches *symptoms* — bursts of all-NEW observations — but
        # only AFTER a bad cone match has already snapped pose. The
        # sanity check below catches the *cause*: an iSAM2 update
        # that pushes pose far from where IMU prediction says we are.
        # When it fires, a strong prior at the IMU-predicted pose is
        # added and the graph is re-optimized; pose at this step lands
        # near IMU prediction instead of where the bad cone factor
        # tried to drag it.
        # Gate the sanity check the same way the cascade detector is
        # gated: only fire after the early-discovery phase
        # (`step > 30`). Reason: iSAM2 refines the IMU bias estimate
        # over the first ~30 scans, during which the optimized pose
        # legitimately deviates from the IMU prediction by tens of cm
        # as it incorporates the first cone constraints. Triggering
        # the corrective prior in that window pins the pose to the
        # uncalibrated-bias prediction and prevents iSAM2 from
        # converging.
        max_pos_dev_m = self.get_parameter("pose_jump_max_pos_m").value
        max_yaw_dev_rad = self.get_parameter("pose_jump_max_yaw_rad").value
        if self._graph.step > 30:
            result, was_corrected = self._graph.commit_with_pose_sanity_check(
                predicted_pose, max_pos_dev_m, max_yaw_dev_rad
            )
            if was_corrected:
                self.get_logger().warn(
                    f"pose-jump rejected: snapped to IMU prediction at "
                    f"step={self._graph.step} "
                    f"(thresholds: {max_pos_dev_m:.2f} m, "
                    f"{np.degrees(max_yaw_dev_rad):.1f}°)"
                )
        else:
            result = self._graph.commit()
        self._prof_marks["commit"] = self._time.monotonic()
        self._latest_result = result
        self._preint.update_bias(result.bias)

        # Refresh the working landmark estimates so the next DA step
        # uses iSAM2-corrected positions, not stale initial guesses.
        self._db.update_from_estimate(self._graph.landmark_position)
        self._prof_marks["dbupd"] = self._time.monotonic()

        self._publish_map_to_odom(stamp, result)
        self._publish_state(stamp, result)
        self._publish_cone_map(stamp)
        self._emit_slam_prof("ok", len(observations))

        # Quiet log every 10 scans (~1 Hz at 10 Hz LiDAR).
        if self._graph.step % 10 == 0:
            self.get_logger().info(
                f"step={self._graph.step} "
                f"obs={len(observations)} new={n_new} assoc={n_assoc} "
                f"map={len(self._db)} "
                f"pose=({result.pose.x():+.2f},{result.pose.y():+.2f},"
                f"yaw={np.degrees(result.pose.rotation().yaw()):+.1f}°)"
            )

        # Bias trajectory dump every 50 scans (~5 s wall, ~5 % of a lap).
        # Tagged so it greps cleanly out of the SLAM log: "BIAS step=…".
        # Used to diagnose whether iSAM2 is letting the bias drift away
        # from the calibration value over the lap, vs. holding it locked
        # by BIAS_RW_SIGMAS being too tight.
        if self._graph.step % 50 == 0:
            ab = result.bias.accelerometer()
            gb = result.bias.gyroscope()
            v = result.velocity
            self.get_logger().info(
                f"BIAS step={self._graph.step} "
                f"accel=({ab[0]:+.5f},{ab[1]:+.5f},{ab[2]:+.5f}) m/s² "
                f"gyro=({gb[0]:+.6f},{gb[1]:+.6f},{gb[2]:+.6f}) rad/s "
                f"vel=({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}) m/s "
                f"|v|={float(np.linalg.norm(v)):.3f}"
            )

    # ----- helpers ----------------------------------------------------------

    def _accumulate_obs_diag(self, per_scan: dict) -> None:
        """Accumulate per-scan obs/assoc/new counters and emit a
        per-second SLAM_OBS log line. Compares observations entering
        SLAM with what survives data association."""
        for k, v in per_scan.items():
            self._obs_diag[k] = self._obs_diag.get(k, 0) + v
        self._obs_n_scans += 1
        now_ns = self.get_clock().now().nanoseconds
        if self._obs_last_log_ns == 0:
            self._obs_last_log_ns = now_ns
            return
        if now_ns - self._obs_last_log_ns < 1_000_000_000:
            return
        n = self._obs_n_scans
        if n <= 0:
            return
        d = self._obs_diag

        def _avg(key: str) -> float:
            return d.get(key, 0) / n

        self.get_logger().info(
            f"SLAM_OBS (avg/scan over {n}): "
            f"obs={_avg('obs_total'):4.1f} "
            f"assoc={_avg('assoc'):4.1f} "
            f"new={_avg('new'):3.1f} "
            f"vetoed={_avg('vetoed'):.1f} "
            f"frozen={_avg('frozen_drop'):.1f} "
            f"skip={_avg('skipped'):.1f}"
        )
        self._obs_diag = {}
        self._obs_n_scans = 0
        self._obs_last_log_ns = now_ns

    @staticmethod
    def _stamp_msg(msg: MarkerArray):
        """Return the first non-DELETE marker's header.stamp, or None
        if the array is empty / DELETEALL only.

        MarkerArray has no top-level header, but Cone_Detection sets
        every marker's stamp from the originating LiDAR scan, so any
        of them is fine.
        """
        for m in msg.markers:
            if m.action != Marker.DELETEALL:
                return m.header.stamp
        return None

    @staticmethod
    def _observations_from_markers(msg: MarkerArray) -> list[Observation]:
        out: list[Observation] = []
        for m in msg.markers:
            # The first marker in the array is action=DELETEALL with
            # placeholder pose — Cone_Detection uses it to clear stale
            # markers in RViz/Foxglove. Skip it.
            if m.action == Marker.DELETEALL:
                continue
            x = m.pose.position.x
            y = m.pose.position.y
            if (x * x + y * y) > MAX_OBSERVATION_RANGE_M * MAX_OBSERVATION_RANGE_M:
                continue
            height = m.scale.z if m.scale.z > 0 else 0.0
            # Detection encodes per-cone σ_xy (metres) on scale.x; the
            # legacy 0.1 default flags "no σ reported" and SLAM falls
            # back to its range-only formula for backward compat.
            sigma_xy = m.scale.x if (m.scale.x > 0.0 and m.scale.x != 0.1) else -1.0
            out.append(
                Observation(
                    body_x=x,
                    body_y=y,
                    height=height,
                    sigma_xy=sigma_xy,
                )
            )
        return out

    @staticmethod
    def _body_to_world(
        body_x: float,
        body_y: float,
        pose_x: float,
        pose_y: float,
        pose_yaw: float,
    ) -> np.ndarray:
        """Project a body-frame xy into world-frame xyz (z=0)."""
        c = np.cos(pose_yaw)
        s = np.sin(pose_yaw)
        return np.array(
            [
                pose_x + body_x * c - body_y * s,
                pose_y + body_x * s + body_y * c,
                0.0,
            ]
        )

    @staticmethod
    def _odom_world_velocity(msg: Odometry, map_yaw: float) -> np.ndarray:
        """SLAM-frame (map / nav-frame) velocity from a nav_msgs/Odometry.

        GTSAM's V(k) lives in the navigation frame, but nav_msgs/Odometry
        twist is expressed in child_frame_id (base_link / body). Rotate
        the planar body velocity through ``map_yaw`` — the car's yaw in
        the SLAM map frame (the motion-predicted yaw at X(k)) — NOT the
        yaw carried in ``msg.pose`` (that one is in the ``odom`` frame,
        which differs from ``map`` by the map→odom drift correction).
        z is taken as-is (flat track). Used to seed and softly anchor
        V(k) in ``motion_model='odom'`` so the velocity node stays
        determined without an IMU factor.
        """
        yaw = float(map_yaw)
        vb = msg.twist.twist.linear
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                c * vb.x - s * vb.y,
                s * vb.x + c * vb.y,
                float(vb.z),
            ]
        )

    # ----- output ------------------------------------------------------------

    def _on_supervisor_odom(self, msg: Odometry) -> None:
        """Cache sim_supervisor's latest /odom sample for map→odom math."""
        self._latest_supervisor_odom = msg

    def _ekf_holdover_result(self) -> Optional[ScanResult]:
        """EKF-dead-reckoned pose for a scan with no cone correction.

        Holds the most recent `map→odom` correction constant and
        recomposes it with the *current* supervisor odom
        (`odom→base_link`), so `map→base` keeps advancing on the EKF's
        dead-reckoning instead of freezing. This is the inverse of
        `compute_map_to_odom`:

            T_map_base = T_map_odom · T_odom_base
            ⇒ base = Δpos + R(Δyaw) · sup_pos,  yaw = Δyaw + sup_yaw

        Why this is needed: republishing the held *absolute* pose would
        make `_publish_map_to_odom` recompute the correction against the
        moved supervisor odom — absorbing and cancelling the EKF motion,
        i.e. a freeze. Holding the *correction* constant and letting
        `map→base` move is what keeps downstream pose lookups (path
        planning) alive during cone-poor windows.

        Returns None when there's no cached correction, no supervisor
        odom, or no prior result to clone velocity/bias from.
        """
        if (
            self._last_correction is None
            or self._latest_supervisor_odom is None
            or self._latest_result is None
        ):
            return None
        dx, dy, dyaw = self._last_correction
        sup = self._latest_supervisor_odom.pose.pose
        sup_x = sup.position.x
        sup_y = sup.position.y
        sup_yaw = 2.0 * np.arctan2(sup.orientation.z, sup.orientation.w)
        c, s = np.cos(dyaw), np.sin(dyaw)
        bx = dx + c * sup_x - s * sup_y
        by = dy + s * sup_x + c * sup_y
        byaw = (dyaw + sup_yaw + np.pi) % (2.0 * np.pi) - np.pi
        pose = gtsam.Pose3(
            gtsam.Rot3.Yaw(float(byaw)),
            gtsam.Point3(float(bx), float(by), self._latest_result.pose.z()),
        )
        return ScanResult(
            pose=pose,
            velocity=self._latest_result.velocity,
            bias=self._latest_result.bias,
        )

    def _publish_map_to_odom(self, stamp, result: ScanResult) -> None:
        """Broadcast the dynamic `map → odom` transform (Phase 2 #382).

        We need a SE(2) transform that, composed with the supervisor's
        latest `odom → base_link`, yields slam's absolute pose in map:

            T_map_base   = T_map_odom · T_odom_base
            ⇒ T_map_odom = T_map_base · T_odom_base⁻¹

        In 2D this is:
            Δyaw = slam_yaw - odom_yaw
            Δpos = slam_pos - R(Δyaw) · odom_pos

        Fallback: until the supervisor's `/odom` is flowing (the
        first ~3 s after activate, during the filter's stationary
        calibration window), broadcast map → odom as identity. This
        keeps Lichtblick's TF tree rooted; the chain map→odom→base_link
        is fully dynamic post-Phase-2, no /tf_static needed.
        """
        slam_pose = result.pose
        slam_x = slam_pose.x()
        slam_y = slam_pose.y()
        slam_yaw = slam_pose.rotation().yaw()

        if self._latest_supervisor_odom is None:
            # Identity fallback during the supervisor calibration window
            # ^ if supervisor hasn't started, treat its odom frame as
            # coincident with map; map→odom = slam_pose itself, so
            # downstream consumers still see slam's pose at the leaf.
            dx, dy, dyaw = slam_x, slam_y, slam_yaw
        else:
            sup = self._latest_supervisor_odom.pose.pose
            sup_x = sup.position.x
            sup_y = sup.position.y
            # Supervisor's quaternion is axis-z only (2D yaw); the
            # full-precision recovery is 2·atan2(qz, qw). q.w can be
            # negative but yaw stays in (-π, π] from atan2.
            sup_yaw = 2.0 * np.arctan2(sup.orientation.z, sup.orientation.w)
            dx, dy, dyaw = compute_map_to_odom(
                slam_x,
                slam_y,
                slam_yaw,
                sup_x,
                sup_y,
                sup_yaw,
            )

        # Cache the correction so a no-cone scan can hold it constant and
        # recompose it with the live EKF odom (see _ekf_holdover_result),
        # keeping map→base moving when there's no cone correction to apply.
        self._last_correction = (float(dx), float(dy), float(dyaw))

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = float(dx)
        t.transform.translation.y = float(dy)
        t.transform.translation.z = 0.0
        half = 0.5 * float(dyaw)
        t.transform.rotation.w = float(np.cos(half))
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(np.sin(half))
        self._tf_broadcaster.sendTransform(t)

        # --- Per-scan latency / correction diagnostic --------------------
        # Distinguishes the two corner-failure mechanisms (see #382 thread):
        #   proc_ms  — wall time spent processing this scan. Climbs into a
        #              corner ⇒ real-time backlog (single-threaded executor
        #              falling behind), which makes the published stamp stale.
        #   age_ms   — ROS-time gap between now and this scan's stamp; this
        #              is the staleness path planning's TF lookup actually
        #              sees. Grows with the backlog.
        #   corr     — magnitude of the map→odom drift correction. A spike
        #              here (with age flat) means a genuine SLAM pose jump,
        #              not a timing problem.
        #   dt_pub   — wall gap since the previous publish (effective scan
        #              cadence).
        # Logged every scan; grep `SLAM_LAT` and plot the columns.
        now_mono = self._time.monotonic()
        proc_ms = (now_mono - getattr(self, "_on_cones_t0", now_mono)) * 1e3
        dt_pub_ms = (now_mono - getattr(self, "_last_publish_mono", now_mono)) * 1e3
        self._last_publish_mono = now_mono
        scan_s = stamp.sec + stamp.nanosec * 1e-9
        age_ms = (self.get_clock().now().nanoseconds * 1e-9 - scan_s) * 1e3
        corr_mag = float(np.hypot(dx, dy))
        self.get_logger().info(
            f"SLAM_LAT proc={proc_ms:6.1f}ms age={age_ms:7.1f}ms "
            f"dt_pub={dt_pub_ms:6.1f}ms "
            f"corr=({dx:+.2f},{dy:+.2f}|{corr_mag:.2f}m,"
            f"{np.degrees(dyaw):+.1f}deg)"
        )
        hz = 1000.0 / dt_pub_ms if dt_pub_ms > 1e-3 else 0.0
        self._publish_f32(self._pub_hz, hz)
        self._publish_f32(self._pub_latency_ms, proc_ms)
        self._publish_f32(self._pub_age_ms, age_ms)
        self._publish_f32(self._pub_map_size, float(len(self._db)))

    def _emit_slam_prof(self, tag: str, n_obs: int) -> None:
        """Emit the per-scan latency breakdown (SLAM_PROF) so we can see
        WHICH stage eats the time as the map grows.

        Segments (wall-clock, ms):
          pre    — motion staging + marker parse (entry → assoc start)
          assoc  — data association (associate())
          spawn  — cascade detection + spawn/veto loop (incl. the O(map)
                   proximity veto run per would-be-new cone)
          commit — graph commit; broken out into the two iSAM2 primitives
                   it actually runs: upd=both isam.update() calls,
                   est=calculateEstimate() back-substitution, x{n} flushes
                   (2 when the pose-jump sanity check re-solves)
          db     — update_from_estimate sweep over every landmark
          pub    — map→odom / state / cone-map publishes

        Grep `SLAM_PROF` and plot against map size to find the slope.
        """
        m = getattr(self, "_prof_marks", None)
        if not m or "entry" not in m:
            return
        now = self._time.monotonic()

        def seg(a: str, b: str) -> float:
            if a in m and b in m:
                return (m[b] - m[a]) * 1e3
            return float("nan")

        total = (now - m["entry"]) * 1e3
        pre = seg("entry", "assoc0")
        assoc = seg("assoc0", "assoc1")
        # spawn loop only runs on the "ok" path; absent on cascade skip.
        spawn = seg("spawn0", "spawn1")
        # commit segment starts wherever the spawn loop ended ("ok") or
        # right after association ("skip").
        commit_start = "spawn1" if "spawn1" in m else "assoc1"
        commit = seg(commit_start, "commit")
        db = seg("commit", "dbupd")
        pub = (now - m["dbupd"]) * 1e3 if "dbupd" in m else float("nan")

        # Structured sink for offline profiling (run_pipeline_benchmark):
        # the replay sets `_prof_sink` to a callable and receives exactly
        # the numbers logged below, without having to parse SLAM_PROF
        # lines. Never set in production (attribute defaults to None).
        sink = getattr(self, "_prof_sink", None)
        if sink is not None:
            sink(
                {
                    "tag": tag,
                    "total_ms": total,
                    "pre_ms": pre,
                    "assoc_ms": assoc,
                    "spawn_ms": spawn,
                    "commit_ms": commit,
                    "isam_update_ms": self._graph.prof_update_ms,
                    "calc_estimate_ms": self._graph.prof_calcest_ms,
                    "flushes": float(self._graph.prof_flushes),
                    "db_ms": db,
                    "pub_ms": pub,
                    "n_obs": float(n_obs),
                    "map_size": float(len(self._db)),
                }
            )

        self.get_logger().info(
            f"SLAM_PROF[{tag}] total={total:6.1f}ms "
            f"pre={pre:5.1f} assoc={assoc:5.1f} spawn={spawn:5.1f} "
            f"commit={commit:6.1f}(upd={self._graph.prof_update_ms:6.1f} "
            f"est={self._graph.prof_calcest_ms:6.1f} "
            f"x{self._graph.prof_flushes}) "
            f"db={db:5.1f} pub={pub:5.1f} "
            f"obs={n_obs} map={len(self._db)}"
        )
        if commit == commit:  # skip NaN (cascade-skip path has no commit)
            self._publish_f32(self._pub_commit_ms, commit)
        self._publish_f32(self._pub_n_obs, float(n_obs))

    def _publish_f32(self, pub, value: float) -> None:
        if pub is None:
            return
        getter = getattr(pub, "get_subscription_count", None)
        if getter is not None:
            try:
                if int(getter()) <= 0:
                    return
            except Exception:
                pass
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)

    def _publish_state(self, stamp, result: ScanResult) -> None:
        msg = Odometry()
        msg.header.stamp = stamp
        # Phase 2 (#382): pose is SLAM's absolute estimate — map frame,
        # not odom. Twist still lives in child_frame_id (base_link).
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.base_frame
        pose = result.pose
        msg.pose.pose.position.x = pose.x()
        msg.pose.pose.position.y = pose.y()
        msg.pose.pose.position.z = pose.z()
        q = pose.rotation().toQuaternion()
        msg.pose.pose.orientation.w = q.w()
        msg.pose.pose.orientation.x = q.x()
        msg.pose.pose.orientation.y = q.y()
        msg.pose.pose.orientation.z = q.z()
        # nav_msgs/Odometry semantics: twist is expressed in child_frame_id
        # (here base_link), NOT the header frame. GTSAM's NavState carries
        # velocity in the navigation (odom) frame, so we project to body
        # frame before publishing — otherwise consumers like Control read
        # `twist.linear.x` as longitudinal speed and instead get an axis-
        # aligned world component, which is wrong as soon as the car is
        # not pointing along world +X.
        v_world = result.velocity
        c, s = np.cos(pose.rotation().yaw()), np.sin(pose.rotation().yaw())
        # R_w2b = [[ c, s, 0], [-s, c, 0], [0, 0, 1]]; vertical untouched.
        msg.twist.twist.linear.x = float(c * v_world[0] + s * v_world[1])
        msg.twist.twist.linear.y = float(-s * v_world[0] + c * v_world[1])
        msg.twist.twist.linear.z = float(v_world[2])
        self._state_pub.publish(msg)

        # GT-aligned diagnostic: publish where the ground truth says
        # the car is, expressed in SLAM's anchored body-frame world,
        # plus the position-error magnitude. Both are zero at t=0 by
        # construction; non-zero values are real SLAM drift, not a
        # frame-mismatch artifact.
        self._publish_gt_aligned(stamp, msg)

    def _publish_gt_aligned(self, stamp, slam_msg: Odometry) -> None:
        """Re-express ground truth in SLAM's anchored frame and publish.

        SLAM internally anchors at identity, so the SLAM frame is:
            origin = car spawn (the ENU pose at calibration end)
            +x axis = car's initial forward direction
            +y axis = car's initial left direction
        The bridge publishes /testing_only/odom in ENU. To compare like-
        for-like, we subtract the snapshot pose from each GT sample and
        rotate by -initial_yaw_enu.
        """
        if self._gt_init_pose is None or self._latest_gt is None:
            return
        gt_now = _odom_to_pose3(self._latest_gt)
        # gt_in_slam_frame = init⁻¹ · gt_now
        gt_aligned = self._gt_init_pose.inverse().compose(gt_now)

        out = Odometry()
        out.header.stamp = stamp
        # GT-aligned diagnostic lives in the same frame as /slam/pose
        # so Lichtblick can plot them on the same axis post-Phase-2.
        out.header.frame_id = self.map_frame
        out.child_frame_id = "gt"
        out.pose.pose.position.x = gt_aligned.x()
        out.pose.pose.position.y = gt_aligned.y()
        out.pose.pose.position.z = gt_aligned.z()
        q = gt_aligned.rotation().toQuaternion()
        out.pose.pose.orientation.w = q.w()
        out.pose.pose.orientation.x = q.x()
        out.pose.pose.orientation.y = q.y()
        out.pose.pose.orientation.z = q.z()
        self._gt_aligned_pub.publish(out)

        # Position-error magnitude in metres. Plot this on a second
        # axis to track drift growth without having to do the
        # subtraction in the visualiser.
        dx = slam_msg.pose.pose.position.x - gt_aligned.x()
        dy = slam_msg.pose.pose.position.y - gt_aligned.y()
        err = float(np.hypot(dx, dy))
        err_msg = Float32()
        err_msg.data = err
        self._gt_error_pub.publish(err_msg)

    def _localize_scan(self, msg, stamp, predicted_pose, v_world) -> None:
        """Process one scan in frozen-map localization mode.

        Replaces the growing-graph SLAM update once the lap is closed: it
        runs the same data association as the SLAM path, then solves a
        fixed-size pose-only problem against the frozen landmarks
        (FactorGraph.localize) instead of advancing iSAM2. No graph
        growth, no relinearization — so the loop-closure latency burst
        that stalled the pose feed and drove the car off track is gone.
        Cone factors still correct accumulated drift on top of the
        EKF-odom motion prediction; unmatched observations are dropped
        (the map is final). The published /slam/pose, map→odom TF and
        /Conos crop are byte-identical in shape to the SLAM path, so
        downstream consumers see no change beyond a steadier feed.
        """
        self._graph.reset_prof()
        self._prof_marks = {"entry": self._on_cones_t0}

        if self._debug_gt_cones:
            observations = self._gt_observations()
        else:
            observations = self._observations_from_markers(msg)
        per_scan = {"obs_total": len(observations)}

        pred_x = predicted_pose.x()
        pred_y = predicted_pose.y()
        pred_yaw = predicted_pose.rotation().yaw()

        # Monotonic step continuing past the frozen graph step so DA's
        # staleness gating and landmark last-seen bookkeeping keep ticking.
        self._loc_scan += 1
        cur_step = self._graph.step + self._loc_scan

        self._prof_marks["assoc0"] = self._time.monotonic()
        matches = associate(
            observations, pred_x, pred_y, pred_yaw, self._db,
            current_step=cur_step,
        )
        self._prof_marks["assoc1"] = self._time.monotonic()

        # Collect matched observations + their frozen world positions for
        # the localization solve. Unmatched obs are dropped (map is final).
        self._prof_marks["spawn0"] = self._time.monotonic()
        cone_obs = []
        n_assoc = 0
        n_drop = 0
        for o, m in zip(observations, matches):
            if m.landmark_id == -1:
                n_drop += 1
                continue
            lm = self._db.get(m.landmark_id)
            self._db.mark_observed(m.landmark_id, cur_step)
            cone_obs.append((o.body_x, o.body_y, o.sigma_xy, lm.position))
            n_assoc += 1
        self._prof_marks["spawn1"] = self._time.monotonic()

        sigma_xy = float(self.get_parameter("loc_prior_sigma_xy_m").value)
        sigma_yaw = float(self.get_parameter("loc_prior_sigma_yaw_rad").value)
        prior_sigmas = np.array([
            0.05, 0.05, sigma_yaw,   # roll, pitch, yaw
            sigma_xy, sigma_xy, 0.05,  # x, y, z
        ])
        result = self._graph.localize(
            predicted_pose, cone_obs, v_world, self._latest_result.bias,
            prior_sigmas,
        )

        # Safety clamp: a cluster of bad associations could still pull the
        # solve past the prior. If the solved pose deviates from the motion
        # prediction beyond the pose-jump thresholds, fall back to pure
        # dead-reckoning this scan (mirrors commit_with_pose_sanity_check).
        max_pos = float(self.get_parameter("pose_jump_max_pos_m").value)
        max_yaw = float(self.get_parameter("pose_jump_max_yaw_rad").value)
        pos_dev = float(np.linalg.norm(
            result.pose.translation() - predicted_pose.translation()))
        yaw_dev = abs(float(
            predicted_pose.rotation().between(result.pose.rotation()).yaw()))
        if pos_dev > max_pos or yaw_dev > max_yaw:
            self.get_logger().warn(
                f"localize pose-jump rejected: dev=({pos_dev:.2f} m, "
                f"{np.degrees(yaw_dev):.1f}°) > ({max_pos:.2f} m, "
                f"{np.degrees(max_yaw):.1f}°) — holding motion prediction"
            )
            result = ScanResult(
                pose=predicted_pose, velocity=v_world,
                bias=self._latest_result.bias)
        self._prof_marks["commit"] = self._time.monotonic()
        self._latest_result = result
        self._prof_marks["dbupd"] = self._time.monotonic()

        self._publish_map_to_odom(stamp, result)
        self._publish_state(stamp, result)
        self._publish_cone_map(stamp)

        per_scan["new"] = 0
        per_scan["assoc"] = n_assoc
        per_scan["vetoed"] = 0
        per_scan["frozen_drop"] = n_drop
        self._accumulate_obs_diag(per_scan)
        self._emit_slam_prof("loc", len(observations))

    def _update_loop_closure(self, pose_x: float, pose_y: float) -> None:
        """Detect lap completion and latch the mapping freeze.

        The SLAM frame origin is the car spawn (≈ start/finish line), so
        distance from origin traces the lap: it climbs as the car drives
        out, then falls back toward 0 as it returns. We declare loop
        closure the first time the pose comes back within
        loop_close_radius_m of origin AFTER having been at least
        loop_close_min_radius_m away (so start-line jitter can't trip it).
        Once latched, _mapping_frozen stays true for the rest of the run.
        """
        if self._mapping_frozen:
            return
        if not bool(self.get_parameter("freeze_mapping_on_loop_close").value):
            return

        dist = float(np.hypot(pose_x, pose_y))
        if dist > self._loop_max_dist:
            self._loop_max_dist = dist

        min_radius = float(self.get_parameter("loop_close_min_radius_m").value)
        close_radius = float(self.get_parameter("loop_close_radius_m").value)
        if self._loop_max_dist >= min_radius and dist <= close_radius:
            self._mapping_frozen = True
            self.get_logger().info(
                f"LOOP_CLOSED — mapping frozen at step={self._graph.step} "
                f"(pose {dist:.1f} m from origin, peaked at "
                f"{self._loop_max_dist:.1f} m, map={len(self._db)}). No new "
                f"landmarks will be spawned; observations re-associate only."
            )

    def _resolve_finish_config(self) -> LapCounterConfig:
        """Per-mission `/slam/finished` criteria, keyed by the behavior
        string (== mission name from mode_registry). The `laps_to_finish` /
        `finish_distance_m` params override the per-mission default: -1
        means "use the mission default", >= 0 forces the value.

        Defaults: autocross = 1 lap, trackdrive = 10 laps, accel = 75 m
        (distance, since accel never returns to origin). trackdrive is
        wired but won't fire until control_node holds its stop-anchor to
        the final lap (follow-up) — today the car stops after ~lap 1 so the
        count never reaches 10. skidpad has no criterion yet (figure-8) and
        so never auto-finishes.
        """
        defaults = {
            "autocross": (1, 0.0),
            "trackdrive": (10, 0.0),
            "accel": (0, 75.0),
        }
        laps_def, dist_def = defaults.get(self._behavior, (0, 0.0))
        laps = int(self.get_parameter("laps_to_finish").value)
        dist = float(self.get_parameter("finish_distance_m").value)
        cfg = LapCounterConfig(
            laps_to_finish=laps if laps >= 0 else laps_def,
            finish_distance_m=dist if dist >= 0.0 else dist_def,
            arm_radius_m=float(
                self.get_parameter("loop_close_min_radius_m").value),
            close_radius_m=float(
                self.get_parameter("loop_close_radius_m").value),
            standstill_mps=float(
                self.get_parameter("finish_standstill_speed_mps").value),
        )
        self.get_logger().info(
            f"mission-finish: behavior='{self._behavior}' -> "
            f"laps={cfg.laps_to_finish} dist={cfg.finish_distance_m:.0f}m "
            f"(standstill <= {cfg.standstill_mps} m/s)"
        )
        return cfg

    def _current_speed(self) -> float:
        """Planar speed (m/s) from the latest SLAM velocity estimate.
        Returns +inf when there is no solve yet so the standstill gate
        can't trip before the car has even localized."""
        if self._latest_result is None:
            return float("inf")
        v = self._latest_result.velocity
        return float(np.hypot(float(v[0]), float(v[1])))

    def _publish_finished(self) -> None:
        """Latch `/slam/finished = true` — mission complete and stopped."""
        self.get_logger().info(
            f"MISSION FINISHED — {self._lap_counter.summary()} "
            f"-> /slam/finished=true"
        )
        if self._finished_pub is not None:
            self._finished_pub.publish(Bool(data=True))

    def _publish_final_lap(self, force: bool = False) -> None:
        """Publish `/slam/final_lap` on change (latched, so edges suffice).

        Edge-triggered rather than per-scan: the topic is TRANSIENT_LOCAL, so a
        late or re-activating control_node inherits the last value anyway, and
        republishing at 10 Hz would add nothing but bag weight. `force` fires
        the initial value on activate regardless of the cached state.
        """
        if self._final_lap_pub is None:
            return
        value = self._lap_counter.final_lap
        if not force and value == self._final_lap_published:
            return
        self._final_lap_published = value
        self._final_lap_pub.publish(Bool(data=value))
        self.get_logger().info(
            f"/slam/final_lap → {value} ({self._lap_counter.summary()}) — "
            f"control_node may now arm its stop anchor"
            if value else
            f"/slam/final_lap → {value} ({self._lap_counter.summary()}) — "
            f"stop anchor held off until the closing lap"
        )

    def _publish_stop_request(self, force: bool = False) -> None:
        """Publish `/slam/stop_request` on change — "criterion met, still moving".

        Rises when the mission target is met and the car has not yet stopped;
        stays true through the stop (the counter latches `target_met`), so a
        flicker can never release a stop already under way. Edge-triggered: the
        topic is TRANSIENT_LOCAL, so a late consumer inherits the value.
        """
        if self._stop_request_pub is None:
            return
        value = bool(self._lap_counter.target_met)
        if not force and value == self._stop_request_published:
            return
        self._stop_request_published = value
        self._stop_request_pub.publish(Bool(data=value))
        if value:
            self.get_logger().info(
                f"/slam/stop_request → true ({self._lap_counter.summary()}) — "
                f"mission criterion met, requesting hard stop to standstill"
            )

    def _publish_cone_map(self, stamp) -> None:
        """Publish the persistent cone landmark database to /Conos.

        Filters out single-shot / under-observed landmarks. A cone
        whose factor count is < `min_observations_for_publish` is
        kept in the SLAM solve (still in self._db, still a factor in
        the graph) but withheld from /Conos so the planner never
        navigates around phantoms. Cf. autocross_track_20260404_
        013721_20260517_232000: /Conos grew 13 → 130 over 68 s; a
        real autocross has ~60–100 cones, so ~30–40 of those are
        single-shot ghosts from imperfect DA accumulating against
        /odom yaw drift.

        marker.id encodes the persistent SLAM landmark id so downstream
        consumers (path_planning) can track cone identity across scans.
        """
        if len(self._db) == 0:
            return

        min_obs = int(self.get_parameter("min_observations_for_publish").value)

        # Local-crop radius: publish only landmarks within `radius_m` of
        # the car (squared compare, Z ignored). Keeps the per-scan publish
        # cost O(local) instead of O(map) — see the parameter docstring.
        # 0 (or no pose yet) disables the crop.
        radius_m = float(self.get_parameter("cone_map_publish_radius_m").value)
        if radius_m > 0.0 and self._latest_result is not None:
            car = self._latest_result.pose
            car_x, car_y = car.x(), car.y()
            radius_sq = radius_m * radius_m
        else:
            radius_sq = None

        out = MarkerArray()
        # First marker is DELETEALL so visualizers refresh cleanly.
        # Phase 2 (#382): /Conos now lives in map frame, not odom.
        # Landmark positions in self._db were always SLAM-absolute;
        # only the frame label needed updating.
        delete_all = Marker()
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = self.map_frame
        delete_all.action = Marker.DELETEALL
        out.markers.append(delete_all)

        # Single neutral colour for every landmark — SLAM has no
        # per-cone colour anymore. Yellow for visibility on dark
        # backgrounds; the path planner ignores the colour anyway and
        # routes everything through ConeTypes.UNKNOWN (#268). The
        # per-cone marker.id still encodes the persistent landmark id
        # so downstream consumers (path_planning) can identify cones
        # across scans.
        n_published = 0
        n_filtered = 0
        n_cropped = 0
        for lm in self._db:
            if lm.n_observations < min_obs:
                n_filtered += 1
                continue
            if radius_sq is not None:
                ddx = float(lm.position[0]) - car_x
                ddy = float(lm.position[1]) - car_y
                if ddx * ddx + ddy * ddy > radius_sq:
                    n_cropped += 1
                    continue
            out.markers.append(self._cone_marker(lm, stamp))
            n_published += 1

        self._cones_pub.publish(out)

        # Full uncropped map for visualization on /Conos_full. Published
        # every scan as a DELTA (only new/moved cones) so the cost is
        # O(changed); a periodic keyframe (DELETEALL + full map) every
        # cone_map_full_publish_every_n_scans scans resyncs late-joining
        # viewers and recovers any dropped VOLATILE frames.
        keyframe_every = int(
            self.get_parameter("cone_map_full_publish_every_n_scans").value
        )
        # Use the monotonic scan counter, not the bare graph step: once
        # mapping is frozen and we're in localization-only mode the iSAM2
        # step stops advancing, so `step % N` would either never fire again
        # or fire on every scan for the rest of the run.
        scan_idx = self._graph.step + self._loc_scan
        keyframe = keyframe_every > 0 and (scan_idx % keyframe_every == 0)
        self._publish_full_cone_map(stamp, min_obs, keyframe=keyframe)

        # Periodic diagnostic — every ~5 s of /Conos publishes,
        # surface the filter stats so the operator can confirm the
        # gate is doing what it should (filtering some, keeping most).
        if not hasattr(self, "_conos_log_counter"):
            self._conos_log_counter = 0
        self._conos_log_counter += 1
        if self._conos_log_counter % 50 == 0:
            self.get_logger().info(
                f"CONOS_FILTER published={n_published} "
                f"filtered={n_filtered} cropped={n_cropped} "
                f"(min_obs={min_obs}, radius={radius_m:.0f}m, map={len(self._db)})"
            )

    def _cone_marker(self, lm, stamp) -> Marker:
        """Build one CYLINDER marker for a landmark. marker.id encodes the
        persistent SLAM landmark id so consumers can track identity across
        scans. Single neutral yellow — SLAM has no per-cone colour."""
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.map_frame
        m.id = lm.id
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(lm.position[0])
        m.pose.position.y = float(lm.position[1])
        m.pose.position.z = float(lm.position[2])
        m.pose.orientation.w = 1.0
        m.scale.x = 0.2
        m.scale.y = 0.2
        m.scale.z = 0.3
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        return m

    def _publish_full_cone_map(
        self, stamp, min_obs: int, keyframe: bool = False
    ) -> None:
        """Publish the full cone map to /Conos_full (visualization only).

        Markers are keyed by landmark id and persist in the viewer until
        replaced or deleted, so we send only what changed since the last
        publish: new landmarks, ones whose smoothed position moved more
        than EPS_M, and DELETE markers for ones that dropped out (e.g.
        fell below min_observations). This keeps the per-scan cost
        O(changed) instead of O(map) — the O(map) Marker build/serialize
        that was the loop-closure regression now happens only on a
        keyframe, and at loop closure the delta naturally covers exactly
        the landmarks that actually moved.

        keyframe=True forces a DELETEALL + full resend and resets the
        delta cache, so a late-joining viewer (VOLATILE QoS, no history)
        gets a complete map within one keyframe interval.
        """
        if self._cones_full_pub is None or len(self._db) == 0:
            return

        # Position-change threshold below which a cone is "unchanged" and
        # not re-sent. 5 cm — well under the 0.2 m marker size, so any
        # visually meaningful shift still propagates.
        EPS_M = 0.05
        eps_sq = EPS_M * EPS_M

        out = MarkerArray()
        if keyframe:
            self._full_pub_last = {}
            delete_all = Marker()
            delete_all.header.stamp = stamp
            delete_all.header.frame_id = self.map_frame
            delete_all.action = Marker.DELETEALL
            out.markers.append(delete_all)

        current_ids = set()
        for lm in self._db:
            if lm.n_observations < min_obs:
                continue
            current_ids.add(lm.id)
            pos = (float(lm.position[0]), float(lm.position[1]), float(lm.position[2]))
            last = self._full_pub_last.get(lm.id)
            if last is None:
                out.markers.append(self._cone_marker(lm, stamp))
                self._full_pub_last[lm.id] = pos
            else:
                dx, dy, dz = pos[0] - last[0], pos[1] - last[1], pos[2] - last[2]
                if dx * dx + dy * dy + dz * dz > eps_sq:
                    out.markers.append(self._cone_marker(lm, stamp))
                    self._full_pub_last[lm.id] = pos

        # Landmarks we previously published that are now gone (or fell
        # below min_obs): DELETE them so the viewer drops them. Skipped on
        # keyframes — the DELETEALL above already cleared everything.
        if not keyframe:
            for rid in list(self._full_pub_last.keys()):
                if rid not in current_ids:
                    d = Marker()
                    d.header.stamp = stamp
                    d.header.frame_id = self.map_frame
                    d.id = rid
                    d.action = Marker.DELETE
                    out.markers.append(d)
                    del self._full_pub_last[rid]

        # Nothing changed this scan → skip the publish entirely.
        if out.markers:
            self._cones_full_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ConeGraphSlamNode()
    # NOTE: We tried MultiThreadedExecutor + callback groups to drain
    # IMU samples while the cone callback ran iSAM2 (the "skip scan: no
    # IMU samples" warnings suggested the executor was bottlenecked).
    # That broke standstill drastically (77 m drift in 28 s of being
    # parked at origin) — GTSAM's iSAM2 holds non-thread-safe state
    # even when our two callback groups never run graph code in
    # parallel. Stick with single-threaded until we have a way to
    # offload IMU buffering without touching the graph from a second
    # thread (e.g. a separate node + IPC).
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
