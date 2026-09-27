"""IFSSIM autonomy control node — clean rewrite (feat/34).

Single ROS node that:
  1. Subscribes to /Path (planner), /slam/pose (SLAM Odometry),
     /Conos_Orange (finish-gate detection)
  2. Builds VehicleState + ReferenceTrajectory each tick
  3. Calls a DriveController (default: CompositeDriveController wrapping
     lateral + longitudinal strategies selected by params / mode_manager)
  4. Publishes /ctrl/cmd_internal (fs_msgs/ControlCommand) — the
     supervisor relays this onto /fsds/control_command via the
     RuntimeControl action; see #384.
  5. Publishes /ctrl/emergency (latched std_msgs/Bool) — the
     supervisor latches /signal/ebs on a rising edge of this. The
     autonomy never touches /signal/ebs or /signal/ebs_reset
     directly anymore.

Keeps strategies behind ABCs. Decoupled stacks use ``drive_controller=composite``;
a future LQR/MPC registers as its own :class:`DriveController` and returns a
single :class:`ActuationCommand` per tick. The node does no path geometry, no
slip math, no PID — those live in the strategy implementations.

Wire format note: the bridge translates ControlCommand{throttle, steering,
brake} → setVehicleCommand{throttle, steering, regen} (feat/33). We emit
explicit unsigned channels: ControlCommand.throttle is motor torque demand
[0,1] and ControlCommand.brake is regen demand [0,1]. The deadband inside
the longitudinal controller guarantees only one of them is non-zero.
"""
from __future__ import annotations
import math
import time
from typing import Optional

import rclpy
from node_base.base_lifecycle_node import BaseLifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn, State as LifecycleState
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from fs_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from transforms3d.euler import quat2euler

from control.state import VehicleState
from control.reference import ReferenceTrajectory
from control.controllers.base import DriveController, LateralController, LongitudinalController
from control.controllers.composite_drive import CompositeDriveController
from control.controllers.pure_pursuit import PurePursuit
from control.controllers.stanley import Stanley
from control.controllers.pi_velocity import PIVelocity
from control.models.bicycle import KinematicBicycle


class ControlNode(BaseLifecycleNode):
    """Lifecycle-managed vehicle controller.

    Lifecycle layout:
      on_configure   declare parameters, build strategies, create
                     lifecycle publishers + TF listener.
      on_activate    create the 3 subscriptions, the 40 Hz tick timer,
                     and the EBS-reset retry timer (publishes the
                     initial reset latched). All per-run state
                     (travelled distance, last commands, stop latch)
                     reset here so deactivate→activate looks like a
                     fresh run.
      on_deactivate  destroy subscriptions + timers; pubs go quiet via
                     super().
      on_cleanup     destroy publishers, drop TF + strategies.
    """

    NODE_NAME = "control_node"
    PUBLISH_RATE_HZ = 40.0

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._declare_params()

        # Built in on_configure once params / mode_manager behavior are visible.
        self._drive: Optional[DriveController] = None

        # Latest input state. Reset on every activate.
        # /slam/pose and /odom are split sources post-#360/#382:
        #   _latest_pose ← /slam/pose (was /slam/pose pre-#382),
        #     used for absolute pose in map frame, drift-corrected via
        #     SLAM cone associations
        #   _latest_odom ← /odom, used for body-frame twist (high-rate
        #     dead-reckoning from sim_supervisor's IMU+RPM filter, or
        #     from the uDV on the real car).
        # Pre-#360 these were the same topic which coupled velocity
        # estimate quality to SLAM DA stability — a cone-only-DA
        # cascade would corrupt velocity feeding control.
        self._latest_pose: Optional[Odometry] = None
        self._latest_odom: Optional[Odometry] = None
        self._latest_path_xs: list[float] = []
        self._latest_path_ys: list[float] = []
        # Per-pose curvature from path_planning, smuggled through
        # `pose.position.z` (FaSTTUBe analytical κ; see path_planning's
        # _pose_stamped). Empty when no path has arrived yet; the
        # ReferenceTrajectory builder falls back to its own finite-
        # difference κ when this list is empty or all-zero.
        self._latest_path_kappas: list[float] = []
        # Big-orange forward distances (base_link frame) — captured at the
        # tick when the gate is first detected, then frozen as a stop anchor
        # in odom frame (see _on_orange / _stop_distance).
        self._stop_anchor_xy: Optional[tuple[float, float]] = None
        self._stop_latched: bool = False
        # Gate for the stop anchor — see _on_orange. Defaults TRUE so a stack
        # with no /slam/final_lap publisher behaves exactly as it did before
        # this gate existed (stop at the first gate past stop_latch_min_travel).
        # Defaulting False would mean "never stop", which is the failure that
        # actually hurts.
        self._final_lap: bool = True
        # Travel distance (odom-frame) used as the gate-latch guard.
        self._travelled: float = 0.0
        self._last_pose_xy: Optional[tuple[float, float]] = None

        # Last published command — input to the actuator slew limiter.
        self._last_throttle: float = 0.0
        self._last_regen:    float = 0.0
        self._last_steering: float = 0.0
        self._tick_count: int = 0

        # I/O handles, set in on_configure / on_activate.
        self._tf_buffer: Optional[Buffer] = None
        self._tf_listener: Optional[TransformListener] = None
        self._cmd_pub = None
        # Post-#384: emergency requests go onto a latched Bool topic
        # that mission_control_node subscribes to and surfaces via
        # RuntimeControl Feedback. The supervisor publishes
        # /signal/ebs on the rising edge from the bridge side.
        self._emergency_pub = None
        self._v_set_pub = None
        self._kappa_max_pub = None
        self._latency_pub = None
        self._sub_path = None
        self._sub_pose = None
        self._sub_odom = None
        self._sub_orange = None
        self._sub_final_lap = None
        self._tick_timer = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        ret = super().on_configure(state)
        if ret != TransitionCallbackReturn.SUCCESS:
            return ret
        self.get_logger().info("on_configure: strategies + publishers + TF")
        self._build_strategies()

        # Publishers — lifecycle-aware so they go silent when deactivated.
        # Post-#384: command flows through mission_control's
        # RuntimeControl action, not /fsds/control_command directly.
        # The supervisor relays each Feedback frame back onto the
        # bridge topic. Topic-name change makes the new chain
        # observable in ros2 topic list and prevents the bridge from
        # mistakenly subscribing to two sources.
        self._cmd_pub = self.create_lifecycle_publisher(
            ControlCommand, "/ctrl/cmd_internal", 10)
        # Emergency channel — latched Bool, mission_control subscribes
        # and propagates rising edges to the supervisor via the
        # RuntimeControl Feedback.emergency flag. We always publish a
        # latched default-false on activate so a late-joining
        # mission_control sees the initial state.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._emergency_pub = self.create_lifecycle_publisher(
            Bool, "/ctrl/emergency", latched)

        # Diagnostic publishers (#260 follow-up). v_set tracks the
        # longitudinal controller's setpoint each tick; kappa_max
        # exposes the local curvature it's reacting to. Plotted
        # alongside SLAM v in slam_debug.json — if v_set doesn't drop
        # going into a corner, the velocity profile isn't being honoured
        # and that's why the car carries too much speed into the apex.
        self._v_set_pub = self.create_lifecycle_publisher(
            Float32, "/control/v_set_mps", 10)
        self._kappa_max_pub = self.create_lifecycle_publisher(
            Float32, "/control/kappa_max_per_m", 10)
        self._latency_pub = self.create_lifecycle_publisher(
            Float32, "/control/latency_ms", 10)

        # TF listener — post-#382 sim_supervisor publishes
        # odom→base_link (100 Hz dead-reckoning) and slam_node
        # publishes map→odom (drift correction). The chain
        # map→odom→base_link gives the leaf-pose used for waypoint
        # projection.
        self._tf_buffer = Buffer()
        # spin_thread=False: this process already rclpy.spin()s the node.
        # Humble's TransformListener default (True) adds the same node to a
        # second executor, so every /tf sample is inserted twice and the
        # second hit logs TF_OLD_DATA (equal timestamps are "from the past").
        self._tf_listener = TransformListener(
            self._tf_buffer, self, spin_thread=False)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info(
            f"on_activate: subs + 40 Hz timer "
            f"(drive={type(self._drive).__name__})")

        # Reset all per-run state so deactivate→activate is a fresh run.
        self._latest_pose = None
        self._latest_odom = None
        self._latest_path_xs = []
        self._latest_path_ys = []
        self._latest_path_kappas = []
        self._stop_anchor_xy = None
        self._stop_latched = False
        self._final_lap = True
        self._travelled = 0.0
        self._last_pose_xy = None
        self._last_throttle = 0.0
        self._last_regen = 0.0
        self._last_steering = 0.0
        self._tick_count = 0

        # Controller-internal state (PI integrator, Stanley steer-rate
        # warm start) lives in the strategy objects, which are built in
        # on_configure and survive deactivate→activate. Clear them too,
        # or a wound-up integral from the previous run produces a
        # throttle/regen transient on the next start.
        if self._drive is not None:
            self._drive.reset()

        # Post-#384: the bridge's EBS gate is owned by the supervisor;
        # the supervisor publishes /signal/ebs_reset itself on
        # Phase 1 ready (mirrors what the uDV firmware does on the
        # real car). control_node only signals intent via
        # /ctrl/emergency now — publish the initial default-false
        # latched value so a late-joining mission_control sees a
        # defined state immediately.
        # LifecyclePublisher.publish() is a silent no-op until the base
        # class on_activate() enables the managed publishers, so this
        # one-shot publish happens after super() below.

        # Subscriptions
        self._sub_path = self.create_subscription(
            Path, "Path", self._on_path, 10)
        # Absolute pose — comes from SLAM at LiDAR tick rate (~10 Hz).
        # Used for x/y/yaw and the gate-latch body→world projection.
        self._sub_pose = self.create_subscription(
            Odometry, "/slam/pose", self._on_pose, 10)
        # Body-frame twist — comes from sim_supervisor (or uDV on the
        # real car) at ~100 Hz from the IMU+RPM complementary filter.
        # Drifty over time but high-rate, so the controller's velocity
        # tracking has fresh data every 40 Hz tick.
        self._sub_odom = self.create_subscription(
            Odometry, "/odom", self._on_odom, 10)
        self._sub_orange = self.create_subscription(
            MarkerArray, "/Conos_Orange", self._on_orange, 10)
        # /slam/final_lap — "the next finish gate is the closing one". SLAM
        # publishes it TRANSIENT_LOCAL; match durability or we would never
        # receive the value it latched before we activated, and trackdrive
        # would sit at _final_lap=True (the safe default) and stop on lap 1.
        final_lap_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._sub_final_lap = self.create_subscription(
            Bool, "/slam/final_lap", self._on_final_lap, final_lap_qos)

        # 40 Hz tick
        self._tick_timer = self.create_timer(
            1.0 / self.PUBLISH_RATE_HZ, self._tick)

        ret = super().on_activate(state)
        if ret != TransitionCallbackReturn.SUCCESS:
            return ret
        self._emergency_pub.publish(Bool(data=False))
        return ret

    def on_deactivate(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate: dropping timers + subs")
        for sub in (self._sub_path, self._sub_pose, self._sub_odom,
                    self._sub_orange, self._sub_final_lap):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_path = None
        self._sub_pose = None
        self._sub_odom = None
        self._sub_orange = None
        self._sub_final_lap = None
        if self._tick_timer is not None:
            self.destroy_timer(self._tick_timer)
        self._tick_timer = None
        return super().on_deactivate(state)

    def on_cleanup(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup: destroying publishers + TF")
        for sub in (self._sub_path, self._sub_pose, self._sub_odom,
                    self._sub_orange, self._sub_final_lap):
            if sub is not None:
                self.destroy_subscription(sub)
        self._sub_path = None
        self._sub_pose = None
        self._sub_odom = None
        self._sub_orange = None
        self._sub_final_lap = None
        if self._tick_timer is not None:
            self.destroy_timer(self._tick_timer)
        self._tick_timer = None
        for pub in (self._cmd_pub, self._emergency_pub,
                    self._v_set_pub, self._kappa_max_pub, self._latency_pub):
            if pub is not None:
                self.destroy_publisher(pub)
        self._cmd_pub = None
        self._emergency_pub = None
        self._v_set_pub = None
        self._kappa_max_pub = None
        self._latency_pub = None
        self._tf_listener = None
        self._tf_buffer = None
        self._drive = None
        return super().on_cleanup(state)

    def on_shutdown(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        return super().on_shutdown(state)

    # ------------------------------------------------------------------ params

    def _declare_params(self) -> None:
        self.declare_parameter("drive_controller", "composite")
        self.declare_parameter("lateral_controller", "pure_pursuit")
        self.declare_parameter("longitudinal_controller", "pi_velocity")
        # Tunables — passed to the strategy constructors. Keep flat: each
        # strategy reads what it needs, ignores the rest.
        self.declare_parameter("v_max", 3.0)  # tuning experiment — completable-lap baseline
        # a_lat_max governs corner braking: v_corner = sqrt(R · a_lat_max).
        # 6.0 m/s² (~0.6 g lateral) was the original default — fine for
        # a real FS car on slicks, too generous for the IFS-08 sim's
        # tire envelope. With a_lat_max=6 and v_max=3, even a 1.5 m
        # radius hairpin yields v_corner ≥ v_max → setpoint never drops
        # → controller carries v_max into the apex and goes off (#260
        # follow-up). Dropped to 3.0 m/s² (~0.3 g): v_corner < v_max
        # whenever R < 3 m, so any FS-style hairpin actually triggers
        # a setpoint drop. Tune up later once tire physics is known.
        self.declare_parameter("a_lat_max", 3.0)
        self.declare_parameter("a_dec_max", 4.0)
        # Lowered 1.5 → 1.0 to widen the band where the β·R radius cap
        # (in pure_pursuit.py) can actually bind. Previous default left
        # hairpins between R ≈ 1.4 and 2.1 m unguarded — the cap was
        # silently re-floored back to L_min and Pure Pursuit overshot
        # the apex (test_submodule first hairpin, #260 follow-up).
        self.declare_parameter("lookahead_min", 1.0)
        self.declare_parameter("lookahead_k", 0.5)
        # Stanley lateral controller parameters. Retuned for the IFS-08
        # sim's operating speed (v_max=3.0, a_lat_max=3.0 → ~1-3 m/s in
        # practice). The original alt_pipeline gains (k=4.0, k_soft=6.0)
        # were sized for a much faster car: k_soft is a velocity (m/s)
        # that dominates the cross-track denominator at low speed, so at
        # 1-3 m/s the cross-track time constant τ≈(k_soft+v)/(k·v) ran
        # 0.75-3.3 s — sluggish and worse the slower the car went.
        # Dropping k_soft to 1.5 makes τ≈0.4-0.7 s across the regime and
        # far more speed-stable. The lower k_soft makes the controller
        # more reactive to planner-yaw noise, so k_damp_steer=0.2 hedges
        # the single-tick swings the 40 Hz slew limiter won't fully
        # absorb. mode_manager pushes these per-mode via SetParameters
        # before the configure transition; see
        # pipeline/mode_manager/mode_manager/mode_registry.py.
        #   stanley_k:           cross-track gain (1/s scaling on cte)
        #   stanley_k_soft:      softening term (m/s) keeping the
        #                        cte denominator stable at v→0
        #   stanley_k_yaw_rate:  optional yaw-rate damping coefficient
        #                        (rad / (rad/s)); 0.0 disables
        #   stanley_k_damp_steer: optional single-tick damping against
        #                         the previous command; 0.0 disables
        # NB k=3.5 with k_yaw_rate=0.0 (no yaw damping) limit-cycled the
        # lateral loop at 1-3 m/s: the heading-error term is fed straight into
        # steering and nothing damped the overshoot, so on any heading
        # disturbance (corner) the steering bang-banged ±1. Measured on bag
        # autocross_track_20260711_184630: steering at full lock 25-53% of the
        # run, ±1 rad/s yaw limit cycle, steering leading yaw by 0.15 s
        # (closed-loop instability). The #306 actuator slew limiter softened
        # but did not remove it. Reverted to the previously-validated k=2.0 +
        # k_yaw_rate=0.5 (Hoffmann yaw-rate damping) — the damping term is the
        # lever that breaks the cycle. Confirmed on a live autocross run.
        self.declare_parameter("stanley_k", 2.0)
        self.declare_parameter("stanley_k_soft", 1.5)
        self.declare_parameter("stanley_k_yaw_rate", 0.5)
        self.declare_parameter("stanley_k_damp_steer", 0.2)
        # Kinematic-bicycle geometry. Used by every lateral controller
        # via the shared KinematicBicycle model. Default matches the
        # IFS-08 chassis (also the default inside bicycle.py).
        self.declare_parameter("wheelbase", 1.627)
        self.declare_parameter("max_steer_deg", 18.2)  # road-wheel ceiling = column limit 100 deg / effective ratio 5.5; MUST match uDV MAX_STEER_ROADWHEEL_DEG (#59, uDV#172)
        self.declare_parameter("kp_v", 0.5)
        self.declare_parameter("ki_v", 0.05)
        # deadband_v must stay well below throttle_max: with both at 0.2 the
        # PI output was binary {0, throttle_max} — no proportional region —
        # producing the throttle chatter/glitch spikes seen on the bench
        # (found with tools/long_tuning/long_harness.py).
        self.declare_parameter("deadband_v", 0.05)
        self.declare_parameter("throttle_max", 0.2)
        # v_meas conditioning for the longitudinal PI — real-car spike
        # rejection at the INPUT (see pi_velocity._condition_velocity).
        # Complements the deadband_v tune above: the deadband widens the
        # region where small errors coast instead of flip-flopping, while
        # this rejects the single-sample velocity glitches that make kp
        # slam the throttle for a tick. tau is the EMA time constant;
        # max_accel is the plausibility-clamp ceiling, set well above the
        # true accel/decel envelope so it only rejects non-physical jumps.
        self.declare_parameter("v_meas_tau", 0.15)
        self.declare_parameter("v_meas_max_accel", 20.0)
        # Actuator slew limits (units = command-units per second). The
        # sim takes commands instantaneously; real actuators don't.
        # These rate-limit the published command at the boundary so
        # every strategy benefits without changing the strategies. See
        # #306 (GT-as-SLAM diagnostic) for the bang-bang traces these
        # caps are designed to absorb. Values approximate IFS-08:
        #   throttle: 0→1 in 0.5 s (EMRAX inverter ramp + throttle map)
        #   regen:    0→1 in 0.3 s (regen torque response is faster)
        #   steering: ~90° rack travel in 0.2 s ≈ 5.0 normalized/s
        # Each can be raised by parameter for debugging without code
        # changes; setting to a very large number disables that cap.
        self.declare_parameter("throttle_rate", 2.0)
        self.declare_parameter("regen_rate", 3.33)
        self.declare_parameter("steering_rate", 5.0)
        # Stop-latch guard. The FS start gate is also big-orange, so we
        # need to drive at least one lap-ish before the first orange
        # detection counts as the finish. Trackdrive courses are >100 m
        # per lap; 30 m is safely past any start-gate proximity.
        self.declare_parameter("stop_latch_min_travel", 30.0)

    def _p(self, name: str):
        return self.get_parameter(name).value

    # ------------------------------------------------------------------ factory

    def _build_strategies(self) -> None:
        """Pick controllers from mode_manager behavior (stanley / pure_pursuit)."""
        dc = str(self._p("drive_controller"))
        if dc != "composite":
            raise ValueError(
                f"unknown drive_controller={dc!r}; supported: 'composite'. "
                "Add a joint DriveController (e.g. LQR, MPC) here when ready."
            )

        lat_name = self._behavior
        if lat_name not in ("stanley", "pure_pursuit"):
            lat_name = self._p("lateral_controller")
        lon_name = self._p("longitudinal_controller")

        # Shared kinematic-bicycle model. Built from params so
        # mode_manager can override wheelbase / max_steer per mode
        # without touching the controller files themselves.
        bicycle = KinematicBicycle(
            wheelbase=float(self._p("wheelbase")),
            max_steer_rad=math.radians(float(self._p("max_steer_deg"))),
        )

        lateral: LateralController
        if lat_name == "pure_pursuit":
            lateral = PurePursuit(
                lookahead_min=self._p("lookahead_min"),
                lookahead_k=self._p("lookahead_k"),
                model=bicycle,
            )
        elif lat_name == "stanley":
            lateral = Stanley(
                k=self._p("stanley_k"),
                k_soft=self._p("stanley_k_soft"),
                k_yaw_rate=self._p("stanley_k_yaw_rate"),
                k_damp_steer=self._p("stanley_k_damp_steer"),
                model=bicycle,
            )
        else:
            raise ValueError(f"unknown lateral_controller={lat_name!r}")

        longitudinal: LongitudinalController
        if lon_name == "pi_velocity":
            longitudinal = PIVelocity(
                v_max=self._p("v_max"),
                a_lat_max=self._p("a_lat_max"),
                a_dec_max=self._p("a_dec_max"),
                kp=self._p("kp_v"),
                ki=self._p("ki_v"),
                deadband=self._p("deadband_v"),
                throttle_max=self._p("throttle_max"),
                v_meas_tau=self._p("v_meas_tau"),
                v_meas_max_accel=self._p("v_meas_max_accel"),
            )
        else:
            raise ValueError(f"unknown longitudinal_controller={lon_name!r}")

        self._drive = CompositeDriveController(lateral, longitudinal)

    # ----------------------------------------------------------- subscriptions

    def _on_pose(self, msg: Odometry) -> None:
        """SLAM absolute pose (drift-corrected). Twist field on this
        message is ignored — we read body-frame velocity from /odom
        instead (see _on_odom)."""
        self._latest_pose = msg

    def _on_odom(self, msg: Odometry) -> None:
        """Dead-reckoning Odometry from sim_supervisor (or uDV on the
        real car). Pose field on this message drifts over time; we
        consume only the twist for body-frame velocity."""
        self._latest_odom = msg

    def _on_path(self, msg: Path) -> None:
        self._latest_path_xs = [p.pose.position.x for p in msg.poses]
        self._latest_path_ys = [p.pose.position.y for p in msg.poses]
        # Side-channel: pose.position.z carries the planner's per-pose
        # curvature (FaSTTUBe analytical κ). See path_planning's
        # _pose_stamped for the encoding rationale.
        self._latest_path_kappas = [p.pose.position.z for p in msg.poses]

    def _on_final_lap(self, msg: Bool) -> None:
        """Track SLAM's 'the next gate is the closing one' signal.

        Only ever consulted by _on_orange. Note this is NOT latched-once: SLAM
        resets it per run, and control resets to True on activate, so a
        deactivate→activate cycle re-reads the current value from the latched
        topic rather than inheriting the last run's.
        """
        value = bool(msg.data)
        if value != self._final_lap:
            self.get_logger().info(
                f"/slam/final_lap → {value}"
                + ("" if value else " — holding stop anchor until the closing lap"))
        self._final_lap = value

    def _on_orange(self, msg: MarkerArray) -> None:
        """Latch a stop anchor on the first tick we see ≥2 big-orange cones,
        AFTER the car has travelled at least stop_latch_min_travel metres
        from origin. The minimum-travel gate exists because the FS start
        gate is also big-orange; without it, we latch on the start cones
        at t=0 and the controller immediately tries to brake to a stop.

        Once latched, never unlatch — transient detector flicker (audit
        P0 item #2) cannot release the stop. The anchor is the centroid
        of the orange cones, projected from the car's current odom-frame
        pose. After that, _stop_distance() is the residual euclidean
        distance from the car to the anchor in odom frame.

        The cones come in base_link (vehicle frame), so we transform via
        the latest *absolute* pose — close enough at the moment of latch
        since the car is still ~10 m from the gate. Uses /slam/pose
        rather than /odom because the stop anchor is a long-lived
        world-frame point and dead-reckoning drift would walk it
        across laps."""
        if self._stop_latched or self._latest_pose is None:
            return
        # cone_detection_node leads every array with a pose-less DELETEALL
        # marker (viewer refresh); only ADD markers are cones. Including the
        # leader in the centroid would pull the gate toward the origin.
        cones = [m for m in msg.markers if m.action != Marker.DELETEALL]
        if len(cones) < 2:
            return
        if self._travelled < self.get_parameter("stop_latch_min_travel").value:
            return
        # Lap gate: only latch on the CLOSING lap's gate. Without this,
        # trackdrive brakes to a stop at the first big-orange gate past
        # stop_latch_min_travel — i.e. the end of lap 1 — and can never reach
        # its 10 laps. SLAM owns the lap count and publishes /slam/final_lap;
        # this node stays dumb about mission rules.
        #
        # Defaults TRUE (see _final_lap init) so a missing publisher degrades
        # to the historical behaviour — stop at the first gate — rather than
        # to "never stop", which would be the dangerous failure. For autocross
        # (1 lap), accel and skidpad, SLAM publishes true immediately, so this
        # gate is transparent and only trackdrive's behaviour changes.
        if not self._final_lap:
            return
        # Centroid in base_link
        n = len(cones)
        sx = sum(m.pose.position.x for m in cones) / n
        sy = sum(m.pose.position.y for m in cones) / n
        # base_link → odom using current absolute pose from SLAM
        o = self._latest_pose
        q = o.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        ax = o.pose.pose.position.x + cos_y * sx - sin_y * sy
        ay = o.pose.pose.position.y + sin_y * sx + cos_y * sy
        self._stop_anchor_xy = (ax, ay)
        self._stop_latched = True
        self.get_logger().info(
            f"stop latched at odom=({ax:.2f}, {ay:.2f}) "
            f"from {n} big-orange cones"
        )

    def _stop_distance(self, state: VehicleState) -> float:
        """Euclidean distance from car to the latched stop anchor. Returns
        +inf when no stop is latched (controller treats this as 'no cap')."""
        if not self._stop_latched or self._stop_anchor_xy is None:
            return float("inf")
        ax, ay = self._stop_anchor_xy
        return math.hypot(ax - state.x, ay - state.y)

    # ------------------------------------------------------------- tick

    def _tick(self) -> None:
        # Default: zero output. Anything that fails below leaves the car
        # commanding nothing rather than the previous tick's cached
        # response — fail-safe under SLAM/path dropout.
        t0 = time.perf_counter()
        cmd = ControlCommand()
        cmd.throttle = 0.0
        cmd.steering = 0.0
        cmd.brake = 0.0

        state = self._build_state()
        ref = ReferenceTrajectory.from_xy(
            self._latest_path_xs, self._latest_path_ys,
            kappa=self._latest_path_kappas or None,
        )

        if state is None or ref.empty:
            self._cmd_pub.publish(cmd)
            self._publish_latency_ms(t0)
            return

        # Accumulate travel distance — used by the stop-latch guard to
        # ignore the start gate's big-orange cones until we've driven a
        # full lap-ish away from spawn.
        if self._last_pose_xy is None:
            self._last_pose_xy = (state.x, state.y)
        else:
            dx = state.x - self._last_pose_xy[0]
            dy = state.y - self._last_pose_xy[1]
            self._travelled += math.hypot(dx, dy)
            self._last_pose_xy = (state.x, state.y)

        # Stop semantics — populated here so both controllers see the same
        # snapshot. Distance is to the latched orange-gate anchor in odom
        # frame; treated as +inf until ≥2 big-orange cones have been seen
        # at least once.
        ref.stop_distance = self._stop_distance(state)
        ref.stop_latched = self._stop_latched

        # Strategies own all algorithm logic. Per-tick state is immutable;
        # any internal accumulator (PI integral) lives on the drive controller.
        if self._drive is None:
            self._cmd_pub.publish(cmd)
            self._publish_latency_ms(t0)
            return

        act = self._drive.compute(state, ref)

        cmd.throttle = float(max(0.0, min(1.0, act.throttle)))
        cmd.brake = float(max(0.0, min(1.0, act.regen)))
        steering_norm = act.steering_normalized
        # Sign convention boundary. The strategy contract declares positive
        # steering = LEFT (math convention, matches Pure Pursuit's curvature
        # κ = 2·sin(α)/Ld where α is atan2(body_y, body_x) with body_y =
        # left). UE5 / Chaos SetSteeringInput uses the automotive convention:
        # positive = RIGHT (clockwise). Flip here so every controller can be
        # written in math without each one needing to know about the wire
        # convention.
        # Sign convention boundary already applied (positive = LEFT in
        # the strategy contract → flip to UE5/Chaos convention here).
        # Soft cap during the first metre of motion: the planner publishes
        # a centerline computed from the first cone observations that's
        # often off-axis (FaSTTUBe's L/R sort can be asymmetric on the
        # initial scan, FOV asymmetry, or virtual-cone insertion on a
        # sparse side), and pure pursuit at Ld≈1 m saturates whenever the
        # lookahead point is more than ~14° off-centre. The result is a
        # visible startup yank that goes away once the car has moved
        # enough for the planner to see both rows symmetrically. The
        # ramp clips |steer| to a value that grows linearly from 0.3 at
        # travelled=0 to 1.0 at travelled≥1 m. Not a real fix — that
        # belongs in the planner — but the saturated startup value is a
        # degenerate-geometry artifact, not a useful command, so capping
        # it loses no information. See #298.
        STARTUP_RAMP_DIST_M = 1.0
        STARTUP_STEER_MIN = 0.3
        if self._travelled < STARTUP_RAMP_DIST_M:
            ramp = self._travelled / STARTUP_RAMP_DIST_M
            startup_cap = STARTUP_STEER_MIN + (1.0 - STARTUP_STEER_MIN) * ramp
        else:
            startup_cap = 1.0
        cmd.steering = float(max(-startup_cap, min(startup_cap, -steering_norm)))

        # ----- actuator slew limiter (#306) ------------------------
        # Rate-limit each channel at the publish boundary. The
        # PI velocity loop's "u >= deadband → throttle = u, u <=
        # -deadband → regen = -u" rule is bang-bang as soon as
        # measured velocity oscillates around v_set, and Pure Pursuit
        # commands large steering swings the moment the path's
        # endpoint jitters. Both produce visible thrash in
        # /control_command without any actuator model in between.
        # The sim takes commands instantaneously, so without this cap
        # the wheels and motor torque step infinitely fast — not what
        # any real car would do. Values declared in _declare_params.
        dt = 1.0 / self.PUBLISH_RATE_HZ
        thr_step  = self._p("throttle_rate") * dt
        regen_step = self._p("regen_rate")   * dt
        steer_step = self._p("steering_rate") * dt
        cmd.throttle = _slew(self._last_throttle, cmd.throttle, thr_step)
        cmd.brake    = _slew(self._last_regen,    cmd.brake,    regen_step)
        cmd.steering = _slew(self._last_steering, cmd.steering, steer_step)
        self._last_throttle = cmd.throttle
        self._last_regen    = cmd.brake
        self._last_steering = cmd.steering

        self._cmd_pub.publish(cmd)

        # Diagnostic publish (#260 follow-up). v_set vs SLAM v shows
        # whether the velocity controller is honouring corner-radius
        # braking; kappa_max shows whether it's even seeing the corner.
        # Both expose internal state that's otherwise only visible in
        # the per-tick log line — having them as topics lets Lichtblick
        # plot them against time alongside the SLAM speed trace.
        v_set_msg = Float32()
        v_set_msg.data = float(getattr(self._drive, "last_v_set", 0.0))
        self._v_set_pub.publish(v_set_msg)
        kappa_msg = Float32()
        kappa_msg.data = float(getattr(self._drive, "last_kappa_max", 0.0))
        self._kappa_max_pub.publish(kappa_msg)

        # Heartbeat — every ~0.5 s. Tells us at a glance whether each
        # stage is producing what we expect.
        self._tick_count = getattr(self, "_tick_count", 0) + 1
        if self._tick_count % 20 == 0:
            self.get_logger().info(
                f"v={state.speed:.2f} travelled={self._travelled:.1f}m -> "
                f"thr={cmd.throttle:+.3f} regen={cmd.brake:+.3f} "
                f"steer={cmd.steering:+.3f} | "
                f"path_n={len(ref.x)} path_len={ref.length:.1f}m "
                f"stop_d={ref.stop_distance:.1f} latched={ref.stop_latched}"
            )
        self._publish_latency_ms(t0)

    def _publish_latency_ms(self, t0: float) -> None:
        if self._latency_pub is None:
            return
        getter = getattr(self._latency_pub, "get_subscription_count", None)
        if getter is not None:
            try:
                if int(getter()) <= 0:
                    return
            except Exception:
                pass
        msg = Float32()
        msg.data = (time.perf_counter() - t0) * 1000.0
        self._latency_pub.publish(msg)

    def _build_state(self) -> Optional[VehicleState]:
        """Compose VehicleState from two sources:
          • pose (x, y, yaw) from /slam/pose — SLAM's drift-
            corrected absolute pose, ~10 Hz.
          • twist (vx, vy, yaw_rate) from /odom — sim_supervisor's
            (or uDV's) dead-reckoning IMU+RPM filter, ~100 Hz.

        Both must be present before the controller can make a
        decision; the strategies need both pose (for path
        projection) and twist (for velocity tracking). Returns
        None until both topics have produced their first message —
        the tick fail-safes to zero command in that window."""
        if self._latest_pose is None or self._latest_odom is None:
            return None
        p = self._latest_pose
        t = self._latest_odom
        q = p.pose.pose.orientation
        _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
        return VehicleState(
            x=p.pose.pose.position.x,
            y=p.pose.pose.position.y,
            yaw=yaw,
            vx=t.twist.twist.linear.x,
            vy=t.twist.twist.linear.y,
            yaw_rate=t.twist.twist.angular.z,
        )


def _slew(prev: float, target: float, max_step: float) -> float:
    """Clamp `target` so it differs from `prev` by no more than
    `max_step`. `max_step` should be positive (rate × dt). Returns the
    rate-limited value to publish AND to feed back as `prev` next tick.
    """
    if target > prev + max_step:
        return prev + max_step
    if target < prev - max_step:
        return prev - max_step
    return target


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
