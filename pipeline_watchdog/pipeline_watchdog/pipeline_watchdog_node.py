"""pipeline_watchdog_node — independent supervisor for the autonomy stack.

Deliberately NOT a lifecycle node and NOT in AUTONOMY_NODE_ORDER: mode_manager
must not be able to configure, deactivate or tear it down. It comes up with the
management trio and runs for the whole session, exactly so it still supervises
when the thing it supervises is broken. Same reasoning as the uDV running its
watchdog outside the mission logic it watches.

WHAT IT COVERS (and what it does not)
-------------------------------------
The uDV already watchdogs `/dv/status` for DVPC liveness (stale > 400 ms →
uDV trips to its safe state). That covers the pipeline being **dead**.

This node covers the pipeline being **alive but sick** — mission_control
publishing DV_RUNNING at 20 Hz while the data underneath it has gone stale.
That is the documented runaway (see `pi_velocity.throttle_max` history):
"pose froze, controller kept commanding full throttle, real car ran away to
~24 m/s and crashed". `control_node` caches its last pose/odom forever and has
no staleness check of its own, so nothing else in the stack notices.

    uDV watchdog      : "is the DVPC alive?"        heartbeat on /dv/status
    pipeline watchdog : "is the DVPC's data sane?"  this node

The two are complementary. If mission_control itself hangs, THIS node's
emergency would not get relayed — but that exact case is already covered by the
uDV's heartbeat watchdog going stale. Neither watchdog needs to cover both.

WHICH TOPICS ARE SUPERVISED, AND WHY ONLY THESE
-----------------------------------------------
Only topics whose staleness leaves the car **still driving on stale data**:

  /slam/pose         frozen pose → control drives blind → the documented runaway
  /odom              frozen speed → PI reads "too slow" → commands more throttle
  /ctrl/cmd_internal control_node died → mission_control stops relaying

Deliberately NOT supervised, because they already fail safe:
  /Path        control_node publishes a zeroed command when the reference is
               empty, so a planner dropout coasts rather than runs away.
  /Conos_raw   its loss surfaces as /slam/pose or /Path staleness one hop later;
               supervising it directly would only add false-trip surface.

Adding a topic here is a safety decision, not a monitoring nicety: every extra
supervised topic is another way to fire the EBS at speed on a false positive.

ON TRIP
-------
Latches `/watchdog/emergency = true`. mission_control subscribes, and on the
rising edge calls `/force_ebs` (with its own ack-retry) and publishes
`DV_EMERGENCY` — the uDV then drives the AS state machine to AS Emergency. The
pipeline never touches the AS machine directly; it only ever asks.
"""
from __future__ import annotations

import math

import rclpy
from fs_msgs.msg import ControlCommand
from nav_msgs.msg import Odometry
from pipeline_watchdog.health_monitor import (
    HealthMonitor,
    PoseProgressSpec,
    TopicSpec,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, UInt8

# DV_RUNNING — the one byte that means "activated and relaying /ctrl/cmd".
# Imported rather than redefined so the watchdog can never drift from the
# reconciler's own notion of running.
from mission_control.interface_contract import DV_RUNNING, TOPIC_DV_STATUS


# /dv/status is published latched (RELIABLE + TRANSIENT_LOCAL) by
# mission_control (_STATUS_QOS). DDS request-vs-offered matching means we MUST
# mirror it: a BEST_EFFORT or VOLATILE reader silently receives nothing and the
# watchdog would sit disarmed forever, which is a silent safety failure rather
# than a loud one. See mission_control.interface_qos for the full rationale.
_LATCHED_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# The emergency channel mirrors /ctrl/emergency's latched profile so a
# late-joining mission_control still sees a raised emergency.
_EMERGENCY_QOS = _LATCHED_QOS

TOPIC_EMERGENCY = "/watchdog/emergency"

_WATCHDOG_HZ = 20.0


class PipelineWatchdogNode(Node):
    def __init__(self) -> None:
        super().__init__("pipeline_watchdog_node")

        # Budgets are generous multiples of each topic's nominal period: a
        # false trip fires the EBS at speed, which is both dangerous and a DNF,
        # so these are sized to catch a genuine stall rather than a hiccup.
        # At v_max = 3 m/s, 0.6 s of blindness is ~1.8 m of travel.
        self.declare_parameter("pose_max_silence_s", 0.6)     # ~10 Hz → 6 missed
        self.declare_parameter("odom_max_silence_s", 0.5)     # high rate → 0.5 s is many
        self.declare_parameter("cmd_max_silence_s", 0.5)      # 40 Hz → 20 missed
        # Spin-up allowance: Numba JIT warm-up, first LiDAR scan, first solve.
        self.declare_parameter("grace_period_s", 5.0)
        self.declare_parameter("pose_progress_enabled", True)
        self.declare_parameter("pose_progress_min_speed_mps", 1.0)
        self.declare_parameter("pose_progress_min_travel_m", 0.5)
        self.declare_parameter("pose_progress_window_s", 1.5)

        specs = (
            TopicSpec("/slam/pose", self._p("pose_max_silence_s"),
                      "SLAM stopped solving — control drives on a stale pose"),
            TopicSpec("/odom", self._p("odom_max_silence_s"),
                      "odometry filter stalled — the velocity loop is blind"),
            TopicSpec("/ctrl/cmd_internal", self._p("cmd_max_silence_s"),
                      "control_node stopped commanding"),
        )
        pose_cfg = PoseProgressSpec(
            min_speed_mps=self._p("pose_progress_min_speed_mps"),
            min_travel_m=self._p("pose_progress_min_travel_m"),
            window_s=self._p("pose_progress_window_s"),
            enabled=bool(
                self.get_parameter("pose_progress_enabled").value),
        )
        self._monitor = HealthMonitor(
            specs,
            grace_period_s=self._p("grace_period_s"),
            pose_progress=pose_cfg,
        )

        self._emergency_pub = self.create_publisher(
            Bool, TOPIC_EMERGENCY, _EMERGENCY_QOS)
        # Publish the default-false immediately so a late-joining
        # mission_control sees a defined state rather than nothing.
        self._emergency_pub.publish(Bool(data=False))
        self._raised = False

        self.create_subscription(
            UInt8, TOPIC_DV_STATUS, self._on_dv_status, _LATCHED_QOS)
        self.create_subscription(
            Odometry, "/slam/pose", self._on_pose, 10)
        self.create_subscription(
            Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(
            ControlCommand, "/ctrl/cmd_internal", self._on_cmd, 10)

        self.create_timer(1.0 / _WATCHDOG_HZ, self._tick)
        self.get_logger().info(
            "pipeline_watchdog up — supervising "
            f"{', '.join(self._monitor.supervised_topics)} "
            f"(armed only while /dv/status == DV_RUNNING)")

    def _p(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _now(self) -> float:
        """Monotonic seconds. Uses the ROS clock so a bag replay with
        use_sim_time drives the watchdog on sim time like everything else."""
        return self.get_clock().now().nanoseconds * 1e-9

    # -- subscriptions -------------------------------------------------
    def _on_dv_status(self, msg: UInt8) -> None:
        was = self._monitor.armed
        self._monitor.set_running(int(msg.data) == DV_RUNNING, self._now())
        if self._monitor.armed != was:
            state = "ARMED" if self._monitor.armed else "disarmed"
            self.get_logger().info(f"watchdog {state} (/dv/status={msg.data})")
            if not self._monitor.armed and self._raised:
                # New run: drop the latch in lockstep with the monitor, and
                # say so on the wire so mission_control's fresh cycle starts
                # from a defined false rather than a stale true.
                self._raised = False
                self._emergency_pub.publish(Bool(data=False))

    def _on_pose(self, msg: Odometry) -> None:
        now = self._now()
        self._monitor.record("/slam/pose", now)
        p = msg.pose.pose.position
        self._monitor.record_pose(p.x, p.y, now)

    def _on_odom(self, msg: Odometry) -> None:
        now = self._now()
        self._monitor.record("/odom", now)
        v = msg.twist.twist.linear
        self._monitor.record_speed(math.hypot(v.x, v.y), now)

    def _on_cmd(self, msg: ControlCommand) -> None:
        self._monitor.record("/ctrl/cmd_internal", self._now())

    # -- tick ----------------------------------------------------------
    def _tick(self) -> None:
        verdict = self._monitor.evaluate(self._now())
        if not verdict.tripped or self._raised:
            return
        self._raised = True
        self.get_logger().error(
            f"PIPELINE WATCHDOG TRIPPED — {verdict.summary()} "
            f"→ {TOPIC_EMERGENCY}=true (requesting EBS via mission_control)")
        self._emergency_pub.publish(Bool(data=True))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineWatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
