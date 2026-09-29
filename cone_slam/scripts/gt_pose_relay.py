#!/usr/bin/env python3
"""GT-as-SLAM diagnostic node.

For one-shot experiments to isolate "is SLAM the bottleneck?".
Replaces `slam_node` in the running pipeline:

  - Subscribes:
      /testing_only/odom        (sim ground truth, ENU)
      /Conos_raw                (per-scan cone observations, base_link)
      /odom                     (odometry_filter_node's EKF dead-reckoning,
                                 odom → base_link — needed to compute the
                                 map → odom correction exactly like slam_node)

  - Publishes (matches slam_node's post-#382 contract):
      /slam/pose                (Odometry, GT-aligned pose in `map`,
                                 body-frame twist in `base_link`)
      /tf                       (map → odom, dynamic: gt_pose ⊖ latest /odom.
                                 odom → base_link is owned by
                                 odometry_filter_node and NOT published here —
                                 two writers on one TF edge is last-writer-wins.)
      /Conos                    (MarkerArray in `map`, /Conos_raw projected
                                 through the GT pose; no landmark identity,
                                 just current-scan cones at world positions)

Anchor: the first GT message received becomes the calibration-end anchor.
Subsequent GT poses are expressed in `gt_init.inverse() * gt_now`, which
matches the SLAM-anchored `map` frame the rest of the pipeline expects.
Until the first /odom sample arrives, map → odom is broadcast as the GT
pose itself (same fallback slam_node uses during the EKF calibration
window) so the TF tree stays rooted.

Usage inside the container (slam_node must not be running — the lifecycle
manager will otherwise bring it back up; deactivate it or run this against a
bag replay):
    docker compose exec -T dv_pipeline_stack bash -lc \\
      'source /opt/ros/humble/setup.bash && \\
       source /dv_pipeline_stack_ws/install/setup.bash && \\
       ros2 lifecycle set /slam_node deactivate ; \\
       python3 /dv_pipeline_stack_ws/src/cone_slam/scripts/gt_pose_relay.py'

This is a diagnostic-only node. NEVER deploy this to the real car.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

# Same SE(2) map→odom math slam_node uses (pure-Python, no rclpy import).
from cone_slam.tf_math import compute_map_to_odom

MAP_FRAME = "map"
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for a rotation about z."""
    h = 0.5 * yaw
    return (math.cos(h), 0.0, 0.0, math.sin(h))


class GTPoseRelay(Node):
    def __init__(self) -> None:
        super().__init__("slam_node")  # take SLAM's name so consumers don't care

        # Anchor: first /testing_only/odom received.
        self._gt_init_x: Optional[float] = None
        self._gt_init_y: Optional[float] = None
        self._gt_init_yaw: Optional[float] = None

        # Latest EKF /odom sample (odom → base_link) for the map→odom math.
        self._latest_odom: Optional[Odometry] = None

        # Latest aligned pose (in SLAM-anchored map frame).
        self._aligned_x: float = 0.0
        self._aligned_y: float = 0.0
        self._aligned_yaw: float = 0.0
        self._aligned_vx_body: float = 0.0
        self._aligned_vy_body: float = 0.0
        self._latest_gt_stamp = None

        # Previous-tick state for finite-difference velocity. Historically
        # the bridge published /testing_only/odom with twist=0 (Cesium/UE5
        # didn't fill the twist field), so this relay derived body-frame
        # velocity ourselves from successive pose deltas. PR #315 closed
        # that gap for twist.linear; the angular-twist follow-up populates
        # twist.angular from RootComponent->GetPhysicsAngularVelocityInRadians()
        # so /testing_only/odom is now fully GT (pose + linear + angular).
        # The finite-difference is kept as a self-contained fallback /
        # sanity check — if the bridge ever regresses on twist again, the
        # controller still reads a non-zero velocity here and doesn't
        # floor the throttle the way it did the one time before #315.
        self._prev_aligned_x: Optional[float] = None
        self._prev_aligned_y: Optional[float] = None
        self._prev_stamp_ns: Optional[int] = None
        # Light EMA on velocity to swallow per-tick numerical noise from
        # 12 ms-spaced finite differences. α=0.6 is responsive enough to
        # follow real acceleration; the controller's PI tolerates more
        # smoothing than this.
        self._VEL_EMA_ALPHA = 0.6

        # Publishers (same names + QoS as slam_node).
        self._tf_broadcaster = TransformBroadcaster(self)
        self._state_pub = self.create_publisher(Odometry, "/slam/pose", 10)
        self._cones_pub = self.create_publisher(MarkerArray, "/Conos", 10)

        # GT subscription (BEST_EFFORT, matches slam_node).
        gt_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Odometry, "/testing_only/odom", self._on_gt, gt_qos)

        # EKF /odom (RELIABLE, matches slam_node's subscription).
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Odometry, "/odom", self._on_odom, odom_qos)

        # Cone observations — reliable (same QoS as slam_node).
        self.create_subscription(
            MarkerArray, "/Conos_raw", self._on_cones, 10)

        self.get_logger().info(
            "gt_pose_relay started — DIAGNOSTIC MODE, GT pose as SLAM output")

    # ------------------------------------------------------------------ /odom

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    # ------------------------------------------------------------------ GT

    def _on_gt(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q.w, q.x, q.y, q.z)
        if self._gt_init_x is None:
            self._gt_init_x = x
            self._gt_init_y = y
            self._gt_init_yaw = yaw
            self.get_logger().info(
                f"anchor captured: ({x:+.2f}, {y:+.2f}, "
                f"yaw={math.degrees(yaw):+.1f}°)")

        # Express current GT pose in the anchor's frame (this is the same
        # `gt_init.inverse().compose(gt_now)` the SLAM diagnostic uses).
        dx = x - self._gt_init_x
        dy = y - self._gt_init_y
        c = math.cos(-self._gt_init_yaw)
        s = math.sin(-self._gt_init_yaw)
        self._aligned_x = c * dx - s * dy
        self._aligned_y = s * dx + c * dy
        self._aligned_yaw = yaw - self._gt_init_yaw
        # Wrap to (-π, π].
        while self._aligned_yaw > math.pi:
            self._aligned_yaw -= 2.0 * math.pi
        while self._aligned_yaw < -math.pi:
            self._aligned_yaw += 2.0 * math.pi

        # Body-frame velocity by finite-differencing aligned pose. The
        # bridge fills /testing_only/odom twist since #315, so this is a
        # self-contained fallback / sanity check — see __init__.
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 \
            + msg.header.stamp.nanosec
        if (self._prev_aligned_x is not None
                and self._prev_stamp_ns is not None):
            dt = (stamp_ns - self._prev_stamp_ns) * 1e-9
            if dt > 1e-4:  # ignore degenerate or back-in-time stamps
                vx_world = (self._aligned_x - self._prev_aligned_x) / dt
                vy_world = (self._aligned_y - self._prev_aligned_y) / dt
                # World → body via current yaw (R_w2b = [[c,s],[-s,c]]).
                cy = math.cos(self._aligned_yaw)
                sy = math.sin(self._aligned_yaw)
                vx_body =  cy * vx_world + sy * vy_world
                vy_body = -sy * vx_world + cy * vy_world
                a = self._VEL_EMA_ALPHA
                self._aligned_vx_body = (
                    a * vx_body + (1.0 - a) * self._aligned_vx_body)
                self._aligned_vy_body = (
                    a * vy_body + (1.0 - a) * self._aligned_vy_body)
        self._prev_aligned_x = self._aligned_x
        self._prev_aligned_y = self._aligned_y
        self._prev_stamp_ns = stamp_ns

        self._latest_gt_stamp = msg.header.stamp

        # Publish on every GT tick (~80 Hz) so consumers always have fresh
        # state. Cone re-projection happens on /Conos_raw arrival instead.
        self._publish_state(msg.header.stamp)
        self._publish_tf(msg.header.stamp)

    def _publish_state(self, stamp) -> None:
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = MAP_FRAME
        msg.child_frame_id = BASE_FRAME
        msg.pose.pose.position.x = self._aligned_x
        msg.pose.pose.position.y = self._aligned_y
        msg.pose.pose.position.z = 0.0
        qw, qx, qy, qz = _yaw_to_quat(self._aligned_yaw)
        msg.pose.pose.orientation.w = qw
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.twist.twist.linear.x = float(self._aligned_vx_body)
        msg.twist.twist.linear.y = float(self._aligned_vy_body)
        msg.twist.twist.linear.z = 0.0
        self._state_pub.publish(msg)

    def _publish_tf(self, stamp) -> None:
        """Broadcast map → odom so that map → odom → base_link resolves to
        the GT-aligned pose at the leaf, exactly like slam_node's
        _publish_map_to_odom. odom → base_link comes from
        odometry_filter_node; we never write that edge."""
        if self._latest_odom is None:
            # No EKF sample yet (its 3 s calibration window): treat odom as
            # coincident with map so the chain still resolves to our pose.
            dx, dy, dyaw = self._aligned_x, self._aligned_y, self._aligned_yaw
        else:
            sup = self._latest_odom.pose.pose
            sup_yaw = 2.0 * math.atan2(sup.orientation.z, sup.orientation.w)
            dx, dy, dyaw = compute_map_to_odom(
                self._aligned_x, self._aligned_y, self._aligned_yaw,
                sup.position.x, sup.position.y, sup_yaw,
            )
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = MAP_FRAME
        t.child_frame_id = ODOM_FRAME
        t.transform.translation.x = float(dx)
        t.transform.translation.y = float(dy)
        t.transform.translation.z = 0.0
        qw, qx, qy, qz = _yaw_to_quat(float(dyaw))
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        self._tf_broadcaster.sendTransform(t)

    # ------------------------------------------------------------------ Cones

    def _on_cones(self, msg: MarkerArray) -> None:
        if self._gt_init_x is None:
            return  # no anchor yet
        out = MarkerArray()
        # Clear previous frame's cones in visualizers.
        clear = Marker()
        clear.header.frame_id = MAP_FRAME
        clear.action = Marker.DELETEALL
        out.markers.append(clear)

        c = math.cos(self._aligned_yaw)
        s = math.sin(self._aligned_yaw)
        for i, m in enumerate(msg.markers):
            if m.action == Marker.DELETEALL:
                continue
            bx = m.pose.position.x
            by = m.pose.position.y
            wx = self._aligned_x + c * bx - s * by
            wy = self._aligned_y + s * bx + c * by
            mk = Marker()
            mk.header.stamp = m.header.stamp
            mk.header.frame_id = MAP_FRAME
            mk.id = i
            mk.type = Marker.CYLINDER
            mk.action = Marker.ADD
            mk.pose.position.x = float(wx)
            mk.pose.position.y = float(wy)
            mk.pose.position.z = 0.0
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.2
            mk.scale.y = 0.2
            mk.scale.z = 0.3
            mk.color.r = 1.0
            mk.color.g = 1.0
            mk.color.b = 0.0
            mk.color.a = 1.0
            out.markers.append(mk)
        self._cones_pub.publish(out)


def main() -> None:
    rclpy.init()
    node = GTPoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
