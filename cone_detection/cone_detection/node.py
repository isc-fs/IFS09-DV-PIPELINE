"""ROS 2 lifecycle node for LiDAR cone detection.

Subscribes to /fsds/lidar/Lidar1 (sensor_msgs/PointCloud2) and publishes:

  - /Conos_raw     — every detected cone (MarkerArray, base_link frame).
                     marker.scale.x carries per-cone σ_xy for SLAM;
                     marker.scale.z carries cluster / template height for the
                     downstream big-orange-vs-small classifier.
  - /Conos_Orange  — big-orange cones only, kept on a separate stream so the
                     autonomous-stop control logic does not need to filter the
                     full cone list.
  - /lidar_points/ground — cropped scan after the same ground-frame rotation used
                     for clustering, floor at z=0 so it overlays /Conos_raw
                     (includes ground inliers). Published only if subscribed.
  - /lidar_points/above_ground — non-ground points only (the points clustering
                     actually sees). Published only if subscribed.
  - /cone_detection/{hz,latency_ms,ground_ms,dbscan_ms,fit_ms,n_accepted}
                     — per-scan / per-second Float32 diagnostics for Lichtblick.

Perception algorithms live in :class:`~cone_detection.strategies.ConeDetectionStrategy`
implementations (``detect_cones`` → :class:`DetectionResult`). This node parses
PointCloud2, runs the strategy, builds markers, publishes, and owns diagnostics.

Held in its own ROS node (rather than fused into cone_graph_slam) so the Numba JIT
compile happens once at configure and persists across SLAM/control restarts during dev.

Lifecycle layout (driven by mode_manager → change_state):

  on_configure    create lifecycle publishers + strategy ``configure()`` (Numba
                  warmup — the expensive ~10–20 s step). Lands during Phase 1
                  ``warming_up``; supervisor heartbeats absorb the delay so
                  downstream timeouts do not trip.
  on_activate     create LiDAR subscription, reset per-second diag counters,
                  ``super().on_activate()`` flips publishers to emitting state.
  on_deactivate   destroy LiDAR subscription so a deactivated node does not burn
                  CPU parsing dropped scans.
  on_cleanup      destroy publishers and strategy. Re-configuring re-runs warmup
                  (rare; normally the container holds the node inactive between
                  missions).
"""

from __future__ import annotations

import dataclasses
import time

import rclpy
import numpy as np
from typing import TYPE_CHECKING

from builtin_interfaces.msg import Time
from node_base.base_lifecycle_node import BaseLifecycleNode
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn, State
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from visualization_msgs.msg import MarkerArray

from cone_detection.conversions import cones_to_markers, pointcloud2_to_xyz, xyz_to_pointcloud2

if TYPE_CHECKING:
    from cone_detection.strategies.cone_detection_strategy import (
        ConeDetectionStrategy,
        DetectionResult,
    )


def _param_default(f: dataclasses.Field):
    """A ``ConeDetectionConfig`` field's default as the ROS parameter's value, typed by the field
    (``x: float = 1`` is a double parameter, so a params file may give it 1.5)."""
    kind = {"float": float, "int": int, "bool": bool}.get(str(f.type), str)
    return kind(f.default)


def _cone_detection_strategy_map() -> dict[str, type]:
    """Loaded in on_configure only — importing earlier pulls sklearn/numba before
    ``~/setup`` exists and mode_manager's wait_for_service times out on cold start.
    """
    from cone_detection.strategies.base_cone_detection import BaseConeDetection

    return {"base": BaseConeDetection}


QOS_LATEST = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.VOLATILE,
)

_PERF_TOPICS = (
    ("hz", "_pub_hz"),
    ("latency_ms", "_pub_latency_ms"),
    ("ground_ms", "_pub_ground_ms"),
    ("dbscan_ms", "_pub_dbscan_ms"),
    ("fit_ms", "_pub_fit_ms"),
    ("n_accepted", "_pub_n_accepted"),
    ("n_clusters", "_pub_n_clusters"),
    ("n_input_points", "_pub_n_input_points"),
    ("n_after_shape", "_pub_n_after_shape"),
    ("n_far_dropped", "_pub_n_far_dropped"),
    ("n_left", "_pub_n_left"),
    ("n_right", "_pub_n_right"),
)


# Every publisher attribute: created in on_configure, destroyed in on_cleanup.
_PUBLISHER_ATTRS = ("_pub_markers", "_pub_orange", "_pub_ground", "_pub_above") + tuple(
    attr for _, attr in _PERF_TOPICS
)


def _has_subs(pub) -> bool:
    """True if anyone is listening. Fail-open if the handle has no count API."""
    if pub is None:
        return False
    getter = getattr(pub, "get_subscription_count", None)
    if getter is None:
        return True
    try:
        return int(getter()) > 0
    except Exception:
        return True


def _publish_f32(pub, value: float) -> None:
    if not _has_subs(pub):
        return
    msg = Float32()
    msg.data = float(value)
    pub.publish(msg)


def _publish_cloud(pub, xyz: np.ndarray, stamp: Time) -> None:
    if not _has_subs(pub):
        return
    pub.publish(xyz_to_pointcloud2(xyz, stamp))


class ConeDetectionNode(BaseLifecycleNode):
    """LifecycleNode publishing per-scan cone observations."""

    NODE_NAME = "cone_detection_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        # One ROS parameter per ConeDetectionConfig field (same name, type and default), so the
        # perception settings are set like every other node's: bringup/config/params.yaml, or a
        # later --params-file. cone_detection.config imports nothing heavy (see its docstring).
        from cone_detection.config import ConeDetectionConfig

        for f in dataclasses.fields(ConeDetectionConfig):
            self.declare_parameter(f.name, _param_default(f))
        # All I/O is created in on_configure / on_activate. Keep __init__
        # side-effect-free so a freshly constructed but unconfigured node
        # holds no resources.
        self._is_active = False
        self._cone_strategy: ConeDetectionStrategy | None = None
        for attr in _PUBLISHER_ATTRS:
            setattr(self, attr, None)
        self._sub = None
        self._reset_diag()

    def _config_from_params(self, base):
        """The strategy's own config (or the defaults) with every parameter that differs from
        its field default applied on top. Returns ``(config, {name: value} applied)``.

        A strategy that tunes a field for its mission keeps it unless a parameter changes it."""
        from cone_detection.config import ConeDetectionConfig

        defaults = ConeDetectionConfig()
        changed = {}
        for f in dataclasses.fields(ConeDetectionConfig):
            value = self.get_parameter(f.name).value
            if value != getattr(defaults, f.name):
                changed[f.name] = value
        return dataclasses.replace(base or defaults, **changed), changed

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        ret = super().on_configure(state)
        if ret != TransitionCallbackReturn.SUCCESS:
            return ret
        try:
            strategy_map = _cone_detection_strategy_map()
            if self._behavior not in strategy_map:
                self.get_logger().error(
                    f"Unknown cone_detection behavior '{self._behavior}' "
                    f"(known: {list(strategy_map)})"
                )
                return TransitionCallbackReturn.FAILURE

            self._pub_markers = self.create_lifecycle_publisher(
                MarkerArray, "Conos_raw", 10,
            )
            # Dedicated stream of big-orange cones (the FS finish-line markers).
            # Kept separate from /Conos_raw so SLAM's blue/yellow classifier
            # does not need to learn about orange, and so downstream consumers
            # (autonomous-stop logic in the control node) can subscribe without
            # filtering the whole cone list. Positions are in the car frame.
            self._pub_orange = self.create_lifecycle_publisher(
                MarkerArray, "Conos_Orange", 10,
            )
            # Ground-aligned viz cloud (same rotation as /Conos_raw).
            self._pub_ground = self.create_lifecycle_publisher(
                PointCloud2, "/lidar_points/ground", QOS_LATEST,
            )
            self._pub_above = self.create_lifecycle_publisher(
                PointCloud2, "/lidar_points/above_ground", QOS_LATEST,
            )
            for topic, attr in _PERF_TOPICS:
                setattr(
                    self,
                    attr,
                    self.create_lifecycle_publisher(
                        Float32, f"/cone_detection/{topic}", 10
                    ),
                )

            # Numba JIT compile — dominant cost of bring-up; lives in
            # strategy.configure() during on_configure (not on_activate) so an
            # inactive→active toggle is instant.
            strategy_cls = strategy_map[self._behavior]
            config, changed = self._config_from_params(
                getattr(strategy_cls, "CONE_DETECTION_CONFIG", None)
            )
            if changed:
                self.get_logger().info(
                    "perception settings from parameters: "
                    + ", ".join(f"{k}={v!r}" for k, v in sorted(changed.items()))
                )
            self._cone_strategy = strategy_cls(self.get_logger(), config=config)
            self._cone_strategy.configure()
            self.get_logger().info(
                f"Cone detection pipeline ready | mode: {self._mode_name} | "
                f"behavior: {self._behavior}"
            )
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._is_active = True
        self._reset_diag()
        self._sub = self.create_subscription(
            PointCloud2,
            "/fsds/lidar/Lidar1",
            self.listener_callback,
            QOS_LATEST,
        )
        self.get_logger().info(f"Activated in mode: {self._mode_name}")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._is_active = False
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        for attr in _PUBLISHER_ATTRS:
            pub = getattr(self, attr, None)
            if pub is not None:
                self.destroy_publisher(pub)
                setattr(self, attr, None)
        self._cone_strategy = None
        return super().on_cleanup(state)

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._cone_strategy = None
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Per-scan callback
    # ------------------------------------------------------------------
    # Kept on the class for callers that decode through the node (tests, sim tools).
    pointcloud2_to_xyz = staticmethod(pointcloud2_to_xyz)

    def listener_callback(self, msg: PointCloud2) -> None:
        # Defensive: should not fire if subscription was destroyed in
        # on_deactivate, but a multi-threaded executor cannot guarantee
        # mid-callback teardown safety.
        if not self._is_active or self._cone_strategy is None:
            return
        if self._pub_markers is None or self._pub_orange is None:
            return

        point_cloud = self.pointcloud2_to_xyz(msg)
        want_ground = _has_subs(self._pub_ground)
        t0 = time.perf_counter()
        result = self._cone_strategy.detect_cones(
            point_cloud, viz_full_cloud=want_ground
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        big_orange_threshold = self._cone_strategy.big_orange_height_threshold_m()
        self._accumulate_diagnostics(result, big_orange_threshold)
        self._publish_perf(result, total_ms)
        if want_ground:
            _publish_cloud(self._pub_ground, result.rotated_xyz, msg.header.stamp)
        _publish_cloud(self._pub_above, result.above_ground_xyz, msg.header.stamp)
        marker_array, orange_array = cones_to_markers(
            result.cones,
            stamp=msg.header.stamp,
            big_orange_threshold_m=big_orange_threshold,
        )
        self._pub_markers.publish(marker_array)
        # Publish even when the current scan has no big-orange cones so
        # downstream consumers do not keep a stale cache when the gate exits
        # the LiDAR FoV at close range (sensor is ~1.4 m forward of the car, so
        # start-gate cones leave the ±60° H-FOV before the car passes them).
        self._pub_orange.publish(orange_array)

    def _accumulate_diagnostics(self, result: DetectionResult, threshold: float) -> None:
        """Sum per-scan filter counters and emit one CONE_FILTER line per second."""
        per_scan = dict(result.debug_counters)
        n_left = n_right = n_centerline = n_bigorange = 0
        # Pre-compute per-side tallies. Detection does not colour-classify (SLAM
        # does downstream), but body-y sign matches SLAM's classify() — a proxy
        # for whether a cone-imbalance in the map (#189) starts here or downstream.
        for cone in result.cones:
            if cone.height_m > threshold:
                n_bigorange += 1
            elif cone.y > 0.5:  # body-y; +Y = left in REP-103
                n_left += 1
            elif cone.y < -0.5:
                n_right += 1
            else:
                n_centerline += 1
        by_side = {
            "accepted_left": n_left,
            "accepted_right": n_right,
            "accepted_centerline": n_centerline,
            "accepted_bigorange": n_bigorange,
        }
        per_scan.update(by_side)
        result.debug_counters.update(by_side)

        if per_scan.get("dbscan_guard_scans", 0):
            # Loud but throttled: this scan was NOT a cone scene (car facing
            # terrain / an object, or a bad ground plane). Without the guard
            # it would have cost seconds and gigabytes — see
            # ConeDetectionConfig.dbscan_max_points.
            self.get_logger().warning(
                "DBSCAN guard: above-ground cloud too large/dense, dropped "
                f"{per_scan.get('dbscan_guard_dropped', 0)} pts before "
                "clustering (car off-track or ground plane mis-fit?)",
                throttle_duration_sec=5.0,
            )

        for k, v in per_scan.items():
            self._diag[k] = self._diag.get(k, 0) + v
        self._diag_n_scans += 1
        now_ns = self.get_clock().now().nanoseconds
        if self._diag_last_log_ns == 0:
            self._diag_last_log_ns = now_ns
        elif (
            now_ns - self._diag_last_log_ns >= 1_000_000_000
            and self._diag_n_scans > 0
        ):
            n = self._diag_n_scans
            dt_s = (now_ns - self._diag_last_log_ns) / 1e9
            hz = n / dt_s if dt_s > 0.0 else 0.0
            _publish_f32(self._pub_hz, hz)
            self.get_logger().info(
                f"CONE_FILTER (avg/scan over {n}): "
                f"pts={self._diag.get('n_input_points', 0) / n:5.0f} "
                f"clusters={self._diag.get('n_clusters', 0) / n:4.1f} "
                f"-> >3pts={self._diag.get('after_min_pts', 0) / n:4.1f} "
                f"-> shape={self._diag.get('after_shape_gate', 0) / n:4.1f} "
                f"-> residual={self._diag.get('after_residual_gate', 0) / n:4.1f} "
                f"accepted={self._diag.get('accepted', 0) / n:4.1f} "
                f"far_dropped={self._diag.get('far_dropped', 0) / n:.1f} "
                f"by-side: L={self._diag.get('accepted_left', 0) / n:4.1f} "
                f"R={self._diag.get('accepted_right', 0) / n:4.1f} "
                f"C={self._diag.get('accepted_centerline', 0) / n:.1f} "
                f"BO={self._diag.get('accepted_bigorange', 0) / n:.1f} "
                f"dbscan_guard={self._diag.get('dbscan_guard_scans', 0)}/{n} "
                f"hz={hz:4.1f}"
            )
            self._reset_diag()
            self._diag_last_log_ns = now_ns

    def _publish_perf(self, result: DetectionResult, total_ms: float) -> None:
        st = result.stage_timings
        _publish_f32(self._pub_latency_ms, total_ms)
        _publish_f32(self._pub_ground_ms, float(st.get("ground_ms", 0.0)))
        _publish_f32(self._pub_dbscan_ms, float(st.get("dbscan_ms", 0.0)))
        _publish_f32(self._pub_fit_ms, float(st.get("fit_ms", 0.0)))
        _publish_f32(self._pub_n_accepted, float(len(result.cones)))
        dc = result.debug_counters
        _publish_f32(self._pub_n_clusters, float(dc.get("n_clusters", 0)))
        _publish_f32(self._pub_n_input_points, float(dc.get("n_input_points", 0)))
        _publish_f32(self._pub_n_after_shape, float(dc.get("after_shape_gate", 0)))
        _publish_f32(self._pub_n_far_dropped, float(dc.get("far_dropped", 0)))
        _publish_f32(self._pub_n_left, float(dc.get("accepted_left", 0)))
        _publish_f32(self._pub_n_right, float(dc.get("accepted_right", 0)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_diag(self) -> None:
        # Per-second diagnostic accumulator for the cluster-filter pipeline.
        # Each scan fills debug_counters via the strategy; we sum across scans
        # and log one summary line per second (helps localise drop-outs, #177).
        self._diag: dict[str, int] = {}
        self._diag_n_scans = 0
        self._diag_last_log_ns = 0


def main(args=None) -> None:
    """Entry point: spin ConeDetectionNode under MultiThreadedExecutor."""
    rclpy.init(args=args)
    node = ConeDetectionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
