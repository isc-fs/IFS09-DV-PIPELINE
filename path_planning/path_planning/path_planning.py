"""Path planning ROS 2 LifecycleNode — thin adapter around FaSTTUBe.

Subscribes:
  /Conos          (visualization_msgs/MarkerArray) — map-frame cone map
                  from slam_node (was odom-frame pre-#382). Each
                  marker carries:
                    - pose.position (x, y, z)        — map-frame cone center
                    - color (r, g, b)                — ignored (cones are position-only)
                    - id                             — persistent landmark id

Publishes:
  /Path           (nav_msgs/Path) — interpolated centerline in `map`
                  frame (was `odom` pre-#382), consumed by control.

Looks up TF:
  map → base_link  for car position + heading. The chain
                   map→odom→base_link resolves to SLAM's drift-
                   corrected absolute pose: slam_node owns map→odom,
                   sim_supervisor owns odom→base_link. If either edge
                   is missing we skip the tick rather than fall back
                   on a stale or wrong frame.

Algorithm: delegated to `fasttube_adapter.FasttubeAdapter`, which wraps
`fsd_path_planning.PathPlanner` (FaSTTUBe / papalotis, MIT). This module
is only the ROS 2 plumbing — color decoding, TF lookup, message-shape
conversion, and per-second instrumentation.

Lifecycle layout:

  on_configure   create lifecycle publishers (/Path + debug overlay),
                 TF buffer/listener, FaSTTUBe adapter, open optional
                 DV_PLANNER_CAPTURE file.
  on_activate    create /Conos subscription, reset rate stats.
  on_deactivate  destroy /Conos subscription.
  on_cleanup     destroy publishers, drop adapter + TF listener, close
                 capture file.

History:
  - PR #243 replaced the in-house Delaunay+best-first walker with the
    FaSTTUBe library. The walker was poisoned by orange-classified false
    positives in the cone soup (independently tracked on fix/241) and
    struggled at one-sided observation regions (#189). FaSTTUBe sorts
    each side independently and matches across sides, eliminating both
    failure classes in one swap. The Delaunay debug overlay
    (`/path_planning/delaunay`) was dropped along with the algorithm —
    the new planner doesn't expose triangulation internals and a stale
    fake overlay would mislead more than help.
  - feat/30 had previously replaced an even older fsd_path_planning-based
    implementation with the from-scratch walker; this PR returns to the
    FaSTTUBe library now that pip install issues are resolved (pinned
    commit, --no-deps in the Dockerfile).
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import rclpy
from node_base.base_lifecycle_node import BaseLifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn, State as LifecycleState

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float32

from transforms3d.euler import quat2euler, euler2quat

from path_planning.core_types import Cone, Pose2D
from path_planning.fasttube_adapter import (
    FasttubeAdapter,
    PlanDebug,
    warmup_numba_planner,
)
from path_planning.planner_strategies import (
    PATH_PLANNING_STRATEGY_MAP,
    PathPlannerStrategy,
)


def _build_debug_markers(debug: PlanDebug) -> MarkerArray:
    """Pack the FaSTTUBe per-side sorted cones into a MarkerArray.

    Three layers, all in `map` frame (was `odom` pre-#382):
      - `left_chain`  blue line strip + spheres   — left_with_virtual
      - `right_chain` yellow line strip + spheres — right_with_virtual
      - DELETEALL leader so previous-frame markers don't accumulate
        when the cone field shrinks (e.g. car drives past cones).

    "with_virtual" is what the library *settled on* after cross-side
    matching, not the raw sorted input. So virtually-matched cones (the
    library's inference for missing-side cones) appear in the chain
    too — useful when debugging a one-sided observation case.
    """
    arr = MarkerArray()

    clear = Marker()
    clear.action = Marker.DELETEALL
    clear.header.frame_id = "map"
    arr.markers.append(clear)

    for ns, color_rgba, idx, xy in (
        ("left_chain",  (0.10, 0.45, 1.0, 0.9), 1, debug.left_with_virtual),
        ("right_chain", (1.0, 0.95, 0.10, 0.9), 2, debug.right_with_virtual),
    ):
        if xy is None or xy.size == 0:
            continue

        line = Marker()
        line.header.frame_id = "map"
        line.ns = ns
        line.id = idx
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.10
        line.color.r, line.color.g, line.color.b, line.color.a = color_rgba
        for x, y in xy:
            p = Point()
            p.x = float(x); p.y = float(y); p.z = 0.05
            line.points.append(p)
        arr.markers.append(line)

        spheres = Marker()
        spheres.header.frame_id = "map"
        spheres.ns = ns
        spheres.id = idx + 10
        spheres.type = Marker.SPHERE_LIST
        spheres.action = Marker.ADD
        spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.30
        spheres.color.r, spheres.color.g, spheres.color.b, spheres.color.a = color_rgba
        for x, y in xy:
            p = Point()
            p.x = float(x); p.y = float(y); p.z = 0.10
            spheres.points.append(p)
        arr.markers.append(spheres)

    return arr


def _pose_stamped(x: float, y: float, yaw: float,
                  curvature: float = 0.0) -> PoseStamped:
    """Build a PoseStamped, smuggling curvature in pose.position.z.

    nav_msgs/Path has no per-pose curvature field. We could add a
    custom message, but that's an invasive multi-package change for a
    single float. Instead: the pipeline runs in a flat-2D world
    (`base_link.z = 0`), so `pose.position.z` is otherwise unused.
    Encode the path's signed curvature there as a side-channel between
    `path_planning` and `control` — the controller's `_on_path` reads
    it back into `ReferenceTrajectory.curvature`. (#260 follow-up)

    Cosmetic side-effect: visualizers will render the path slightly
    above z=0 on tight bends. Acceptable — the displaced height is at
    most ~|κ| metres (typically < 0.5 m), barely visible.
    """
    p = PoseStamped()
    p.header.frame_id = "map"
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = float(curvature)
    qw, qx, qy, qz = euler2quat(0.0, 0.0, yaw)
    p.pose.orientation.w = float(qw)
    p.pose.orientation.x = float(qx)
    p.pose.orientation.y = float(qy)
    p.pose.orientation.z = float(qz)
    return p


class PathPlanningNode(BaseLifecycleNode):
    """Lifecycle-managed FaSTTUBe planner adapter.

    See module docstring for I/O and the lifecycle split.
    """

    NODE_NAME = "path_planning_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)

        # I/O references — populated in on_configure / on_activate.
        self.publisher_path = None
        self.publisher_debug = None
        self._pub_hz = None
        self._pub_latency_ms = None
        self._pub_n_waypoints = None
        self._pub_length_m = None
        self._pub_empty = None
        self._pub_tf_miss = None
        self._sub = None

        self.tf_buffer: Optional[Buffer] = None
        self.tf_listener: Optional[TransformListener] = None

        self._adapter: Optional[FasttubeAdapter] = None

        # Per-second instrumentation. Each callback falls into exactly
        # one bucket; rates are logged every ~1 s in _maybe_log_stats.
        # In a healthy run cb≈publish; everything else is a starvation
        # signal.
        self._stats: dict = {}
        self._stats_prev: dict = {}
        self._stats_last_log_ns = 0

        # Optional per-tick capture: when DV_PLANNER_CAPTURE is set to
        # a writable file path, every callback dumps (cones, pose,
        # n_path) as a JSON line. Lets us replay real failing scenes
        # through the planner offline and write regression tests with
        # actual SLAM-derived data — synthetic geometries miss the
        # frame-to-frame-instability + sparse-cone failure modes that
        # show up in PIE.
        self._capture_path: str = ""
        self._capture_fh = None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        if self._behavior not in PATH_PLANNING_STRATEGY_MAP:
            self.get_logger().error(f"Unknown planner behavior {self._behavior!r}")
            return TransitionCallbackReturn.FAILURE

        ret = super().on_configure(state)
        if ret != TransitionCallbackReturn.SUCCESS:
            return ret

        self.get_logger().info("on_configure: publishers + TF + adapter + capture")

        strategy = PATH_PLANNING_STRATEGY_MAP[self._behavior]()
        if not isinstance(strategy, PathPlannerStrategy):
            self.get_logger().error("Strategy is not a PathPlannerStrategy")
            return TransitionCallbackReturn.FAILURE
        self._strategy = strategy
        mission_type = strategy.get_mission_type()

        self.publisher_path = self.create_lifecycle_publisher(
            Path, "Path", 10)
        # Debug overlay (#254) — three layers in one MarkerArray showing
        # FaSTTUBe's per-side sorted cones (with virtual cones for
        # missing sides). Visualises whether the colour-blind sort is
        # putting cones on the side a human would intuitively call
        # left/right. Frame matches /Path (`odom`).
        self.publisher_debug = self.create_lifecycle_publisher(
            MarkerArray, "/path_planning/debug", 10)
        self._pub_hz = self.create_lifecycle_publisher(
            Float32, "/path_planning/hz", 10)
        self._pub_latency_ms = self.create_lifecycle_publisher(
            Float32, "/path_planning/latency_ms", 10)
        self._pub_n_waypoints = self.create_lifecycle_publisher(
            Float32, "/path_planning/n_waypoints", 10)
        self._pub_length_m = self.create_lifecycle_publisher(
            Float32, "/path_planning/length_m", 10)
        self._pub_empty = self.create_lifecycle_publisher(
            Float32, "/path_planning/empty", 10)
        self._pub_tf_miss = self.create_lifecycle_publisher(
            Float32, "/path_planning/tf_miss", 10)

        self.tf_buffer = Buffer()
        # spin_thread=False: this process already rclpy.spin()s the node.
        # Humble's TransformListener default (True) adds the same node to a
        # second executor, so every /tf sample is inserted twice and the
        # second hit logs TF_OLD_DATA (equal timestamps are "from the past").
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=False)

        self._adapter = FasttubeAdapter(mission_type)
        self.get_logger().info(
            f"PathPlanner for behavior={self._behavior!r} "
            f"mission_type={mission_type.name}"
        )

        # Numba JIT warmup — same rationale as cone_detection_node: pay
        # the fsd_path_planning kernel compile here, during Phase-1
        # warming_up (parallel fan-out, 120 s per-node budget), instead
        # of on the first /Conos callback while the car is driving.
        self.get_logger().info(
            "warming up FaSTTUBe Numba kernels (~0.5 s cached, tens of s first ever)"
        )
        warm_s = warmup_numba_planner(mission_type)
        self.get_logger().info(f"FaSTTUBe Numba warmup complete ({warm_s:.1f} s)")

        # Optional capture file
        self._capture_path = os.environ.get("DV_PLANNER_CAPTURE", "")
        if self._capture_path:
            try:
                self._capture_fh = open(self._capture_path, "w")
                self.get_logger().info(
                    f"DV_PLANNER_CAPTURE → {self._capture_path}")
            except OSError as ex:
                self.get_logger().error(f"capture open failed: {ex}")
                self._capture_fh = None

        self._reset_stats()
        return TransitionCallbackReturn.SUCCESS

    def on_activate(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate: subscribing to /Conos")
        self._reset_stats()
        self._sub = self.create_subscription(
            MarkerArray, "Conos", self._on_cones, 10)
        return super().on_activate(state)

    def on_deactivate(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate: dropping /Conos subscription")
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        return super().on_deactivate(state)

    def on_cleanup(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup: destroying publishers + TF")
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        for pub in (
            self.publisher_path,
            self.publisher_debug,
            self._pub_hz,
            self._pub_latency_ms,
            self._pub_n_waypoints,
            self._pub_length_m,
            self._pub_empty,
            self._pub_tf_miss,
        ):
            if pub is not None:
                self.destroy_publisher(pub)
        self.publisher_path = None
        self.publisher_debug = None
        self._pub_hz = None
        self._pub_latency_ms = None
        self._pub_n_waypoints = None
        self._pub_length_m = None
        self._pub_empty = None
        self._pub_tf_miss = None
        self.tf_listener = None
        self.tf_buffer = None
        self._adapter = None
        if self._capture_fh is not None:
            try:
                self._capture_fh.close()
            except Exception:
                pass
            self._capture_fh = None
        self._reset_stats()
        return super().on_cleanup(state)

    def on_shutdown(
        self, state: LifecycleState
    ) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        if self._capture_fh is not None:
            try:
                self._capture_fh.close()
            except Exception:
                pass
            self._capture_fh = None
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_stats(self) -> None:
        self._stats = {
            "callbacks": 0,
            "no_cones": 0,        # 0 cones after color decode
            "tf_miss": 0,         # TF lookup failed
            "plan_empty": 0,      # adapter returned []
            "publish": 0,
        }
        self._stats_prev = dict(self._stats)
        self._stats_last_log_ns = 0

    def _maybe_log_stats(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if self._stats_last_log_ns == 0:
            self._stats_last_log_ns = now_ns
            return
        if now_ns - self._stats_last_log_ns < 1_000_000_000:
            return
        dt = (now_ns - self._stats_last_log_ns) / 1e9
        d = {k: self._stats[k] - self._stats_prev[k] for k in self._stats}
        hz = d["callbacks"] / dt if dt > 0.0 else 0.0
        self.get_logger().info(
            f"PATH_RATE cb={hz:4.1f}/s "
            f"pub={d['publish']/dt:4.1f}/s "
            f"no_cones={d['no_cones']} tf_miss={d['tf_miss']} "
            f"plan_empty={d['plan_empty']}"
        )
        self._publish_if_sub(self._pub_hz, hz)
        self._stats_prev = dict(self._stats)
        self._stats_last_log_ns = now_ns

    # ------------------------------------------------------------------
    # Cone callback (unchanged behaviour from pre-lifecycle version)
    # ------------------------------------------------------------------
    def _on_cones(self, msg: MarkerArray) -> None:
        # Defensive: subscription is destroyed before publishers in
        # on_deactivate, but a callback already inflight when deactivate
        # fires can race past that.
        if self.publisher_path is None or self._adapter is None:
            return

        t0 = time.perf_counter()
        self._stats["callbacks"] += 1

        cones: List[Cone] = []
        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                continue
            cones.append(Cone(
                x=float(m.pose.position.x),
                y=float(m.pose.position.y),
            ))

        if not cones:
            self._stats["no_cones"] += 1
            self._maybe_log_stats()
            self._publish_plan_diag(n_waypoints=0, length_m=0.0, empty=1.0, tf_miss=0.0)
            self._publish_latency_ms(t0)
            return

        # Pose lookup in map frame (Phase 2 — #382). The chain
        # map→odom→base_link resolves to SLAM's absolute pose; map→odom
        # is owned by slam_node (drift correction), odom→base_link
        # by sim_supervisor (100 Hz dead-reckoning).
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}")
            self._stats["tf_miss"] += 1
            self._maybe_log_stats()
            self._publish_plan_diag(n_waypoints=0, length_m=0.0, empty=1.0, tf_miss=1.0)
            self._publish_latency_ms(t0)
            return

        yaw = quat2euler([
            tf.transform.rotation.w,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
        ])[2]
        pose = Pose2D(
            x=float(tf.transform.translation.x),
            y=float(tf.transform.translation.y),
            yaw=float(yaw),
        )

        path_points, debug = self._adapter.plan(cones, pose)

        # Always publish the debug overlay (even when the path is empty —
        # shows what cones FaSTTUBe assigned to each side, which is the
        # most useful signal when the planner just gave up).
        self.publisher_debug.publish(_build_debug_markers(debug))

        # Tick capture: dump (cones, pose, path xy) for offline replay
        # and path-vs-CSV-centerline analysis (#254). Path is dumped as
        # a flat list of [x, y] pairs in `odom` frame so the offline
        # tool can compute perpendicular distance to the loaded track
        # CSV without re-running the planner.
        if self._capture_fh is not None:
            try:
                self._capture_fh.write(json.dumps({
                    "t_ns": self.get_clock().now().nanoseconds,
                    "pose": [pose.x, pose.y, pose.yaw],
                    "cones": [[c.x, c.y] for c in cones],
                    "n_path": len(path_points),
                    "path": [[p.x, p.y] for p in path_points],
                }) + "\n")
                self._capture_fh.flush()
            except Exception:
                pass

        if not path_points:
            self._stats["plan_empty"] += 1
            self._maybe_log_stats()
            self._publish_plan_diag(n_waypoints=0, length_m=0.0, empty=1.0, tf_miss=0.0)
            self._publish_latency_ms(t0)
            return

        out = Path()
        out.header.frame_id = "map"
        for p in path_points:
            out.poses.append(_pose_stamped(p.x, p.y, p.yaw, p.curvature))

        self.publisher_path.publish(out)
        self._stats["publish"] += 1
        length_m = 0.0
        for i in range(1, len(path_points)):
            dx = path_points[i].x - path_points[i - 1].x
            dy = path_points[i].y - path_points[i - 1].y
            length_m += (dx * dx + dy * dy) ** 0.5
        self._maybe_log_stats()
        self._publish_plan_diag(
            n_waypoints=len(path_points),
            length_m=length_m,
            empty=0.0,
            tf_miss=0.0,
        )
        self._publish_latency_ms(t0)

    def _publish_if_sub(self, pub, value: float) -> None:
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

    def _publish_latency_ms(self, t0: float) -> None:
        self._publish_if_sub(self._pub_latency_ms, (time.perf_counter() - t0) * 1000.0)

    def _publish_plan_diag(
        self,
        *,
        n_waypoints: int,
        length_m: float,
        empty: float,
        tf_miss: float,
    ) -> None:
        self._publish_if_sub(self._pub_n_waypoints, float(n_waypoints))
        self._publish_if_sub(self._pub_length_m, float(length_m))
        self._publish_if_sub(self._pub_empty, float(empty))
        self._publish_if_sub(self._pub_tf_miss, float(tf_miss))


def main(args=None) -> None:
    """Entry point: spin PathPlanningNode until SIGINT."""
    rclpy.init(args=args)
    node = PathPlanningNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
