#!/usr/bin/env python3
"""Offline replay harness for cone_graph_slam (P2 of issue #282).

Reads a rosbag2 capture (`/imu` + `/Conos_raw` + `/motor_rpm` + `/odom` +
`/testing_only/odom`) and drives `ConeGraphSlamNode` through it
deterministically. Compares SLAM pose to GT (re-anchored to SLAM's
calibration-end frame, the same way the live diagnostic does) and
emits per-scan residuals plus a pass/fail summary.

This is the gate every cone_slam-touching PR has to pass before
merging onto dev. Catches IndeterminantLinearSystemException, runaway
drift, DA cascades, all without spinning UE5.

## Capturing a bag

Inside the container, while the live pipeline is running:

    docker compose exec dv_pipeline_stack bash -lc \
      'source /opt/ros/humble/setup.bash && \
       ros2 bag record -o /tmp/slam_bag \
         /imu /Conos_raw /motor_rpm /odom /testing_only/odom'

Then drive a representative session, Ctrl-C the recorder when done.
Copy the bag out if you want it on the host:

    docker compose cp dv_pipeline_stack:/tmp/slam_bag ./slam_bag

## Running the replay

    docker compose exec -T dv_pipeline_stack bash -lc \
      'source /opt/ros/humble/setup.bash && \
       source /dv_pipeline_stack_ws/install/setup.bash && \
       python3 /dv_pipeline_stack_ws/src/cone_slam/scripts/replay_slam.py \
         /tmp/slam_bag --csv /tmp/residuals.csv'

Exit 0  = pose-vs-GT residual stayed within `--max-pose-err-m` and
          SLAM never raised.
Exit 1  = SLAM crashed mid-replay OR exceeded the residual threshold.
Exit 2  = bag missing required topics / file not found.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Cone observations: /cones_raw (dv_msgs/ConeArray) in new bags, /Conos_raw
# (MarkerArray) in older ones. slam_node._on_cones takes either.
CONE_TOPICS = ("/cones_raw", "/Conos_raw")
REQUIRED_TOPICS = {
    "/imu",
    "/motor_rpm",
    "/odom",
    "/testing_only/odom",
}


def _open_bag(path: str):
    # Imports deferred so `--help` works outside the container.
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id="sqlite3"),
        ConverterOptions("", ""),
    )
    return reader


def _topic_msg_classes(reader) -> dict:
    from rosidl_runtime_py.utilities import get_message
    out = {}
    for tt in reader.get_all_topics_and_types():
        out[tt.name] = get_message(tt.type)
    return out


def _odom_to_pose3(msg):
    """Match `cone_graph_slam_node._odom_to_pose3`."""
    import gtsam
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z),
        np.array([p.x, p.y, p.z]),
    )


def _yaw_err(slam_yaw: float, gt_yaw: float) -> float:
    """Wrap to [-pi, pi]."""
    d = slam_yaw - gt_yaw
    return float(np.arctan2(np.sin(d), np.cos(d)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline replay of cone_slam against a rosbag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("bag", help="path to rosbag2 directory")
    ap.add_argument(
        "--max-pose-err-m", type=float, default=1.5,
        help="fail if SLAM-vs-GT position residual exceeds this (default 1.5 m)")
    ap.add_argument(
        "--max-yaw-err-deg", type=float, default=15.0,
        help="fail if SLAM-vs-GT yaw residual exceeds this (default 15°)")
    ap.add_argument(
        "--csv", type=str, default=None,
        help="dump per-scan residuals to this CSV path")
    ap.add_argument(
        "--max-scans", type=int, default=0,
        help="stop after N cone scans (default 0 = no limit)")
    ap.add_argument(
        "--quiet", action="store_true",
        help="suppress per-scan progress lines")
    ap.add_argument(
        "--veto-m", type=float, default=None,
        help="override `new_landmark_proximity_veto_m`; pass 0 to "
             "disable the cascade-guard veto for baseline comparison")
    args = ap.parse_args()

    if not Path(args.bag).exists():
        print(f"bag not found: {args.bag}", file=sys.stderr)
        return 2

    reader = _open_bag(args.bag)
    topic_classes = _topic_msg_classes(reader)
    missing = REQUIRED_TOPICS - set(topic_classes.keys())
    # One cone topic only (a bag may hold both; feeding both would double every scan).
    cone_topic = next((t for t in CONE_TOPICS if t in topic_classes), None)
    if cone_topic is None:
        missing = missing | {" or ".join(CONE_TOPICS)}
    wanted = REQUIRED_TOPICS | {cone_topic}
    if missing:
        print(f"bag missing required topics: {sorted(missing)}",
              file=sys.stderr)
        print(f"  found topics: {sorted(topic_classes.keys())}",
              file=sys.stderr)
        return 2

    # Heavy ROS imports past the arg-parse / file-presence checks.
    import rclpy
    from rclpy.serialization import deserialize_message

    rclpy.init()
    node = None
    try:
        from cone_slam.cone_graph_slam_node import ConeGraphSlamNode

        node = ConeGraphSlamNode()

        if args.veto_m is not None:
            from rclpy.parameter import Parameter
            node.set_parameters([Parameter(
                "new_landmark_proximity_veto_m",
                Parameter.Type.DOUBLE,
                float(args.veto_m))])

        # Lifecycle: ConeGraphSlamNode is a LifecycleNode; its _preint
        # / _graph / _db only exist after on_configure(), and the
        # subscription handlers only run their full logic past
        # on_activate. The live pipeline issues these transitions via
        # mode_manager; replay drives them directly. The state objects
        # are unused by the callbacks, hence the simple stub.
        from rclpy.lifecycle import State as LifecycleState
        from lifecycle_msgs.msg import State as StateMsg
        _stub = LifecycleState(StateMsg.PRIMARY_STATE_UNCONFIGURED,
                               "unconfigured")
        node.on_configure(_stub)
        node.on_activate(_stub)

        residuals: list[dict] = []

        def _record_scan(t: float) -> None:
            """Snapshot SLAM pose vs GT (re-anchored). Mirrors the
            node's `_publish_gt_aligned` math."""
            if node._latest_result is None:
                return
            if node._gt_init_pose is None or node._latest_gt is None:
                return
            gt_now = _odom_to_pose3(node._latest_gt)
            gt_aligned = node._gt_init_pose.inverse().compose(gt_now)
            slam = node._latest_result.pose
            err = float(np.hypot(
                slam.x() - gt_aligned.x(),
                slam.y() - gt_aligned.y()))
            yaw_err = _yaw_err(
                slam.rotation().yaw(),
                gt_aligned.rotation().yaw())
            residuals.append({
                "t": t,
                "step": node._graph.step,
                "slam_x": slam.x(),
                "slam_y": slam.y(),
                "slam_yaw": slam.rotation().yaw(),
                "gt_x": gt_aligned.x(),
                "gt_y": gt_aligned.y(),
                "gt_yaw": gt_aligned.rotation().yaw(),
                "err_m": err,
                "yaw_err_rad": yaw_err,
            })

        n_imu = n_cones = n_rpm = n_sup_odom = n_gt = 0

        # Read messages in chronological order. rosbag2's
        # SequentialReader interleaves topics by recording timestamp,
        # which is the right order for replay.
        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            if topic not in wanted:
                continue
            cls = topic_classes[topic]
            msg = deserialize_message(raw, cls)
            try:
                if topic == "/imu":
                    node._on_imu(msg)
                    n_imu += 1
                elif topic == "/motor_rpm":
                    node._on_rpm(msg)
                    n_rpm += 1
                elif topic == "/odom":
                    node._on_supervisor_odom(msg)
                    n_sup_odom += 1
                elif topic == "/testing_only/odom":
                    node._on_gt_odom(msg)
                    n_gt += 1
                elif topic == cone_topic:
                    node._on_cones(msg)
                    n_cones += 1
                    _record_scan(t_ns * 1e-9)
                    if not args.quiet and n_cones % 50 == 0 and residuals:
                        last = residuals[-1]
                        print(f"  scan {n_cones} step={last['step']} "
                              f"err={last['err_m']:.2f}m "
                              f"yaw_err={np.degrees(last['yaw_err_rad']):+.1f}°")
                    if args.max_scans and n_cones >= args.max_scans:
                        break
            except Exception as ex:
                print(f"\nSLAM crashed during {topic} (scan #{n_cones}): "
                      f"{type(ex).__name__}: {ex}", file=sys.stderr)
                return 1

        # ----- Summary -----
        print(f"\nReplay summary:")
        print(f"  imu:   {n_imu} samples")
        print(f"  cones: {n_cones} scans")
        print(f"  rpm:   {n_rpm} samples")
        print(f"  /odom: {n_sup_odom} samples")
        print(f"  gt:    {n_gt} samples")
        if n_sup_odom == 0:
            print("  WARNING: no /odom in bag — SLAM EKF pose prior (#545) was not exercised",
                  file=sys.stderr)

        if not residuals:
            print("  no residuals collected — calibration may not have "
                  "completed before the bag ended", file=sys.stderr)
            return 1

        errs = np.array([r["err_m"] for r in residuals])
        yaw_errs_deg = np.array(
            [abs(np.degrees(r["yaw_err_rad"])) for r in residuals])

        max_err_idx = int(errs.argmax())
        max_yaw_idx = int(yaw_errs_deg.argmax())

        print(f"  pose-vs-GT: "
              f"mean={errs.mean():.2f}m "
              f"max={errs.max():.2f}m "
              f"@scan={max_err_idx}/step={residuals[max_err_idx]['step']}")
        print(f"  yaw-vs-GT:  "
              f"mean={yaw_errs_deg.mean():.2f}° "
              f"max={yaw_errs_deg.max():.2f}° "
              f"@scan={max_yaw_idx}/step={residuals[max_yaw_idx]['step']}")

        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(residuals[0].keys()))
                w.writeheader()
                w.writerows(residuals)
            print(f"  residuals csv: {args.csv}")

        failed = False
        if errs.max() > args.max_pose_err_m:
            print(f"FAIL: max pose error {errs.max():.2f}m "
                  f"> threshold {args.max_pose_err_m:.2f}m", file=sys.stderr)
            failed = True
        if yaw_errs_deg.max() > args.max_yaw_err_deg:
            print(f"FAIL: max yaw error {yaw_errs_deg.max():.2f}° "
                  f"> threshold {args.max_yaw_err_deg:.2f}°", file=sys.stderr)
            failed = True
        return 1 if failed else 0
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
