"""Conversions between ROS messages and the detector's numpy / dataclass types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

if TYPE_CHECKING:
    from cone_detection.strategies.cone_detection_strategy import ConeObservation


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Decode PointCloud2 to ``(N, 3)`` float32 (x, y, z).

    Reads x/y/z by their declared byte offsets over a ``point_step`` stride,
    so ANY layout works regardless of whether ``point_step`` is a whole
    number of float32s.

    Do NOT reshape by ``point_step // 4``: the Hesai ATX driver publishes a
    *packed* 26-byte point (x,y,z,intensity f32 + ring u16 + timestamp f64),
    and 26 is not divisible by 4 — ``frombuffer(...).reshape(N, 26//4)`` then
    raised ``ValueError`` on every scan, so the node silently emitted zero
    cones (the sim's 16-byte cloud hid this). Offset slicing over a uint8
    view handles packed-26, aligned-32, and the sim layouts alike.
    """
    num_points = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        num_points, msg.point_step
    )
    off = {f.name: f.offset for f in msg.fields}
    try:
        ox, oy, oz = off["x"], off["y"], off["z"]
    except KeyError as exc:  # pragma: no cover - malformed cloud
        raise ValueError(
            f"PointCloud2 missing x/y/z field: have {sorted(off)}"
        ) from exc
    # Strided float32 views straight over the message buffer (stride =
    # point_step, valid unaligned), each written once into the (N, 3)
    # result: one allocation, no per-column .copy() / stack temporaries.
    out = np.empty((num_points, 3), dtype=np.float32)
    for col, off in enumerate((ox, oy, oz)):
        out[:, col] = np.ndarray(
            (num_points,), dtype=np.float32, buffer=raw,
            offset=off, strides=(msg.point_step,),
        )
    return out


def xyz_to_pointcloud2(
    xyz: np.ndarray,
    stamp: Time,
    frame_id: str = "base_link",
) -> PointCloud2:
    """Pack an ``(N, 3)`` xyz array as a dense xyz32 PointCloud2."""
    pts = np.ascontiguousarray(np.asarray(xyz, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[1] < 3:
        pts = np.zeros((0, 3), dtype=np.float32)
    else:
        pts = pts[:, :3]
    n = int(pts.shape[0])
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = n
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = pts.tobytes()
    return msg


def cones_to_markers(
    cones: list[ConeObservation],
    *,
    stamp: Time,
    big_orange_threshold_m: float,
) -> tuple[MarkerArray, MarkerArray]:
    """Build /Conos_raw and /Conos_Orange MarkerArrays from detections.

    Both arrays start with a dedicated pose-less ``DELETEALL`` leader
    so RViz/Foxglove drop the previous scan's cubes, followed by one
    ``ADD`` marker per cone. The leader carries no cone: every
    consumer (slam_node, control_node, path_planning) skips markers
    with ``action == DELETEALL``, so a cone must never ride on it.
    (Before this, the first *real* cone was emitted as the DELETEALL
    marker itself and SLAM silently dropped one cone per scan.)
    """
    marker_array = MarkerArray()
    orange_array = MarkerArray()

    for arr in (marker_array, orange_array):
        clear = Marker()
        clear.header.frame_id = "base_link"
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

    for marker_id, cone in enumerate(cones):
        # /Conos_raw is published in the body frame; SLAM transforms to map.
        # marker.scale carries per-cone metadata for downstream SLAM:
        #   scale.x → σ_xy in metres (position uncertainty); sigma_xy < 0
        #             means unknown — SLAM uses its range-only fallback
        #   scale.y → reserved (visualization width; kept default)
        #   scale.z → height_m (template apex or fitted height)
        # SLAM reads scale.x in cone_graph_slam_node._observations_from_markers.
        marker_array.markers.append(
            _cone_cube(
                cone,
                stamp,
                marker_id=marker_id,
                scale_xy=(cone.sigma_xy if cone.sigma_xy > 0.0 else 0.1, 0.1),
                rgb=(1.0, 0.0, 1.0),
            )
        )
        if cone.height_m > big_orange_threshold_m:
            # Same pose, distinct marker list, orange colour for RViz.
            orange_array.markers.append(
                _cone_cube(
                    cone,
                    stamp,
                    marker_id=len(orange_array.markers) - 1,  # after the leader
                    scale_xy=(0.3, 0.3),
                    rgb=(1.0, 0.5, 0.0),
                )
            )

    return marker_array, orange_array


def _cone_cube(
    cone: ConeObservation,
    stamp: Time,
    *,
    marker_id: int,
    scale_xy: tuple[float, float],
    rgb: tuple[float, float, float],
) -> Marker:
    """``ADD`` cube marker at the cone's ground position in ``base_link``."""
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = stamp
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.id = marker_id
    marker.pose.position.x = cone.x
    marker.pose.position.y = cone.y
    marker.pose.position.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x, marker.scale.y = scale_xy
    marker.scale.z = max(0.1, cone.height_m)
    marker.color.a = 1.0
    marker.color.r, marker.color.g, marker.color.b = rgb
    return marker
