"""The node applies the deskew only to clouds with point times and a fresh /odom.

The sim cloud has no ``timestamp`` field: its cones must come out untouched and
no /odom reader may be created.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
from sensor_msgs.msg import PointCloud2, PointField  # noqa: E402

from cone_detection import cone_detection_node as cdn  # noqa: E402
from cone_detection.strategies.cone_detection_strategy import ConeObservation  # noqa: E402

from test_deskew import OFFSETS, STAMP, STEP, _cloud  # noqa: E402

CONES = [ConeObservation(x=6.0, y=1.5, height_m=0.3, sigma_xy=0.05)]


class _Log:
    def warn(self, *_a, **_k):
        pass


@pytest.fixture
def node():
    n = object.__new__(cdn.ConeDetectionNode)
    n.created = []
    n.create_subscription = lambda t, topic, cb, qos: n.created.append(topic) or object()
    params = {"deskew_cones": True, "deskew_odom_topic": "/odom"}
    n.get_parameter = lambda name: SimpleNamespace(value=params[name])
    n.get_logger = lambda: _Log()
    n._sub_odom = None
    n._odom = None
    n._deskew_warned = False
    return n


def _msg(with_time: bool) -> PointCloud2:
    data, npts = _cloud(8.0, 0.0, True)
    m = PointCloud2()
    m.header.stamp.sec = int(STAMP)
    m.header.stamp.nanosec = int(round((STAMP - int(STAMP)) * 1e9))
    m.height, m.width, m.point_step = 1, npts, STEP
    m.fields = [PointField(name=k, offset=o, datatype=t, count=1)
                for k, (o, t) in OFFSETS.items() if with_time or k != "timestamp"]
    m.data = data
    return m


def test_sim_cloud_is_untouched_and_opens_no_odom_reader(node):
    assert node._deskew(_msg(with_time=False), CONES) is CONES
    assert node.created == []


def test_car_cloud_waits_for_odom_then_moves_cones(node):
    msg = _msg(with_time=True)
    assert node._deskew(msg, CONES) is CONES  # no /odom yet
    assert node.created == ["/odom"]
    node._odom = (time.monotonic(), 8.0, 0.0)
    out = node._deskew(msg, CONES)
    # seen later than the stamp while driving forward: it was further ahead at the stamp
    assert out[0].x > CONES[0].x + 0.2
    assert out[0].height_m == CONES[0].height_m and out[0].sigma_xy == CONES[0].sigma_xy
    node._odom = (time.monotonic() - 1.0, 8.0, 0.0)  # stale
    assert node._deskew(msg, CONES) is CONES
