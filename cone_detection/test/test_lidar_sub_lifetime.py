"""Detection keeps ONE LiDAR reader for its whole life; activations only gate the callback.

Why: re-creating the reader on every mission activation meant a new DDS reader per run. In HIL
(sim bridge on Fast DDS, pipeline on Cyclone) the remote writer sometimes never matched the new
reader after an emergency, so the next run got no scans and the watchdog tripped on
"/slam/pose never published". The contract pinned here:
  * configure creates the reader; activate/deactivate never create or destroy it;
  * an inactive node ignores scans before decoding them;
  * cleanup destroys it, and a later configure creates a fresh one.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
from rclpy.lifecycle import TransitionCallbackReturn  # noqa: E402

from cone_detection import cone_detection_node as cdn  # noqa: E402
from node_base.base_lifecycle_node import BaseLifecycleNode  # noqa: E402


class _Log:
    def info(self, *_a, **_k):
        pass

    error = warn = info


@pytest.fixture
def node(monkeypatch):
    """A ConeDetectionNode without rclpy: DDS calls are counted, the base transitions succeed."""
    for name in ("on_activate", "on_deactivate", "on_cleanup"):
        monkeypatch.setattr(
            BaseLifecycleNode, name, lambda self, state: TransitionCallbackReturn.SUCCESS
        )
    n = object.__new__(cdn.ConeDetectionNode)
    n.created, n.destroyed = [], []

    def create_subscription(msg_type, topic, cb, qos):
        sub = object()
        n.created.append((topic, sub))
        return sub

    n.create_subscription = create_subscription
    n.destroy_subscription = lambda sub: n.destroyed.append(sub)
    n.destroy_publisher = lambda pub: None
    n.get_logger = lambda: _Log()
    n._mode_name = "trackdrive"
    n._is_active = False
    n._sub = None
    n._cone_strategy = None
    n._pub_markers = n._pub_orange = None
    n._reset_diag = lambda: None
    return n


def test_one_reader_across_mission_restarts(node):
    node._ensure_lidar_sub()  # what on_configure does
    assert [t for t, _ in node.created] == ["/fsds/lidar/Lidar1"]
    first = node._sub
    for _ in range(3):  # three missions, each ended by deactivate (emergency or finish)
        assert node.on_activate(None) == TransitionCallbackReturn.SUCCESS
        assert node._is_active and node._sub is first
        assert node.on_deactivate(None) == TransitionCallbackReturn.SUCCESS
        assert not node._is_active and node._sub is first
    assert len(node.created) == 1 and node.destroyed == []


def test_activate_without_configure_reader_still_creates_one(node):
    node.on_activate(None)
    assert len(node.created) == 1 and node._sub is not None


def test_inactive_node_ignores_scans_before_decoding(node, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("decoded a scan while inactive")

    monkeypatch.setattr(cdn.ConeDetectionNode, "pointcloud2_to_xyz", staticmethod(boom))
    node._ensure_lidar_sub()
    node._cone_strategy = object()
    node._pub_markers = node._pub_orange = object()
    node.listener_callback(object())  # inactive: returns early, no decode


def test_cleanup_drops_the_reader_and_configure_makes_a_new_one(node):
    node._ensure_lidar_sub()
    first = node._sub
    node.on_activate(None)
    node.on_deactivate(None)
    node.on_cleanup(None)
    assert node.destroyed == [first] and node._sub is None
    node._ensure_lidar_sub()
    assert len(node.created) == 2 and node._sub is not first
