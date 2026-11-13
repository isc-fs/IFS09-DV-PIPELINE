"""RK3588 core pinning through the launch layer (DV_CPU_PIN).

Pins two properties:
  * unset (every host except the RK3588) = no prefix, so the DVPC, the sim CI
    and developer laptops launch exactly as before;
  * set = each listed autonomy node starts under `taskset -c <cpus>`, and a
    malformed value stops the launch instead of silently not pinning.

Needs launch_ros (the ROS 2 install), like the launch files themselves.
"""
from __future__ import annotations

import pytest

pytest.importorskip("launch_ros")

from bringup import launch_common as lc  # noqa: E402

RK3588 = ("cone_detection_node=4 slam_node=5 path_planning_node=6 "
          "control_node=6 odometry_filter_node=7")


def test_unset_means_no_pinning():
    assert lc.cpu_pin_prefix("slam_node", env={}) is None
    assert lc.parse_cpu_pin(None) == {}


def test_rk3588_layout():
    env = {"DV_CPU_PIN": RK3588}
    assert lc.cpu_pin_prefix("cone_detection_node", env) == "taskset -c 4"
    assert lc.cpu_pin_prefix("slam_node", env) == "taskset -c 5"
    assert lc.cpu_pin_prefix("control_node", env) == "taskset -c 6"
    assert lc.cpu_pin_prefix("odometry_filter_node", env) == "taskset -c 7"
    assert lc.cpu_pin_prefix("mission_control_node", env) is None


def test_cpu_lists_and_ranges():
    assert lc.parse_cpu_pin("a=0-3 b=6,7 c=1,4-5") == {"a": "0-3", "b": "6,7", "c": "1,4-5"}


@pytest.mark.parametrize("bad", ["slam_node", "slam_node=", "=5", "slam_node=five", "slam_node=5;rm"])
def test_malformed_entry_stops_the_launch(bad):
    with pytest.raises(ValueError):
        lc.parse_cpu_pin(bad)


def _built(monkeypatch):
    """autonomy_actions("car") with LifecycleNode recorded: name -> kwargs."""
    seen = {}
    monkeypatch.setattr(lc, "LifecycleNode", lambda **kw: seen.setdefault(kw["name"], kw))
    lc.autonomy_actions("car")
    return seen


def test_autonomy_nodes_get_the_prefix(monkeypatch):
    monkeypatch.setenv("DV_CPU_PIN", RK3588)
    seen = _built(monkeypatch)
    assert {n: kw.get("prefix") for n, kw in seen.items()} == {
        "odometry_filter_node": "taskset -c 7",
        "cone_detection_node": "taskset -c 4",
        "slam_node": "taskset -c 5",
        "path_planning_node": "taskset -c 6",
        "control_node": "taskset -c 6",
    }


def test_autonomy_nodes_unpinned_by_default(monkeypatch):
    monkeypatch.delenv("DV_CPU_PIN", raising=False)
    seen = _built(monkeypatch)
    assert len(seen) == 5
    assert all("prefix" not in kw for kw in seen.values())


FULL = RK3588 + (" mode_manager_node=0 mission_control_node=1 "
                 "pipeline_watchdog_node=2 sim_supervisor_node=3")


def test_management_and_watchdog_get_the_prefix(monkeypatch):
    """The A55 half of the layout: one management process per little core."""
    monkeypatch.setenv("DV_CPU_PIN", FULL)
    seen = {}
    monkeypatch.setattr(lc, "LifecycleNode", lambda **kw: seen.setdefault(kw["name"], kw))
    monkeypatch.setattr(lc, "Node", lambda **kw: seen.setdefault(kw["name"], kw))
    monkeypatch.setattr(lc, "EmitEvent", lambda **kw: None)
    monkeypatch.setattr(lc, "RegisterEventHandler", lambda *a, **kw: None)
    monkeypatch.setattr(lc, "OnStateTransition", lambda **kw: None)
    monkeypatch.setattr(lc, "ChangeState", lambda **kw: None)
    monkeypatch.setattr(lc, "matches_action", lambda n: None)
    lc.management_actions(include_sim_supervisor=True)
    assert {n: kw.get("prefix") for n, kw in seen.items()} == {
        "mode_manager_node": "taskset -c 0",
        "mission_control_node": "taskset -c 1",
        "pipeline_watchdog_node": "taskset -c 2",
        "sim_supervisor_node": "taskset -c 3",
    }
