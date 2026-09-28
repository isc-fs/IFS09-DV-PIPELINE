"""Shared launch-file helpers and IFSSIM topic remap constants.

Every launch file under `bringup/launch/` composes the same node-graph
fragments — management trio, autonomy lifecycle nodes, /fsds/* remap
table. Keeping the shared pieces here avoids three near-identical
copies drifting out of sync.

Public API used by the launch files:

  REMAP_*                IFSSIM /fsds/* surface → pipeline-side names.
                         Single source of truth.

  auto_active(...)       Build a LifecycleNode that auto-drives itself
                         from `unconfigured` to `active` at launch
                         start. Used for the management trio whose
                         action/service endpoints must be live before
                         mode_manager fans change_state out to the
                         autonomy nodes.

  autonomy_lifecycle(...) Build a LifecycleNode that parks in
                          `unconfigured` until mode_manager drives the
                          configure/activate transitions in response
                          to a StartMission/activate_mode call.

  autonomy_actions()     The pre-baked list of autonomy
                         lifecycle-nodes (odometry_filter,
                         cone_detection, slam, path_planning,
                         control) with their remappings. Bring-up
                         order is owned by mode_manager's registry.
                         Lets each launch file insert them with one
                         `actions += ...`.

  management_actions(include_sim_supervisor=True)
                         The management trio pre-baked with the right
                         remaps. The sim_supervisor flag toggles
                         between the sim layout (supervisor included,
                         sits between control_node and the bridge) and
                         the real-car layout (supervisor omitted, the
                         on-vehicle uDV handles the same role).
                         Also brings up the independent pipeline
                         watchdog on both profiles.

  watchdog_action()      The independent pipeline supervisor — a plain
                         Node (never lifecycle) so mode_manager cannot
                         tear down the thing that watches it.

  node_parameters(...)   The `parameters=` every node gets: use_sim_time
                         plus bringup/config/params.yaml, the nodes'
                         default parameters (see that file's header).
"""
from __future__ import annotations

import os
from typing import Iterable

from launch.actions import EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.parameter_descriptions import ParameterValue

from lifecycle_msgs.msg import Transition

from bringup.topic_contract import (
    REMAP_IMU,
    REMAP_RPM,
    REMAP_STEERING,
    REMAP_BRAKE,
    REMAP_CMD,
    autonomy_remaps,
)


def use_sim_time_params() -> list:
    """`parameters=` fragment that binds use_sim_time to the launch arg.

    Every pipeline node reads its clock through this so the whole graph
    shares one toggle: sim launches declare `use_sim_time:=true` (the
    bridge publishes /clock from UE sim time), the car build declares
    `false` (real sensors carry real stamps, no /clock). Every launch
    file that inserts these actions MUST DeclareLaunchArgument(
    "use_sim_time") or the substitution is unresolved at launch.

    value_type=bool is required: LaunchConfiguration yields the string
    "true"/"false", and rclcpp would otherwise reject a string for the
    bool use_sim_time parameter.
    """
    return [{
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool),
    }]


def params_file() -> str:
    """bringup/config/params.yaml: from the installed package, or from the source
    tree when running out of a checkout (no install)."""
    try:
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(get_package_share_directory("bringup"), "config", "params.yaml")
        if os.path.isfile(path):
            return path
    except (ImportError, LookupError):  # PackageNotFoundError is a KeyError
        pass
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "params.yaml")


def node_parameters(extra: list | None = None) -> list:
    """`parameters=` for a pipeline node: use_sim_time, then params.yaml, then
    `extra` (later entries win, so a launch argument like mission_control's
    free_run still overrides the file).

    params.yaml has one section per node; a node only reads its own, so every
    node can be given the whole file.
    """
    return use_sim_time_params() + [params_file()] + list(extra or [])


# Topic remap constants live in bringup.topic_contract (a dependency-free
# module so the contract is unit-testable without a ROS install). The
# names used directly below by management_actions are imported at the top
# of this file; autonomy_actions pulls its per-node table from
# topic_contract.autonomy_remaps().


def auto_active(
    package: str,
    executable: str,
    name: str,
    remappings: Iterable[tuple[str, str]] | None = None,
    parameters: list | None = None,
) -> list:
    """Return [LifecycleNode, configure_event, activate_handler].

    The named node is auto-driven from `unconfigured` to `active` at
    launch start. Used for the management trio whose action/service
    endpoints must be live before mode_manager fans out change_state
    to the autonomy nodes.

    `parameters` are appended after use_sim_time and params.yaml (e.g.
    mission_control's free_run flag); each launch file owns what it passes.
    """
    node = LifecycleNode(
        package=package,
        executable=executable,
        name=name,
        namespace="",
        output="screen",
        remappings=list(remappings or []),
        parameters=node_parameters(parameters),
    )
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(node),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    # When configure completes (state → 'inactive'), emit activate.
    activate = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=node,
        goal_state="inactive",
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        ))],
    ))
    return [node, activate, configure]


def autonomy_lifecycle(
    package: str,
    executable: str,
    name: str,
    remappings: Iterable[tuple[str, str]] | None = None,
) -> LifecycleNode:
    """Return a LifecycleNode parked in 'unconfigured'.

    mode_manager drives the configure/activate transitions via
    change_state fan-out when StartMission arrives; until then the
    node holds no subscriptions and emits no traffic.
    """
    return LifecycleNode(
        package=package,
        executable=executable,
        name=name,
        namespace="",
        output="screen",
        remappings=list(remappings or []),
        parameters=node_parameters(),
    )


def autonomy_actions(profile: str = "sim") -> list:
    """Pre-baked autonomy lifecycle-node list with their remaps.

    Order in the LaunchDescription doesn't constrain bring-up order;
    mode_manager.AUTONOMY_NODE_ORDER owns that. Listed here in the
    same order for readable `ros2 node list` output.

    Args:
        profile: "sim" wires the autonomy onto the IFSSIM /fsds/* bridge
            surface (the historical default). "car" wires it onto the
            real-vehicle surface: only the LiDAR is a pure remap onto the
            Hesai topic (REMAP_LIDAR_CAR). IMU is canonical /imu on both
            sides (the uDV publishes /imu and the nodes subscribe to /imu
            in code), so it is NOT remapped; likewise /steering_angle and
            /motor_rpm arrive on their canonical names from the uDV
            directly (steering converted to rad on-board, /motor_rpm from
            the inverter). The sim-only ground-truth debug taps
            (REMAP_GT / REMAP_TRACK on slam) are dropped on the car —
            nothing publishes /fsds/testing_only/* there.
    """
    # Per-node remap table (pure data; raises on unknown profile).
    remaps = autonomy_remaps(profile)

    return [
        autonomy_lifecycle(
            "odometry_filter_node", "odometry_filter_node",
            "odometry_filter_node",
            remappings=remaps["odometry_filter_node"],
        ),
        autonomy_lifecycle(
            "cone_detection", "cone_detection_node", "cone_detection_node",
            remappings=remaps["cone_detection_node"],
        ),
        autonomy_lifecycle(
            "cone_slam", "slam_node", "slam_node",
            remappings=remaps["slam_node"],
        ),
        autonomy_lifecycle(
            "path_planning", "path_planning_node", "path_planning_node",
            remappings=remaps["path_planning_node"],
        ),
        autonomy_lifecycle(
            "control", "control_node", "control_node",
            # control_node no longer publishes /fsds/control_command
            # directly — its output flows on /ctrl/cmd_internal to
            # mission_control_node, which (while RUNNING) republishes it
            # as a normalised geometry_msgs/Twist on /ctrl/cmd for the
            # uDV / sim_supervisor emulator to scale and actuate. No
            # bridge-facing remap needed (empty list for both profiles).
            remappings=remaps["control_node"],
        ),
    ]


def watchdog_action() -> list:
    """The independent pipeline supervisor (pipeline_watchdog_node).

    Deliberately a PLAIN Node — not a LifecycleNode, and not part of
    autonomy_actions(): it comes up with the management trio and stays up for
    the whole session, so mode_manager can never configure, deactivate or tear
    down the thing that supervises it. It needs no launch-time mission wiring
    because it arms itself off /dv/status == DV_RUNNING.

    Included in BOTH the sim and car profiles: a watchdog that only exists on
    one of them is a watchdog whose false-trip behaviour is discovered on the
    vehicle.
    """
    return [Node(
        package="pipeline_watchdog",
        executable="pipeline_watchdog_node",
        name="pipeline_watchdog_node",
        namespace="",
        output="screen",
        parameters=node_parameters(),
    )]


def management_actions(
    include_sim_supervisor: bool = True,
    free_run=None,
    hard_stop_on_finish=None,
) -> list:
    """Pre-baked management trio, all auto-active.

    Bring-up order: mode_manager first (its activate_mode service must
    be registered before mission_control's StartMission handler tries
    to call into it), mission_control second, sim_supervisor third
    (its StartMission server only opens after mission_control is ready
    to receive its downstream call).

    Args:
        include_sim_supervisor: True for sim/full builds (the
            supervisor sits between control_node and the bridge);
            False for car builds (the on-vehicle uDV replaces it).
        free_run: if given (a LaunchConfiguration or bool), sets
            mission_control's `free_run` parameter — the always-on
            data-collection floor. None (sim/full) leaves the node
            default (off).
        hard_stop_on_finish: if given, sets mission_control's
            `hard_stop_on_finish` parameter — emit DV_STOPPING (an ASB
            hard stop to standstill) at mission end. SAFETY-CRITICAL:
            leave None / false for every normal run; only ever set true
            for a bench-validated run with byte-7 firmware flashed. See
            docs/HARD_STOP.md and docs/HARD_STOP_BENCH.md. None leaves the
            node default (false).
    """
    actions: list = []
    actions += auto_active("mode_manager", "mode_manager_node", "mode_manager_node")
    mc_params_dict = {}
    if free_run is not None:
        mc_params_dict["free_run"] = ParameterValue(free_run, value_type=bool)
    if hard_stop_on_finish is not None:
        mc_params_dict["hard_stop_on_finish"] = ParameterValue(
            hard_stop_on_finish, value_type=bool)
    mc_params = [mc_params_dict] if mc_params_dict else None
    actions += auto_active(
        "mission_control", "mission_control_node", "mission_control_node",
        parameters=mc_params,
    )
    # Independent supervisor, after mission_control so /dv/status has a
    # publisher to latch from at discovery (DDS would cope either way).
    actions += watchdog_action()
    if include_sim_supervisor:
        # sim_supervisor needs /imu + /motor_rpm + /steering + /brake +
        # /control_command remapped onto /fsds/* so its OdometryFilter
        # (when enabled) and command relay see the bridge's sensor and
        # actuator topics. The IMU/RPM/steering/brake remaps stay even
        # though the C++ odometry_filter_node owns those subscriptions
        # by default (use_external_odometry_filter=True) — the remap
        # is harmless when no subscriber asks for the topic, and lets
        # `--ros-args -p use_external_odometry_filter:=false` fall back
        # to the Python filter without re-adding the remaps.
        actions += auto_active(
            "sim_supervisor", "sim_supervisor_node", "sim_supervisor_node",
            remappings=[REMAP_IMU, REMAP_RPM, REMAP_STEERING, REMAP_BRAKE,
                        REMAP_CMD],
        )
    return actions
