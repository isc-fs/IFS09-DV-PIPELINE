"""
mode_manager_node — lifecycle fan-out for the autonomy stack.

Hosts one service: `activate_mode` (dv_msgs/ActivateMode). On request
it calls ~/setup then drives `change_state` for each autonomy
LifecycleNode (odometry_filter, perception, slam, path_planning,
control) so each node activates with the correct strategy.

Lifecycle of mode_manager itself is trivial — it stays `active` for
the whole container lifetime; it's the autonomy nodes underneath that
move between unconfigured → inactive → active.

Skeleton: the activate_mode handler returns ok=true without touching
any change_state services. Real wiring lands in step 5.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

from lifecycle_msgs.msg import State as LifecycleStateMsg, Transition
from lifecycle_msgs.srv import ChangeState, GetState

from dv_msgs.msg import LifecycleProgress
from dv_msgs.srv import ActivateMode, Setup

from mode_manager.mode_registry import (
    AUTONOMY_NODE_ORDER,
    MODE_REGISTRY,
    node_config_for,
)


# Mapping from a lifecycle transition ID → the state ID it lands in.
# Used by _drive_transition's pre-check: if the node is already in
# the target state (or past it on the path we're walking), skip the
# change_state call. Makes activate_mode idempotent — calling
# activate_mode("trackdrive") twice in a row, or activate_mode("")
# on an already-unconfigured stack, succeeds without invalid-
# transition errors.
_TRANSITION_TARGET_STATE: dict[int, int] = {
    Transition.TRANSITION_CONFIGURE:  LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
    Transition.TRANSITION_ACTIVATE:   LifecycleStateMsg.PRIMARY_STATE_ACTIVE,
    Transition.TRANSITION_DEACTIVATE: LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
    Transition.TRANSITION_CLEANUP:    LifecycleStateMsg.PRIMARY_STATE_UNCONFIGURED,
}

# Per bring-up / tear-down direction, the set of states from which a
# transition is a no-op (we're already at or past the goal).
#
# bring_up (CONFIGURE then ACTIVATE):
#   • CONFIGURE: skip if already in inactive or active
#   • ACTIVATE:  skip if already in active
# tear_down (DEACTIVATE then CLEANUP):
#   • DEACTIVATE: skip if already in inactive or unconfigured
#   • CLEANUP:    skip if already in unconfigured
_TRANSITION_SKIP_STATES: dict[int, frozenset[int]] = {
    Transition.TRANSITION_CONFIGURE: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
        LifecycleStateMsg.PRIMARY_STATE_ACTIVE,
    }),
    Transition.TRANSITION_ACTIVATE: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_ACTIVE,
    }),
    Transition.TRANSITION_DEACTIVATE: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_INACTIVE,
        LifecycleStateMsg.PRIMARY_STATE_UNCONFIGURED,
    }),
    Transition.TRANSITION_CLEANUP: frozenset({
        LifecycleStateMsg.PRIMARY_STATE_UNCONFIGURED,
    }),
}


# Names of the autonomy LifecycleNodes that mode_manager fans out to.
# Order matters when bringing up: odometry before perception before
# slam before planning before control. Reverse order on tear-down.
#
# Sourced from the registry's AUTONOMY_NODE_ORDER (single source of
# truth). The tuple shape `(name, lifecycle_managed)` is preserved so
# downstream code can still distinguish real LifecycleNodes from
# legacy plain Nodes via the second field. Today every node is
# lifecycle-managed; a node temporarily reverting to a plain Node
# can be flagged `False` here without touching the registry itself.
AUTONOMY_LIFECYCLE_NODES: tuple[tuple[str, bool], ...] = tuple(
    (name, True) for name in AUTONOMY_NODE_ORDER
)

# Per-service-call timeout when waiting for a change_state response.
# The expensive transition is on_configure for cone_detection_node
# (Numba JIT, ~10-60 s on Docker hosts). 120 s matches alt_pipeline
# prepare-phase budget so SetMission does not abort mid-JIT.
_CHANGE_STATE_TIMEOUT_S: float = 120.0

# Timeout for the pre-configure ~/setup call: waiting for the service to
# appear *and* for the Setup response. cone_detection_node loads sklearn +
# the cone_detection pipeline before BaseLifecycleNode registers ~/setup;
# on cold Docker/WSL imports alone often exceed 5 s.
_SETUP_TIMEOUT_S: float = 60.0

# How long to wait for one Setup reply before asking again. The handler only
# stores mode + behavior, so a node answers in milliseconds; a reply that
# hasn't come by then was lost (the server can answer before it has
# discovered a brand-new client's reply reader, and DDS drops it, #25).
# Asking again on the same client gets through. Total wait is still capped
# by _SETUP_TIMEOUT_S.
_SETUP_REPLY_TIMEOUT_S: float = 5.0

# Sentinel transition ID for the setup step on /mode_manager/progress.
TRANSITION_SETUP: int = 255

# Mission strings accepted by activate_mode. Built from the registry
# so adding a new mode is a single edit in mode_registry.py.
VALID_MISSIONS: frozenset[str] = frozenset(MODE_REGISTRY.keys())


class ModeManagerNode(LifecycleNode):
    """Lifecycle fan-out service host. See module docstring."""

    NODE_NAME = "mode_manager_node"

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._prepared_mode: str | None = None
        self._active_mode: str | None = None
        self._activate_mode_srv = None
        # #387 — per-node-per-transition progress publisher. One event
        # per change_state call. Created in on_configure.
        self._progress_pub = None
        # Reentrant group so the activate_mode service callback can
        # call out to lifecycle clients and wait on their futures
        # while the same node's executor dispatches the responses.
        self._cb_group = ReentrantCallbackGroup()
        # change_state / get_state clients are *cached lazily* —
        # created on first use inside _drive_transition /
        # _get_current_state and reused. We never created them
        # up-front in on_configure because doing so before the
        # autonomy nodes are up costs a DDS discovery race: clients
        # made too early can miss the server's late-joining endpoint
        # announcement and stay stale for the lifetime of the node.
        # Lazy creation guarantees the client is built only after
        # the activate_mode call lands, by which time the autonomy
        # nodes have been alive for the full container-startup
        # window and discovery is settled. Setup clients (~/setup)
        # are created *fresh per call* inside _call_setup — they
        # are one-shot per mission, so caching buys nothing.
        self._change_state_clients: dict[str, rclpy.client.Client] = {}
        self._get_state_clients: dict[str, rclpy.client.Client] = {}

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_configure: creating activate_mode service")
        self._activate_mode_srv = self.create_service(
            ActivateMode,
            "activate_mode",
            self._handle_activate_mode,
            callback_group=self._cb_group,
        )
        # #387 — progress topic. Depth=10 KEEP_LAST: bursty during the
        # ~4-6 transitions of a single activate_mode call, idle the
        # rest of the time. A late-joining subscriber misses the most
        # recent burst but that's fine — mission_control_node's
        # heartbeat fills the gap. Default RELIABLE is correct (small
        # payload, drops would defeat the point).
        self._progress_pub = self.create_publisher(
            LifecycleProgress, "/mode_manager/progress", 10,
        )
        # Lifecycle clients are created lazily — see __init__ comment
        # for the discovery-race rationale.
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Lazy client accessors
    # ------------------------------------------------------------------
    def _get_change_state_client(self, node_name: str) -> "rclpy.client.Client":
        cli = self._change_state_clients.get(node_name)
        if cli is None:
            cli = self.create_client(
                ChangeState,
                f"/{node_name}/change_state",
                callback_group=self._cb_group,
            )
            self._change_state_clients[node_name] = cli
        return cli

    def _get_get_state_client(self, node_name: str) -> "rclpy.client.Client":
        cli = self._get_state_clients.get(node_name)
        if cli is None:
            cli = self.create_client(
                GetState,
                f"/{node_name}/get_state",
                callback_group=self._cb_group,
            )
            self._get_state_clients[node_name] = cli
        return cli

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_activate")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_deactivate")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_cleanup")
        if self._activate_mode_srv is not None:
            self.destroy_service(self._activate_mode_srv)
        self._activate_mode_srv = None
        if self._progress_pub is not None:
            self.destroy_publisher(self._progress_pub)
            self._progress_pub = None
        for cli in self._change_state_clients.values():
            self.destroy_client(cli)
        self._change_state_clients.clear()
        for cli in self._get_state_clients.values():
            self.destroy_client(cli)
        self._get_state_clients.clear()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("on_shutdown")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Service handler
    # ------------------------------------------------------------------
    def _handle_activate_mode(
        self,
        request: ActivateMode.Request,
        response: ActivateMode.Response,
    ) -> ActivateMode.Response:
        """
        Prepare (setup+configure) or activate prepared nodes, or tear down.

        activate=false → SetMission prepare path (nodes stay inactive).
        activate=true  → RuntimeControl open (activate prepared stack).
        mission=""     → full tear-down regardless of activate flag.
        """
        mission = request.mission
        do_activate = bool(request.activate)
        # Nodes to clean-cycle (deactivate->activate) on this activate so they
        # re-run on_activate fresh. Empty for every legacy caller. Ignored on
        # teardown ("" tears everything down regardless).
        reset = set(request.reset_nodes)

        if mission == "":
            self.get_logger().info("activate_mode: tearing down autonomy")
            ok, msg = self._fan_out(
                reversed(AUTONOMY_LIFECYCLE_NODES),
                (Transition.TRANSITION_DEACTIVATE, Transition.TRANSITION_CLEANUP),
                "tear_down",
                mission=None,
            )
            if ok:
                self._prepared_mode = None
                self._active_mode = None
            response.ok = ok
            response.message = msg if not ok else "autonomy torn down"
            return response

        if mission not in VALID_MISSIONS:
            response.ok = False
            response.message = (
                f"unknown mission {mission!r}; must be one of "
                f"{sorted(VALID_MISSIONS)}"
            )
            self.get_logger().warning(response.message)
            return response

        if not do_activate:
            return self._prepare_mode(mission, response)

        return self._activate_prepared_mode(mission, response, reset)

    def _prepare_mode(
        self, mission: str, response: ActivateMode.Response,
    ) -> ActivateMode.Response:
        """Setup + configure without activating (SetMission / prepare)."""
        self.get_logger().info(f"activate_mode: preparing mission={mission!r}")

        if self._active_mode == mission:
            response.ok = True
            response.message = f"Mode {mission!r} already active"
            return response

        if self._prepared_mode == mission and self._active_mode is None:
            response.ok = True
            response.message = f"Mode {mission!r} already prepared"
            return response

        if self._active_mode and self._active_mode != mission:
            self.get_logger().info(
                f"Deactivating previous active mode: {self._active_mode}"
            )
            ok, msg = self._fan_out(
                reversed(AUTONOMY_LIFECYCLE_NODES),
                (Transition.TRANSITION_DEACTIVATE, Transition.TRANSITION_CLEANUP),
                "tear_down",
                mission=None,
            )
            if not ok:
                response.ok = False
                response.message = msg
                return response
            self._active_mode = None
            self._prepared_mode = None

        if self._prepared_mode and self._prepared_mode != mission:
            self.get_logger().info(
                f"Cleaning up previous prepared mode: {self._prepared_mode}"
            )
            ok, msg = self._fan_out(
                reversed(AUTONOMY_LIFECYCLE_NODES),
                (Transition.TRANSITION_CLEANUP,),
                "cleanup_prepared",
                mission=None,
            )
            if not ok:
                response.ok = False
                response.message = msg
                return response
            self._prepared_mode = None

        ok, msg = self._fan_out(
            AUTONOMY_LIFECYCLE_NODES,
            (Transition.TRANSITION_CONFIGURE,),
            "prepare",
            mission=mission,
        )
        if not ok:
            response.ok = False
            response.message = msg
            return response

        self._prepared_mode = mission
        response.ok = True
        response.message = f"prepared {mission!r}"
        return response

    def _activate_prepared_mode(
        self, mission: str, response: ActivateMode.Response,
        reset: set[str],
    ) -> ActivateMode.Response:
        """Activate nodes configured during prepare (RuntimeControl open).

        `reset` names nodes to clean-cycle first: each is DEACTIVATEd before
        the activate fan-out re-activates it, so it re-runs on_activate with
        fresh per-run state while every other node is left as-is (already-
        active nodes are idempotently skipped). The free-run go hand-off uses
        reset={control_node} so control — which ran through the OFF/manual
        floor logging its would-be commands — starts the real run clean, with
        no SLAM reset or Numba re-JIT of the perception stack.
        """
        self.get_logger().info(
            f"activate_mode: activating mission={mission!r}"
            + (f" (reset={sorted(reset)})" if reset else "")
        )

        if self._prepared_mode != mission:
            response.ok = False
            response.message = (
                f"Mode {mission!r} has not been prepared "
                f"(prepared: {self._prepared_mode!r}). "
                "Complete SetMission before RuntimeControl."
            )
            self.get_logger().error(response.message)
            return response

        # A reset request always fans out (the clean-cycle must run even when
        # the mode is already fully active); without one, an already-active
        # mode is a no-op.
        if self._active_mode == mission and not reset:
            response.ok = True
            response.message = f"Mode {mission!r} already active"
            return response

        # Clean-cycle the reset nodes: DEACTIVATE first (idempotently skipped
        # if already inactive), so the ACTIVATE fan-out below re-runs their
        # on_activate fresh.
        if reset:
            reset_nodes = tuple(
                (name, managed)
                for name, managed in AUTONOMY_LIFECYCLE_NODES
                if name in reset
            )
            ok, msg = self._fan_out(
                reset_nodes,
                (Transition.TRANSITION_DEACTIVATE,),
                "reset",
                mission=None,
            )
            if not ok:
                response.ok = False
                response.message = msg
                return response

        ok, msg = self._fan_out(
            AUTONOMY_LIFECYCLE_NODES,
            (Transition.TRANSITION_ACTIVATE,),
            "activate",
            mission=None,
        )
        if not ok:
            response.ok = False
            response.message = msg
            return response

        self._active_mode = mission
        response.ok = True
        response.message = f"activated {mission!r}"
        return response

    # ------------------------------------------------------------------
    # Lifecycle fan-out (parallel)
    # ------------------------------------------------------------------
    def _fan_out(
        self,
        nodes: Iterable[tuple[str, bool]],
        transitions: tuple[int, ...],
        phase: str,
        mission: str | None,
    ) -> tuple[bool, str]:
        """
        Run each managed node's (optional setup + transitions) work
        concurrently. Bring-up time is bounded by the slowest node
        (Numba JIT in cone_detection_node), not the sum.

        All worker threads run on a ThreadPoolExecutor while the
        activate_mode callback that launched _fan_out blocks here.
        The outer ReentrantCallbackGroup + MultiThreadedExecutor lets
        the rclpy executor dispatch service responses to the worker
        threads' futures in parallel. Each node owns its own client
        objects, so the per-node clients are never touched from more
        than one worker.

        Returns (True, "") iff every managed node finished its full
        transition chain. On any failure, blocks until every other
        in-flight worker finishes (so the system isn't left half-
        transitioned), then returns (False, "; "-joined diagnostics).

        When `mission` is set (bring-up direction), calls ~/setup on
        each node before its first transition. Tear-down skips setup.
        Skipped unmanaged nodes are noted in the log but don't fail
        the call.
        """
        call_setup = (
            mission is not None
            and Transition.TRANSITION_CONFIGURE in transitions
        )

        managed_nodes: list[str] = []
        for name, managed in nodes:
            if not managed:
                self.get_logger().info(
                    f"  [{phase}] {name}: not lifecycle-managed yet, skipping"
                )
                continue
            managed_nodes.append(name)

        if not managed_nodes:
            return True, ""

        def _process_node(node_name: str) -> tuple[str, bool, str]:
            if call_setup:
                cfg = node_config_for(mission, node_name)
                ok, err = self._call_setup(node_name, mission, cfg.behavior)
                if not ok:
                    return node_name, False, err
            for t in transitions:
                ok, err = self._drive_transition(node_name, t)
                if not ok:
                    return node_name, False, err
            return node_name, True, ""

        failures: list[str] = []
        with ThreadPoolExecutor(
            max_workers=len(managed_nodes),
            thread_name_prefix=f"mode_manager-{phase}",
        ) as pool:
            futures = [pool.submit(_process_node, n) for n in managed_nodes]
            for fut in as_completed(futures):
                try:
                    name, ok, err = fut.result()
                except Exception as ex:  # noqa: BLE001 — defensive
                    failures.append(f"<worker>: {ex!r}")
                    continue
                if not ok:
                    failures.append(f"{name}: {err}")

        if failures:
            return False, "; ".join(failures)
        return True, ""

    # ------------------------------------------------------------------
    # Future waiting
    # ------------------------------------------------------------------
    @staticmethod
    def _wait_for_future(future, timeout_sec: float) -> bool:
        """Block until `future` is done or `timeout_sec` elapses.

        Uses `add_done_callback` + threading.Event so the calling
        thread doesn't busy-poll while another executor thread
        dispatches the response. Required when many futures are
        in flight from parallel _fan_out workers — a sleep-poll
        loop per worker would burn CPU and add latency.

        Returns True iff the future completed within the deadline.
        """
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        return done.wait(timeout=timeout_sec) and future.done()

    # ------------------------------------------------------------------
    # Setup (pre-configure step)
    # ------------------------------------------------------------------
    def _call_setup(
        self, node_name: str, mode_name: str, behavior: str,
    ) -> tuple[bool, str]:
        """Call /<node_name>/setup with mode + behavior. Blocks.

        Fresh client per call (no caching): setup is one-shot per
        mission, and creating it on demand sidesteps the
        startup-time DDS discovery race that bit clients built in
        on_configure before the autonomy nodes were up.
        """
        self._publish_progress(node_name, TRANSITION_SETUP, "starting", "")

        cli = self.create_client(
            Setup,
            f"/{node_name}/setup",
            callback_group=self._cb_group,
        )
        try:
            return self._call_setup_with_client(
                cli, node_name, mode_name, behavior,
            )
        finally:
            self.destroy_client(cli)

    def _call_setup_with_client(
        self,
        cli: "rclpy.client.Client",
        node_name: str,
        mode_name: str,
        behavior: str,
    ) -> tuple[bool, str]:
        if not cli.wait_for_service(timeout_sec=_SETUP_TIMEOUT_S):
            err = (
                f"/{node_name}/setup did not appear within "
                f"{_SETUP_TIMEOUT_S}s"
            )
            self._publish_progress(node_name, TRANSITION_SETUP, "timeout", err)
            return False, err

        req = Setup.Request()
        req.mode_name = mode_name
        req.behavior = behavior
        deadline = time.monotonic() + _SETUP_TIMEOUT_S
        tries = 0
        while True:
            tries += 1
            future = cli.call_async(req)
            wait = min(_SETUP_REPLY_TIMEOUT_S, max(0.0, deadline - time.monotonic()))
            if self._wait_for_future(future, wait):
                break
            try:
                cli.remove_pending_request(future)
            except Exception:
                pass
            if time.monotonic() >= deadline:
                err = f"timeout waiting for setup response ({tries} tries)"
                self._publish_progress(node_name, TRANSITION_SETUP, "timeout", err)
                return False, err
            self.get_logger().warning(
                f"  [{node_name}] no setup reply after "
                f"{_SETUP_REPLY_TIMEOUT_S:g}s, asking again"
            )

        result = future.result()
        if result is None:
            err = "setup returned None (rclpy error)"
            self._publish_progress(node_name, TRANSITION_SETUP, "failed", err)
            return False, err
        if not result.success:
            err = result.message or "setup failed"
            self._publish_progress(node_name, TRANSITION_SETUP, "failed", err)
            return False, err

        self.get_logger().info(
            f"  [{node_name}] setup ok: mode={mode_name!r} behavior={behavior!r}"
        )
        self._publish_progress(node_name, TRANSITION_SETUP, "ok", "")
        return True, ""

    def _drive_transition(self, node_name: str, transition_id: int) -> tuple[bool, str]:
        """Call /<node_name>/change_state with `transition_id`. Blocks.

        Idempotency pre-check: if the node is already in (or past)
        the transition's target state, return success without calling
        change_state. Lets activate_mode("trackdrive") be invoked
        repeatedly without invalid-transition errors, and lets
        activate_mode("") succeed on an already-unconfigured stack
        (the common case at fresh-container startup).

        Publishes /mode_manager/progress events ("starting", then one
        of "ok"/"failed"/"timeout"/"skipped") so mission_control_node
        can echo per-node progress into StartMission feedback (#387).
        """
        cli = self._get_change_state_client(node_name)

        # State-check: skip if already at/past target. The get_state
        # call is cheap (~ms) and removes the need for callers to
        # track autonomy state externally.
        current = self._get_current_state(node_name)
        if current is None:
            # get_state failed — proceed to attempt the transition;
            # the change_state error will be more informative than
            # bailing here.
            self.get_logger().debug(
                f"  [{node_name}] get_state unavailable; "
                f"attempting transition {transition_id} blind"
            )
        else:
            skip_states = _TRANSITION_SKIP_STATES.get(transition_id, frozenset())
            if current in skip_states:
                self.get_logger().info(
                    f"  [{node_name}] already at/past target for "
                    f"transition {transition_id} (state={current}); skipping"
                )
                self._publish_progress(node_name, transition_id, "skipped", "")
                return True, ""

        # Announce we're about to start. Subscribers can latch this and
        # surface "configuring <node>" before the (potentially long)
        # change_state call returns — that's the whole point of #387.
        self._publish_progress(node_name, transition_id, "starting", "")

        if not cli.wait_for_service(timeout_sec=_CHANGE_STATE_TIMEOUT_S):
            err = (
                f"/{node_name}/change_state did not appear within "
                f"{_CHANGE_STATE_TIMEOUT_S}s"
            )
            self._publish_progress(node_name, transition_id, "timeout", err)
            return False, err

        req = ChangeState.Request()
        req.transition.id = transition_id
        future = cli.call_async(req)
        # The outer MultiThreadedExecutor dispatches the response into
        # the future via the ReentrantCallbackGroup that owns this
        # client; this worker thread just blocks on the future's done
        # event. Critical for parallel _fan_out: with many futures in
        # flight, sleep-polling each one would burn CPU and bound the
        # bring-up time below by the polling cadence.
        if not self._wait_for_future(future, _CHANGE_STATE_TIMEOUT_S):
            try:
                cli.remove_pending_request(future)
            except Exception:
                pass
            err = f"timeout waiting for transition {transition_id}"
            self._publish_progress(node_name, transition_id, "timeout", err)
            return False, err
        result = future.result()
        if result is None:
            err = "change_state returned None (rclpy error)"
            self._publish_progress(node_name, transition_id, "failed", err)
            return False, err
        if not result.success:
            err = f"transition {transition_id} returned success=False"
            self._publish_progress(node_name, transition_id, "failed", err)
            return False, err
        self.get_logger().info(
            f"  [{node_name}] transition {transition_id} ok"
        )
        self._publish_progress(node_name, transition_id, "ok", "")
        return True, ""

    def _publish_progress(
        self,
        node_name: str,
        transition_id: int,
        phase: str,
        error: str,
    ) -> None:
        """Emit one LifecycleProgress event. Best-effort — publish
        failures are logged at debug and never bubble up; the lifecycle
        fan-out itself is the source of truth, the progress topic is
        diagnostic-grade only.
        """
        if self._progress_pub is None:
            return
        msg = LifecycleProgress()
        msg.node_name = node_name
        msg.transition_id = int(transition_id)
        msg.phase = phase
        msg.error = error
        msg.stamp = self.get_clock().now().to_msg()
        try:
            self._progress_pub.publish(msg)
        except Exception as ex:
            self.get_logger().debug(
                f"_publish_progress: publish failed for "
                f"{node_name}/{transition_id}/{phase}: {ex}"
            )

    def _get_current_state(self, node_name: str) -> int | None:
        """Query /<node_name>/get_state. Returns the state ID (an int
        from lifecycle_msgs.msg.State.PRIMARY_STATE_*), or None if the
        service is unavailable or the call timed out.

        Short timeout: this is a fast intra-container service call,
        and we use it from inside _drive_transition's hot path. If
        it doesn't respond in 2 s the node is probably stuck and
        the change_state call will fail with a more informative
        error anyway.
        """
        cli = self._get_get_state_client(node_name)
        if not cli.service_is_ready():
            # No wait — if the service isn't already there, we won't
            # wait for it. The downstream change_state will gate on
            # wait_for_service.
            return None

        future = cli.call_async(GetState.Request())
        if not self._wait_for_future(future, 2.0):
            try:
                cli.remove_pending_request(future)
            except Exception:
                pass
            return None
        result = future.result()
        if result is None:
            return None
        return int(result.current_state.id)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ModeManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
