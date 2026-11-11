"""Base lifecycle node with mode/behavior support for autonomy nodes."""
from __future__ import annotations

from abc import ABC, abstractmethod

import rclpy
from dv_msgs.srv import Setup
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn

from node_base.gc_tuning import freeze_heap


class ExecutionStrategy(ABC):
    """Abstract base for per-mission execution strategies."""

    @abstractmethod
    def execute(self) -> None:
        """Strategy-specific logic (optional hook)."""
        pass


class TrackdriveStrategy(ExecutionStrategy):
    def execute(self) -> None:
        pass


class AutocrossStrategy(ExecutionStrategy):
    def execute(self) -> None:
        pass


class AccelStrategy(ExecutionStrategy):
    def execute(self) -> None:
        pass


class SkidpadStrategy(ExecutionStrategy):
    def execute(self) -> None:
        pass


# Mission-name behaviors pushed by mode_manager for perception/SLAM nodes.
STRATEGY_MAP: dict[str, type[ExecutionStrategy]] = {
    "trackdrive": TrackdriveStrategy,
    "autocross": AutocrossStrategy,
    "accel": AccelStrategy,
    "skidpad": SkidpadStrategy,
}


class BaseLifecycleNode(LifecycleNode):
    """Lifecycle node with ~/setup for mode_manager.

    Subclasses override on_configure/on_activate/etc. and call
    super() first so mode_name/behavior/strategy are ready.
    """

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)
        self._strategy: ExecutionStrategy | None = None
        self._mode_name: str | None = None
        self._behavior: str = "default"

        self.create_service(Setup, "~/setup", self._handle_setup)

    def _handle_setup(
        self, request: Setup.Request, response: Setup.Response
    ) -> Setup.Response:
        self._mode_name = request.mode_name
        self._behavior = request.behavior
        response.success = True
        response.message = (
            f"Ready: mode={request.mode_name} behavior={request.behavior}"
        )
        self.get_logger().info(response.message)
        return response

    @property
    def mode_name(self) -> str | None:
        return self._mode_name

    @property
    def behavior(self) -> str:
        return self._behavior

    @property
    def strategy(self) -> ExecutionStrategy | None:
        return self._strategy

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        try:
            if self._behavior in STRATEGY_MAP:
                self._strategy = STRATEGY_MAP[self._behavior]()
            self.get_logger().info(
                f"Configured | mode: {self._mode_name} | behavior: {self._behavior}"
            )
            return TransitionCallbackReturn.SUCCESS
        except Exception as ex:
            self.get_logger().error(f"Failed to configure: {ex}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        # Chain to LifecycleNode.on_activate so every
        # create_lifecycle_publisher on this node flips from INACTIVE
        # to ACTIVE. Without this call .publish() is a silent no-op
        # and downstream nodes never see /Conos_raw, /slam/pose,
        # /Path, /ctrl/cmd_internal — the autonomy looks "active" via
        # transition 3 ok but the data plane stays dark.
        self.get_logger().info(f"Activated in mode: {self._mode_name}")
        # Configure (warm-ups, JIT) is done: move everything alive now out of
        # reach of full GC passes, which otherwise land on a scan as a
        # 130-300 ms stall (see node_base.gc_tuning).
        frozen, ms = freeze_heap()
        self.get_logger().info(f"GC: froze {frozen} startup objects ({ms:.0f} ms)")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # Same reasoning as on_activate: chain to LifecycleNode so
        # lifecycle publishers flip back to INACTIVE and don't keep
        # emitting after on_deactivate returns.
        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        try:
            self._strategy = None
            self._mode_name = None
            self._behavior = "default"
            self.get_logger().info("Cleaned up")
            return TransitionCallbackReturn.SUCCESS
        except Exception as ex:
            self.get_logger().error(f"Cleanup failed: {ex}")
            return TransitionCallbackReturn.FAILURE

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._strategy = None
        self.get_logger().info("Shutdown")
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().error(f"Error state: {state}")
        return TransitionCallbackReturn.SUCCESS
