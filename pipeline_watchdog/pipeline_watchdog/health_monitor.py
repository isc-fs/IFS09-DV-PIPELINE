"""Pure, ROS-free health core for the independent pipeline watchdog.

Mirrors the firmware's pure `as_transition` / `as_actuation` cores and this
repo's other pure cores (`mission_control.reconcile`, `cone_slam.lap_counter`):
no rclpy, no numpy, injected clock — so every branch is unit-testable off-node.

WHY THIS EXISTS
---------------
The uDV already runs a liveness watchdog on `/dv/status` (>= 2 Hz, stale at
`HEARTBEAT_STALE_S` = 400 ms → uDV trips to its safe state). That covers the
pipeline being **dead** — process crash, DDS silence, DVPC power loss.

It does NOT cover the pipeline being **alive but sick**: mission_control
happily publishing `DV_RUNNING` at 20 Hz while SLAM has stopped solving. That
is not hypothetical — it is the documented runaway (see
`control/control/controllers/pi_velocity.py`, throttle_max rationale):

    "pose froze, controller kept commanding full throttle, real car ran away
     to ~24 m/s and crashed"

`control_node` caches the last pose forever and has no staleness check of its
own, so a frozen `/slam/pose` leaves it driving blind on stale state while the
uDV's heartbeat watchdog sees a perfectly healthy DVPC. This core closes that
gap. The two watchdogs are complementary, not redundant:

    uDV watchdog      : "is the DVPC alive?"      (heartbeat on /dv/status)
    pipeline watchdog : "is the DVPC's data sane?" (this module)

TWO INDEPENDENT CHECKS
----------------------
1. **Liveness** — a critical topic has gone silent past its budget. This is
   what catches the documented incident: when SLAM strands (cascade-skip,
   see cone_graph_slam_node "cones lose alignment ... /slam/pose freezes"),
   the scan path returns before publishing, so the topic simply stops.

2. **Pose progress** — `/slam/pose` is still arriving, but its position is not
   advancing while `/odom` says the car is moving. Catches the nastier variant
   where SLAM keeps republishing a stale solve. Liveness alone would miss it.

Both are latching: once tripped, a run never un-trips. Clearing requires a new
run (mission_control resets on a fresh cycle). This is deliberate — an
intermittent sensor that recovers for one scan must not silently rearm the car.

ARMING
------
The monitor only supervises while the pipeline is genuinely driving. It is
armed off `/dv/status == DV_RUNNING`, which is the pipeline's own statement
that it is activated and relaying `/ctrl/cmd`. Consequences of arming off that
byte, all intended:
  * Free-run / data-collection floor (autonomy up, human driving) never trips
    EBS — the floor never reaches DV_RUNNING.
  * A torn-down or idle stack never trips.
  * It reuses the one byte both sides already agree on, rather than inventing
    a second notion of "are we running".

After arming, a grace window absorbs node spin-up (JIT warm-up, first LiDAR
scan, first solve) before any budget applies. Within the grace window nothing
can trip. After it, a topic that has *never* published is treated as stale —
"SLAM never started" is as fatal as "SLAM stopped".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TopicSpec:
    """One supervised topic and how long it may go silent.

    topic:         resolved topic name, as the watchdog subscribes to it.
    max_silence_s: trip if no message for longer than this, once armed and
                   past the grace window.
    why:           human rationale, rendered into the trip reason so the log
                   (and the person reading the bag afterwards) says what broke.
    """

    topic: str
    max_silence_s: float
    why: str


@dataclass(frozen=True)
class PoseProgressSpec:
    """Config for the 'SLAM is publishing but not moving' check.

    min_speed_mps:   only judge progress when /odom says we are moving at
                     least this fast — a legitimately stopped car must never
                     trip.
    min_travel_m:    over the window, the pose must move at least this far.
                     Sized well under what the speed floor implies so noise
                     and a slow solve can't false-trip.
    window_s:        how long the "moving but not progressing" condition must
                     hold continuously before tripping: counted from the
                     later of the pose's last advance and /odom first
                     reaching min_speed_mps, so a car that drives off after
                     a long wait gets the full window too (#31).
    enabled:         escape hatch — liveness alone still works with this off.
    """

    min_speed_mps: float = 1.0
    min_travel_m: float = 0.5
    window_s: float = 1.5
    enabled: bool = True


@dataclass(frozen=True)
class Verdict:
    """Outcome of one evaluate() call."""

    tripped: bool
    reasons: tuple[str, ...] = ()

    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "healthy"


_OK = Verdict(tripped=False)


@dataclass
class _PoseProgressState:
    """Rolling anchor for the pose-progress check."""

    anchor_x: float = 0.0
    anchor_y: float = 0.0
    anchor_t: float = 0.0
    valid: bool = False


class HealthMonitor:
    """Latching health core. Drive it with:

        record(topic, now)              on every supervised message
        record_pose(x, y, now)          on every /slam/pose
        record_speed(speed_mps, now)    on every /odom
        set_running(is_running, now)    on every /dv/status
        evaluate(now) -> Verdict        on every tick

    All times are a monotonic clock in seconds, injected by the caller so tests
    drive time directly and never sleep.
    """

    def __init__(
        self,
        specs: tuple[TopicSpec, ...],
        grace_period_s: float = 3.0,
        pose_progress: PoseProgressSpec | None = None,
    ) -> None:
        self._specs = specs
        self._grace_period_s = float(grace_period_s)
        self._pose_cfg = pose_progress or PoseProgressSpec()
        self._last_seen: dict[str, float] = {}
        self._armed_at: float | None = None
        self._tripped: bool = False
        self._reasons: tuple[str, ...] = ()
        self._speed_mps: float = 0.0
        # When /odom last went from below min_speed_mps to at or above it;
        # None while below. The stall can't have started before this.
        self._moving_since: float | None = None
        self._pose = _PoseProgressState()

    # -- introspection -------------------------------------------------
    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def supervised_topics(self) -> tuple[str, ...]:
        return tuple(s.topic for s in self._specs)

    # -- inputs --------------------------------------------------------
    def set_running(self, is_running: bool, now: float) -> None:
        """Arm on the rising edge of DV_RUNNING, disarm on the falling edge.

        Disarming clears the trip latch and all per-run memory: the next run
        starts from a clean slate, matching mission_control resetting its own
        emergency latch on a fresh cycle. Re-arming from an already-armed
        state is a no-op — we must NOT restart the grace window every tick.
        """
        if is_running:
            if self._armed_at is None:
                self._armed_at = now
            return
        if self._armed_at is not None:
            self._armed_at = None
            self._tripped = False
            self._reasons = ()
            self._last_seen.clear()
            self._pose = _PoseProgressState()
            self._speed_mps = 0.0
            self._moving_since = None

    def record(self, topic: str, now: float) -> None:
        """Note that `topic` produced a message at `now`."""
        self._last_seen[topic] = now

    def record_speed(self, speed_mps: float, now: float) -> None:
        """Latest planar speed from /odom (the high-rate, always-live source).

        Deliberately taken from /odom and not /slam/pose: the whole point is
        to catch SLAM lying, so the speed reference must not come from SLAM.
        """
        self._speed_mps = abs(float(speed_mps))
        if self._speed_mps < self._pose_cfg.min_speed_mps:
            self._moving_since = None
        elif self._moving_since is None:
            self._moving_since = now

    def record_pose(self, x: float, y: float, now: float) -> None:
        """Feed one /slam/pose sample into the progress check.

        Keeps a rolling anchor: whenever the pose has moved at least
        min_travel_m from the anchor, the anchor jumps forward. The check
        trips when the anchor goes stale (car moving, anchor not advancing).
        """
        if not self._pose.valid:
            self._pose = _PoseProgressState(x, y, now, True)
            return
        dx = x - self._pose.anchor_x
        dy = y - self._pose.anchor_y
        if (dx * dx + dy * dy) ** 0.5 >= self._pose_cfg.min_travel_m:
            self._pose = _PoseProgressState(x, y, now, True)

    # -- evaluation ----------------------------------------------------
    def evaluate(self, now: float) -> Verdict:
        """One supervision step. Returns the latched verdict.

        Returns healthy while disarmed or inside the grace window. Once
        tripped, keeps returning the same tripped verdict until disarmed.
        """
        if self._tripped:
            return Verdict(True, self._reasons)
        if self._armed_at is None:
            return _OK
        if now - self._armed_at < self._grace_period_s:
            return _OK

        reasons: list[str] = []
        reasons.extend(self._liveness_reasons(now))
        pose_reason = self._pose_progress_reason(now)
        if pose_reason is not None:
            reasons.append(pose_reason)

        if not reasons:
            return _OK
        self._tripped = True
        self._reasons = tuple(reasons)
        return Verdict(True, self._reasons)

    def _liveness_reasons(self, now: float) -> list[str]:
        """Topics silent past budget. A topic that never published is stale
        from the end of the grace window, not from the start of time."""
        out: list[str] = []
        for spec in self._specs:
            last = self._last_seen.get(spec.topic)
            if last is None:
                # Never seen: measure silence from arming, so the grace
                # window is the allowance and nothing more.
                silence = now - float(self._armed_at)
                if silence > spec.max_silence_s:
                    out.append(
                        f"{spec.topic} never published "
                        f"({silence:.2f}s since armed, "
                        f"budget {spec.max_silence_s:.2f}s) — {spec.why}")
                continue
            silence = now - last
            if silence > spec.max_silence_s:
                out.append(
                    f"{spec.topic} silent {silence:.2f}s "
                    f"(budget {spec.max_silence_s:.2f}s) — {spec.why}")
        return out

    def _pose_progress_reason(self, now: float) -> str | None:
        """SLAM publishing but not advancing while /odom says we're moving."""
        cfg = self._pose_cfg
        if not cfg.enabled or not self._pose.valid:
            return None
        if self._speed_mps < cfg.min_speed_mps or self._moving_since is None:
            return None
        # A pose that stood still while the car stood still isn't stale: only
        # count the time the car has been moving (#31).
        stalled_for = now - max(self._pose.anchor_t, self._moving_since)
        if stalled_for <= cfg.window_s:
            return None
        return (
            f"/slam/pose not advancing: <{cfg.min_travel_m:.2f}m of travel "
            f"in {stalled_for:.2f}s while /odom reads "
            f"{self._speed_mps:.2f} m/s — SLAM solve is stale/frozen")
