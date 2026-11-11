"""Adapter for the FaSTTUBe Formula Student path planning library.

Wraps `fsd_path_planning.PathPlanner` (https://github.com/papalotis/ft-fsd-path-planning,
MIT, by FaSTTUBe — papalotis is the maintainer's GitHub handle) to match the
input/output contract the ROS node expects: list of `Cone` + `Pose2D` in,
list of `PathPoint` (+ optional debug info) out.

Why FaSTTUBe replaces our Delaunay+walker planner (PR #243):
  - The walker was poisoned by spurious orange-classified cones in the cone
    soup (see fix/241 SLAM audit) — Delaunay edges through ghost orange cones
    looked cross-colour to the walker, so it picked midpoints on the wrong
    side of the track.
  - The walker also struggled at one-sided observation regions (PR #189) —
    when only one side's cones are visible, only same-colour Delaunay edges
    exist and the walker's midpoints sit on the outside arc.
  - FaSTTUBe sorts each side independently and matches across sides, so
    orange ghosts get dropped naturally and one-sided regions are inferred
    from the present side's geometry.

This adapter is intentionally thin. The real algorithm lives upstream;
we own the type translation, the empty/exception guards, and the body-frame
yaw recomputation the controller wants.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from fsd_path_planning import MissionTypes, PathPlanner

from path_planning.core_types import Cone, PathPoint, Pose2D

logger = logging.getLogger(__name__)

# Every cone routes to FaSTTUBe's UNKNOWN slot — the pipeline carries
# no real colour signal (UE5 sets `bReturnPhysicalMaterial=false` at
# the LiDAR raycast site, the wire format is XYZ-only). The library's
# geometric sort assigns left/right by corridor topology.

# Cone cull window applied in the adapter (#254). FaSTTUBe is designed
# to consume a per-tick local view of the cone field, not the full
# persistent SLAM landmark map. Without culling, scoring passes over
# the entire history (lap-1 cones still in the map on lap 2) which
# poisons the per-side sort score on dense maps. 25 m radius covers
# the LiDAR's effective range with margin; the body_x > 0 cull drops
# behind-the-car cones that the planner has no use for.
_CULL_RANGE_M = 25.0

# Maximum path arc-length we publish on /Path (#260). FaSTTUBe extrapolates
# missing-side cones via cross-side matching when only one side is
# observed; on tight turns (one-sided LiDAR observation typical at corner
# exits) the extrapolation can run 30+ m past actual cone returns. The
# tail of the path is then geometrically unsupported — the controller's
# Pure Pursuit lookahead can land in that extrapolated region. Capping
# the published path at 12 m of arc length keeps the controller looking
# only at well-supported geometry. 12 m comfortably exceeds Pure
# Pursuit's adaptive Ld at any speed we'd reach at FS-Driverless events
# (Ld ≤ 8 m for the lookahead_max default), so the controller is never
# starved.
_MAX_PATH_ARC_M = 12.0


@dataclass
class PlanDebug:
    """Per-tick intermediate results from FaSTTUBe (`fix/254`).

    Every field is a `(N, 2)` array of world-frame xy points. Empty
    arrays when the planner failed; arrays may be different lengths
    per side (left vs right) since each side's sort is independent
    and one side may include virtually-matched cones the other doesn't.
    """
    left_sorted: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    right_sorted: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    left_with_virtual: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    right_with_virtual: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

# FaSTTUBe expects exactly 5 cone arrays indexed 0..4 (one per ConeTypes).
# Slot 0 is UNKNOWN (cones with no colour info) — we don't currently
# produce those because cone_slam always assigns a colour at landmark
# creation. Kept as an empty array for forward compatibility.
_NUM_CONE_TYPES = 5


class FasttubeAdapter:
    """Wraps a single `PathPlanner` instance and translates I/O.

    Lifetime: construct once, reuse across plan calls. The library is
    stateless for trackdrive (the mission we ship), so the same instance
    serves every cone callback. Skidpad/acceleration are stateful and
    tracked separately in issue #243 — out of scope for this PR.
    """

    def __init__(self, mission: MissionTypes = MissionTypes.trackdrive) -> None:
        # `use_unknown_cones=True` is already the default in the
        # library's trackdrive cone-sorting config (verify via
        # `fsd_path_planning.config.get_cone_sorting_config`), so we
        # don't have to opt in explicitly. We use that path because
        # the upstream pipeline carries no real cone-colour signal
        # (see _cones_to_arrays — every cone is packed into the
        # UNKNOWN slot).
        self._planner = PathPlanner(mission)
        self._mission = mission
        # Rate-limited error logging — papalotis can raise on degenerate
        # cone fields (very few cones, all same colour, malformed
        # geometry). We must never let that take down the node, but we
        # also don't want a crash loop spamming syslog.
        self._last_error_log_time: float = 0.0

    def plan(
        self, cones: List[Cone], pose: Pose2D,
        stage_timings: "dict[str, float] | None" = None,
    ) -> "tuple[List[PathPoint], PlanDebug]":
        """Compute centerline path + per-tick debug payload.

        Path points are in the same world frame as `cones` and `pose`
        (the SLAM `odom` frame in our pipeline). Yaw is recomputed from
        finite differences along the path because FaSTTUBe outputs
        `(s, x, y, curvature)` and the downstream controller only reads
        `(x, y)` from the published `nav_msgs/Path` anyway.

        The `PlanDebug` payload carries the library's intermediate
        per-side sorted cones (with virtually-matched cones for missing
        sides) so the node can publish a `/path_planning/debug` overlay.
        Returns `([], PlanDebug())` on degenerate input or library
        failure.

        `stage_timings`, when given, is filled with per-stage wall times
        (cull_ms / pack_ms / fsd_plan_ms / postprocess_ms) — same pattern
        as cone_detection's `detect_cones`. Used by the offline pipeline
        benchmark; no cost when omitted.
        """
        if not cones:
            return [], PlanDebug()

        _t0 = time.perf_counter()
        # Cull cones to a forward-facing local window (#254). The library
        # is designed for a per-tick local view, not the full persistent
        # SLAM map — without culling, scoring passes over the entire
        # history poison the per-side sort.
        cones = _cull_cones(cones, pose, _CULL_RANGE_M)
        if stage_timings is not None:
            stage_timings["cull_ms"] = (time.perf_counter() - _t0) * 1000.0
        if not cones:
            return [], PlanDebug()

        _t0 = time.perf_counter()
        global_cones = self._cones_to_arrays(cones)
        car_position = np.array([pose.x, pose.y], dtype=np.float64)
        # Unit vector form (preferred over yaw float per the README
        # example — fewer wrap-around bugs).
        car_direction = np.array(
            [np.cos(pose.yaw), np.sin(pose.yaw)], dtype=np.float64
        )
        if stage_timings is not None:
            stage_timings["pack_ms"] = (time.perf_counter() - _t0) * 1000.0

        _t0 = time.perf_counter()
        try:
            result = self._planner.calculate_path_in_global_frame(
                global_cones, car_position, car_direction,
                return_intermediate_results=True,
            )
        except Exception as e:  # pylint: disable=broad-except
            # Library may raise on tiny/degenerate cone fields. Return []
            # so the node logs its own plan_empty counter and the
            # controller keeps the previous reference. Rate-limit at one
            # log per ~5 s.
            now = time.monotonic()
            if now - self._last_error_log_time > 5.0:
                logger.warning(
                    "fasttube planner raised %s: %s (n_cones=%d)",
                    type(e).__name__, e, len(cones),
                )
                self._last_error_log_time = now
            return [], PlanDebug()
        if stage_timings is not None:
            stage_timings["fsd_plan_ms"] = (time.perf_counter() - _t0) * 1000.0
        _t0 = time.perf_counter()

        # Tuple unpack: (path, left_sorted, right_sorted,
        #                left_with_virtual, right_with_virtual,
        #                left_indices, right_indices).
        path, l_sorted, r_sorted, l_virt, r_virt, _, _ = result
        debug = PlanDebug(
            left_sorted=l_sorted, right_sorted=r_sorted,
            left_with_virtual=l_virt, right_with_virtual=r_virt,
        )

        if path is None or path.size == 0:
            return [], debug

        # path is (M, 4) with columns (s, x, y, curvature). Cap the arc
        # length we publish so the controller never chases the
        # virtually-extrapolated tail (#260). `s` is monotonically
        # increasing from 0 at the path's first point, so we just slice
        # to the prefix where s ≤ _MAX_PATH_ARC_M.
        s = np.asarray(path[:, 0], dtype=np.float64)
        within_cap = s <= _MAX_PATH_ARC_M
        # Always keep at least 2 points so the controller has a tangent.
        # On a degenerate path that's all virtual past 0 m, this still
        # gives Pure Pursuit something to chase rather than zero-out
        # steering.
        n_kept = int(np.sum(within_cap))
        if n_kept < 2:
            n_kept = min(2, path.shape[0])
            within_cap = np.zeros_like(within_cap)
            within_cap[:n_kept] = True
        xy = np.asarray(path[within_cap, 1:3], dtype=np.float64)
        # Curvature is the 4th column. FaSTTUBe computes it analytically
        # from the path B-spline, which is much smoother than the
        # finite-difference recompute the controller would do otherwise.
        # We pass it through so the longitudinal controller can see
        # corners earlier and more reliably.
        kappa = np.asarray(path[within_cap, 3], dtype=np.float64)
        if xy.shape[0] < 2:
            return [], debug
        points = _xy_to_path_points(xy, kappa)
        if stage_timings is not None:
            stage_timings["postprocess_ms"] = (time.perf_counter() - _t0) * 1000.0
        return points, debug

    @staticmethod
    def _cones_to_arrays(cones: List[Cone]) -> List[np.ndarray]:
        """Pack cones into the 5-array list FaSTTUBe expects.

        Every cone goes into slot 0 (UNKNOWN); the other four slots
        stay empty. The library's colour-blind geometric sort handles
        the assignment.
        """
        unknown_xy = [[c.x, c.y] for c in cones]
        empty = np.zeros((0, 2), dtype=np.float64)
        return [
            np.asarray(unknown_xy, dtype=np.float64).reshape(-1, 2),
            empty, empty, empty, empty,
        ]


def _cull_cones(cones: List[Cone], pose: Pose2D, max_range_m: float) -> List[Cone]:
    """Cull cones to a forward-facing local window around the car (#254).

    Keeps cones with body-frame `x > 0` (in front of the car) and
    `range <= max_range_m`. The transform is the standard 2D rotation
    of (cone.xy - pose.xy) by `-pose.yaw` so we can read body-frame
    coordinates directly.
    """
    if not cones:
        return cones
    cos_y = float(np.cos(-pose.yaw))
    sin_y = float(np.sin(-pose.yaw))
    out: List[Cone] = []
    for c in cones:
        dx = c.x - pose.x
        dy = c.y - pose.y
        # Range check first — cheaper than the rotation, and most far
        # cones in a persistent map fail it.
        rng = (dx * dx + dy * dy) ** 0.5
        if rng > max_range_m:
            continue
        body_x = cos_y * dx - sin_y * dy
        # 1 mm tolerance avoids dropping/keeping cones based on float
        # noise from the rotation when body_x is essentially zero.
        if body_x <= 1e-3:
            continue
        out.append(c)
    return out


def warmup_numba_planner(
    mission: MissionTypes = MissionTypes.trackdrive,
) -> float:
    """Eager-run fsd_path_planning's Numba kernels; returns elapsed seconds.

    The library JIT-compiles ~90 ``cache=True`` kernels (cone sorting,
    cross-side matching, path parameterization) lazily on the first
    ``calculate_path_in_global_frame`` call. Without this warmup that
    compile lands on the first /Conos callback — mid-mission, while the
    car is already driving. Cost there: ~0.3-0.5 s when the on-disk cache
    is warm, tens of seconds when it is cold (fresh install, rebuilt
    volume, numba upgrade, or an unwritable cache location — see
    NUMBA_CACHE_DIR in deploy/dv-pipeline.service and docker-compose).
    Calling this from ``PathPlanningNode.on_configure`` moves the cost
    into Phase-1 ``warming_up``, where mode_manager's 120 s per-node
    budget (parallel fan-out, same as cone_detection's warmup) absorbs it.

    Replays the REAL adapter chain (cull → pack → plan) on a synthetic
    two-row corridor instead of poking kernels directly: Numba
    specializes per call pattern, so only the live call chain guarantees
    the live signatures get compiled (an argument left to its default is
    a different specialization than the same value passed explicitly).

    Uses a throwaway adapter instance: skidpad/acceleration planners are
    stateful, and the live instance must not see synthetic cone history.
    The module-level Numba dispatchers being warmed are shared, so the
    live adapter still benefits.

    Never raises — warmup is best-effort; on failure the first real tick
    simply pays the JIT as before.
    """
    t0 = time.perf_counter()
    try:
        adapter = FasttubeAdapter(mission)
        # Gentle left-curving corridor, two rows 3 m apart, 4 m spacing —
        # enough cones that the sorter, cross-side matcher, and spline
        # fit all run (verified: produces a full path for trackdrive).
        cones: List[Cone] = []
        for i in range(12):
            s = 4.0 * i + 2.0
            theta = 0.02 * s
            cx = s * math.cos(theta)
            cy = s * math.sin(theta) + 0.5 * theta * s
            hx, hy = math.cos(theta), math.sin(theta)
            cones.append(Cone(x=cx - 1.5 * hy, y=cy + 1.5 * hx))
            cones.append(Cone(x=cx + 1.5 * hy, y=cy - 1.5 * hx))
        pose = Pose2D(x=0.0, y=0.0, yaw=0.05)
        # Twice: the first call JIT-compiles / loads the disk cache; the
        # second exercises warm-path branches (per-planner caches built
        # during the first tick).
        adapter.plan(cones, pose)
        adapter.plan(cones, pose)
    except Exception:  # pylint: disable=broad-except
        logger.warning("fsd_path_planning Numba warmup failed", exc_info=True)
    # The 24-cone corridor above is too small and too clean to reach every
    # signature the live planner hits (SW-04). Two were measured missing after
    # it, both compiled on the first real /Conos callback:
    # * vec_dot / norm_of_last_axis / vec_angle_between on 3-D non-contiguous
    #   ('A' layout) float64 arrays, reached only on a realistic map (5.5 s on
    #   an M1, about 10 s on the RK3588, cold cache);
    # * angle_difference on 0-d arrays (a single-vector angle), reached on a
    #   noisy field.
    # Numba specialises on rank and layout, not just dtype, so these need
    # their own warm-up. Both steps below are best-effort, like the one above.
    try:
        _warm_realistic_fields(mission)
    except Exception:  # pylint: disable=broad-except
        logger.warning("fsd_path_planning realistic-field warmup failed", exc_info=True)
    try:
        _warm_layout_signatures()
    except Exception:  # pylint: disable=broad-except
        logger.warning("fsd_path_planning signature warmup failed", exc_info=True)
    return time.perf_counter() - t0


def _loop_track_cones(rng: np.random.Generator, noise_m: float):
    """A closed ~250 m loop, 3 m wide, cones every 4.5 m per side (~110 cones)
    plus a few false positives: the size and mess of a real SLAM map.

    Returns (cones, centreline (M, 2), unit tangents (M, 2)).
    """
    t = np.linspace(0.0, 2.0 * np.pi, 2000, endpoint=False)
    r = 40.0 * (1.0 + 0.15 * np.sin(2.0 * t + 0.4) + 0.08 * np.sin(3.0 * t + 1.3))
    c = np.c_[r * np.cos(t), r * np.sin(t)]
    seg = np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1)
    idx = np.searchsorted(np.cumsum(seg), np.arange(0.0, seg.sum(), 4.5))
    idx = idx[idx < len(c)]
    tan = np.gradient(c, axis=0)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    nrm = np.c_[-tan[:, 1], tan[:, 0]]
    pts = np.vstack([c[idx] + 1.5 * nrm[idx], c[idx] - 1.5 * nrm[idx]])
    pts += rng.normal(0.0, noise_m, pts.shape)
    ghosts = c[rng.choice(len(c), 6, replace=False)] + rng.normal(0.0, 3.0, (6, 2))
    pts = np.vstack([pts, ghosts])
    return [Cone(x=float(x), y=float(y)) for x, y in pts], c, tan


def _warm_realistic_fields(mission: MissionTypes) -> None:
    """Plan on a realistic map from several poses: clean, noisy and few-cone."""
    rng = np.random.default_rng(0)
    for noise_m in (0.02, 0.15):
        adapter = FasttubeAdapter(mission)
        cones, centre, tan = _loop_track_cones(rng, noise_m)
        for k in range(0, len(centre), len(centre) // 6):
            pose = Pose2D(
                x=float(centre[k, 0]),
                y=float(centre[k, 1]),
                yaw=float(math.atan2(tan[k, 1], tan[k, 0])),
            )
            adapter.plan(cones, pose)
    # Few cones ahead of the car (start line, one-sided view).
    adapter = FasttubeAdapter(mission)
    few = [Cone(x=4.0, y=1.5), Cone(x=4.0, y=-1.5), Cone(x=8.5, y=1.6)]
    adapter.plan(few, Pose2D(x=0.0, y=0.0, yaw=0.0))


def _warm_layout_signatures() -> None:
    """Compile the two signature families measured missing after the warm-up
    (see warmup_numba_planner), by calling the kernels on arrays of that rank
    and layout. The calls are cheap; the compile is what we want."""
    from fsd_path_planning.utils import math_utils as mu

    base = np.random.default_rng(1).normal(size=(4, 3, 4))
    a3 = base[:, :, :2]  # 3-D, non-contiguous: Numba layout 'A'
    c3 = np.ascontiguousarray(base[:, :, 2:])  # 3-D, contiguous: layout 'C'
    mu.norm_of_last_axis(a3)
    mu.norm_of_last_axis(c3)
    # Every layout pair: the live path mixes them (measured: a sliced 'A' view
    # against a 'C' array), and each pair is its own signature.
    for x, y in ((a3, a3), (a3, c3), (c3, a3), (c3, c3)):
        mu.vec_dot(x, y)
        # Call sites may pass clip_cos_theta or leave it at its default, and
        # Numba compiles those as different signatures: warm both.
        mu.vec_angle_between(x, y)
        mu.vec_angle_between(x, y, True)
    zero_d = np.asarray(0.5)
    mu.angle_difference(zero_d, np.asarray(-0.25))


class _LateCompileLogger:
    """Logs every Numba compile that happens after the warm-up."""

    def __init__(self, log_fn):
        from numba.core import event

        log = log_fn

        class _Listener(event.Listener):
            def on_start(self, ev):
                disp = ev.data.get("dispatcher")
                func = getattr(disp, "py_func", None)
                module = getattr(func, "__module__", "") or ""
                if module == "numba" or module.startswith("numba."):
                    return  # numba's own helpers, compiled inside a user kernel's compile
                name = f"{module}.{func.__qualname__}" if func is not None else repr(disp)
                log(f"Numba compile after warm-up (stalls planning): {name}{ev.data.get('args')}")

            def on_end(self, ev):
                pass

        self.listener = _Listener()
        event.register("numba:compile", self.listener)

    def close(self) -> None:
        from numba.core import event

        event.unregister("numba:compile", self.listener)


def log_late_numba_compiles(log_fn) -> _LateCompileLogger | None:
    """After the warm-up, report any compile on the hot path through ``log_fn``.

    A compile there is a multi-second stall on a cold cache. This makes a
    missed signature visible in the log on the car and in CI, instead of only
    as a latency spike. Returns the handle (``close()`` removes it), or None if
    this numba has no event API.
    """
    try:
        return _LateCompileLogger(log_fn)
    except Exception:  # pylint: disable=broad-except
        logger.warning("could not install the Numba compile listener", exc_info=True)
        return None


def _xy_to_path_points(xy: np.ndarray,
                       kappa: Optional[np.ndarray] = None) -> List[PathPoint]:
    """Convert (M, 2) world-frame samples + optional κ to PathPoints.

    Yaw is recomputed from finite differences (FaSTTUBe outputs include
    yaw implicitly via consecutive xy, no need to pass it through).
    Curvature comes through verbatim from the library when provided;
    defaults to 0 otherwise (the controller will fall back to its own
    finite-difference computation if every PathPoint reads κ=0).
    """
    n = xy.shape[0]
    out: List[PathPoint] = []
    for i in range(n):
        if i + 1 < n:
            dx = xy[i + 1, 0] - xy[i, 0]
            dy = xy[i + 1, 1] - xy[i, 1]
        else:
            # Last point: reuse previous segment's heading so the path
            # tangent is continuous at the tip.
            dx = xy[i, 0] - xy[i - 1, 0]
            dy = xy[i, 1] - xy[i - 1, 1]
        yaw = float(np.arctan2(dy, dx))
        k = float(kappa[i]) if kappa is not None else 0.0
        out.append(PathPoint(x=float(xy[i, 0]), y=float(xy[i, 1]),
                             yaw=yaw, curvature=k))
    return out
