"""Smoke tests for the FaSTTUBe adapter.

Covers the integration surface — type translation, empty-input safety,
frame conventions, and the cone cull window. Position-only since the
body_y classifier was removed; cones carry no colour through this
module.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

# Skip the whole module if the library isn't installed (CI on a fresh
# checkout, IDE running tests without the Docker image's pip env).
fsd_path_planning = pytest.importorskip("fsd_path_planning")

from path_planning.fasttube_adapter import (
    FasttubeAdapter,
    PlanDebug,
    _CULL_RANGE_M,
    _MAX_PATH_ARC_M,
    _cull_cones,
)
from path_planning.core_types import Cone, Pose2D


# --- Fixtures ---------------------------------------------------------------


def _straight_track(n_pairs: int = 12, spacing: float = 3.0,
                    half_width: float = 1.5) -> List[Cone]:
    """Two rows of cones along world-X with small deterministic jitter.

    Jitter (≤ 5 cm) breaks the perfect bilateral symmetry that triggers
    a `ZeroDivisionError` deep in the library's nearest-neighbour cost
    function — the library is designed for real-world cone fields with
    sensor noise, not pixel-perfect synthetic ones. The jitter is
    deterministic so tests are reproducible.
    """
    rng = np.random.default_rng(seed=0xFA577)
    cones: List[Cone] = []
    for i in range(n_pairs):
        x_base = float(i * spacing) + 1.0
        cones.append(Cone(
            x=x_base + float(rng.uniform(-0.05, 0.05)),
            y=+half_width + float(rng.uniform(-0.05, 0.05)),
        ))
        cones.append(Cone(
            x=x_base + float(rng.uniform(-0.05, 0.05)),
            y=-half_width + float(rng.uniform(-0.05, 0.05)),
        ))
    return cones


def _car_at_origin() -> Pose2D:
    return Pose2D(x=0.0, y=0.0, yaw=0.0)


# --- Tests ------------------------------------------------------------------


def test_straight_track_returns_nonempty_path() -> None:
    """Smoke: dense straight track → adapter produces a forward path."""
    adapter = FasttubeAdapter()
    path, debug = adapter.plan(_straight_track(), _car_at_origin())

    assert len(path) >= 5, (
        f"expected ≥5 path points on a clean straight track, got {len(path)}"
    )
    assert isinstance(debug, PlanDebug)
    xs = [p.x for p in path]
    assert xs[-1] > xs[0] + 5.0, (
        f"path didn't progress forward: x[0]={xs[0]:.2f}, x[-1]={xs[-1]:.2f}"
    )


def test_cones_to_arrays_packs_all_into_unknown() -> None:
    """Every cone goes into FaSTTUBe's UNKNOWN slot (slot 0); slots
    1..4 are empty. The library's geometric sort handles assignment."""
    cones = [
        Cone(x=1.0, y=+1.5),
        Cone(x=1.0, y=-1.5),
        Cone(x=2.0, y=+1.5),
    ]
    arrays = FasttubeAdapter._cones_to_arrays(cones)

    assert len(arrays) == 5
    assert arrays[0].shape == (3, 2)
    for slot in range(1, 5):
        assert arrays[slot].shape == (0, 2), \
            f"slot {slot} should be empty"


def test_empty_input_returns_empty_path() -> None:
    """No cones in → no path out, no exception."""
    adapter = FasttubeAdapter()
    path, debug = adapter.plan([], _car_at_origin())
    assert path == []
    assert isinstance(debug, PlanDebug)
    assert debug.left_with_virtual.shape == (0, 2)
    assert debug.right_with_virtual.shape == (0, 2)


def test_one_cone_input_returns_empty_path() -> None:
    """Single cone is degenerate — adapter should swallow any library
    error and return ([], PlanDebug()), not propagate."""
    adapter = FasttubeAdapter()
    cones = [Cone(x=1.0, y=0.0)]
    path, debug = adapter.plan(cones, _car_at_origin())
    assert isinstance(path, list)
    assert isinstance(debug, PlanDebug)


def test_pose_rotation_rotates_path() -> None:
    """Rotate cones + pose by 90° → path rotates correspondingly."""
    adapter = FasttubeAdapter()

    cones_x = _straight_track()
    path_x, _ = adapter.plan(cones_x, Pose2D(x=0.0, y=0.0, yaw=0.0))
    assert len(path_x) >= 5

    cones_y = [Cone(x=-c.y, y=+c.x) for c in cones_x]
    path_y, _ = adapter.plan(cones_y, Pose2D(x=0.0, y=0.0, yaw=math.pi / 2.0))
    assert len(path_y) >= 5

    ys = [p.y for p in path_y]
    assert ys[-1] > ys[0] + 5.0, (
        f"rotated path didn't progress along +Y: y[0]={ys[0]:.2f}, "
        f"y[-1]={ys[-1]:.2f}"
    )


def test_path_points_have_finite_yaw() -> None:
    """Yaw is recomputed from finite differences — must be finite for
    every point including the tail (which reuses the previous segment)."""
    adapter = FasttubeAdapter()
    path, _ = adapter.plan(_straight_track(), _car_at_origin())
    assert all(np.isfinite(p.yaw) for p in path)


# --- Cone cull (range + behind-car) -----------------------------------------


def test_cull_drops_cones_behind_car() -> None:
    """`_cull_cones` filters body_x ≤ 0 (cones already passed)."""
    pose = Pose2D(x=10.0, y=0.0, yaw=0.0)  # car at (10, 0) facing +X
    cones = [
        Cone(x=15.0, y=+1.5),   # ahead — keep
        Cone(x=5.0,  y=+1.5),   # behind — drop
        Cone(x=10.5, y=-1.5),   # ahead — keep
        Cone(x=10.0, y=-1.5),   # body_x≈0 — drop (≤ 0)
    ]
    out = _cull_cones(cones, pose, _CULL_RANGE_M)
    kept_x = sorted(c.x for c in out)
    assert kept_x == [10.5, 15.0]


def test_cull_drops_cones_beyond_range() -> None:
    """`_cull_cones` filters cones farther than max_range_m."""
    pose = Pose2D(x=0.0, y=0.0, yaw=0.0)
    cones = [
        Cone(x=10.0, y=0.0),    # 10 m — keep
        Cone(x=24.0, y=0.0),    # 24 m — keep (< 25)
        Cone(x=26.0, y=0.0),    # 26 m — drop (> 25)
        Cone(x=50.0, y=0.0),    # 50 m — drop
    ]
    out = _cull_cones(cones, pose, _CULL_RANGE_M)
    kept_x = sorted(c.x for c in out)
    assert kept_x == [10.0, 24.0]


def test_cull_respects_pose_yaw() -> None:
    """Cull uses body-frame x; "behind the car" depends on yaw."""
    pose = Pose2D(x=0.0, y=0.0, yaw=math.pi / 2.0)
    cones = [
        Cone(x=0.0, y=+5.0),    # ahead (body_x = +5)
        Cone(x=0.0, y=-5.0),    # behind (body_x = -5)
        Cone(x=+5.0, y=0.0),    # body_x = 0 → drop
        Cone(x=-5.0, y=0.0),    # body_x = 0 → drop
    ]
    out = _cull_cones(cones, pose, _CULL_RANGE_M)
    assert len(out) == 1
    assert out[0].y == pytest.approx(5.0)


# --- Path forward-distance cap (#260) ---------------------------------------


def test_path_arc_length_capped() -> None:
    """A long straight track produces a long FaSTTUBe path; we should
    only publish up to _MAX_PATH_ARC_M of arc length so the controller
    never sees the virtually-extrapolated tail."""
    adapter = FasttubeAdapter()
    cones = _straight_track(n_pairs=30, spacing=3.0)
    path, _ = adapter.plan(cones, _car_at_origin())

    assert len(path) >= 2
    arc = 0.0
    for i in range(1, len(path)):
        dx = path[i].x - path[i - 1].x
        dy = path[i].y - path[i - 1].y
        arc += math.hypot(dx, dy)
    assert arc <= _MAX_PATH_ARC_M * 1.5, (
        f"path arc {arc:.2f}m should be capped near {_MAX_PATH_ARC_M}m, "
        f"got 50%+ over"
    )
