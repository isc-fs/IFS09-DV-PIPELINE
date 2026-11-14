"""Configure-time Numba warmup, so the first real scan pays no JIT cost."""

from __future__ import annotations

import math

import numpy as np

from cone_detection.config import ConeDetectionConfig
from cone_detection.detector import RealtimeConeDetector
from cone_detection.fitting.alternative_fits import cone_fit_2params
from cone_detection.fitting.cone_classes import CONE_BIG_C, CONE_BIG_D, CONE_SMALL_C, CONE_SMALL_D
from cone_detection.fitting.template_fit import cone_fit_template_dispatch
from cone_detection.ground.methods.mrf_methods import warm_mrf_graph



def warmup_numba_functions(
    config: ConeDetectionConfig | None = None,
    *,
    also_warm_two_param: bool = False,
) -> None:
    """Warm every Numba kernel on the live hot path at configure time.

    All perception kernels compile with ``cache=True``, so the compiled
    machine code persists in ``__pycache__`` next to the sources (or in
    ``NUMBA_CACHE_DIR`` when set). Warmup still runs on every configure: the
    first call per process is what loads the disk cache, and it transparently
    recompiles on a cache miss (numba/Python upgrade, source change, cold
    volume). Cached start ≈ 1–2 s; first-ever start pays the full 10–20 s JIT.

    Coverage matters more than volume here. Numba specializes per *call
    pattern*: an argument left to its default compiles a different
    specialization (``Omitted`` type) than the same value passed explicitly,
    so hand-written kernel calls alone can miss the signature the node
    actually uses. The end-to-end block below therefore replays the real
    call chain — ``RealtimeConeDetector.detect`` on a synthetic scan, with
    the caller's ``config`` — twice, covering both the no-previous-plane and
    warm-plane RANSAC entries exactly as the live node hits them.

    Args:
        config: The production config the node will run with (solver choice,
            fit backend, ground method). ``None`` = module defaults.
        also_warm_two_param: Force-warm ``cone_fit_2params`` even when the
            config does not select ``fit_backend="two_param"``.
    """
    cfg = config or ConeDetectionConfig()
    rng = np.random.default_rng(0)

    def _synthetic_cone_cloud(
        n: int, noise_m: float, c_true: float, d_true: float
    ) -> np.ndarray:
        """N points on a cone-like sweep; float32 like decoded PointCloud2."""
        th = np.linspace(0, 2 * np.pi, n)
        rr = np.linspace(0.05, 0.10, n)
        a0, b0 = 5.0, 0.5
        x = (a0 + rr * np.cos(th)).astype(np.float32)
        y = (b0 + rr * np.sin(th)).astype(np.float32)
        z = (d_true - c_true * rr + rng.normal(0.0, noise_m, size=n)).astype(
            np.float32
        )
        return np.column_stack([x, y, z])

    def _near_face_cluster(
        cx: float, cy: float, c_true: float, d_true: float, n: int
    ) -> np.ndarray:
        """One-sided near-face cluster like a vertical-beam LiDAR sees."""
        z = rng.uniform(0.06, d_true - 0.03, size=n)
        gamma = (d_true - z) / c_true
        norm = math.hypot(cx, cy)
        ux, uy = cx / norm, cy / norm
        psi = rng.uniform(-1.0, 1.0, size=n)
        px = cx - gamma * (np.cos(psi) * ux - np.sin(psi) * uy)
        py = cy - gamma * (np.cos(psi) * uy + np.sin(psi) * ux)
        px += rng.normal(0.0, 0.003, size=n)
        py += rng.normal(0.0, 0.003, size=n)
        z = z + rng.normal(0.0, 0.008, size=n)
        return np.column_stack([px, py, z])

    # --- direct kernel warms: both solver paths (gn = Numba kernel, scipy
    # kept for A/B) plus whatever non-default method the config selects, and
    # the collinear closed-form dispatch branch.
    solvers = {"gn", "L-BFGS-B", cfg.cone_fit_solver}
    for n in (8, 24, 48):
        for solver in solvers:
            cone_fit_template_dispatch(
                _synthetic_cone_cloud(n, 0.012, 5.0, 0.35), solver=solver
            )
            cone_fit_template_dispatch(
                _synthetic_cone_cloud(n, 0.012, 5.5, 0.55), solver=solver
            )

    line_cluster = _synthetic_cone_cloud(6, 0.012, 5.0, 0.35)
    line_cluster[:, :2] = np.linspace(
        (5.0, 0.0), (5.2, 0.03), num=line_cluster.shape[0]
    )
    cone_fit_template_dispatch(line_cluster)

    if also_warm_two_param or cfg.fit_backend == "two_param":
        for n in (8, 24, 48):
            cone_fit_2params(_synthetic_cone_cloud(n, 0.012, 5.0, 0.35))

    # --- end-to-end warm: replay the live call chain (decode-shaped float32
    # cloud -> ground removal -> rotate -> DBSCAN -> floor cull -> gates -> fit) with
    # the production config. A tilted ground plane at z ~ 0 plus one cluster
    # per template keeps every gate passable so the fit stage really runs.
    n_ground = 2000
    gx = rng.uniform(1.0, 14.0, n_ground)
    gy = rng.uniform(-4.0, 4.0, n_ground)
    gz = 0.02 * gx + rng.normal(0.0, 0.01, size=n_ground)
    small = _near_face_cluster(5.0, 1.0, CONE_SMALL_C, CONE_SMALL_D, 24)
    big = _near_face_cluster(8.0, -2.0, CONE_BIG_C, CONE_BIG_D, 24)
    small[:, 2] += 0.02 * small[:, 0]
    big[:, 2] += 0.02 * big[:, 0]
    scan = np.vstack(
        [np.column_stack([gx, gy, gz]), small, big]
    ).astype(np.float32)
    detector = RealtimeConeDetector(cfg)
    detector.detect(scan)  # first scan: no previous plane
    detector.detect(scan)  # second scan: warm-plane RANSAC branch
    if cfg.ground_method == "mrf":
        # The synthetic scan above is fully decided before the flow solve, so the
        # graph-cut kernels need their own input (see warm_mrf_graph).
        warm_mrf_graph()
