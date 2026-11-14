"""The C controller (control_cpp/src/control_core.c) against this package.

control_core is a port of the Python controllers for the C++ node and, later,
the uDV STM32H7. These tests build it with the system C compiler and compare
it with the Python reference through ctypes: each piece on random inputs, and
(when rclpy is available) the whole control_node tick in a closed loop.
"""
from __future__ import annotations

import ctypes as C
import math
import pathlib
import random
import shutil
import subprocess

import pytest

from control.controllers.pi_velocity import PIVelocity
from control.controllers.pure_pursuit import PurePursuit
from control.controllers.stanley import Stanley
from control.models.bicycle import KinematicBicycle
from control.odom_history import OdomHistory, project_pose
from control.reference import ReferenceTrajectory
from control.state import VehicleState

ROOT = pathlib.Path(__file__).resolve().parents[2] / "control_cpp"
TOL = 1e-9  # libm hypot vs Python's math.hypot may differ in the last bit
# -fno-builtin: clang fuses sin(a) and cos(a) into one sincos call whose result
# can differ from Python's separate calls in the last bit (seen on macOS).

D = C.c_double
PD = C.POINTER(C.c_double)


class Params(C.Structure):
    _fields_ = [
        ("wheelbase", D), ("max_steer_rad", D), ("lateral", C.c_int),
        ("lookahead_min", D), ("lookahead_k", D), ("lookahead_max", D),
        ("lookahead_radius_factor", D),
        ("stanley_k", D), ("stanley_k_soft", D), ("stanley_k_yaw_rate", D),
        ("stanley_k_damp_steer", D), ("stanley_yaw_rate_loop_gain", D),
        ("v_max", D), ("a_lat_max", D), ("a_dec_max", D), ("kp", D), ("ki", D),
        ("deadband", D), ("throttle_max", D),
        ("regen_power_w", D), ("vehicle_mass", D), ("unseen_corner_radius", D),
        ("brake_latency_s", D), ("v_meas_tau", D), ("v_meas_max_accel", D),
        ("throttle_rate", D), ("throttle_release_rate", D), ("regen_rate", D),
        ("steering_rate", D), ("stop_latch_min_travel", D), ("stop_past_anchor_m", D),
        ("stop_on_anchor", C.c_int), ("latency_comp_s", D),
    ]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("cc") / "libcontrol_core_test.so"
    subprocess.run(
        [cc, "-std=c99", "-O2", "-ffp-contract=off", "-fno-builtin", "-shared", "-fPIC",
         f"-I{ROOT / 'include'}", str(ROOT / "src" / "control_core.c"),
         str(ROOT / "test" / "core_shim.c"), "-o", str(out), "-lm"],
        check=True)
    L = C.CDLL(str(out))
    L.ctrl_params_default.argtypes = [C.POINTER(Params)]
    L.shim_ref.argtypes = [PD, PD, PD, C.c_int, PD, PD, PD]
    L.shim_pp.restype = D
    L.shim_pp.argtypes = [C.POINTER(Params), PD, PD, PD, PD, C.c_int]
    L.shim_stanley.restype = D
    L.shim_stanley.argtypes = [C.POINTER(Params), PD, PD, PD, PD, C.c_int, PD]
    L.shim_pi_new.restype = C.c_void_p
    L.shim_pi_step.argtypes = [C.POINTER(Params), C.c_void_p, PD, PD, PD, PD, C.c_int,
                               D, C.c_int, PD]
    L.shim_hist_new.restype = C.c_void_p
    L.ctrl_hist_add.argtypes = [C.c_void_p, D, D, D, D]
    L.ctrl_hist_at.restype = C.c_int
    L.ctrl_hist_at.argtypes = [C.c_void_p, D, PD]
    L.ctrl_project_pose.argtypes = [PD, PD, PD, PD]
    L.ctrl_predict_pose.argtypes = [PD, D, PD, C.c_int, C.c_int, D, D, D, PD]
    L.ctrl_brake_entry_speed.restype = D
    L.ctrl_brake_entry_speed.argtypes = [C.POINTER(Params), D, D]
    L.ctrl_quat_yaw.restype = D
    L.ctrl_quat_yaw.argtypes = [D, D, D, D]
    L.shim_node_new.restype = C.c_void_p
    L.shim_node_new.argtypes = [C.POINTER(Params)]
    L.ctrl_node_on_pose.argtypes = [C.c_void_p, D, D, D, D, D, D, D]
    L.ctrl_node_on_odom.argtypes = [C.c_void_p, D, D, D, D, D, D, D, D]
    L.ctrl_node_on_path.restype = C.c_int
    L.ctrl_node_on_path.argtypes = [C.c_void_p, PD, PD, PD, C.c_int]
    L.ctrl_node_on_final_lap.restype = C.c_int
    L.ctrl_node_on_final_lap.argtypes = [C.c_void_p, C.c_int]
    L.shim_node_final_lap.restype = C.c_int
    L.shim_node_final_lap.argtypes = [C.c_void_p]
    L.ctrl_node_on_orange.restype = C.c_int
    L.ctrl_node_on_orange.argtypes = [C.c_void_p, PD, PD, C.c_int, PD, PD, PD, PD]
    L.shim_node_tick.argtypes = [C.c_void_p, PD]
    L.shim_free.argtypes = [C.c_void_p]
    return L


def arr(xs):
    return (D * max(1, len(xs)))(*xs)


def params(lib, **kw) -> Params:
    p = Params()
    lib.ctrl_params_default(C.byref(p))
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def random_path(rng: random.Random, n: int):
    """A path ahead of a car near the origin: piecewise-constant curvature."""
    x, y, h = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.5, 0.5)
    xs, ys = [], []
    k = rng.uniform(-0.4, 0.4)
    for i in range(n):
        if rng.random() < 0.15:
            k = rng.choice([0.0, rng.uniform(-0.5, 0.5)])
        xs.append(x + rng.gauss(0, 0.01))
        ys.append(y + rng.gauss(0, 0.01))
        step = 0.5 if rng.random() > 0.05 else 0.0  # a repeated point now and then
        h += k * step
        x += step * math.cos(h)
        y += step * math.sin(h)
    return xs, ys


def random_state(rng: random.Random) -> VehicleState:
    return VehicleState(x=rng.uniform(-2, 2), y=rng.uniform(-2, 2),
                        yaw=rng.uniform(-math.pi, math.pi), vx=rng.uniform(-1, 13),
                        vy=rng.uniform(-0.5, 0.5), yaw_rate=rng.uniform(-1, 1))


def s6(st: VehicleState):
    return arr([st.x, st.y, st.yaw, st.vx, st.vy, st.yaw_rate])


def kappas(rng, n):
    r = rng.random()
    if r < 0.3:
        return None
    if r < 0.4:
        return [0.0] * n  # the "no curvature" sentinel
    return [rng.uniform(-0.6, 0.6) for _ in range(n)]


def test_reference_geometry(lib):
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(2, 80)
        xs, ys = random_path(rng, n)
        ks = kappas(rng, n)
        ref = ReferenceTrajectory.from_xy(xs, ys, kappa=ks)
        yaw, curv, s = (D * n)(), (D * n)(), (D * n)()
        lib.shim_ref(arr(xs), arr(ys), arr(ks) if ks else None, n, yaw, curv, s)
        assert max(abs(a - b) for a, b in zip(ref.yaw, yaw)) == 0.0
        assert max(abs(a - b) for a, b in zip(ref.s, s)) < TOL
        assert max(abs(a - b) for a, b in zip(ref.curvature, curv)) < TOL


@pytest.mark.parametrize("seed", range(4))
def test_lateral_controllers(lib, seed):
    rng = random.Random(seed)
    p = params(lib)
    bike = KinematicBicycle(wheelbase=p.wheelbase, max_steer_rad=p.max_steer_rad)
    pp = PurePursuit(lookahead_min=1.0, lookahead_k=0.5, model=bike)
    stan = Stanley(k=2.0, k_soft=1.5, k_yaw_rate=0.0, k_damp_steer=0.2, model=bike,
                   yaw_rate_loop_gain=0.2)
    prev = D(0.0)
    worst = 0.0
    for _ in range(500):
        n = rng.randint(0, 70)
        xs, ys = random_path(rng, n)
        ks = kappas(rng, n)
        st = random_state(rng)
        ref = ReferenceTrajectory.from_xy(xs, ys, kappa=ks)
        a = pp.compute(st, ref)
        b = lib.shim_pp(C.byref(p), s6(st), arr(xs), arr(ys), arr(ks) if ks else None, n)
        c = stan.compute(st, ref)
        d = lib.shim_stanley(C.byref(p), s6(st), arr(xs), arr(ys),
                             arr(ks) if ks else None, n, C.byref(prev))
        worst = max(worst, abs(a - b), abs(c - d), abs(stan._prev_steer_norm - prev.value))
    assert worst < TOL


@pytest.mark.parametrize("unseen,power", [(4.5, 11000.0), (0.0, 11000.0), (4.5, 0.0)])
def test_pi_velocity_sequence(lib, unseen, power):
    rng = random.Random(7)
    p = params(lib, unseen_corner_radius=unseen, regen_power_w=power, v_max=12.0)
    py = PIVelocity(v_max=12.0, a_lat_max=3.0, a_dec_max=4.0, kp=0.5, ki=0.05,
                    deadband=0.05, throttle_max=1.0, regen_power_w=power,
                    vehicle_mass=275.0, unseen_corner_radius=unseen,
                    brake_latency_s=0.3, v_meas_tau=0.15, v_meas_max_accel=20.0)
    pi = lib.shim_pi_new()
    out = (D * 4)()
    v = 0.0
    worst = 0.0
    for i in range(3000):
        n = rng.randint(2, 60)
        xs, ys = random_path(rng, n)
        ks = kappas(rng, n)
        v = max(-1.0, v + rng.gauss(0.05, 0.3)) if rng.random() > 0.02 else 0.0  # glitches
        st = VehicleState(x=xs[0], y=ys[0], yaw=0.0, vx=v)
        ref = ReferenceTrajectory.from_xy(xs, ys, kappa=ks)
        latched = i > 2000
        ref.stop_distance = (40.0 - (i - 2000) * 0.05) if latched else float("inf")
        ref.stop_latched = latched
        thr, regen = py.compute(st, ref)
        lib.shim_pi_step(C.byref(p), pi, s6(st), arr(xs), arr(ys), arr(ks) if ks else None,
                         n, ref.stop_distance, int(latched), out)
        worst = max(worst, abs(thr - out[0]), abs(regen - out[1]),
                    abs(py.last_v_set - out[2]), abs(py.last_kappa_max - out[3]))
    lib.shim_free(pi)
    assert worst < TOL


def test_brake_entry_speed(lib):
    rng = random.Random(3)
    for power in (11000.0, 0.0):
        p = params(lib, regen_power_w=power)
        py = PIVelocity(regen_power_w=power)
        for _ in range(2000):
            ve, s = rng.uniform(0, 15), rng.uniform(-5, 200)
            assert lib.ctrl_brake_entry_speed(C.byref(p), ve, s) == py.brake_entry_speed(ve, s)


def test_odom_history_and_projection(lib):
    rng = random.Random(5)
    h = lib.shim_hist_new()
    py = OdomHistory()
    t = 100.0
    out = (D * 3)()
    for i in range(5000):
        if rng.random() < 0.002:
            t -= 50.0  # clock restart
        t += rng.choice([0.01, 0.01, 0.01, 0.0, 0.3])
        x, y, yaw = rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-math.pi, math.pi)
        py.add(t, x, y, yaw)
        lib.ctrl_hist_add(h, t, x, y, yaw)
        q = t - rng.uniform(-0.4, 2.5)
        a = py.at(q)
        ok = lib.ctrl_hist_at(h, q, out)
        assert (a is None) == (ok == 0)
        if a is not None:
            assert tuple(out) == a
            now = (x, y, yaw)
            pose = (rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-3, 3))
            res = (D * 3)()
            lib.ctrl_project_pose(arr(pose), arr(a), arr(now), res)
            assert tuple(res) == project_pose(pose, a, now)
    lib.shim_free(h)


def test_predict_pose(lib):
    from control.odom_history import predict_pose
    rng = random.Random(4)
    res = (D * 3)()
    for _ in range(3000):
        pose = (rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-math.pi, math.pi))
        v = rng.uniform(-1, 15)
        hist = [rng.uniform(-1, 1) for _ in range(rng.randint(0, 16))]
        n = rng.randint(-1, 6)
        lib.ctrl_predict_pose(arr(pose), v, arr(hist), len(hist), n, 1.627,
                              math.radians(18.2), 0.025, res)
        assert tuple(res) == predict_pose(pose, v, hist, n, 1.627, math.radians(18.2), 0.025)


def test_quaternion_yaw(lib):
    t3d = pytest.importorskip("transforms3d.euler")
    rng = random.Random(9)
    for _ in range(2000):
        q = [rng.gauss(0, 1) for _ in range(4)]
        if rng.random() < 0.5:
            q[1] = q[2] = 0.0
        assert lib.ctrl_quat_yaw(*q) == t3d.quat2euler(q)[2]
    assert lib.ctrl_quat_yaw(0.0, 0.0, 0.0, 0.0) == t3d.quat2euler([0, 0, 0, 0])[2]


# --------------------------------------------------------------------------
# Whole node: the Python ControlNode and the C node on the same messages.
# --------------------------------------------------------------------------

class _Rec:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)

    def get_subscription_count(self):
        return 0


def _track(kind: str):
    """Centreline sampler for a closed track: (x, y) at arc length s."""
    if kind == "accel":
        return lambda s: (s, 0.0), 1e9
    r, straight = 9.0, 30.0
    per = 2 * straight + 2 * math.pi * r

    def f(s):
        s %= per
        if s < straight:
            return s, 0.0
        s -= straight
        if s < math.pi * r:
            a = s / r
            return straight + r * math.sin(a), r - r * math.cos(a)
        s -= math.pi * r
        if s < straight:
            return straight - s, 2 * r
        a = (s - straight) / r
        return -r * math.sin(a), r + r * math.cos(a)
    return f, per


def test_final_lap_rise_before_moving_is_ignored(lib):
    """A False -> True edge before stop_latch_min_travel is the last mission's
    latched value (HIL v11, trackdrive 4 stopped after lap 1): ignored."""
    cn = lib.shim_node_new(C.byref(params(lib)))
    try:
        assert lib.shim_node_final_lap(cn) == 1          # safe default
        assert lib.ctrl_node_on_final_lap(cn, 0) == 1    # this mission: not yet
        assert lib.ctrl_node_on_final_lap(cn, 1) == -1   # stale True, at 0 m
        assert lib.shim_node_final_lap(cn) == 0
        assert lib.ctrl_node_on_final_lap(cn, 0) == 0
    finally:
        lib.shim_free(cn)


@pytest.mark.parametrize("mode,behavior,latency", [
    ("trackdrive", "pure_pursuit", 0.0), ("autocross", "stanley", 0.0),
    ("skidpad", "stanley", 0.0), ("accel", "pure_pursuit", 0.0),
    ("autocross", "stanley", 0.05), ("trackdrive", "pure_pursuit", 0.075)])
def test_node_closed_loop(lib, mode, behavior, latency):
    rclpy = pytest.importorskip("rclpy")
    from nav_msgs.msg import Odometry, Path
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool
    from dv_msgs.msg import ConeArray
    from control.control_node import ControlNode
    from rclpy.parameter import Parameter

    rclpy.init()
    try:
        node = ControlNode()
        node.set_parameters([Parameter("latency_comp_s", value=latency)])
        node._mode_name = mode
        node._behavior = behavior
        node._build_strategies()
        cmd, vset, kap = _Rec(), _Rec(), _Rec()
        node._cmd_pub, node._v_set_pub, node._kappa_max_pub = cmd, vset, kap
        node._latency_pub = None

        p = params(lib, lateral=1 if behavior == "stanley" else 0,
                   v_max=12.0 if mode == "accel" else 5.0,
                   unseen_corner_radius=0.0 if mode == "accel" else 4.5,
                   stop_on_anchor=1 if mode == "skidpad" else 0, latency_comp_s=latency)
        cn = lib.shim_node_new(C.byref(p))
        out = (D * 8)()

        track, per = _track(mode)
        rng = random.Random(11)
        # Plant: kinematic bicycle driven by the Python node's command.
        x, y, h, v = 0.0, 0.0, 0.0, 0.0
        s_car = 0.0
        hist = []  # (t, x, y, h) for the delayed SLAM pose
        final_lap = mode != "trackdrive"
        worst = 0.0
        ticks = 0
        steps = 100 * 120  # 120 s at 100 Hz
        for k in range(steps):
            t = 1000.0 + k * 0.01
            # plant step
            thr = node._last_throttle
            brk = node._last_regen
            steer = -node._last_steering * node._drive._lateral.model.max_steer_rad
            v = max(0.0, v + (6.0 * thr - 4.0 * brk - 0.05 * v) * 0.01)
            h += v * math.tan(steer) / 1.627 * 0.01
            x += v * math.cos(h) * 0.01
            y += v * math.sin(h) * 0.01
            hist.append((t, x, y, h))
            # progress along the track (nearest of a local search)
            best = min((math.hypot(track(s_car + d)[0] - x, track(s_car + d)[1] - y), d)
                       for d in [i * 0.1 for i in range(-5, 30)])
            s_car += best[1]

            # /odom 100 Hz, with drift and the odd glitch
            o = Odometry()
            o.header.stamp.sec, o.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9))
            st = o.header.stamp.sec + o.header.stamp.nanosec * 1e-9
            o.pose.pose.position.x = x * 1.01 + 0.3
            o.pose.pose.position.y = y * 0.99 - 0.2
            o.pose.pose.orientation.w = math.cos((h + 0.05) / 2)
            o.pose.pose.orientation.z = math.sin((h + 0.05) / 2)
            o.twist.twist.linear.x = 0.0 if rng.random() < 0.003 else v + rng.gauss(0, 0.02)
            o.twist.twist.linear.y = rng.gauss(0, 0.01)
            o.twist.twist.angular.z = v * math.tan(steer) / 1.627
            node._on_odom(o)
            lib.ctrl_node_on_odom(cn, st, o.pose.pose.position.x, o.pose.pose.position.y,
                                  o.pose.pose.orientation.w, o.pose.pose.orientation.z,
                                  o.twist.twist.linear.x, o.twist.twist.linear.y,
                                  o.twist.twist.angular.z)

            if k % 10 == 5 and k > 20:  # /slam/pose 10 Hz, pose 0.1 s old
                tp, xp, yp, hp = hist[-11]
                m = Odometry()
                m.header.stamp.sec = int(tp)
                m.header.stamp.nanosec = int(round((tp % 1) * 1e9))
                stp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                m.pose.pose.position.x = xp + rng.gauss(0, 0.02)
                m.pose.pose.position.y = yp + rng.gauss(0, 0.02)
                m.pose.pose.orientation.w = math.cos(hp / 2)
                m.pose.pose.orientation.x = rng.gauss(0, 0.001)
                m.pose.pose.orientation.y = rng.gauss(0, 0.001)
                m.pose.pose.orientation.z = math.sin(hp / 2)
                node._on_pose(m)
                q = m.pose.pose.orientation
                lib.ctrl_node_on_pose(cn, stp, m.pose.pose.position.x, m.pose.pose.position.y,
                                      q.w, q.x, q.y, q.z)

            if k % 10 == 7 and k > 30:  # /Path 10 Hz: 40 points ahead
                pm = Path()
                xs, ys, zs = [], [], []
                for i in range(40):
                    px, py_ = track(s_car + i * 0.5)
                    ps = PoseStamped()
                    ps.pose.position.x = px + rng.gauss(0, 0.03)
                    ps.pose.position.y = py_ + rng.gauss(0, 0.03)
                    ps.pose.position.z = 0.0 if mode == "accel" else rng.uniform(-0.1, 0.12)
                    pm.poses.append(ps)
                    xs.append(ps.pose.position.x)
                    ys.append(ps.pose.position.y)
                    zs.append(ps.pose.position.z)
                node._on_path(pm)
                lib.ctrl_node_on_path(cn, arr(xs), arr(ys), arr(zs), len(xs))

            if mode == "trackdrive" and k == 4:
                # The previous mission's latched True arriving after this
                # mission's False (HIL v11): both nodes must ignore it.
                node._on_final_lap(Bool(data=True))
                assert lib.ctrl_node_on_final_lap(cn, 1) == -1
                assert node._final_lap is False
                assert lib.shim_node_final_lap(cn) == 0
            if mode == "trackdrive" and not final_lap and s_car > per:
                final_lap = True
            if k % 10 == 3:
                b = Bool(data=final_lap)
                node._on_final_lap(b)
                lib.ctrl_node_on_final_lap(cn, int(final_lap))

            if k % 10 == 9:  # big-orange cones near the start/finish line
                gx = (s_car % per) if mode != "accel" else s_car
                ca = ConeArray()
                if mode == "accel" and 60 < s_car < 75 or mode != "accel" and gx > per - 12:
                    d = (per - gx) if mode != "accel" else 75 - s_car
                    ca.x = [d + 0.1, d - 0.1, d + 0.05]
                    ca.y = [1.6, -1.5, 1.7]
                node._on_orange(ca)
                lib.ctrl_node_on_orange(cn, arr(list(ca.x)), arr(list(ca.y)), len(ca.x),
                                        None, None, None, None)

            if k % 5 in (0, 2):  # 40 Hz tick (two per 50 ms)
                node._tick()
                lib.shim_node_tick(cn, out)
                ticks += 1
                c = cmd.msgs[-1]
                worst = max(worst, abs(c.throttle - out[0]), abs(c.brake - out[1]),
                            abs(c.steering - out[2]), abs(node._travelled - out[6]))
                if out[3]:
                    worst = max(worst, abs(vset.msgs[-1].data - out[4]),
                                abs(kap.msgs[-1].data - out[5]))
                assert bool(node._stop_latched) == bool(out[7])
        lib.shim_free(cn)
        assert ticks > 4000
        assert node._travelled > 40.0, "the closed loop never drove"
        assert worst < 1e-6, worst
    finally:
        rclpy.shutdown()
