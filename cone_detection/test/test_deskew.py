"""Per-cone motion compensation (cone_detection/deskew.py).

A synthetic rolling scan: the car drives at v with yaw rate r while the sensor
sweeps -60..+60 deg (either direction) in 90 ms; each static cone is recorded
in the car frame of the instant the beam hit it. Deskewing must put every cone
back where it is in the car frame at the stamp, whatever the sweep direction.
"""
import numpy as np
import pytest

from cone_detection.deskew import deskew_xy, scan_time_model

STEP = 26  # Hesai ATX packed point: x,y,z,intensity f32, ring u16, timestamp f64
OFFSETS = {"x": (0, 7), "y": (4, 7), "z": (8, 7), "intensity": (12, 7),
           "ring": (16, 4), "timestamp": (18, 8)}
STAMP = 1_758_375_927.25
SWEEP_S = 0.090


def _pose(tau, v, r):
    """Car pose at tau in the frame at the stamp (constant v, r)."""
    psi = r * tau
    return v * tau * np.cos(psi / 2), v * tau * np.sin(psi / 2), psi


def _to_car(px, py, tau, v, r):
    cx, cy, psi = _pose(tau, v, r)
    dx, dy = px - cx, py - cy
    c, s = np.cos(psi), np.sin(psi)
    return c * dx + s * dy, -s * dx + c * dy


def _beam_time(az, forward):
    """Time the beam points at sensor azimuth az (rad)."""
    frac = (az + np.pi / 3) / (2 * np.pi / 3)
    return SWEEP_S * (frac if forward else 1 - frac)


def _cloud(v, r, forward, n=8000, seed=0):
    """Points as the sensor reports them: in the car frame of their beam time."""
    rng = np.random.default_rng(seed)
    az = np.linspace(-np.pi / 3, np.pi / 3, n)
    tau = _beam_time(az, forward)
    rng_m = rng.uniform(2, 25, n)
    x, y = rng_m * np.cos(az), rng_m * np.sin(az)
    buf = np.zeros((n, STEP), np.uint8)
    buf[:, 0:4] = x.astype(np.float32).view(np.uint8).reshape(n, 4)
    buf[:, 4:8] = y.astype(np.float32).view(np.uint8).reshape(n, 4)
    buf[:, 18:26] = (STAMP + tau).view(np.uint8).reshape(n, 8)
    buf[::97, 0:8] = 0  # empty returns at the origin
    return buf.tobytes(), n


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("v,r", [(8.0, 0.0), (6.0, 0.8), (3.0, -1.5)])
def test_cones_return_to_the_stamp_frame(forward, v, r):
    data, n = _cloud(v, r, forward)
    model = scan_time_model(data, n, STEP, OFFSETS, STAMP)
    assert model is not None
    a, b, t_min, t_max = model
    assert abs(abs(b) - SWEEP_S / (2 * np.pi / 3)) < 1e-6

    # cones in the stamp frame, seen at the beam time of their azimuth
    true = np.array([[5.0, 1.6], [9.0, -1.5], [14.0, 2.5], [4.0, -2.2], [18.0, -6.0]])
    seen = true.copy()
    for _ in range(10):  # the beam time depends on where the cone is seen
        tau = _beam_time(np.arctan2(seen[:, 1], seen[:, 0]), forward)
        seen = np.column_stack(_to_car(true[:, 0], true[:, 1], tau, v, r))
    err_before = np.hypot(*(seen - true).T).max()

    x0, y0 = deskew_xy(seen[:, 0], seen[:, 1], model, v, r)
    err_after = np.hypot(x0 - true[:, 0], y0 - true[:, 1]).max()
    assert err_before > 0.15
    assert err_after < 0.02, (err_before, err_after)


def test_no_timestamp_field_means_no_model():
    data, n = _cloud(5.0, 0.0, True)
    off = {k: v for k, v in OFFSETS.items() if k != "timestamp"}
    assert scan_time_model(data, n, STEP, off, STAMP) is None


def test_standstill_changes_nothing():
    data, n = _cloud(0.0, 0.0, True)
    model = scan_time_model(data, n, STEP, OFFSETS, STAMP)
    x = np.array([5.0, 9.0])
    y = np.array([1.6, -1.5])
    x0, y0 = deskew_xy(x, y, model, 0.0, 0.0)
    assert np.allclose(x0, x) and np.allclose(y0, y)
