"""Per-cone motion compensation for a rolling LiDAR scan.

The Hesai ATX sweeps its 120 deg field of view in ~90 ms and stamps the cloud
with the start of the sweep; each point carries its own ``timestamp`` (f64,
seconds). The sweep direction alternates every scan. Without compensation every
cone is placed where the car saw it up to 90 ms after ``header.stamp`` while
SLAM treats it as seen at ``header.stamp``: at 6 m/s a cone moves ~30 cm
between consecutive scans depending on which edge the sweep started from.

Deskewing the cloud point by point is not needed: a cone covers a few degrees,
so one time per cone is enough. Point time is linear in azimuth within a scan
(car bag 20 Sep 2026: 0.740 ms/deg, fit residual p50 0.8 ms, p99 3 ms), so a
line fitted to a sparse subsample of points (every 256th: 0.6 ms per scan on
an RK3588 A76 core) gives each cone's time from its azimuth. The cone is then
moved from the car frame at that time to the car frame at ``header.stamp`` with
the ego speed and yaw rate (constant over the sweep).

Car bag, 179 consecutive scans at 7-11 m/s: a cone's scan-to-scan jump after
ego motion went from 29.6 cm median (p90 61.8) to 3.9 cm (p90 8.5).

The sim cloud has no ``timestamp`` field: nothing changes there.
"""
from __future__ import annotations

import numpy as np

# sensor_msgs/PointField.FLOAT64
_FLOAT64 = 8


def scan_time_model(
    data,
    n_points: int,
    point_step: int,
    offsets: dict[str, tuple[int, int]],
    stamp_s: float,
    stride: int = 256,
    min_range_m: float = 0.5,
) -> tuple[float, float, float, float] | None:
    """Fit ``t - stamp = a + b * azimuth`` over every ``stride``-th point.

    ``offsets`` maps field name -> (byte offset, datatype). Returns
    ``(a, b, t_min, t_max)`` in seconds (b in s/rad), or None when the cloud has
    no f64 ``timestamp`` field or too few valid points.
    """
    ts = offsets.get("timestamp")
    if ts is None or ts[1] != _FLOAT64 or n_points < 2 * stride:
        return None
    raw = np.frombuffer(data, dtype=np.uint8, count=n_points * point_step)
    sub = raw.reshape(n_points, point_step)[::stride]
    ox, oy, ot = offsets["x"][0], offsets["y"][0], ts[0]
    x = sub[:, ox:ox + 4].copy().view(np.float32).ravel()
    y = sub[:, oy:oy + 4].copy().view(np.float32).ravel()
    t = sub[:, ot:ot + 8].copy().view(np.float64).ravel() - stamp_s
    # Empty returns sit at the origin with a real time: they would pull the fit.
    ok = (x * x + y * y > min_range_m * min_range_m) & np.isfinite(t)
    if ok.sum() < 20:
        return None
    az = np.arctan2(y[ok], x[ok]).astype(np.float64)
    t = t[ok]
    az_c = az - az.mean()
    var = float(az_c @ az_c)
    if var <= 0.0:
        return None
    b = float(az_c @ (t - t.mean())) / var
    a = float(t.mean()) - b * float(az.mean())
    return a, b, float(t.min()), float(t.max())


def deskew_xy(
    x: np.ndarray,
    y: np.ndarray,
    model: tuple[float, float, float, float],
    v_mps: float,
    yaw_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move points seen at their sweep time into the car frame at the stamp.

    Car frame at time tau relative to the stamp: rotated by psi = r*tau and
    displaced by the arc chord v*tau*(cos(psi/2), sin(psi/2)).
    """
    a, b, t_min, t_max = model
    tau = np.clip(a + b * np.arctan2(y, x), t_min, t_max)
    psi = yaw_rate * tau
    c, s = np.cos(psi), np.sin(psi)
    d = v_mps * tau
    return (c * x - s * y + d * np.cos(0.5 * psi),
            s * x + c * y + d * np.sin(0.5 * psi))


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


__all__ = ["scan_time_model", "deskew_xy", "stamp_seconds"]
