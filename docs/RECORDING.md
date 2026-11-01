# Rosbag recording on the car

DV testing time is scarce, so we record a bag on **every** run — including
manual driving — for offline replay (perception / SLAM / odom development).

## TL;DR

Recording captures **everything available on the graph** (`-a`), which differs
by mode because the pipeline only runs for real missions:

| You run | Unit | Records | Bag name |
|---|---|---|---|
| `dv race` for a mission | `dv-pipeline` + `dv-record` | **everything** — sensors + cones + SLAM + path + control commands (`/ctrl/cmd_internal`) | `<mission>_<ts>` (accel / skidpad / autocross / trackdrive) |
| `dv manual` (manual driving) | `dv-manual` | **sensors only** — `/imu`, `/lidar_points`, `/motor_rpm`, `/steering_angle` (no pipeline → no autonomy topics exist) | `manual_<ts>` |
| `dv stop` | — | stops both | — |

The pipeline bag is named by the **confirmed AMI mission** (falls back to
`race_`/`umbilical_` from `/run/dv_mode` if the mission can't be read). Manual
bags are `manual_`. So every bag's filename tells you what it is.

Bags land in `/home/isc/bags/`. **Always `dv stop` before copying a bag off the
car** — a hard kill leaves an unfinalized mcap (no footer/index).

## What gets recorded

`ros2 bag record -s mcap -a` — **every topic on the graph**, including the raw
`/lidar_points` cloud (~45 MB/s ≈ 2.7 GB/min; a 5-min run ≈ 13 GB). The cloud is
the one signal that lets us debug perception offline, so it is recorded by
default. Trim it with `DV_RECORD_EXCLUDE='^/lidar_points$'` if disk is tight.

**QoS matters:** the Hesai driver (and the uDV) publish best-effort. A default
(reliable) rosbag reader is QoS-incompatible with a best-effort publisher and
records **zero** messages, silently. The recorder forces a best-effort reader on
those topics (`DV_RECORD_BEST_EFFORT`); don't remove that or the bag comes back
empty for the cloud.

## When does it start?

Not on a mission — on the **graph being up**. The recorder waits for a warmup
topic to appear, then records:

- **race** (`dv-record`): waits for `/dv/status` (mission_control's heartbeat —
  the last node up). It is **pipeline-gated, not mission-gated**: it never reads
  `/ami/mission` and records the same way for every mission.
- **manual** (`dv-manual`): waits for `/lidar_points` to be advertised (the
  sensor layer is up). No pipeline, no `/dv/status`.

## Manual driving (no pipeline)

`dv manual` starts `dv-manual.service`, which brings up **only** the sensor layer
(`car_sensors.launch.py` — Hesai driver + static TFs) and records. It does **not**
start `mission_control` or the autonomy nodes.

```bash
dv manual         # start: sensors + recorder, bag = manual_<timestamp>
dv manual stop    # stop just the manual recorder
dv stop           # stop everything (pipeline + manual)
dv status         # shows dv-manual + dv-automission state + last bag
```

### Recording is gated on mission selection

You don't run `dv manual` by hand. `dv-automission.service` (enabled at boot)
watches `/ami/mission` and starts the manual recorder **only when you confirm
Manual** on the AMI — recording is gated on the mission you select:

| `/ami/mission` | Meaning | Automission does |
|---|---|---|
| **0** | Manual **confirmed** | **starts** `dv-manual` → sensor bag |
| **1–9** | a real mission confirmed | **stops** `dv-manual` (`dv race` records the full bag) |
| **-1** | up, nothing confirmed | leaves state (not Manual — no recording) |
| not seen | uDV/agent silent | leaves state (no thrash) |
| `/etc/dv/norecord` | opt-out | stops + stays idle |

**To record a manual bag: select + confirm Manual on the AMI.** Nothing records
until you do — so there's no stray power-on bag and no clash with `dv race` over
the lidar. Confirm a real mission instead → the manual recorder releases and
`dv race` records the full mission bag (`dv race` also stops `dv-manual` first,
belt-and-braces, so the two never fight over the Hesai driver).

> Per uDV#189 the AMI emits an index only on **confirm** (not dial position), and
> `/ami/mission` publishes continuously with the ASMS off — so confirming Manual
> works during manual driving with no pipeline. `dv manual` by hand still works
> too.

**Opt out:** `touch /etc/dv/norecord` (honoured live) or
`sudo systemctl disable --now dv-automission.service`.

### The full car sensor set

A manual bag captures **all four** on-vehicle sensors, from two sources:

| Topic | Type | Source | Started by |
|---|---|---|---|
| `/lidar_points` | PointCloud2 | Hesai ATX driver | `dv-manual` (car_sensors) |
| `/imu` | Imu | uDV | `microros-agent.service` |
| `/motor_rpm` | Float32 | uDV | `microros-agent.service` |
| `/steering_angle` | Float32 | uDV | `microros-agent.service` |

The lidar cloud alone is not enough for offline odom/SLAM replay — you need the
uDV's IMU + wheel-speed + steering too. Those come over `microros-agent.service`,
which `dv-manual` pulls in (`Wants=`), so it starts even during pure manual
driving with no pipeline.

`dv_manual.sh` **verifies the full set is live before recording** and logs
present/missing. If the uDV topics are absent (agent down / uDV link dead) it
records anyway (a cloud bag beats no bag) but **warns loudly** in the journal —
so a proprioceptive-less bag is never a silent surprise. It only refuses to
start if *nothing* is present.

> After `dv manual`, sanity-check: `dv status` (dv-manual active), then
> `ros2 bag info` on the `manual_<ts>` bag — confirm **non-zero message counts**
> for `/imu`, `/motor_rpm`, `/steering_angle` **and** `/lidar_points`, not just
> that the topics exist. The best-effort QoS override is what makes those counts
> non-zero; a wrong reader records the topic with zero messages.

### Other notes
- **Don't run `dv manual` while the pipeline is up** — both launch the Hesai
  driver. `dv manual` refuses to start if `dv-pipeline` is active; `dv stop`
  first.
- Same disk floor (`DV_RECORD_MIN_FREE_GB`, default 20) and `/etc/dv/norecord`
  opt-out as the race recorder.
- No GPS/GSS/brake sensor exists on the car — the four above are the complete
  proprioceptive + perception set the EKF and SLAM consume.

## Architecture (why it's split this way)

`car_bringup.launch.py` = `car_sensors.launch.py` + `car_pipeline.launch.py`.
The split lets `dv-manual` reuse the exact sensor layer without the autonomy
stack, and lets the manual and race recorders share one record core
(`dv_record.sh`) — manual just overrides the warmup topic/method and the bag
label via `DV_RECORD_*` env. The race path defaults are unchanged, so the split
is behaviour-neutral for `dv race`.

Files: `deploy/dv` (verbs), `deploy/dv-manual.service` + `deploy/dv_manual.sh`
(manual path), `deploy/dv-record.service` + `deploy/dv_record.sh` (shared core),
`bringup/launch/car_sensors.launch.py` (sensor layer).

## Missing lidar clouds

If a bag is missing `/lidar_points` messages while `/lidar_packets_loss` shows no UDP loss,
the clouds were lost in ROS 2 delivery, not in the driver. See
[DDS_TUNING.md](DDS_TUNING.md) (issue #21) for the cause, the fix and how to measure it.
