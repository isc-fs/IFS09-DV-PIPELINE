# Car adaptation — running the DV pipeline on the real vehicle

**Status:** stock-typed uDV ↔ mission_control interface landed · firmware
gaps flagged below
**Last verified against:** `IFS08-DV-uDV @ feat/14-bridge-parse-fixes`

This document is the source of truth for the interface between the
autonomy pipeline (the DVPC) and the car, and the **open gaps** that must
be closed before an on-track run.

## 0. The big picture — one pipeline, two "uDV"s

The autonomy stack (mode_manager, mission_control, odometry_filter,
cone_detection, slam, path_planning, control) runs **identical code in
sim and on the car**. The only thing that differs is *who plays the uDV*:

* **Real car:** the IFS-08 uDV (a micro-ROS endpoint) + the Hesai driver.
* **Sim:** `sim_supervisor` emulates the uDV; the IFSSIM bridge provides
  the sensors.

`mission_control` is the seam. It exchanges **only standard ROS 2
interface types** with the uDV, so the micro-ROS firmware needs **no
custom messages and no library recompile**, and the sim emulator can
speak the exact same surface. There are **no DVPC-side car adapter
nodes** — the old `car_sensor_bridge` and `car_supervisor` are gone;
their unit conversions, mission/AS logic and actuation scaling now live
in the uDV firmware (and, mirrored, in the sim emulator).

---

## 1. The stock-typed interface (uDV ↔ mission_control)

Single source of truth: `mission_control/interface_contract.py`. Every
type here is in the standard micro-ROS interface set.

### uDV → mission_control (uplink)
| Topic | Type | Notes |
|---|---|---|
| `/assi/state` | `std_msgs/UInt8` | AS state machine byte (FS-Rules T14.9). **Publish ≥2 Hz** — it is mission_control's liveness heartbeat. |
| `/ami/mission` | `std_msgs/Int32` | selected AMI mission **index** (0..9). |
| `/imu` | `sensor_msgs/Imu` | 400 Hz → consumed directly by autonomy `/imu` (no remap; canonical on both sides). |
| `/steering_angle` | `std_msgs/Float32` | **radians** — converted on-board from the deg sensor. |
| `/motor_rpm` | `std_msgs/Float32` | motor-shaft RPM from the inverter. |
| `/lidar_points` | `sensor_msgs/PointCloud2` | Hesai → autonomy `/fsds/lidar/Lidar1` (pure remap). |

### mission_control → uDV (downlink)
| Name | Type | Notes |
|---|---|---|
| `/dv/status` | `std_msgs/UInt8` | pipeline lifecycle byte (IDLE/PREPARING/READY/RUNNING/FINISHED/EMERGENCY/FAILED). The prepare/run **handshake** — the uDV gates "go" on `READY`. **Publish ≥2 Hz** (the uDV's liveness watchdog on the DVPC). |
| `/ctrl/cmd` | `geometry_msgs/Twist` | normalised command: `linear.x` = signed drive demand `throttle − regen` in [-1,1] (x>0 → throttle, x<0 → regen/brake of magnitude −x; the controller never asks for both at once), `angular.z`=steering [-1,1]. The uDV splits `linear.x` back into the two channels, scales to physical units + clamps + actuates **only while AS Driving**. |
| `/force_ebs` | `std_srvs/SetBool` (service, **served by the uDV**) | mission_control requests EBS here on emergency. |

The two `UInt8` byte topics are each other's heartbeats: a stale
`/assi/state` makes mission_control reconcile to torn-down; a stale
`/dv/status` holds the uDV in a safe state. This bidirectional liveness
replaces the old action goal's implicit connection.

### How a mission runs (the handshake)
1. AMI selects a mission → uDV publishes `/ami/mission`.
2. Operator arms → uDV asserts **AS Ready** → mission_control configures
   the autonomy → publishes `/dv/status = PREPARING → READY`.
3. **The uDV gates the RES go on `/dv/status == READY`** — it won't enter
   AS Driving until autonomy is genuinely prepared.
4. RES go → uDV → **AS Driving** → mission_control activates → publishes
   `/dv/status = RUNNING` and streams `/ctrl/cmd`.
5. `slam` finished → mission_control `/dv/status = FINISHED`; emergency →
   `/dv/status = EMERGENCY` + `/force_ebs` call.

mission_control is a **reconciler**: it reads `/assi/state` +
`/ami/mission` level-triggered and drives `mode_manager` so the autonomy
lifecycle converges to what the AS state demands. Decision logic is the
pure, unit-tested `mission_control/reconcile.py`.

### Free-run — always-on data collection (`free_run`, default **on**)

`car_pipeline.launch.py` declares `free_run:=true` by default. With it on,
mission_control raises an autonomy **floor** whenever the uDV is powered on
(heartbeat alive) — *regardless of AS state*, including AS OFF and manual
driving with the ASMS unpowered:

- The **whole** autonomy stack runs — perception, SLAM, planning **and
  `control_node`**. control computes its would-be commands onto
  `/ctrl/cmd_internal`, which the `bag_recorder_node` captures in a
  `freerun_<UTC>` bag alongside the pilot's actuals (`/steering_angle`,
  `/motor_rpm`) — a ready-made **pilot-vs-autonomy** comparison dataset. The
  pipeline does **not** relay those commands (see invariants).
- The floor prepares the **operator-selected** mission when one is dialed in,
  else **autocross** (`FREE_RUN_MISSION_ID`) — the most general profile
  (SLAM maps an unknown track). This is what makes the arm hand-off *warm*:
  the floor already runs the mission the driver will arm with.
- **Hand-off:** arming to AS Ready keeps the floor up (no teardown); the go
  edge to AS Driving **clean-cycles `control_node`** (`ActivateMode.reset_nodes`:
  deactivate→activate control only) so the real run starts with fresh
  controller state — no SLAM reset, no Numba re-JIT of perception. Changing
  the AMI selection mid-floor re-preps (accepted cost).
- **Safety invariants:** the floor never publishes `/ctrl/cmd` (the relay
  opens only at `ActiveLevel.RUNNING`, i.e. a live armed run), and
  `/dv/status` still tracks the *real* AS state (OFF/manual → `IDLE`,
  standalone missions → `IDLE`), so the byte the uDV gates its go on is
  byte-for-byte the pre-free-run handshake — free-run cannot perturb arm →
  Ready → Driving. AS Emergency/Finished still win over the floor.

Turn it off for a lighter-CPU competition build:
`ros2 launch bringup car_pipeline.launch.py free_run:=false`. Sim/full
pipelines default it off (mission_control's node default).

---

## 2. What replaced car_sensor_bridge / car_supervisor

| Old DVPC adapter responsibility | Now lives in |
|---|---|
| steering deg→rad | uDV firmware (publishes `/steering_angle` in rad) |
| inverter→`/motor_rpm` | uDV firmware (reads inverter CAN) |
| AS-state → mission/actuation | uDV firmware AS state machine + `/ctrl/cmd` subscriber |
| `[-1,1]`→deg steering scaling + clamp | uDV firmware |
| "actuate only while Driving" gate | uDV firmware (it owns AS state) |
| AMI index→registry mission_id | mission_control (`interface_contract`, Python, testable) |
| EBS request | mission_control calls the uDV's `/force_ebs` |

In sim, `sim_supervisor` performs the firmware half in Python (AS state
machine in `sim_supervisor/as_state_machine.py`, `/ctrl/cmd`→bridge
relay, `/force_ebs` server), driven by the backend/CLI over the sim
operator-panel topics (`/sim/mission`, `/sim/intent`, `/sim/estop`).

---

## 3. ⚠️ OPEN GAPS — must close before on-track running

These are **firmware** items (IFS08-DV-uDV), flagged here so they aren't
missed. Each is a bounded change.

### G1 — Inverter `/motor_rpm` source `[BLOCKER for odometry quality]`
The uDV exposes **no wheel-speed / motor-RPM topic** yet. The EKF needs
`/motor_rpm`; without it odometry is IMU+steering dead-reckoning and
drifts. The uDV is on the vehicle CAN bus, so it is the natural publisher.
Action: read the inverter CAN, confirm **eRPM vs mechanical RPM** /
units / sign / pole-pairs, publish `/motor_rpm` (motor-shaft RPM). Set
`ekf.rpm_to_ms` on `odometry_filter_node` for the real wheel radius /
gear ratio (the sim default `0.00821` is almost certainly wrong).

### G2 — Throttle actuation sink
`/ctrl/cmd` carries the signed drive demand (`linear.x` = throttle − regen)
+ steering (`angular.z`); the uDV currently has **no throttle ROS
subscriber** (only steering). Add the inverter torque/accel path (e.g. the
`0x507` accel frame) and map `linear.x < 0` to regen torque — the
controller relies on it for the corner-radius speed cap and the
finish-gate stop, so a throttle-only sink cannot slow the car. Hydraulic
proportional braking is out of scope; only emergency EBS is wired
(`/force_ebs`).

### G3 — Steering scaling + units `[SAFETY]`
`/ctrl/cmd.angular.z` is normalised [-1,1]; the uDV scales it to degrees,
clamps to a safety limit **under STEERING's 70° cutoff**, and applies
only while AS Driving. Measure the real full-lock command and the sign
(positive δ ⇒ positive yaw rate) and bake them into firmware. Likewise
confirm the `/steering_angle` deg→rad conversion measures road-wheel vs
column angle.

### G4 — Mission-finished path to the uDV ✅ CONFIRMED (uDV#177)
**Byte 4 drives DRIVING→FINISHED. Implemented firmware-side; nothing to do.**
`app_task.cpp` builds `as_in.dv_finished = dv_fresh && (dv_status ==
DV_STATUS_FINISHED)`; `as_transition.hpp` consumes it. AS Finished then fires
the EBS and opens the SDC, latching until ASMS-off.

> ⚠️ **The uDV does NOT check standstill.** Send `FINISHED` while the car is
> rolling and it enters AS Finished *immediately* — EBS fired and **SDC opened
> at speed**. The rules only allow AS Finished at standstill, so **the
> standstill gate is entirely ours**. It lives in `cone_slam.lap_counter`
> (`finished` = criterion met AND `speed <= standstill_mps`). Do not weaken it,
> and do not let anything else publish byte 4.

⚠️ **Byte 4 only applies to pipeline missions** (`mission_needs_pipeline`):
AMI 1–4 yes; AMI 5–6 no — standalone missions end on their own
`mission_complete` and ignore every byte we send.

**Correction:** the earlier "or its own RES/standstill logic" alternative was
wrong. `state_manager.cpp` (`StateManager::updateState()`) *looks* like it gates
FINISHED on standstill, but it is **dead code** — its `getState()` is read only
by firmware host tests. The live AS state is `as_next_state()`, which has no
standstill term at all. Do not rely on it.

### G5 — AMI → mission mapping ✅ MOSTLY CONFIRMED (uDV#178)
Indices 0–9 **confirmed against firmware source**; our table is correct
(1=accel, 2=skidpad, 3=autocross, 4=trackdrive).

**The authority is `Core/Inc/mission.h` + `Core/Src/mission_registry.cpp`, NOT
`ws2812.c`** — which this doc used to name and which has no mission mapping in
it at all (it is the ASSI LED UART bridge). Track those two files.

Two items remain open:

- ⚠️ **AMI 5 (EBS test) is NOT confirmed standalone** — treat it as *TBD,
  currently inert*. `mission_ebstest.cpp` is a stub: wheels straight,
  `requests_r2d=false` (it won't even enable the inverter), never
  self-finishes. It cannot move the car, so `needs_pipeline=false` is true only
  in the trivial sense that a mission doing nothing needs nothing. The real
  FS EBS test must reach a set speed autonomously — an open design decision
  (standalone vs pipeline-driven). If it goes pipeline, our mapping changes.
- ⚠️ **Autonomous Demo has no AMI index.** The `AmiMission` enum stops at
  `SHUTDOWN=7`; 8/9 are unassigned. Unmapped fails safe both sides (firmware
  refuses GO; we tear down), and `mission_control` now logs a warning rather
  than idling silently. **Blocked on the AMI owners: does the board emit a Demo
  selection, and on what index?**

**AMI 6 (Inspection) IS confirmed standalone** and genuinely implemented —
open-loop steering sweep, 15% torque via the real ECU R2D handshake,
self-finishes on a 30 s timer. No pipeline at any point.

**`needs_pipeline` is the column that predicts our behaviour**: codes 1–4 the
uDV listens to `/dv/status`; codes 5–6 **every byte we send is ignored**.

### G6 — IMU / LiDAR frames + TF
`cone_detection` reads `header.frame_id`; provide the static TFs
(`base_link → hesai_lidar`, `base_link → imu_link`) and re-tune the
RANSAC ground / DBSCAN params on real ATX data.

### G7 — micro-ROS entity budget
The new uDV-facing topics (`/ctrl/cmd` sub, `/dv/status` sub,
`/motor_rpm` + `/steering_angle` pubs) add entities. Confirm they fit the
firmware's `RMW_UXRCE_MAX_*` budget; if not, bump `colcon.meta` and
rebuild (a config rebuild, **not** a custom-type integration).

---

## 4. Launching

```bash
# Car (NEVER use_sim_time:=true; the uDV is the mission_control peer):
ros2 launch bringup car_pipeline.launch.py

# Sim (sim_supervisor is the uDV emulator; backend/CLI drive the panel):
ros2 launch bringup full_pipeline.launch.py use_sim_time:=true
ros2 run sim_supervisor supervisor_cli run 1     # e.g. trackdrive
```

Per the repo workflow, only validated code on `main` runs on the car.

---

## 5. Tests

Pure-logic suites (no ROS install required):

```bash
python -m pytest pipeline/bringup/test/test_topic_contract.py \
                 pipeline/mission_control/test/ \
                 ros2/src/sim_supervisor/test/test_as_state_machine.py -q
```

Covers: the remap table, the AS/DV byte contract + AMI map, the
mission_control reconciler decision table, and the sim emulator AS state
machine (which shares the canonical bytes with `interface_contract`, so
the test also guards that the two packages agree).
