# Pipeline watchdog

`pipeline_watchdog_node` — an independent supervisor that trips AS Emergency
when the autonomy stack goes stale **while the car is driving**.

## Why it exists

We already have a watchdog. The uDV watches `/dv/status`: if the byte goes
stale past `DV_STATUS_STALE_MS` (400 ms) the uDV trips to its safe state. That
covers the pipeline being **dead** — process crash, DDS silence, DVPC power
loss.

It does not cover the pipeline being **alive but sick**. `mission_control` will
happily publish `DV_RUNNING` at 20 Hz while SLAM has stopped solving
underneath it. Nothing else in the stack notices, because `control_node` caches
its last pose and odom forever and has no staleness check of its own.

That is not hypothetical. It is the documented runaway, recorded in
`control/control/controllers/pi_velocity.py` as the reason `throttle_max`
was once capped at 0.2:

> pose froze, controller kept commanding full throttle, real car ran away to
> ~24 m/s and crashed

A throttle cap only bounds how *fast* that divergence blows up. It does not
stop it. This node does, so the cap is now off (`throttle_max` 1.0).

| Watchdog | Question it answers | Mechanism |
|---|---|---|
| uDV (firmware) | Is the DVPC **alive**? | `/dv/status` heartbeat staleness |
| pipeline (this) | Is the DVPC's **data sane**? | data liveness + pose progress |

The two are complementary, and neither needs to cover the other's case. If
`mission_control` itself hangs, this node's emergency would not get relayed —
but that is exactly when the uDV's heartbeat watchdog fires.

## How it works

```
/dv/status == DV_RUNNING  ──arm──▶  pipeline_watchdog_node
                                          │  supervises
                                          ▼
                          /slam/pose · /odom · /ctrl/cmd_internal
                                          │  stale / not advancing
                                          ▼
                              /watchdog/emergency = true  (latched)
                                          │
                                          ▼
                       mission_control._raise_emergency()
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
            /dv/status = DV_EMERGENCY               /force_ebs (SetBool)
            (what the uDV LATCHES on)               (redundant, non-latching)
                        │
                        ▼
                 uDV → AS Emergency + EBS
```

The pipeline never touches the AS state machine — the uDV owns it. We only
ever ask.

### Arming

Armed **only** while `/dv/status == DV_RUNNING`, which is the pipeline's own
statement that it is activated and relaying `/ctrl/cmd`. Consequences, all
intended:

- The free-run / data-collection floor never trips the EBS — it never reaches
  `DV_RUNNING`, so a human can drive with a stale autonomy stack in the
  background.
- A torn-down or idle stack never trips.
- It reuses the one byte both sides already agree on rather than inventing a
  second notion of "running".

After arming, a **grace window** (default 5 s) absorbs node spin-up: Numba JIT
warm-up, first LiDAR scan, first solve. Nothing can trip inside it. After it, a
topic that has *never* published is treated as stale — "SLAM never started" is
as fatal as "SLAM stopped".

### The two checks

1. **Liveness** — a supervised topic went silent past its budget. This is what
   catches the documented incident: when SLAM strands, the scan path returns
   before publishing, so `/slam/pose` simply stops.
2. **Pose progress** — `/slam/pose` is still arriving but the position is not
   advancing while `/odom` says the car is moving. Catches the nastier variant
   where SLAM republishes a stale solve. Liveness alone would miss it.
   The speed reference deliberately comes from `/odom`, **not** SLAM: the whole
   point is to catch SLAM lying, so the audit cannot use SLAM as its source.
   The stall is timed from the later of the pose's last advance and `/odom`
   reaching the speed floor, so a car that stood still (waiting for its first
   path, say) and then drives off gets the full window before it can trip (#31).

Both latch. Once tripped, a run never un-trips; clearing requires a new run
(leaving `DV_RUNNING`). An intermittent sensor that recovers for one scan must
not silently rearm the car.

### What is supervised, and what is deliberately not

Only topics whose staleness leaves the car **still driving on stale data**:

| Topic | Budget | Why it is dangerous when stale |
|---|---|---|
| `/slam/pose` | 0.6 s | frozen pose → control drives blind → the documented runaway |
| `/odom` | 0.5 s | frozen speed → PI reads "too slow" → commands more throttle |
| `/ctrl/cmd_internal` | 0.5 s | control_node died → mission_control stops relaying |

**Not** supervised, because they already fail safe:

- `/Path` — `control_node` publishes a zeroed command when the reference is
  empty, so a planner dropout coasts rather than runs away.
- `/Conos_raw` — its loss surfaces as `/slam/pose` or `/Path` staleness one hop
  later. Supervising it directly would only add false-trip surface.

**Adding a topic here is a safety decision, not a monitoring nicety.** Every
extra supervised topic is another way to fire the EBS at speed on a false
positive. `test_supervised_set_is_only_the_drive_on_stale_data_topics` exists
to make that argument explicit.

Budgets are generous multiples of each topic's nominal period, for the same
reason. At `v_max = 3 m/s`, 0.6 s of blindness is ~1.8 m of travel.

## Design notes

**Not a lifecycle node, and not in `AUTONOMY_NODE_ORDER`.** It comes up with the
management trio (`launch_common.watchdog_action`) and runs for the whole
session, so `mode_manager` can never configure, deactivate or tear down the
thing that supervises it. Same reasoning as the uDV running its watchdog
outside the mission logic it watches.

**Launched on both the sim and car profiles.** A watchdog that only exists on
one of them is a watchdog whose false-trip behaviour gets discovered on the
vehicle.

**QoS is load-bearing.** `/dv/status` is published latched (RELIABLE +
TRANSIENT_LOCAL). DDS request-vs-offered matching means a BEST_EFFORT or
VOLATILE reader silently receives *nothing* — the watchdog would sit disarmed
forever, which is a silent safety failure rather than a loud one. The
subscription mirrors the publisher's profile deliberately; see
`mission_control/interface_qos.py` for the full rationale.

**Separate topic from `/ctrl/emergency`.** Control raising an emergency and the
watchdog catching control being stale are different faults, and a bag must say
which one fired. Both land on the same handler in `mission_control` — the
response is identical.

## Tuning

All budgets are ROS params on the node (`pose_max_silence_s`,
`odom_max_silence_s`, `cmd_max_silence_s`, `grace_period_s`,
`pose_progress_*`). Note the repo currently ships **no YAML param files** and
`launch_common` passes no overrides, so the code defaults are the flight
config — retuning means editing `pipeline_watchdog_node.py`.

To disable the pose-progress check without touching liveness:
`pose_progress_enabled:=false`.

## Testing

```bash
python -m pytest pipeline_watchdog/test/ -q
```

Three layers, all ROS-free:

- `test_health_monitor.py` — the pure decision core, on an injected clock.
- `test_watchdog_node_behavior.py` — the **node**, driven through a fake-ROS
  harness: arming, the frozen-pose scenario, the stopped-car false-positive
  case, latch-once semantics. Catches wiring bugs, not transport bugs.
- `test_watchdog_contract.py` — the cross-package agreements (topic name,
  `DV_RUNNING` byte, supervised set) that fail *silently* if they drift.

Both the core and the node have been mutation-checked: breaking the latch, the
arming byte, the pose/speed wiring, or the trip publish each fail tests.

**Still required: a bench run.** These tests cannot catch a DDS QoS mismatch —
the one failure mode that makes this node silently do nothing. Before trusting
it on the car, confirm on the bench that `pipeline_watchdog_node` logs
`watchdog ARMED` when the stack goes to `DV_RUNNING`. If it never logs that,
the `/dv/status` subscription is not matching.
