"""Stock-typed uDV ↔ mission_control interface — the single source of truth.

The DV pipeline and the "uDV" (the real micro-ROS endpoint on the car, or
sim_supervisor emulating it in the sim) exchange ONLY standard ROS 2
interface types, so the firmware needs no custom messages and no
micro-ROS library recompile. mission_control runs one identical reconciler
against this surface in both worlds; the sim/car difference is only *who*
plays the uDV.

  uDV → mission_control (uplink, the "request"):
    /assi/state   std_msgs/UInt8   AS state machine byte (FS-Rules T14.9)
    /ami/mission  std_msgs/Int32   selected AMI mission INDEX (0..9, raw)

  mission_control → uDV (downlink, the "ack" + command):
    /dv/status    std_msgs/UInt8   pipeline lifecycle byte — the stock
                                   stand-in for the old SetMission /
                                   RuntimeControl action Results
    /ctrl/cmd     geometry_msgs/Twist  normalised command:
                                   linear.x = throttle - regen [-1, 1]
                                     (x > 0 → throttle = x, regen = 0;
                                      x < 0 → throttle = 0, regen = -x),
                                   angular.z = steering [-1, 1]
    /force_ebs    std_srvs/SetBool (service)  emergency-brake request

Both byte topics MUST be published at a steady cadence (>= ~2 Hz, not
edge-only): each is the other side's liveness heartbeat (a stale
/assi/state deactivates the pipeline; a stale /dv/status holds the uDV in
a safe state).

This module is pure data + pure functions — NO rclpy / launch imports —
so it unit-tests in plain pytest and can be imported by both
mission_control (the reconciler) and sim_supervisor (the emulator). The
same byte values are mirrored in the uDV firmware (C); this module is the
canonical Python source.
"""
from __future__ import annotations


# AS state bytes on /assi/state (FS-Rules T14.9 / uDV MMEE). The uDV —
# never the DVPC — owns this state machine; mission_control only reacts.
AS_OFF       = 0
AS_EMERGENCY = 1
AS_READY     = 2
AS_DRIVING   = 3
AS_FINISHED  = 4

# /dv/status bytes — the pipeline's own lifecycle, reported back to the
# uDV as the prepare/run handshake. The uDV gates AS Ready on DV_READY
# (won't honour "go" until autonomy is genuinely prepared) and reacts to
# DV_FINISHED / DV_EMERGENCY / DV_FAILED.
DV_IDLE      = 0   # nothing prepared
DV_PREPARING = 1   # configure / JIT in flight (was SetMission feedback)
DV_READY     = 2   # prepared OK (was SetMission Result success=true)
DV_RUNNING   = 3   # activated, emitting /ctrl/cmd
DV_FINISHED  = 4   # mission complete (was RuntimeControl outcome=finished)
DV_EMERGENCY = 5   # pipeline raised EBS (was outcome=emergency)
DV_FAILED    = 6   # prepare/activate error (was Result success=false/error)
# DV_STOPPING — "mission over, bring the car to rest NOW, but this is not an
# emergency". Requested when the mission criterion is met while the car is
# still rolling; the pipeline holds it until standstill, then reports
# DV_FINISHED and the uDV runs the normal AS Finished actuation.
#
# *** IT IS AN **ASB** ACTUATION. IT DOES NOT "FIRE THE EBS". ***
# This distinction is not pedantry — it IS the compliance argument, and getting
# it backwards is what makes people think this byte is illegal (it isn't).
#
#   FS-Rules 2026 T15.1.1: the vehicle has an **ASB** (Autonomous System Brake)
#   "that features an EBS ... as part of it". The ASB is the brake system; the
#   EBS is its fail-safe subset. T14.4.1 names the concept we use directly:
#   "autonomous brake actuation".
#
#   T14.8.1: the EBS is "activated" ONLY IF **the T15.2.2 power supply path is
#   cut** — NOT merely when the brakes are applied. T15.2.2 lists that path
#   exhaustively: LVMS + ASMS + the RES-bypass relay's NO contact + a relay
#   supplied by the SDC. The uDV's actuator lines (D1/D2) are NOT in it.
#
# So with all four supply elements intact — and in particular **the SDC CLOSED**
# — asserting the actuators applies brake pressure while the EBS is NOT
# "activated". Figure 15 (T14.8.3) then takes its left branch, R2D is still
# true, and the car is legitimately in **AS Driving** while braking.
#
# The existence proof is AS Ready: T14.4.1 closes the SDC precisely BECAUSE
# "sufficient brake pressure is built up, i.e. brakes are closed". Brakes on +
# SDC closed + EBS not activated is the state every FS car sits in before every
# run. If actuating the brakes counted as EBS activation, no car could ever arm.
#
# Corollary, and why DV_FINISHED is different: AS Finished REQUIRES
# "EBS activated? yes" (Figure 15, right branch). So the uDV opening the SDC at
# standstill is not incidental — cutting the supply path is what makes AS
# Finished reachable at all. DV_FINISHED genuinely does activate the EBS.
# DV_STOPPING genuinely does not. Do not conflate them.
#
# *** FULL BRAKE PRESSURE. END-OF-MISSION ONLY. NEVER TO MODULATE SPEED. ***
# There is NO separate service brake on the IFS08 (uDV#176): the ASB's only
# actuators are the EBS's actuators, and they are BINARY — apply or release,
# nothing to ramp or PWM. So this byte is not "brake a bit", it is "stop, now,
# hard". It buys exactly one thing: reaching standstill at the end of a mission
# so AS Finished becomes reachable. Using it mid-run to trim speed would be a
# full-pressure stop at every application, and the uDV team asked explicitly
# that `hard_stop_on_finish` stay strictly end-of-mission. `LapCounter.
# target_met` enforces that — it only latches on the mission's criterion.
#
# The uDV team will still run this past scrutineering before a real run; if
# push-back comes, it stays gated off. Note T14.11.1 requires the ASF to
# document the entire AS "including ASB" — so describe this as an ASB actuation
# there too, not as an EBS trigger.
#
# *** SAFE TO SHIP AHEAD OF FIRMWARE — but keep the gate closed for now. ***
# The firmware compares /dv/status for EQUALITY against only the bytes it acts
# on (READY=2, FINISHED=4, EMERGENCY=5, FAILED=6), so byte 7 is inert on today's
# build — RUNNING=3 has always been "unknown" to it in exactly the same way. No
# lockstep flash needed. `hard_stop_on_finish` still defaults FALSE, now for a
# different reason than when it was written: not "unknown byte, unknown risk"
# (that is resolved) but "the pairing has not been bench-validated yet".
#
# ONE EXCEPTION, and it is a footgun: in AS READY an unrecognised byte makes
# `dv_ready` false and the uDV REFUSES GO — the car silently never launches.
# So this byte must never be emitted while arming. mission_control gates its
# emission on the real AS state being AS_DRIVING; see _current_dv_status.
DV_STOPPING  = 7   # hard stop requested — brake to standstill, SDC stays closed

# Free-run (always-on data-collection) default mission. When the free_run
# flag is set, mission_control brings the autonomy floor — everything but
# control_node — up even while the uDV is OFF / in manual driving, so
# perception, SLAM and planning run and a rosbag records without the car
# ever being armed (ASMS need not be powered). The floor prepares the
# operator-selected mission when one is dialed in on /ami/mission, else
# this default. autocross (registry id 2) is the most general profile:
# SLAM maps an unknown track as it goes, no prior map assumed. Kept in
# lockstep with mode_registry by hand (autocross.mission_id == 2).
FREE_RUN_MISSION_ID = 2

# Interface topic / service names — single source of truth so the
# reconciler, the emulator and any tooling never drift.
TOPIC_ASSI_STATE  = "/assi/state"
TOPIC_AMI_MISSION = "/ami/mission"
TOPIC_DV_STATUS   = "/dv/status"
TOPIC_CTRL_CMD    = "/ctrl/cmd"
SERVICE_FORCE_EBS = "/force_ebs"

# DVPC-INTERNAL: slam → mission_control. "The mission criterion is met but the
# car is still rolling" — the trigger for the DV_STOPPING hard stop. Distinct
# from /slam/finished, which additionally requires standstill; see
# cone_slam.lap_counter for why the two must not be conflated.
TOPIC_SLAM_STOP_REQUEST = "/slam/stop_request"

# DVPC-INTERNAL emergency channel — not part of the uDV wire contract above.
# pipeline_watchdog (the independent supervisor) latches this true when the
# autonomy stack goes stale while running; mission_control turns the rising
# edge into /force_ebs + DV_EMERGENCY, exactly as it does for /ctrl/emergency.
# It lives here — rather than in pipeline_watchdog — because pipeline_watchdog
# already depends on this module for DV_RUNNING, and defining it the other way
# round would make the two packages circular.
TOPIC_WATCHDOG_EMERGENCY = "/watchdog/emergency"

# Heartbeat liveness bound (seconds) — the canonical value shared by both
# directions. Each byte topic above is the other side's liveness heartbeat;
# a side reconciles/trips to its safe state when the other's heartbeat has
# been silent for longer than this. At the >= 10 Hz publish cadence that is
# 4 missed cycles (jitter-safe), and it sits strictly under
# HEARTBEAT_STALE_CAP_S — the FS-Rules T11.9.4 bound for detecting a lost
# safety-critical message and entering the safe state.
#
# This is the single source of truth. The uDV firmware MUST mirror it as
# DV_STATUS_STALE_MS in dv_interface.h (currently 400 ms) — the two repos
# have no build-time link, so keep them in lockstep by hand. The pipeline
# test suite pins HEARTBEAT_STALE_S <= HEARTBEAT_STALE_CAP_S; the firmware
# static_asserts DV_STATUS_STALE_MS < DV_STATUS_STALE_CAP_MS.
#
# Detection budget: each watcher only evaluates staleness on its own tick,
# so its worst-case detection latency is HEARTBEAT_STALE_S + one tick
# period — and THAT sum, not the bare window, must stay under the cap.
# Firmware side: AppTask loops ~1 ms, trivially fine. Pipeline side: the
# reconcile tick (_RECONCILE_HZ in mission_control_node) is sized so
# window + tick leaves real reaction margin, pinned by
# test_detection_budget_leaves_reaction_margin.
HEARTBEAT_STALE_CAP_S = 0.5    # FS-Rules T11.9.4 detect-and-safe cap
HEARTBEAT_STALE_S     = 0.4    # == firmware DV_STATUS_STALE_MS (400 ms)

# Sim operator panel — the AMI board + RES buttons stand-in. SIM-ONLY
# (Linux↔Linux): the backend/CLI drive sim_supervisor's emulated AS state
# machine through these; the real car has no equivalent (the AMI board /
# RES produce /ami/mission + the AS transitions directly). /sim/mission
# carries an AMI INDEX (not a registry id) so it forwards straight onto
# /ami/mission. The intent byte values are mirrored by
# as_state_machine.OperatorIntent.
TOPIC_SIM_MISSION = "/sim/mission"
TOPIC_SIM_INTENT  = "/sim/intent"
TOPIC_SIM_ESTOP   = "/sim/estop"
SIM_INTENT_OFF    = 0   # disarmed
SIM_INTENT_READY  = 1   # armed / prepare (RES go not pressed)
SIM_INTENT_GO     = 2   # RES go — run


# AMI mission INDEX → pipeline registry mission_id (mode_registry:
# trackdrive=1, autocross=2, accel=3, skidpad=4). 0 = no autonomy mission /
# tear down. mission_control maps the raw /ami/mission index through this so
# the uDV stays dumb (no registry numbering baked into firmware).
#
# CONFIRMED against firmware source 2026-07-15 (uDV#178). The authority is
# `Core/Inc/mission.h` (the AmiMission enum) + `Core/Src/mission_registry.cpp`
# (the k_by_code[] dispatch table) — NOT ws2812.c, which this comment used to
# name and which contains no mission mapping at all (it is the ASSI LED UART
# bridge). Diffing against it was diffing against an empty set.
#
# `needs_pipeline` is the column that actually predicts our behaviour: it is
# the firmware's answer to "does the uDV listen to the pipeline at all for this
# mission". For codes 1–4 it is true, and our /dv/status bytes drive the run
# (FINISHED ends it, EMERGENCY/FAILED and a stale heartbeat trip Emergency,
# STOPPING brakes once uDV#176 lands). For codes 5–6 it is false and **every
# byte we send is ignored** — those missions end on their own logic.
#
# Idx 0 is MANUAL, not a mission: the uDV deliberately refuses GO on it (manual
# R2D is the ECU start-button path). 7 = SHUTDOWN, 8/9 are unassigned aux menu
# entries. All are nullptr missions firmware-side, so GO is refused.
#
# Autonomous Demo has NO index assigned (the AmiMission enum stops at
# SHUTDOWN=7). Unmapped indices fail safe on both sides — firmware returns a
# nullptr mission → mission_valid=false → GO refused; we return 0 → torn down.
# BLOCKED on the AMI owners: does the board actually emit a Demo selection, and
# on what index? Nobody has claimed one. See uDV#178.
#
# ⚠️ Index 5 (EBS test) is **TBD, currently inert** — NOT a confirmed standalone
# mission. Today `mission_ebstest.cpp` is a stub: it holds the wheels straight,
# sets requests_r2d=false (it will not even enable the inverter), and never
# self-finishes. It cannot move the car at all, so needs_pipeline=false is true
# only in the trivial sense that a mission which does nothing needs nothing.
# The FS EBS test requires driving to a set speed autonomously and verifying
# deceleration — that is an on-car TODO, and whether it ends up standalone or
# pipeline-driven is an OPEN DESIGN DECISION. If it goes the pipeline route,
# this mapping changes. The uDV team will consult us before implementing it.
#
# Index 6 (Inspection) IS confirmed standalone, and genuinely so:
# mission_inspection.cpp is fully implemented, sweeps the steering open-loop,
# drives 15% torque through the real ECU R2D handshake, and self-finishes on
# its own 30 s timer. No /dv/status, no /ctrl/cmd, no pipeline at any point.
DEFAULT_AMI_TO_MISSION_ID: dict[int, int] = {
    0: 0,   # MISSION_MANUAL     → no mission (human drives; uDV refuses GO)
    1: 3,   # MISSION_ACCEL      → accel        (needs_pipeline=true)
    2: 4,   # MISSION_SKIDPAD    → skidpad      (needs_pipeline=true)
    3: 2,   # MISSION_AUTOCROSS  → autocross    (needs_pipeline=true)
    4: 1,   # MISSION_TRACKDRIVE → trackdrive   (needs_pipeline=true)
    5: 0,   # MISSION_EBS_TEST   → no mission — TBD, currently an inert stub;
            #                      may become pipeline-driven. See above.
    6: 0,   # MISSION_INSPECTION → no mission (confirmed standalone, real)
    7: 0,   # MISSION_SHUTDOWN   → no mission (not a drive mission)
    8: 0,   # aux1               → no mission (AMI menu only, unassigned)
    9: 0,   # aux2               → no mission (AMI menu only, unassigned)
}


def is_known_ami_index(ami_index: int) -> bool:
    """True if the AMI index is one the firmware actually defines.

    Distinguishes "the operator picked a non-pipeline mission" (0, 5-9 — all
    legitimate) from "we have no idea what this index is". Both map to mission
    0 and both fail safe, so this exists purely so the caller can SAY which one
    happened: an unmapped index means the AMI is sending something the table
    has never heard of — e.g. an Autonomous Demo selection, which has no
    assigned index (uDV#178). Firmware-side that shows up as a refused GO, and
    a car that won't launch with no stated reason is a bad afternoon.
    """
    return int(ami_index) in DEFAULT_AMI_TO_MISSION_ID


def ami_index_to_mission_id(
    ami_index: int,
    mapping: dict[int, int] | None = None,
) -> int:
    """Translate an AMI mission index to a pipeline registry mission_id.

    Returns 0 (no autonomy mission / tear down) for any index not in the
    mapping, including the non-autonomy AMI slots (Manual, Shutdown,
    Aux). Never raises — an out-of-range index is treated as "no
    mission" so a glitchy /ami/mission can't start an unintended run.
    """
    table = DEFAULT_AMI_TO_MISSION_ID if mapping is None else mapping
    return int(table.get(int(ami_index), 0))


# Inverse map (registry mission_id → AMI index), first-wins so each
# runnable mission gets one canonical AMI index. Used ONLY by the sim
# operator panel (backend/CLI): users pick a registry mission, but
# /ami/mission must carry the AMI index in both sim and car so
# mission_control maps it identically. On the real car the AMI board
# produces the index directly — this inverse is never used there.
DEFAULT_MISSION_ID_TO_AMI_INDEX: dict[int, int] = {}
for _ami, _mid in DEFAULT_AMI_TO_MISSION_ID.items():
    if _mid != 0 and _mid not in DEFAULT_MISSION_ID_TO_AMI_INDEX:
        DEFAULT_MISSION_ID_TO_AMI_INDEX[_mid] = _ami


def mission_id_to_ami_index(
    mission_id: int,
    mapping: dict[int, int] | None = None,
) -> int:
    """Translate a registry mission_id to a sim AMI index for /sim/mission.

    Returns 0 (Manual / no autonomy mission) for mission_id 0 or any id
    without a runnable AMI slot. Round-trips through
    ami_index_to_mission_id for the runnable missions.
    """
    table = DEFAULT_MISSION_ID_TO_AMI_INDEX if mapping is None else mapping
    return int(table.get(int(mission_id), 0))
