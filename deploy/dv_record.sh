#!/usr/bin/env bash
# =============================================================================
# dv_record.sh — race rosbag recorder (run by dv-record.service).
#
# Started automatically whenever dv-pipeline.service starts (race boot,
# `dv race`, `dv restart`) via the unit's WantedBy=dv-pipeline.service;
# PartOf= stops it with the pipeline. Waits for the pipeline warmup —
# /dv/status is published by mission_control once the stack is up, so its
# appearance means the topic graph exists — then records EVERY topic,
# INCLUDING the raw lidar cloud (/lidar_points), so a run can be replayed and
# you can see exactly what the lidar saw.
#
# ⚠ THROUGHPUT: /lidar_points is ~4.5 MB × 10 Hz ≈ 45 MB/s ≈ 2.7 GB/min. A 5-min
# run is ~13 GB. That is deliberate — the cloud is the one signal that lets us
# debug perception offline (point_step/layout, why a cone was or wasn't
# detected, residual histograms). Set DV_RECORD_EXCLUDE='^/lidar_points$' to go
# back to the small telemetry-only bag if disk gets tight.
#
# ⚠ QoS (why the override below is NOT optional): the Hesai driver publishes
# /lidar_points BEST_EFFORT (SensorDataQoS). A RELIABLE subscriber is
# QoS-INCOMPATIBLE with a best-effort publisher and receives NOTHING — so
# recording the topic without forcing a best-effort reader silently produces a
# /lidar_points topic with ZERO messages. A best-effort reader matches both
# best-effort and reliable publishers, so the override is always safe.
#
# Knobs (systemd drop-in or environment):
#   DV_RECORD_DIR             output dir            (default /home/isc/bags)
#   DV_RECORD_EXCLUDE         record -x regex       (default EMPTY = record all,
#                             incl. /lidar_points). e.g. '^/lidar_points$'
#   DV_RECORD_BEST_EFFORT     space-separated topics to force a best-effort
#                             reader on          (default "/lidar_points")
#   DV_RECORD_WARMUP_TIMEOUT  seconds to wait for the warmup topic (default 90)
#   DV_RECORD_MIN_FREE_GB     refuse to record below this free space (default 20)
#   DV_RECORD_WARMUP_TOPIC    topic whose presence means "graph is up"
#                             (default /dv/status — the pipeline heartbeat)
#   DV_RECORD_WARMUP_METHOD   how to test it: "echo" (default; needs a RELIABLE
#                             topic like /dv/status) or "exists" (topic merely
#                             advertised in `ros2 topic list` — use for
#                             best-effort sensor topics like /lidar_points,
#                             which a reliable `echo` reader never sees)
#   DV_RECORD_MODE_LABEL      bag-name prefix. If unset, the label is the
#                             CONFIRMED AMI mission name (accel / skidpad /
#                             autocross / trackdrive) so pipeline bags are named
#                             by the mission they ran; falls back to /run/dv_mode
#                             (race / umbilical) when unresolved. dv_manual.sh
#                             sets "manual".
# Opt-out: `touch /etc/dv/norecord` (isc-owned dir) disables recording.
#
# Manual driving (no pipeline): deploy/dv_manual.sh launches ONLY the sensor
# layer and execs this script with WARMUP_TOPIC=/lidar_points, METHOD=exists,
# MODE_LABEL=manual — so the same record core captures a manual-driving bag
# (full sensor graph incl. the cloud) without mission_control ever coming up.
# The defaults below preserve the race path exactly.
#
# The unit sends SIGINT on stop (KillSignal) so rosbag2 finalizes the bag. A
# hard kill — or copying the bag while it is still recording — leaves an
# unfinalized mcap (header magic present, footer/index missing). Always
# `dv stop` before scp'ing a bag off the car.
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh — `dv update` alone does NOT refresh it).
# =============================================================================
# No `set -u`: sourcing ROS setup.bash references unbound vars.
set -o pipefail
LOG(){ logger -t dv-record "$*" 2>/dev/null; echo "[dv-record] $*"; }

if [ -f /etc/dv/norecord ]; then
  LOG "disabled by /etc/dv/norecord — not recording."
  exit 0
fi

source /opt/ros/humble/setup.bash 2>/dev/null
source /home/isc/ros2_ws/install/local_setup.bash 2>/dev/null
source /home/isc/dv_ws/install/local_setup.bash 2>/dev/null
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Cyclone DDS large-message tuning (issue #21, docs/DDS_TUNING.md); no file = defaults.
if [ -f /etc/dv/cyclonedds.env ]; then set -a; . /etc/dv/cyclonedds.env; set +a; fi

OUTDIR="${DV_RECORD_DIR:-/home/isc/bags}"
EXCLUDE="${DV_RECORD_EXCLUDE-}"                       # default: exclude NOTHING
BEST_EFFORT="${DV_RECORD_BEST_EFFORT-/lidar_points}"
WARMUP_TIMEOUT="${DV_RECORD_WARMUP_TIMEOUT:-90}"
MIN_FREE_GB="${DV_RECORD_MIN_FREE_GB:-20}"
WARMUP_TOPIC="${DV_RECORD_WARMUP_TOPIC:-/dv/status}"  # race: pipeline heartbeat
WARMUP_METHOD="${DV_RECORD_WARMUP_METHOD:-echo}"      # echo (reliable) | exists

mkdir -p "$OUTDIR"

free_gb=$(( $(df -Pk "$OUTDIR" | awk 'NR==2{print $4}') / 1024 / 1024 ))
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  LOG "only ${free_gb} GB free in $OUTDIR (< ${MIN_FREE_GB}) — refusing to record."
  exit 1
fi

# Warmup: wait for the graph to be up. Default gate is mission_control's
# /dv/status heartbeat (RELIABLE, so `topic echo` works). Manual mode gates on a
# best-effort sensor topic instead, via method=exists (mere advertisement in
# `ros2 topic list`) — a reliable `echo` reader never sees a best-effort topic.
LOG "waiting up to ${WARMUP_TIMEOUT}s for warmup (${WARMUP_TOPIC}, method=${WARMUP_METHOD})…"
up=0
for _ in $(seq 1 "$WARMUP_TIMEOUT"); do
  if [ "$WARMUP_METHOD" = "exists" ]; then
    if ros2 topic list 2>/dev/null | grep -qxF "$WARMUP_TOPIC"; then up=1; break; fi
  else
    if timeout 3 ros2 topic echo --once "$WARMUP_TOPIC" >/dev/null 2>&1; then up=1; break; fi
  fi
  sleep 1
done
if [ "$up" != 1 ]; then
  LOG "graph never warmed up (${WARMUP_TOPIC} absent after ${WARMUP_TIMEOUT}s) — giving up."
  exit 1
fi
sleep 2   # let the rest of the topic graph register with discovery

# Per-topic QoS overrides: force a best-effort reader on the sensor topics that
# are published best-effort (see the QoS note above). depth 10 on a 4.5 MB
# cloud is already a ~45 MB buffer — do not raise it casually.
QOS_ARGS=()
QOS_FILE=""
if [ -n "$BEST_EFFORT" ]; then
  QOS_FILE=$(mktemp /tmp/dv_record_qos.XXXXXX.yaml) || QOS_FILE=""
  if [ -n "$QOS_FILE" ]; then
    for t in $BEST_EFFORT; do
      printf '%s:\n  reliability: best_effort\n  history: keep_last\n  depth: 10\n  durability: volatile\n' "$t"
    done > "$QOS_FILE"
    QOS_ARGS=(--qos-profile-overrides-path "$QOS_FILE")
    trap 'rm -f "$QOS_FILE"' EXIT
    LOG "best-effort reader forced on: $BEST_EFFORT"
  fi
fi

# Bag-name label. Priority:
#   1. DV_RECORD_MODE_LABEL if set  — the manual path forces "manual".
#   2. the CONFIRMED AMI mission     — so a pipeline run is named by the mission
#      it ran (accel / skidpad / autocross / trackdrive), which is exactly what
#      differentiates one pipeline bag from another.
#   3. /run/dv_mode                  — fallback (race / umbilical) when the
#      mission can't be resolved (agent down, or nothing confirmed = -1).
mission_label_from_ami(){
  # /ami/mission is BEST_EFFORT and carries the CONFIRMED mission index
  # (uDV#189). Map index → name; mirrors interface_contract
  # DEFAULT_AMI_TO_MISSION_ID. Empty on -1 / unknown / unresolved.
  local idx
  idx=$(timeout 3 ros2 topic echo --once --qos-reliability best_effort /ami/mission \
        2>/dev/null | awk '/^data:/{gsub(/[^0-9-]/,"",$2); print $2; exit}')
  case "$idx" in
    1) echo accel ;;
    2) echo skidpad ;;
    3) echo autocross ;;
    4) echo trackdrive ;;
    0) echo manual ;;
    *) echo "" ;;
  esac
}

if [ -n "${DV_RECORD_MODE_LABEL:-}" ]; then
  MODE="$DV_RECORD_MODE_LABEL"
else
  MODE=$(mission_label_from_ami)
  if [ -n "$MODE" ]; then
    LOG "bag named by confirmed AMI mission: ${MODE}"
  else
    MODE=$(cat /run/dv_mode 2>/dev/null || echo run)
    LOG "no confirmed mission on /ami/mission — bag labelled '${MODE}' (fallback)"
  fi
fi
OUT="$OUTDIR/${MODE}_$(date +%Y%m%d_%H%M%S)"
if [ -n "$EXCLUDE" ]; then
  LOG "recording -> $OUT (all topics except '$EXCLUDE')  [${free_gb} GB free]"
  exec ros2 bag record -s mcap -o "$OUT" "${QOS_ARGS[@]}" -a -x "$EXCLUDE"
else
  LOG "recording -> $OUT (ALL topics, incl. /lidar_points ~45 MB/s)  [${free_gb} GB free]"
  exec ros2 bag record -s mcap -o "$OUT" "${QOS_ARGS[@]}" -a
fi
