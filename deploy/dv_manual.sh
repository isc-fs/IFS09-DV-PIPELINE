#!/usr/bin/env bash
# =============================================================================
# dv_manual.sh — record a rosbag during MANUAL driving, WITHOUT the pipeline.
#
# Started by dv-manual.service (`dv manual`). Brings up ONLY the sensor layer
# (Hesai LiDAR driver + static TFs, via car_sensors.launch.py) — NOT
# mission_control, NOT the autonomy lifecycle nodes — then records the full
# sensor graph, exactly like the race recorder:
#
#   /lidar_points  raw cloud     ← Hesai driver (started here)
#   /imu /motor_rpm /steering_angle  ← the uDV, over microros-agent.service
#                                      (independent of this; present if it's up)
#
# Why this exists: DV testing time is scarce, so we want a bag from every run
# including manual laps (offline perception/SLAM/odom development replays the
# manual bag through the pipeline later). The race recorder is bolted to
# dv-pipeline.service and waits for /dv/status, so it can't record manual
# driving. This is the manual-mode twin.
#
# Lifecycle: the sensor launch runs in the background inside this unit's cgroup;
# the recorder runs in the foreground (exec). `dv stop` / `systemctl stop
# dv-manual.service` sends SIGINT to the whole cgroup (KillMode=control-group),
# so rosbag2 finalizes the bag AND the sensor nodes shut down cleanly.
#
# Bags land in the same place as race bags (/home/isc/bags) with a `manual_`
# prefix. Honours /etc/dv/norecord. Same disk floor + best-effort QoS handling
# as dv_record.sh (this reuses it).
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh).
# =============================================================================
set -o pipefail
LOG(){ logger -t dv-manual "$*" 2>/dev/null; echo "[dv-manual] $*"; }

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

WITH_LIDAR="${DV_MANUAL_WITH_LIDAR:-true}"

# Bring up the sensor layer in the background (same cgroup, so it dies with the
# unit). No autonomy: this is car_sensors, not car_bringup.
LOG "starting sensor layer (car_sensors.launch.py, with_lidar=${WITH_LIDAR})…"
ros2 launch bringup car_sensors.launch.py with_lidar:="${WITH_LIDAR}" &
SENSORS_PID=$!

# If the sensor launch dies, don't leave a recorder running against a dead
# graph — take the whole unit down.
trap 'kill "$SENSORS_PID" 2>/dev/null' EXIT

# The uDV's proprioceptive sensors — /imu, /motor_rpm, /steering_angle — come
# over microros-agent.service, NOT from car_sensors. The unit Wants= that agent
# (so starting `dv manual` starts it too), but a manual bag with only the cloud
# and no IMU/wheel/steer is nearly useless for the offline odom/SLAM replay this
# is FOR — so we verify the full car sensor set is live before recording and
# warn loudly if any is missing, rather than silently shipping a lidar-only bag.
UDV_SENSORS="/imu /motor_rpm /steering_angle"
EXPECTED="$UDV_SENSORS"
[ "$WITH_LIDAR" = "true" ] && EXPECTED="/lidar_points $UDV_SENSORS"

# microros-agent status, for the log (Wants= should have started it).
LOG "microros-agent: $(systemctl is-active microros-agent.service 2>/dev/null || echo unknown)"

# Pre-flight: wait up to SENSOR_WAIT s for the expected topics to appear, then
# report present/missing. Best-effort topics only need to be ADVERTISED, so we
# check `ros2 topic list` (a reliable echo reader never sees them).
SENSOR_WAIT="${DV_MANUAL_SENSOR_WAIT:-30}"
LOG "waiting up to ${SENSOR_WAIT}s for the car sensor set: ${EXPECTED}…"
present=""; missing=""
for _ in $(seq 1 "$SENSOR_WAIT"); do
  have=$(ros2 topic list 2>/dev/null)
  present=""; missing=""
  for t in $EXPECTED; do
    if printf '%s\n' "$have" | grep -qxF "$t"; then present="$present $t"; else missing="$missing $t"; fi
  done
  [ -z "$missing" ] && break
  sleep 1
done
LOG "sensors present:$present"
if [ -n "$missing" ]; then
  LOG "⚠ MISSING sensors:$missing — recording anyway, but this bag will not have them."
  case "$missing" in
    *"/imu"*|*"/motor_rpm"*|*"/steering_angle"*)
      LOG "⚠ uDV proprioceptive data missing — check microros-agent.service and the uDV link." ;;
  esac
fi
# Refuse a completely empty run: if NOTHING is up, there is nothing to record.
if [ -z "$present" ]; then
  LOG "no expected sensor topic appeared after ${SENSOR_WAIT}s — giving up (nothing to record)."
  exit 1
fi

# Hand off to the shared record core. Gate its own warmup on a topic we know is
# present (first of the present set), method=exists (best-effort), label manual.
# The recorder execs `ros2 bag record`, replacing this shell; the backgrounded
# sensor launch stays in the cgroup and is reaped by systemd's control-group
# kill on stop.
export DV_RECORD_WARMUP_TOPIC="$(printf '%s' "$present" | awk '{print $1}')"
export DV_RECORD_WARMUP_METHOD="exists"
export DV_RECORD_MODE_LABEL="manual"
# Force best-effort readers on every uDV + lidar topic, else a best-effort
# publisher is invisible to rosbag's default reliable reader and records ZERO.
export DV_RECORD_BEST_EFFORT="${DV_RECORD_BEST_EFFORT-/lidar_points $UDV_SENSORS}"

LOG "handing off to dv_record.sh (warmup ${DV_RECORD_WARMUP_TOPIC} exists, label manual)…"
exec /usr/local/bin/dv_record.sh
