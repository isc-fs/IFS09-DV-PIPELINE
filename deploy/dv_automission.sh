#!/usr/bin/env bash
# =============================================================================
# dv_automission.sh — record a manual sensor bag when Manual is confirmed on AMI.
#
# Always-on watcher (dv-automission.service, enabled at boot). Recording is
# GATED ON MISSION SELECTION: it subscribes to the uDV's /ami/mission and only
# starts the manual recorder when the operator CONFIRMS Manual (index 0).
#
#   mission 0 (Manual confirmed) + pipeline NOT up → start dv-manual.service
#   mission 1..9 (a real mission confirmed)        → stop  dv-manual.service
#   mission -1 (up, nothing confirmed) / not seen  → leave state untouched
#   /etc/dv/norecord present                       → stop + stay idle (opt-out)
#
# Why mission-gated: recording only when a mission is selected means NO spurious
# bag from mere power-on and NO fight over the Hesai driver with `dv race`. To
# record a manual bag you SELECT + CONFIRM Manual on the AMI — one deliberate
# action, exactly as the operator asked. (Per uDV#189 the AMI emits an index
# only on confirm, not dial position; Manual=0 needs a confirm like any mission.
# /ami/mission publishes continuously, ASMS-off, no pipeline — confirmed there —
# so this works with the ASMS off during manual driving.)
#
# It never sends anything to the car; it only reads /ami/mission and starts/stops
# a local systemd unit. No autonomy runs — the manual bag is sensors only.
#
# Reconcile loop (not edge-triggered): re-evaluates every DV_AUTOMISSION_POLL
# seconds so the pipeline-guard is rechecked too — confirm-Manual while the
# pipeline is still up must not start the recorder, but a later `dv stop` must.
#
# Versioned in IFS08-DV-PIPELINE/deploy/ (installed to /usr/local/bin by
# install_dv_pipeline_service.sh).
# =============================================================================
set -o pipefail
LOG(){ logger -t dv-automission "$*" 2>/dev/null; echo "[dv-automission] $*"; }

source /opt/ros/humble/setup.bash 2>/dev/null
source /home/isc/ros2_ws/install/local_setup.bash 2>/dev/null
source /home/isc/dv_ws/install/local_setup.bash 2>/dev/null
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Cyclone DDS large-message tuning (issue #21, docs/DDS_TUNING.md); no file = defaults.
if [ -f /etc/dv/cyclonedds.env ]; then set -a; . /etc/dv/cyclonedds.env; set +a; fi

POLL="${DV_AUTOMISSION_POLL:-2}"

manual_active()  { systemctl is-active --quiet dv-manual.service; }
pipeline_active(){ systemctl is-active --quiet dv-pipeline.service; }

# Latest /ami/mission value as a bare integer, or empty if not seen. /ami/mission
# is BEST_EFFORT (UPLINK_QOS) — a default reliable reader sees NOTHING — so force
# a best-effort reader. Parse `data: N` (robust across ros2 CLI versions).
read_mission(){
  timeout 3 ros2 topic echo --once --qos-reliability best_effort /ami/mission \
    2>/dev/null | awk '/^data:/{gsub(/[^0-9-]/,"",$2); print $2; exit}'
}

LOG "watching /ami/mission (poll ${POLL}s); recording gated on mission select — confirm Manual (0) to record a sensor bag."

while true; do
  if [ -f /etc/dv/norecord ]; then
    manual_active && { LOG "/etc/dv/norecord set → stopping manual recorder"; sudo -n systemctl stop dv-manual.service; }
    sleep "$POLL"; continue
  fi

  m=$(read_mission)
  case "$m" in
    0)
      # Manual CONFIRMED on the AMI → record the sensor bag, unless the pipeline
      # owns the sensors. This is the ONLY thing that starts recording — nothing
      # records until a mission is confirmed, so there is no spurious setup bag.
      if pipeline_active; then
        : # a real run is up; the pipeline's recorder is in charge
      elif ! manual_active; then
        LOG "AMI Manual (0) confirmed, no pipeline → starting manual recording"
        sudo -n systemctl start dv-manual.service || LOG "failed to start dv-manual.service"
      fi ;;
    ""|-*)
      # -1 = uDV+agent up but NO mission confirmed yet (fresh boot / pre-arm);
      # "" = agent silent. Neither is Manual (uDV#189) — leave state untouched:
      # don't start recording (nothing selected) and don't drop an in-progress
      # bag on a transient -1/blip.
      : ;;
    *)
      # A real/other mission (1..9) confirmed → release the manual recorder so
      # `dv race` gets a clean Hesai driver + its own recorder.
      if manual_active; then
        LOG "AMI mission $m confirmed → stopping manual recording"
        sudo -n systemctl stop dv-manual.service || LOG "failed to stop dv-manual.service"
      fi ;;
  esac

  sleep "$POLL"
done
