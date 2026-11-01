#!/usr/bin/env bash
# =============================================================================
# install_cyclonedds_tuning.sh — Cyclone DDS large-message tuning for the DVPC.
# Issue #21; full explanation, measurements and rollback in docs/DDS_TUNING.md.
#
#   sudo deploy/install_cyclonedds_tuning.sh install            # apply (backs up first)
#        deploy/install_cyclonedds_tuning.sh status             # read-only report
#   sudo deploy/install_cyclonedds_tuning.sh rollback [BACKUP]  # undo the latest install not yet
#                                                               # undone; run again to go further back
#
# What install does, in this order (each step only if the previous one worked):
#   1. backs up everything it will touch into ~isc/.isc_backups/<ts>_cyclonedds/,
#      including the live net.core.rmem_max value;
#   2. installs deploy/60-isc-cyclonedds.conf → /etc/sysctl.d/ and applies it, then
#      checks the kernel limit really is ≥ 10 MB (aborts otherwise);
#   3. installs deploy/cyclonedds.xml → /etc/dv/cyclonedds.xml and checks a ROS 2 node
#      can start with it (aborts otherwise);
#   4. installs deploy/cyclonedds.env → /etc/dv/cyclonedds.env. THIS is the switch: the
#      dv-* units and scripts load it if present. Without it they run as before.
#
# Nothing restarts. Running services pick the tuning up on their next start
# (`dv restart`, next boot). The units/scripts that read the switch file are installed
# by install_dv_pipeline_service.sh; status says whether the installed copies have it.
# =============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BK_ROOT=/home/isc/.isc_backups
LOG="$BK_ROOT/cyclonedds_tuning.log"
SYSCTL_FILE=/etc/sysctl.d/60-isc-cyclonedds.conf
XML_FILE=/etc/dv/cyclonedds.xml
ENV_FILE=/etc/dv/cyclonedds.env
NEED_BYTES=10485760
UNITS=(dv-pipeline dv-manual dv-automission dv-record)
SCRIPTS=(dv_manual.sh dv_automission.sh dv_record.sh)

need_root() { [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }; }
rmem() { cat /proc/sys/net/core/rmem_max; }

status() {
  echo "net.core.rmem_max (live)    : $(rmem) bytes (default 212992; tuned 33554432)"
  for f in "$SYSCTL_FILE" "$XML_FILE" "$ENV_FILE"; do
    [ -f "$f" ] && echo "present  $f" || echo "absent   $f"
  done
  [ -f "$ENV_FILE" ] && echo "tuning switch               : ON (services started from now on use it)" \
                     || echo "tuning switch               : OFF (Cyclone defaults)"
  echo "other rmem_max settings     :"
  grep -Hs 'rmem_max' /etc/sysctl.conf /etc/sysctl.d/*.conf /run/sysctl.d/*.conf \
    /usr/lib/sysctl.d/*.conf 2>/dev/null | grep -v "^$SYSCTL_FILE:" | sed 's/^/  /' || echo "  none"
  echo "installed units/scripts that load the switch:"
  for u in "${UNITS[@]}"; do
    f=/etc/systemd/system/$u.service
    if [ -f "$f" ]; then grep -q cyclonedds.env "$f" && echo "  yes  $f" || echo "  NO   $f (re-run install_dv_pipeline_service.sh)"; fi
  done
  for s in "${SCRIPTS[@]}"; do
    f=/usr/local/bin/$s
    if [ -f "$f" ]; then grep -q cyclonedds.env "$f" && echo "  yes  $f" || echo "  NO   $f (re-run install_dv_pipeline_service.sh)"; fi
  done
  for u in "${UNITS[@]}"; do
    pid=$(systemctl show -p MainPID --value "$u.service" 2>/dev/null || echo 0)
    if [ -n "$pid" ] && [ "$pid" != 0 ] && [ -r "/proc/$pid/environ" ]; then
      tr '\0' '\n' < "/proc/$pid/environ" | grep -q '^CYCLONEDDS_URI=' \
        && echo "running  $u: tuning ACTIVE" || echo "running  $u: tuning not active (restart to apply)"
    fi
  done
  echo "UDP sockets, most drops first (rb = receive buffer bytes, d = packets dropped since open):"
  if command -v ss >/dev/null; then
    ss -uamp 2>/dev/null | awk '
      /users:\(\("/ { match($0, /users:\(\("[^"]+/); p = substr($0, RSTART + 9, RLENGTH - 9); next }
      /skmem:/ && p != "" { match($0, /rb[0-9]+/); rb = substr($0, RSTART + 2, RLENGTH - 2)
                            match($0, /,d[0-9]+\)/); d = substr($0, RSTART + 2, RLENGTH - 3)
                            printf "  d=%-8s rb=%-10s %s\n", d, rb, p; p = "" }' \
      | sort -t= -k2 -rn | head -15
  else
    echo "  (ss not available)"
  fi
}

install_tuning() {
  need_root
  local ts bk
  if [ -f "$ENV_FILE" ] && cmp -s "$HERE/cyclonedds.env" "$ENV_FILE" && cmp -s "$HERE/cyclonedds.xml" "$XML_FILE" \
     && cmp -s "$HERE/60-isc-cyclonedds.conf" "$SYSCTL_FILE" && [ "$(rmem)" -ge "$NEED_BYTES" ]; then
    echo "Already installed and identical to this checkout; nothing to do. See 'status'."
    return 0
  fi
  ts=$(date +%Y%m%d_%H%M%S_%N)
  bk="$BK_ROOT/${ts}_cyclonedds"
  mkdir -p "$BK_ROOT"
  mkdir "$bk"   # fails rather than reuse an existing backup
  rmem > "$bk/rmem_max.before"
  : > "$bk/MANIFEST"
  for f in "$SYSCTL_FILE" "$XML_FILE" "$ENV_FILE"; do
    if [ -f "$f" ]; then cp -a "$f" "$bk/"; echo "existed $f" >> "$bk/MANIFEST"; else echo "absent $f" >> "$bk/MANIFEST"; fi
  done
  echo "backup: $bk (rmem_max was $(cat "$bk/rmem_max.before"))"

  # 1. Kernel limit. Applied and verified before anything asks Cyclone for 10 MB.
  install -m 644 -o root -g root "$HERE/60-isc-cyclonedds.conf" "$SYSCTL_FILE"
  if ! sysctl -q -p "$SYSCTL_FILE" || [ "$(rmem)" -lt "$NEED_BYTES" ]; then
    echo "ERROR: net.core.rmem_max is $(rmem) after applying $SYSCTL_FILE (need ≥ $NEED_BYTES)."
    echo "Another sysctl file may override it (see 'status'). Rolling back."
    rollback "$bk"; exit 1
  fi

  # 2. Cyclone config, checked by starting a real ROS 2 node with it.
  install -d -m 755 -o isc -g isc /etc/dv
  install -m 644 -o root -g root "$HERE/cyclonedds.xml" "$XML_FILE"
  if ! sudo -u isc env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="file://$XML_FILE" \
       bash -c 'source /opt/ros/humble/setup.bash && timeout 20 python3 -c "import rclpy; rclpy.init(); rclpy.create_node(\"cyclonedds_tuning_check\"); print(\"node ok\")"' ; then
    echo "ERROR: a ROS 2 node could not start with $XML_FILE. Rolling back."
    rollback "$bk"; exit 1
  fi

  # 3. The switch.
  install -m 644 -o root -g root "$HERE/cyclonedds.env" "$ENV_FILE"
  echo "$(date -Is) install  backup=$bk  rmem_max $(cat "$bk/rmem_max.before") -> $(rmem)  repo=$(git -c safe.directory="$(dirname "$HERE")" -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo ?)" >> "$LOG"
  echo
  echo "Installed. Takes effect when the dv-* services next start (dv restart / reboot)."
  echo "Check:    $HERE/install_cyclonedds_tuning.sh status"
  echo "Undo:     sudo $HERE/install_cyclonedds_tuning.sh rollback $bk"
}

rollback() {
  need_root
  local bk="${1:-}"
  if [ -z "$bk" ]; then
    # Latest backup not yet rolled back: repeated rollbacks walk back one install at a time.
    for d in $(ls -d "$BK_ROOT"/*_cyclonedds 2>/dev/null | sort -r); do
      [ -f "$d/ROLLED_BACK" ] || { bk=$d; break; }
    done
  fi
  [ -n "$bk" ] && [ -d "$bk" ] && [ -f "$bk/MANIFEST" ] || { echo "nothing to undo: no cyclonedds backup left that is not already rolled back ($BK_ROOT/*_cyclonedds)"; exit 1; }
  echo "rolling back from $bk"
  # Switch off first, so nothing started from here on uses a half-restored state.
  for f in "$ENV_FILE" "$XML_FILE" "$SYSCTL_FILE"; do
    if grep -qx "existed $f" "$bk/MANIFEST"; then
      cp -a "$bk/$(basename "$f")" "$f"; echo "restored $f"
    else
      rm -f "$f"; echo "removed  $f"
    fi
  done
  if [ -f "$bk/rmem_max.before" ]; then
    sysctl -q -w net.core.rmem_max="$(cat "$bk/rmem_max.before")"
    echo "net.core.rmem_max back to $(rmem)"
  fi
  date -Is > "$bk/ROLLED_BACK"
  echo "$(date -Is) rollback backup=$bk  rmem_max now $(rmem)" >> "$LOG"
  echo "Services already running keep the tuning until their next start (dv restart / reboot)."
}

case "${1:-}" in
  install)  install_tuning ;;
  status)   status ;;
  rollback) rollback "${2:-}" ;;
  *) awk 'NR > 2 && /^# =====/ { exit } NR > 2' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
