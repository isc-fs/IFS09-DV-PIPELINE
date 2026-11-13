#!/usr/bin/env bash
# =============================================================================
# rk3588_tuning.sh — CPU and interrupt tuning for the RK3588 car computer.
# Why and the measurements: docs/arm_optimization/validation.md (level 5, HIL).
#
#   sudo deploy/rk3588_tuning.sh install    # install files, enable at boot, apply now
#        deploy/rk3588_tuning.sh status     # read-only report
#   sudo deploy/rk3588_tuning.sh apply      # apply now (what the boot unit runs)
#   sudo deploy/rk3588_tuning.sh rollback   # restore the state recorded by install
#
# apply:
#   1. the A76 cores → performance governor (cpufreq policy4 = cores 4-5, policy6 =
#      cores 6-7). ondemand drops them to 408 MHz between scans and needs ~10 ms to
#      ramp back: detection p99 9.6 → 5.9 ms, scan → cones p99 171 → 120 ms, no extra
#      heat on the bench. The A55s (policy0, cores 0-3) keep the board's governor:
#      performance there plus one pinned management node per A55 lost more scans in
#      the HIL A/B (0.89 vs 0.62 gaps > 0.25 s per minute).
#   2. every interrupt of the wired network cards → core 7 (the odometry filter's
#      core, ~24 % busy). By default they all land on core 4, the detector's.
#   3. network card receive ring → its maximum (when ethtool can set it).
#
# install also:
#   - /etc/dv/rk3588.env: core pinning (DV_CPU_PIN) and one-thread numeric libraries;
#   - a drop-in for dv-pipeline.service and dv-manual.service: the launch starts on
#     the A55 cores 0-3, loads rk3588.env, and re-applies this tuning just before
#     the pipeline starts (an interface brought up after boot gets new interrupts);
#   - disables irqbalance if present, so it cannot move the interrupts back.
# Socket receive buffers stay with install_cyclonedds_tuning.sh (status reports them).
# =============================================================================
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

POLICIES=${POLICIES:-"policy4 policy6"}
GOVERNOR=${GOVERNOR:-performance}
IRQ_CPU=${IRQ_CPU:-7}
LAUNCH_CPUS=${LAUNCH_CPUS:-0-3}

STATE=/var/lib/dv/rk3588_tuning
BIN=/usr/local/bin/rk3588_tuning.sh
ENV_FILE=/etc/dv/rk3588.env
UNIT=/etc/systemd/system/dv-rk3588-tuning.service
DROPIN_UNITS="dv-pipeline.service dv-manual.service"
DROPIN_NAME=60-rk3588.conf
CPUFREQ=/sys/devices/system/cpu/cpufreq

need_root() { [ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }; }

# Wired network cards: interfaces backed by a device (skips lo, docker0, veth, …).
nics() { for d in /sys/class/net/*; do [ -e "$d/device" ] && basename "$d"; done; }

# IRQ numbers whose handler name is <iface> or <iface>-<queue>.
nic_irqs() {
  local ifs re
  ifs=$(nics | paste -sd'|' -)
  [ -n "$ifs" ] || return 0
  re="^($ifs)(-.*)?$"
  awk -v re="$re" '$1 ~ /^[0-9]+:$/ && $NF ~ re { sub(":", "", $1); print $1 }' /proc/interrupts
}

apply() {
  need_root
  local p irq n=0 bad=0
  for p in $POLICIES; do
    if [ -w "$CPUFREQ/$p/scaling_governor" ]; then
      echo "$GOVERNOR" > "$CPUFREQ/$p/scaling_governor"
    else
      echo "warning: no $CPUFREQ/$p" >&2; bad=1
    fi
  done
  for irq in $(nic_irqs); do
    if echo "$IRQ_CPU" > "/proc/irq/$irq/smp_affinity_list" 2>/dev/null; then
      n=$((n + 1))
    else
      bad=1; echo "warning: IRQ $irq refused core $IRQ_CPU" >&2
    fi
  done
  if command -v ethtool >/dev/null; then
    local i max cur
    for i in $(nics); do
      max=$(ethtool -g "$i" 2>/dev/null | awk '/^RX:/{print $2; exit}')
      cur=$(ethtool -g "$i" 2>/dev/null | awk '/^RX:/{v=$2} END{print v}')
      if [ -n "$max" ] && [ "$max" != "n/a" ] && [ "$cur" != "$max" ]; then
        ethtool -G "$i" rx "$max" 2>/dev/null || echo "warning: $i rx ring stays at $cur" >&2
      fi
    done
  fi
  echo "rk3588 tuning: $GOVERNOR ($POLICIES), $n NIC interrupts on core $IRQ_CPU"
  return $bad
}

status() {
  local p i irq
  echo "== governors"
  for p in "$CPUFREQ"/policy*; do
    echo "  $(basename "$p") cores $(cat "$p/related_cpus"): $(cat "$p/scaling_governor") $(( $(cat "$p/scaling_cur_freq") / 1000 )) MHz"
  done
  echo "== NIC interrupts (want core $IRQ_CPU)"
  for irq in $(nic_irqs); do
    printf '  %s %s -> %s\n' "$irq" "$(awk -v q="$irq:" '$1==q{print $NF}' /proc/interrupts)" "$(cat "/proc/irq/$irq/smp_affinity_list")"
  done | sort -k3 | uniq -c -f2 | sed 's/^ */  /'
  echo "== irqbalance: $(systemctl is-active irqbalance 2>/dev/null || true) / $(systemctl is-enabled irqbalance 2>/dev/null || echo absent)"
  if command -v ethtool >/dev/null; then
    for i in $(nics); do
      echo "== $i rx ring: $(ethtool -g "$i" 2>/dev/null | awk '/^RX:/{a[++n]=$2} END{print a[2] " of " a[1]}')"
    done
  fi
  echo "== net.core.rmem_max: $(cat /proc/sys/net/core/rmem_max) (install_cyclonedds_tuning.sh)"
  echo "== installed"
  for f in "$BIN" "$ENV_FILE" "$UNIT" $(for u in $DROPIN_UNITS; do echo "/etc/systemd/system/$u.d/$DROPIN_NAME"; done); do
    [ -e "$f" ] && echo "  yes $f" || echo "  no  $f"
  done
  echo "  boot unit: $(systemctl is-enabled dv-rk3588-tuning.service 2>/dev/null || echo absent)"
  [ -d "$STATE" ] && echo "  rollback state: $STATE" || true
}

install_() {
  need_root
  mkdir -p "$STATE"
  if [ ! -f "$STATE/governors" ]; then   # record the pre-install state once
    for p in "$CPUFREQ"/policy*; do echo "$(basename "$p") $(cat "$p/scaling_governor")"; done > "$STATE/governors"
    echo "$(systemctl is-enabled irqbalance 2>/dev/null || echo absent)" > "$STATE/irqbalance"
    echo "recorded the current state in $STATE"
  fi
  install -m 755 "$HERE/rk3588_tuning.sh" "$BIN"
  install -d -m 755 /etc/dv
  install -m 644 "$HERE/rk3588.env" "$ENV_FILE"
  install -m 644 "$HERE/dv-rk3588-tuning.service" "$UNIT"
  for u in $DROPIN_UNITS; do
    install -d -m 755 "/etc/systemd/system/$u.d"
    cat > "/etc/systemd/system/$u.d/$DROPIN_NAME" <<EOF
# Installed by deploy/rk3588_tuning.sh. Remove with: sudo rk3588_tuning.sh rollback
[Service]
# Everything starts on the A55 cores; DV_CPU_PIN gives each pipeline node its core.
CPUAffinity=$LAUNCH_CPUS
EnvironmentFile=$ENV_FILE
# '+' runs it as root although the unit runs as its user; '-' never blocks the start.
ExecStartPre=-+$BIN apply
EOF
  done
  if systemctl cat irqbalance.service >/dev/null 2>&1; then
    systemctl disable --now irqbalance.service >/dev/null 2>&1 || true
    systemctl mask irqbalance.service >/dev/null
    echo "irqbalance disabled and masked"
  fi
  systemctl daemon-reload
  systemctl enable dv-rk3588-tuning.service >/dev/null
  systemctl restart dv-rk3588-tuning.service
  apply >/dev/null || true
  echo "installed; applied now and at every boot. A running pipeline needs a restart for the pinning."
  status
}

rollback() {
  need_root
  [ -f "$STATE/governors" ] || { echo "no recorded state in $STATE, nothing to roll back" >&2; exit 1; }
  systemctl disable dv-rk3588-tuning.service >/dev/null 2>&1 || true
  rm -f "$UNIT" "$ENV_FILE"
  for u in $DROPIN_UNITS; do
    rm -f "/etc/systemd/system/$u.d/$DROPIN_NAME"
    rmdir "/etc/systemd/system/$u.d" 2>/dev/null || true
  done
  while read -r p g; do echo "$g" > "$CPUFREQ/$p/scaling_governor" || true; done < "$STATE/governors"
  # IRQ numbers change between boots, so the NIC interrupts go back to the kernel
  # default (any core) rather than to recorded numbers.
  for irq in $(nic_irqs); do echo "0-$(($(nproc --all) - 1))" > "/proc/irq/$irq/smp_affinity_list" 2>/dev/null || true; done
  case "$(cat "$STATE/irqbalance")" in
    enabled) systemctl unmask irqbalance.service >/dev/null; systemctl enable --now irqbalance.service >/dev/null ;;
    disabled) systemctl unmask irqbalance.service >/dev/null ;;
  esac
  systemctl daemon-reload
  rm -f "$BIN"
  rm -rf "$STATE"
  echo "rolled back: governors, NIC interrupts and irqbalance as before install; files removed"
}

case "${1:-}" in
  apply) apply ;;
  status) status ;;
  install) install_ ;;
  rollback) rollback ;;
  *) sed -n '2,30p' "$0"; exit 2 ;;
esac
