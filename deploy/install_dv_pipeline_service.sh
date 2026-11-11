#!/usr/bin/env bash
# =============================================================================
# install_dv_pipeline_service.sh — switch DVPC race mode to the real pipeline.
#
# Installs (with timestamped backups into ~isc/.isc_backups/<ts>/):
#   deploy/dv-pipeline.service → /etc/systemd/system/   (NOT enabled — started
#                                by dv-mode at race boot, or manually: dv race)
#   deploy/dv-record.service   → /etc/systemd/system/   (enabled: WantedBy=
#                                dv-pipeline.service → records a bag on every
#                                racing start, after pipeline warmup)
#   deploy/dv_record.sh        → /usr/local/bin/        (warmup wait + record)
#   deploy/dv                  → /usr/local/bin/dv      (targets dv-pipeline)
#   deploy/dv_mode_boot.sh     → /usr/local/bin/        (race → dv-pipeline, verified)
#   deploy/dv_detect_mode.sh   → /usr/local/bin/        (mode detect + override)
#   deploy/dv_warm_numba.{sh,py} → /usr/local/bin/      (bake + verify Numba JIT cache;
#                                hooked as ExecStartPost of the auto-update
#                                unit + `dv update` + run once at install)
#   deploy/dv-race.sudoers     → /etc/sudoers.d/dv-race (NOPASSWD dv-pipeline)
# creates the isc-owned override dir /etc/dv (so `dv mode …` needs no sudo),
# and re-points the login prompt (/etc/profile.d/zz-dv-mode-prompt.sh) from
# isc-startup to dv-pipeline. The legacy isc-startup.service (isc_ws stub
# stack) is left installed but nothing references it anymore.
#
# Run ON the DVPC, as root, from the updated checkout:
#   sudo /home/isc/dv_ws/src/IFS08-DV-PIPELINE/deploy/install_dv_pipeline_service.sh
# =============================================================================
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)

TS=$(date +%Y%m%d_%H%M%S)
BK="/home/isc/.isc_backups/$TS"
mkdir -p "$BK"

for f in /etc/systemd/system/dv-pipeline.service \
         /etc/systemd/system/dv-record.service \
         /etc/systemd/system/dv-manual.service \
         /etc/systemd/system/dv-automission.service \
         /usr/local/bin/dv \
         /usr/local/bin/dv_mode_boot.sh \
         /usr/local/bin/dv_detect_mode.sh \
         /usr/local/bin/dv_record.sh \
         /usr/local/bin/dv_manual.sh \
         /usr/local/bin/dv_automission.sh \
         /usr/local/bin/dv_warm_numba.sh \
         /usr/local/bin/dv_warm_numba.py \
         /etc/profile.d/zz-dv-mode-prompt.sh; do
  [ -f "$f" ] && cp -a "$f" "$BK/" && echo "backed up $f"
done

install -m 644 "$HERE/dv-pipeline.service"   /etc/systemd/system/dv-pipeline.service
install -m 644 "$HERE/dv-record.service"     /etc/systemd/system/dv-record.service
install -m 644 "$HERE/dv-manual.service"     /etc/systemd/system/dv-manual.service
install -m 644 "$HERE/dv-automission.service" /etc/systemd/system/dv-automission.service
install -m 755 "$HERE/dv"                     /usr/local/bin/dv
install -m 755 "$HERE/dv_mode_boot.sh"        /usr/local/bin/dv_mode_boot.sh
install -m 755 "$HERE/dv_detect_mode.sh"      /usr/local/bin/dv_detect_mode.sh
install -m 755 "$HERE/dv_record.sh"           /usr/local/bin/dv_record.sh
install -m 755 "$HERE/dv_manual.sh"           /usr/local/bin/dv_manual.sh
install -m 755 "$HERE/dv_automission.sh"      /usr/local/bin/dv_automission.sh
install -m 755 "$HERE/dv_warm_numba.sh"       /usr/local/bin/dv_warm_numba.sh
install -m 755 "$HERE/dv_warm_numba.py"       /usr/local/bin/dv_warm_numba.py

# Enable the auto-record watcher at boot so "dial at Manual → record" is the
# default. It is inert until the uDV publishes /ami/mission with the ASMS off
# (isc-fs/IFS08-DV-uDV#189), and honours /etc/dv/norecord as the opt-out.
systemctl daemon-reload
systemctl enable dv-automission.service
echo "enabled dv-automission.service (auto-record on AMI Manual; opt out: touch /etc/dv/norecord)"

# Boot-mode override dir: isc-owned so `dv mode {race|umbilical|auto}` can
# write /etc/dv/mode without sudo. dv_detect_mode.sh reads it (root) at boot.
install -d -m 755 -o isc -g isc /etc/dv

# Passwordless start/stop/restart of the racing unit for the [r] login prompt
# and `dv race`. This NOPASSWD rule MUST name dv-pipeline.service — the earlier
# isc-startup→dv-pipeline switch left it pointing at the old unit, which made
# `sudo -n systemctl start dv-pipeline.service` prompt for a password and both
# `[r]` and `dv race` fail with "Could not start automatically". Validate with
# visudo before moving into place so a bad rule can never lock out sudo.
[ -f /etc/sudoers.d/dv-race ] && cp -a /etc/sudoers.d/dv-race "$BK/"
visudo -cf "$HERE/dv-race.sudoers"
install -m 0440 -o root -g root "$HERE/dv-race.sudoers" /etc/sudoers.d/dv-race
visudo -c >/dev/null

# Login prompt: the [r] RACE choice + status line must start the real pipeline.
if [ -f /etc/profile.d/zz-dv-mode-prompt.sh ]; then
  sed -i 's/isc-startup/dv-pipeline/g' /etc/profile.d/zz-dv-mode-prompt.sh
  echo "re-pointed zz-dv-mode-prompt.sh at dv-pipeline"
fi

# Bake the Numba JIT cache right after every umbilical auto-update: the
# update's git pull + colcon build rewrites sources, which is exactly what
# invalidates numba's on-disk cache (keyed on source content/mtime). Warming
# on the bench keeps race-mode bring-up on the warm path (~1 s/node) instead
# of a ~1-2 min cold JIT inside mode_manager's configure budget.
# dv-pipeline-update.service itself is NOT versioned here (it predates this
# repo's deploy/), so hook in via a drop-in — and only if the unit exists.
# `ExecStartPost=-` tolerates a warmup failure without failing the update,
# and dv_warm_numba.sh re-execs as isc so the cache stays user-writable.
if systemctl cat dv-pipeline-update.service >/dev/null 2>&1; then
  mkdir -p /etc/systemd/system/dv-pipeline-update.service.d
  cat > /etc/systemd/system/dv-pipeline-update.service.d/50-warm-numba.conf <<'EOF'
# Installed by install_dv_pipeline_service.sh — see deploy/dv_warm_numba.sh.
[Service]
ExecStartPost=-/usr/local/bin/dv_warm_numba.sh
EOF
  echo "installed dv-pipeline-update drop-in (Numba cache bake after auto-update)"
else
  echo "dv-pipeline-update.service not found — skipping warm-numba drop-in" \
       "(dv update + install-time bake still cover it)"
fi

systemctl daemon-reload

# Recorder rides along with the pipeline: enable creates the
# dv-pipeline.service.wants/ symlink (WantedBy=dv-pipeline.service), so every
# racing start also starts dv-record. dv-record itself stays un-started here —
# updating ≠ running, same policy as dv-pipeline.
systemctl enable dv-record.service >/dev/null 2>&1 || systemctl enable dv-record.service

# One-time bake so even the very first race boot after install hits a warm
# cache. Safe to interrupt (best-effort, exits 0) — the pipeline would just
# JIT at its first bring-up instead.
echo "Baking Numba JIT cache (fast if warm, ~1-2 min on a cold cache)…"
/usr/local/bin/dv_warm_numba.sh || true

echo
echo "Done. Race mode / 'dv race' now starts dv-pipeline.service"
echo "(bringup car_bringup.launch.py: Hesai + TFs + real autonomy)."
echo "Rollback: restore from $BK"
