# DVPC changelog

Every change to the car computer (DVPC) itself: system settings, installed units, files
outside the colcon workspaces. Newest first. Each entry says what changed on the box, how
to check it, and how to undo it, so any state can be walked back.

Each entry has a **Status**:
- *in repo*: merged or proposed, not installed on the DVPC;
- *installed (date, by)*: on the box;
- *rolled back (date)*: undone.

When you install, roll back, or find the box differs from an entry, update that entry's
status. Backups made by the installers live in `~isc/.isc_backups/`.

---

## 2026-11-13 · RK3588 CPU and interrupt tuning

**Status:** installed (2026-11-13) on the Orange Pi 5 Plus test board (192.168.0.128), which
runs the HIL check. Not on a car yet. The dv-* drop-ins are in place but inert there: that
board has no dv-pipeline unit.

**Why:** measured in the HIL full check
([validation, level 5](../docs/arm_optimization/validation.md#5-hardware-in-the-loop-the-full-sim-check-on-the-orange-pi)).
`ondemand` drops the cores to 408 MHz between scans; on the A76s `performance` took detection p99
from 9.6 to 5.9 ms and scan → cones p99 from 171 to 120 ms, with no extra heat. The network
card's interrupts all landed on core 4, the detector's. The A55s stay on the board governor
with the management nodes unpinned: a HIL A/B with them on `performance` and one node per A55
lost more scans (0.89 vs 0.62 gaps over 0.25 s per minute, tripped after 3 laps vs 8).

**Changes on the box:**

| Path | Before | After |
|---|---|---|
| Governor, A76 cores 4-7 (cpufreq policy4, policy6) | `ondemand` (board's `/etc/default/cpufrequtils`) | `performance`; A55 cores 0-3 (policy0) unchanged |
| Wired network card interrupts | any core (0-7), in practice core 4 | core 7 |
| Network card receive ring | already at its maximum (1024) on the Orange Pi | maximum |
| irqbalance | absent on the Orange Pi image | disabled and masked, if present |
| `/usr/local/bin/rk3588_tuning.sh` | absent | the tuning (`apply`, `status`, `install`, `rollback`) |
| `/etc/systemd/system/dv-rk3588-tuning.service` | absent | runs `apply` at boot, after cpufrequtils and the network |
| `/etc/dv/rk3588.env` | absent | `DV_CPU_PIN` (one A76 core per per-scan process) and one-thread numeric libraries |
| `/etc/systemd/system/dv-{pipeline,manual}.service.d/60-rk3588.conf` | absent | launch on cores 0-3, load `rk3588.env`, re-run `apply` before start |

Core pinning moves from a script that pinned running processes to the launch: with
`DV_CPU_PIN` set, `bringup` starts each listed node under `taskset -c <core>`. Unset (the
DVPC, the sim CI, laptops) nothing changes. Layout:

| Core | Runs |
|---|---|
| 0-3 (A55) | unpinned: mode manager, mission control, watchdog, sim supervisor, LiDAR driver, recorder, the launch, the OS |
| 4 (A76) | cone detection |
| 5 (A76) | SLAM |
| 6 (A76) | path planning |
| 7 (A76) | control, odometry filter, network interrupts |

Unlisted processes float on cores 0-3.

**Install:** `sudo deploy/rk3588_tuning.sh install` (records the current state, applies now,
enables at boot). Restart the pipeline to pick up the pinning.

**Check:** `rk3588_tuning.sh status`: policy4 and policy6 on `performance`, every network
interrupt on 7. Verified across two reboots on the test board.

**Undo:** `sudo rk3588_tuning.sh rollback`: governors and irqbalance as recorded, network
interrupts back to any core, files and drop-ins removed.

---

## 2026-11-01 · Cyclone DDS large-message tuning (issue #21)

**Status:** in repo, branch `fix/21-cyclone-large-messages`. **Not installed**: the DVPC
was not reachable. Nothing has been measured on the car.

**Why:** 11.9 % of lidar clouds missing from the 2026-09-20 bag, lost in ROS 2 delivery, not
in the driver. Details and measurements: [`docs/DDS_TUNING.md`](../docs/DDS_TUNING.md).

**Changes on the box, once installed:**

| Path | Before | After |
|---|---|---|
| `net.core.rmem_max` (live) | 212992 (kernel default; confirm on the box) | 33554432 |
| `/etc/sysctl.d/60-isc-cyclonedds.conf` | absent | sets `net.core.rmem_max = 33554432` at boot |
| `/etc/dv/cyclonedds.xml` | absent | Cyclone `SocketReceiveBufferSize max="10MB"` |
| `/etc/dv/cyclonedds.env` | absent | `CYCLONEDDS_URI=file:///etc/dv/cyclonedds.xml` (the switch) |
| `/etc/systemd/system/dv-{pipeline,manual,automission,record}.service` | no `EnvironmentFile` | `EnvironmentFile=-/etc/dv/cyclonedds.env` (inert without the switch) |
| `/usr/local/bin/dv_{manual,automission,record}.sh` | — | source the switch file if present (inert without it) |

**Install:** `sudo deploy/install_dv_pipeline_service.sh` (units and scripts), then
`sudo deploy/install_cyclonedds_tuning.sh install`, then `dv restart`. Full procedure with
before/after measurement in `docs/DDS_TUNING.md`.

**Check:** `deploy/install_cyclonedds_tuning.sh status`.

**Undo:** `sudo deploy/install_cyclonedds_tuning.sh rollback`, then `dv restart`. By hand:
delete `/etc/dv/cyclonedds.env` (enough on its own), optionally also the XML and the sysctl
file, and `sudo sysctl -w net.core.rmem_max=<previous>` (recorded in the backup's
`rmem_max.before`). The unit/script changes can stay; to revert them too, restore the
copies that `install_dv_pipeline_service.sh` saved in its backup folder.

**Result on the car:** *to fill in*: `dds_large_msg_check.py` before/after, missing-cloud
rate of the first bag after install.

---

## 2026-07-10 · Hesai driver stamps with lidar time (`use_timestamp_type: 0`)

**Status:** installed 2026-07-10. *Backfilled from
[`docs/lidar_imu_time_sync_mitigation.md`](../docs/lidar_imu_time_sync_mitigation.md).*

**Changed:** `~/ros2_ws/src/HesaiLidar_ROS_2.0/config/config.yaml`, `use_timestamp_type` from
`1` (DVPC receive time, the pre-PTP workaround) to `0` (lidar measurement time).

**Undo:** restore `~/.isc_backups/config.yaml.pre-tstype0` and restart the pipeline.
Fallback `1` remains safe if PTP is lost.

---

## 2026-07-10 · PTP master switched to gPTP

**Status:** installed 2026-07-10. *Backfilled from
[`docs/lidar_imu_time_sync_mitigation.md`](../docs/lidar_imu_time_sync_mitigation.md).*

**Changed:** `/etc/linuxptp/ptp4l.conf` to the 802.1AS / gPTP profile the ATX expects. Versioned
in `deploy/ptp4l.conf`, `deploy/ptp4l.service`, `deploy/install_ptp4l.sh`.

**Undo:** restore `~/.isc_backups/ptp4l.conf.pre-gptp` to `/etc/linuxptp/ptp4l.conf` and
`sudo systemctl restart ptp4l`.

---

Earlier changes to the box predate this changelog. The installers' backup folders in
`~isc/.isc_backups/` are the only record of them.
