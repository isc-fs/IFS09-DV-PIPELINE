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
