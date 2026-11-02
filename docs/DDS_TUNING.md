# Cyclone DDS tuning for large messages

Issue #21. Status: **in the repo, not yet installed on the DVPC** (the car was not reachable
when this was written, 2026-11-01). Every change to the DVPC itself is listed in
[`deploy/CHANGELOG.md`](../deploy/CHANGELOG.md).

## The problem

The manual bag of 2026-09-20 is missing 11.9 % of its lidar clouds (236 of 1,985), mostly
one at a time, at the same rate parked and driving. The Hesai driver lost 0 of 1.52 M UDP
packets, so the clouds are lost after the driver, while ROS 2 delivers them to the recorder.
`cone_detection` subscribes the same way and probably loses clouds too; that has not been
measured.

## What Cyclone DDS is, and why it drops clouds

ROS 2 does not move messages itself: it hands them to a **DDS** (Data Distribution Service)
implementation, chosen with `RMW_IMPLEMENTATION`. The car uses **Eclipse Cyclone DDS**
(`rmw_cyclonedds_cpp`, set in every `dv-*` service).

- A lidar cloud is ~4.5 MB. Cyclone sends it as many UDP fragments, even between two
  processes on the same computer.
- The kernel queues incoming fragments in the receiving socket's buffer. By default that
  buffer can be at most `net.core.rmem_max` = 212,992 bytes, less than 5 % of one cloud.
- A burst that does not fit is discarded by the kernel. With best-effort QoS nothing is
  resent, and one missing fragment discards the whole cloud.

The ROS 2 documentation lists this as a known Cyclone issue and gives the fix (ROS 2 docs,
*How-To Guides → DDS tuning → Cyclone DDS tuning*): raise `net.core.rmem_max`, and make
Cyclone request a large socket receive buffer.

## The fix, and how it differs from the ROS 2 guide

| Piece | Installed as | What it does |
|---|---|---|
| `deploy/60-isc-cyclonedds.conf` | `/etc/sysctl.d/60-isc-cyclonedds.conf` | `net.core.rmem_max = 33554432` (32 MB). Only raises the ceiling; sockets that do not ask for more are unchanged. Applied at every boot. |
| `deploy/cyclonedds.xml` | `/etc/dv/cyclonedds.xml` | `<SocketReceiveBufferSize max="10MB"/>`: Cyclone asks for a 10 MB buffer. |
| `deploy/cyclonedds.env` | `/etc/dv/cyclonedds.env` | `CYCLONEDDS_URI=file:///etc/dv/cyclonedds.xml`. **This file is the on/off switch.** |
| `dv-pipeline`, `dv-manual`, `dv-automission`, `dv-record` units | `/etc/systemd/system/` | `EnvironmentFile=-/etc/dv/cyclonedds.env`. The `-` makes it optional: no file, no change. |
| `dv_manual.sh`, `dv_automission.sh`, `dv_record.sh` | `/usr/local/bin/` | Load the switch file if it exists, so manual runs match the services. |

Two deliberate differences from the ROS 2 guide:

- **`max="10MB"` instead of `min="10MB"`.** Tested: with `min`, if the kernel limit has not
  been raised, Cyclone refuses to create any node ("failed to increase socket receive buffer
  size … failed to create domain"). The whole stack would fail to start. With `max`, Cyclone
  takes the largest size the kernel allows. When the limit is raised, both give the same
  buffer.
- **32 MB ceiling instead of the guide's 2,147,483,647.** The ceiling only needs to exceed the
  10 MB request; a lower one stops other programs from reserving huge buffers.

Not applied: the guide's `ipfrag_time` / `ipfrag_high_thresh` settings. They matter for IP
fragmentation on links with a small MTU. The clouds travel between processes on the same
computer, where that does not happen. Revisit if a DDS subscriber on another machine (not
Foxglove, which uses a websocket) shows losses.

## Measurements (Docker on the development Mac, not the DVPC)

`tools/dds_large_msg_check.py`: 4.52 MB `PointCloud2`, 10 Hz, best effort, depth 1, publisher
and subscriber in separate processes, Cyclone DDS 0.10.5 (ROS 2 Humble).

| Setup | Sent | Lost |
|---|---|---|
| Defaults (`rmem_max` 212,992, no XML) | 200 | 143 (71.5 %) |
| XML only (kernel limit not raised) | 200 | 197 (98.5 %): no help |
| Kernel limit only (no XML; Cyclone asks for its default 1 MB) | 200 | 10 (5.0 %) |
| **Both** | 500 (two runs) | **0** |

Socket receive buffer actually granted (the kernel reports twice the requested size):
425,984 bytes by default; 2,097,152 with the kernel limit only; 20,971,520 with both.

The DVPC loses fewer clouds than this (11.9 %) because its hardware and load differ; the
mechanism is the same, but **the improvement on the car is not yet measured**.

## Installing on the DVPC

The installed units and scripts are copies. Merging this into the repo changes nothing on
the car until the two install steps below are run.

1. Measure **before** (pipeline stopped):
   ```bash
   source /opt/ros/humble/setup.bash
   RMW_IMPLEMENTATION=rmw_cyclonedds_cpp python3 tools/dds_large_msg_check.py
   deploy/install_cyclonedds_tuning.sh status
   ```
2. Install the units and scripts that know about the switch (this installer backs up what it
   replaces):
   ```bash
   sudo deploy/install_dv_pipeline_service.sh
   ```
3. Install the tuning:
   ```bash
   sudo deploy/install_cyclonedds_tuning.sh install
   ```
   It backs up into `~isc/.isc_backups/<timestamp>_cyclonedds/` (including the live
   `rmem_max`), applies and verifies the kernel limit, checks that a ROS 2 node starts with
   the XML, and only then writes the switch file. If any check fails it rolls itself back.
   It restarts nothing.
4. Restart: `dv restart`, or reboot.
5. Measure **after**:
   ```bash
   set -a; . /etc/dv/cyclonedds.env; set +a
   RMW_IMPLEMENTATION=rmw_cyclonedds_cpp python3 tools/dds_large_msg_check.py   # pipeline stopped
   deploy/install_cyclonedds_tuning.sh status                                     # pipeline running
   ```
   `status` shows each running service with the tuning active or not, and every UDP socket's
   receive buffer and drop count. Then record a bag and compare its missing-cloud rate with
   11.9 %.
6. Write the result in `deploy/CHANGELOG.md`.

## Going back

```bash
sudo deploy/install_cyclonedds_tuning.sh rollback     # undo the latest install
dv restart                                            # running services drop the tuning
```

Rollback removes the switch first, then restores or removes the XML and the sysctl file
exactly as they were before the install, and sets `rmem_max` back to the recorded value.
Running it again undoes the install before that. Every install and rollback is logged in
`~isc/.isc_backups/cyclonedds_tuning.log`.

Without the script (for example from a rescue shell):

```bash
sudo rm /etc/dv/cyclonedds.env                       # the switch: enough on its own
sudo rm /etc/dv/cyclonedds.xml /etc/sysctl.d/60-isc-cyclonedds.conf
sudo sysctl -w net.core.rmem_max=212992              # or reboot
```

The unit and script changes do not need reverting: without `/etc/dv/cyclonedds.env` they
behave exactly as before.

## Tested, and not tested

Tested in an Ubuntu 22.04 + Humble container with host networking:

- install;
- identical re-install (no-op);
- update with a changed XML;
- `status` with a node running;
- rollback twice back to the original state, then a third rollback refused;
- an install with a broken XML aborting and rolling itself back;
- pre-existing files restored rather than deleted.

Not tested: systemd itself (no systemd in the container), and anything on the DVPC.
