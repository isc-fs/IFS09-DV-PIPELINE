# DV Pipeline Testing Tools

This directory contains utilities for testing and validating the DV pipeline with rosbag replay.

## Tools

### mission_injector.py
Simulates the uDV (vehicle control unit) interface for pipeline testing. Provides the required `/assi/state` heartbeat and `/ami/mission` commands that the pipeline expects from the real vehicle hardware.

**Use when:** Testing the pipeline with rosbag replay (manual driving bags don't contain mission commands)

### record_pipeline_replay.sh
Automated orchestration script that replays a rosbag, runs the pipeline, and records all outputs for analysis.

**Use when:** You want to capture the complete pipeline behavior for performance analysis

### dds_large_msg_check.py
Publishes lidar-sized (4.5 MB) best-effort messages at 10 Hz from one process and counts how many a second process receives. Measures ROS 2 delivery loss on the current machine without driving.

**Use when:** Checking the Cyclone DDS tuning before and after installing it (see [docs/DDS_TUNING.md](../docs/DDS_TUNING.md)). Stop the pipeline first.

## Quick Start

```bash
# Record complete pipeline behavior from a manual driving bag
cd ~/dv_ws/src/IFS08-DV-PIPELINE/tools
./record_pipeline_replay.sh ~/bags/your_manual_bag trackdrive
```

On the DVPC this is also wired into the `dv` control command, so it can be run
from anywhere without cd-ing into the tools dir:

```bash
dv replay ~/bags/your_manual_bag trackdrive
```

`dv replay` refuses to run while the racing pipeline or manual recorder is up
(`dv stop` them first) — replay spins up its own sim-time pipeline.

See **REPLAY_SETUP_SUMMARY.md** for detailed usage instructions.

## Documentation

- **REPLAY_SETUP_SUMMARY.md** - Complete guide to replay testing
- **TOPIC_RECORDING_INFO.md** - Topic recording configuration reference

## Purpose

These tools enable:
- ✅ Real-time performance testing without full simulator or car
- ✅ Latency and computational load analysis
- ✅ Pipeline behavior recording for offline analysis
- ✅ Development and debugging without hardware access

## Requirements

- ROS 2 Humble
- DV pipeline workspace built and sourced
- Manual driving rosbags with: `/imu`, `/lidar_points`, `/motor_rpm`, `/steering_angle`
