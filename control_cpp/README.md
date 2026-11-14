# control_cpp

The DV vehicle controller in C, so it can move to the uDV STM32H7.

| Part | File | Depends on |
|---|---|---|
| Controller | `src/control_core.c`, `include/control_core/control_core.h` | C99 and `<math.h>` only: no heap, no OS, no ROS |
| ROS 2 node | `src/control_node.cpp` | rclcpp; turns messages into core calls and back |

`control_core` is a port of the Python `control` package: Pure Pursuit or Stanley,
PI velocity with the corner, sight-distance and stop caps, the finish-stop latch,
`/slam/pose` carried forward on `/odom`, the startup steering cap and the actuator
slew. The node keeps the Python node's name, topics, QoS, parameters and lifecycle,
so the launch picks either one (`DV_CONTROL=python` for the Python node, see
`bringup/launch_common.py`).

## Same output as the Python node

`control/test/test_control_core_c.py` compiles the core with the system C
compiler and compares it with the Python package, piece by piece on random inputs
and, where rclpy is installed, the whole node over 120 s closed-loop runs of four
missions. Differences stay below 1e-9 (`hypot` in libm and in Python can differ in
the last bit). Build with `-ffp-contract=off` to keep that: fused multiply-add
changes the rounding.

## Moving it to the STM32H7

1. Add `control_core.c` and its header to the firmware build. Cortex-M7 has a
   double-precision FPU, so keep `double`.
2. Size the buffers: `CTRL_MAX_PATH` (path points, default 512) and
   `CTRL_ODOM_HIST` (odometry samples in 2 s, default 1024) set the size of
   `ctrl_node_t` (66 KB by default; 128 and 256 bring it to 17 KB).
   Define them before including the header.
3. Fill a `ctrl_params_t` (`ctrl_params_default`, then the mission's values)
   and call `ctrl_node_init`, then `ctrl_node_reset` at each mission start.
4. Feed inputs as they arrive: `ctrl_node_on_odom` (the uDV's own odometry),
   `ctrl_node_on_pose`, `ctrl_node_on_path`, `ctrl_node_on_orange` and
   `ctrl_node_on_final_lap` (from the Pi).
5. Call `ctrl_node_tick` at 40 Hz. Its `throttle`, `brake` and `steering` are
   the command; `steering` is positive to the right, as `ControlCommand`.

The Pi then only forwards the path, the pose and the cones.
