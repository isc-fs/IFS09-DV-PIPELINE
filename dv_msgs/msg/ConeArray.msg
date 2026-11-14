# Cones as flat arrays: entry i of every array describes cone i.
#
# Pipeline transport for the cone hops:
#   /cones_raw     detection -> SLAM      per-scan observations, base_link
#   /cones_orange  detection -> control   big-orange cones only, base_link
#   /cones         SLAM -> path planning  landmark map near the car, map frame
# These replace visualization_msgs/MarkerArray on those hops. rclpy decoded a
# 36-cone MarkerArray in 18.7 ms on an RK3588 A76 core (every Marker builds
# ~20 nested Python objects); flat arrays decode as a few contiguous copies.
# The MarkerArray topics (/Conos_raw, /Conos_Orange, /Conos) remain for viewers
# and recorders, built only when something subscribes to them.
#
# header.stamp is the originating LiDAR scan time, also when the arrays are empty.

std_msgs/Header header

float32[] x          # m
float32[] y          # m
float32[] z          # m
float32[] height     # m, cone height (template apex or fitted); 0 = unknown
float32[] sigma_xy   # m, 1-sigma position uncertainty; < 0 = unknown
int32[] id           # persistent SLAM landmark id; -1 = none (detections)
