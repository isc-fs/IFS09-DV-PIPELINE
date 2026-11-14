/*
 * control_core: the DV vehicle controller in plain C99.
 *
 * A line-for-line port of the Python `control` package (control_node._tick,
 * Pure Pursuit, Stanley, PI velocity, stop target, odom history). No heap,
 * no OS, no ROS: fixed-size arrays and <math.h> only, so the same file builds
 * for the RK3588 node (control_cpp/src/control_node.cpp) and for the uDV
 * STM32H7 (Cortex-M7, double-precision FPU) when control moves there.
 *
 * Conventions match the Python package: SI units, x forward, y left, yaw CCW,
 * strategy steering positive = left; the published command flips it.
 */
#ifndef CONTROL_CORE_H
#define CONTROL_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Path points kept per /Path message (the planner sends ~30-60). */
#ifndef CTRL_MAX_PATH
#define CTRL_MAX_PATH 512
#endif
/* /odom samples kept for the 2 s history (100 Hz needs 200). */
#ifndef CTRL_ODOM_HIST
#define CTRL_ODOM_HIST 1024
#endif

#define CTRL_RATE_HZ 40.0

enum { CTRL_LAT_PURE_PURSUIT = 0, CTRL_LAT_STANLEY = 1 };

typedef struct {
  /* kinematic bicycle */
  double wheelbase;
  double max_steer_rad;
  /* lateral */
  int lateral; /* CTRL_LAT_* */
  double lookahead_min, lookahead_k, lookahead_max, lookahead_radius_factor;
  double stanley_k, stanley_k_soft, stanley_k_yaw_rate, stanley_k_damp_steer;
  /* longitudinal (PI velocity) */
  double v_max, a_lat_max, a_dec_max, kp, ki, deadband, throttle_max;
  double regen_power_w, vehicle_mass, unseen_corner_radius, brake_latency_s;
  double v_meas_tau, v_meas_max_accel;
  /* node: actuator slew (units per second) and the finish stop */
  double throttle_rate, throttle_release_rate, regen_rate, steering_rate;
  double stop_latch_min_travel, stop_past_anchor_m;
  int stop_on_anchor; /* 1: stop on the anchor, straight-line distance */
} ctrl_params_t;

/* The Python defaults (PurePursuit/Stanley/PIVelocity constructors and
 * control_node._declare_params). */
void ctrl_params_default(ctrl_params_t *p);

/* ---------------------------------------------------------------- pieces */

typedef struct {
  double x, y, yaw, vx, vy, yaw_rate;
} ctrl_state_t;

typedef struct {
  int n;
  double x[CTRL_MAX_PATH], y[CTRL_MAX_PATH], yaw[CTRL_MAX_PATH];
  double curvature[CTRL_MAX_PATH], s[CTRL_MAX_PATH];
  double stop_distance; /* +inf: no stop */
  int stop_latched;
} ctrl_ref_t;

/* ReferenceTrajectory.from_xy. kappa may be NULL; it is used when it has a
 * value above 1e-6, else curvature comes from finite differences. n < 2 gives
 * an empty reference (n is kept, geometry is not derived). */
void ctrl_ref_build(ctrl_ref_t *r, const double *xs, const double *ys,
                    const double *kappa, int n);
int ctrl_ref_empty(const ctrl_ref_t *r);

double ctrl_wrap_pi(double a);

double ctrl_pure_pursuit(const ctrl_params_t *p, const ctrl_state_t *st,
                         const ctrl_ref_t *r);
/* prev_steer_norm: in/out warm start (Stanley._prev_steer_norm). */
double ctrl_stanley(const ctrl_params_t *p, const ctrl_state_t *st,
                    const ctrl_ref_t *r, double *prev_steer_norm);

typedef struct {
  int seeded;      /* _v_filt is not None */
  double v_filt, v_prev_raw, integral;
  double last_v_set, last_kappa_max;
} ctrl_pi_t;

void ctrl_pi_reset(ctrl_pi_t *pi);
double ctrl_brake_entry_speed(const ctrl_params_t *p, double v_end, double s);
void ctrl_pi_compute(const ctrl_params_t *p, ctrl_pi_t *pi,
                     const ctrl_state_t *st, const ctrl_ref_t *r,
                     double *throttle, double *regen);

typedef struct {
  int head, len;
  double t[CTRL_ODOM_HIST], x[CTRL_ODOM_HIST], y[CTRL_ODOM_HIST],
      yaw[CTRL_ODOM_HIST];
} ctrl_odom_hist_t;

void ctrl_hist_clear(ctrl_odom_hist_t *h);
void ctrl_hist_add(ctrl_odom_hist_t *h, double t, double x, double y,
                   double yaw);
/* 1 and out[3] = (x, y, yaw) at t, or 0 (OdomHistory.at with max_gap 0.25). */
int ctrl_hist_at(const ctrl_odom_hist_t *h, double t, double out[3]);
void ctrl_project_pose(const double pose[3], const double then[3],
                       const double now[3], double out[3]);

/* Yaw of a quaternion as transforms3d.euler.quat2euler (sxyz) gives it. */
double ctrl_quat_yaw(double w, double x, double y, double z);
/* Yaw of a z-only quaternion: 2·atan2(qz, qw). */
double ctrl_quat_yaw_z(double w, double z);

/* ------------------------------------------------------------- the node */

typedef struct {
  ctrl_params_t p;

  int have_pose;
  double pose_x, pose_y, pose_yaw, pose_stamp;
  int have_odom;
  double odom_x, odom_y, odom_yaw, odom_vx, odom_vy, odom_wz;
  ctrl_odom_hist_t hist;

  int path_n;
  double path_x[CTRL_MAX_PATH], path_y[CTRL_MAX_PATH], path_k[CTRL_MAX_PATH];

  int stop_latched;
  double stop_tx, stop_ty, stop_dx, stop_dy;
  int final_lap;
  double travelled, final_lap_from;
  int have_last_xy;
  double last_x, last_y;

  double last_throttle, last_regen, last_steering;
  uint32_t tick_count;

  ctrl_pi_t pi;
  double stanley_prev;
  ctrl_ref_t ref; /* scratch, rebuilt every tick */
} ctrl_node_t;

typedef struct {
  double throttle, brake, steering; /* the published command */
  int computed;                      /* 0: fail-safe zero command */
  double v_set, kappa_max;           /* published only when computed */
  double speed, path_len, stop_distance; /* heartbeat log */
  int path_n, stop_latched;
} ctrl_out_t;

/* on_configure + on_activate: params, then the per-run reset. */
void ctrl_node_init(ctrl_node_t *n, const ctrl_params_t *p);
/* on_activate: per-run state and controller state back to a fresh run. */
void ctrl_node_reset(ctrl_node_t *n);

void ctrl_node_on_pose(ctrl_node_t *n, double stamp, double x, double y,
                       double qw, double qx, double qy, double qz);
void ctrl_node_on_odom(ctrl_node_t *n, double stamp, double x, double y,
                       double qw, double qz, double vx, double vy, double wz);
/* Returns the number of points kept (capped at CTRL_MAX_PATH). */
int ctrl_node_on_path(ctrl_node_t *n, const double *xs, const double *ys,
                      const double *zs, int count);
/* Returns 1 when the value changed. */
int ctrl_node_on_final_lap(ctrl_node_t *n, int value);
/* Big-orange cones in base_link. Returns 1 when this call latched the stop;
 * anchor (ax, ay) and target (tx, ty) are written then (any may be NULL). */
int ctrl_node_on_orange(ctrl_node_t *n, const double *xs, const double *ys,
                        int count, double *ax, double *ay, double *tx,
                        double *ty);
void ctrl_node_tick(ctrl_node_t *n, ctrl_out_t *out);

#ifdef __cplusplus
}
#endif

#endif /* CONTROL_CORE_H */
