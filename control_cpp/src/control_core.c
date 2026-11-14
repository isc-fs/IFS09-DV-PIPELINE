/*
 * control_core.c: see control_core.h. Each function names the Python code it
 * ports; the operation order follows it so results match to the last bit on
 * the same libm (build with -ffp-contract=off: no fused multiply-add).
 */
#include "control_core/control_core.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#define PI_D 3.141592653589793
#define TWO_PI_D (2.0 * PI_D)
#define FLOAT_EPS 2.220446049250313e-16 /* np.finfo(float).eps */
#define HIST_HORIZON_S 2.0
#define HIST_MAX_GAP_S 0.25
#define STARTUP_RAMP_DIST_M 1.0
#define STARTUP_STEER_MIN 0.3

/* Python's min/max of two floats: the first argument wins ties. */
static double pmin(double a, double b) { return b < a ? b : a; }
static double pmax(double a, double b) { return b > a ? b : a; }

/* Python float %: the result takes the sign of the divisor. */
static double py_mod(double a, double b) {
  double m = fmod(a, b);
  if (m != 0.0) {
    if ((b < 0.0) != (m < 0.0)) m += b;
  } else {
    m = copysign(0.0, b);
  }
  return m;
}

double ctrl_wrap_pi(double a) { return py_mod(a + PI_D, TWO_PI_D) - PI_D; }

void ctrl_params_default(ctrl_params_t *p) {
  memset(p, 0, sizeof(*p));
  p->wheelbase = 1.627;
  p->max_steer_rad = 18.2 * (PI_D / 180.0);
  p->lateral = CTRL_LAT_PURE_PURSUIT;
  p->lookahead_min = 1.0;
  p->lookahead_k = 0.5;
  p->lookahead_max = 8.0;
  p->lookahead_radius_factor = 0.7;
  p->stanley_k = 2.0;
  p->stanley_k_soft = 1.5;
  p->stanley_k_yaw_rate = 0.0;
  p->stanley_yaw_rate_loop_gain = 0.2;
  p->stanley_k_damp_steer = 0.2;
  p->v_max = 5.0;
  p->a_lat_max = 3.0;
  p->a_dec_max = 4.0;
  p->kp = 0.5;
  p->ki = 0.05;
  p->deadband = 0.05;
  p->throttle_max = 1.0;
  p->regen_power_w = 11000.0;
  p->vehicle_mass = 275.0;
  p->unseen_corner_radius = 4.5;
  p->brake_latency_s = 0.3;
  p->v_meas_tau = 0.15;
  p->v_meas_max_accel = 20.0;
  p->throttle_rate = 2.0;
  p->throttle_release_rate = 8.0;
  p->regen_rate = 3.33;
  p->steering_rate = 5.0;
  p->stop_latch_min_travel = 30.0;
  p->stop_past_anchor_m = 8.0;
  p->stop_on_anchor = 0;
  p->latency_comp_s = 0.0;
}

/* ------------------------------------------- reference.py: from_xy */

int ctrl_ref_empty(const ctrl_ref_t *r) { return r->n < 2; }

void ctrl_ref_build(ctrl_ref_t *r, const double *xs, const double *ys,
                    const double *kappa, int n) {
  int i, use_kappa = 0;
  if (n > CTRL_MAX_PATH) n = CTRL_MAX_PATH;
  if (n < 0) n = 0;
  r->n = n;
  r->stop_distance = INFINITY;
  r->stop_latched = 0;
  for (i = 0; i < n; ++i) {
    r->x[i] = xs[i];
    r->y[i] = ys[i];
  }
  if (n < 2) return;

  r->s[0] = 0.0;
  for (i = 0; i < n - 1; ++i) {
    const double dx = xs[i + 1] - xs[i];
    const double dy = ys[i + 1] - ys[i];
    r->yaw[i] = atan2(dy, dx);
    r->s[i + 1] = r->s[i] + hypot(dx, dy);
  }
  r->yaw[n - 1] = r->yaw[n - 2];

  if (kappa != NULL) {
    for (i = 0; i < n; ++i) {
      if (fabs(kappa[i]) > 1e-6) {
        use_kappa = 1;
        break;
      }
    }
  }
  if (use_kappa) {
    for (i = 0; i < n; ++i) r->curvature[i] = kappa[i];
  } else {
    for (i = 0; i < n; ++i) r->curvature[i] = 0.0;
    for (i = 1; i < n - 1; ++i) {
      const double ds = r->s[i + 1] - r->s[i - 1];
      if (ds > 1e-6) {
        r->curvature[i] = ctrl_wrap_pi(r->yaw[i + 1] - r->yaw[i - 1]) / ds;
      }
    }
    r->curvature[0] = r->curvature[1];
    r->curvature[n - 1] = r->curvature[n - 2];
  }
}

static double state_speed(const ctrl_state_t *st) {
  return sqrt(st->vx * st->vx + st->vy * st->vy);
}

static int nearest_index(double x, double y, const ctrl_ref_t *r) {
  int i, best_i = 0;
  double best_d2 = INFINITY;
  for (i = 0; i < r->n; ++i) {
    const double dx = r->x[i] - x, dy = r->y[i] - y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best_i = i;
    }
  }
  return best_i;
}

static double normalize_steer(const ctrl_params_t *p, double steer_rad) {
  return pmax(-1.0, pmin(1.0, steer_rad / p->max_steer_rad));
}

/* ------------------------------------------- pure_pursuit.py */

static double max_kappa_in_window(const ctrl_ref_t *r, int nearest,
                                  double window_m) {
  int i;
  double kmax = 0.0, s_anchor;
  if (nearest >= r->n) return 0.0;
  s_anchor = r->s[nearest];
  for (i = nearest; i < r->n; ++i) {
    double k;
    if (r->s[i] - s_anchor > window_m) break;
    k = fabs(r->curvature[i]);
    if (k > kmax) kmax = k;
  }
  return kmax;
}

double ctrl_pure_pursuit(const ctrl_params_t *p, const ctrl_state_t *st,
                         const ctrl_ref_t *r) {
  double ld, kappa_max, s_anchor, tx, ty, dx, dy, cos_y, sin_y, body_x, body_y;
  double d2, kappa;
  int nearest, target_idx, i, found = 0;
  if (ctrl_ref_empty(r)) return 0.0;

  ld = pmax(p->lookahead_min,
            pmin(p->lookahead_max,
                 p->lookahead_min + p->lookahead_k * state_speed(st)));
  nearest = nearest_index(st->x, st->y, r);
  kappa_max = max_kappa_in_window(r, nearest, ld);
  if (kappa_max > 1e-3) {
    ld = pmin(ld, p->lookahead_radius_factor / kappa_max);
    ld = pmax(ld, 0.5);
  }

  s_anchor = r->s[nearest];
  target_idx = nearest;
  for (i = nearest; i < r->n; ++i) {
    if (r->s[i] - s_anchor >= ld) {
      target_idx = i;
      found = 1;
      break;
    }
  }
  if (!found) target_idx = r->n - 1;

  tx = r->x[target_idx];
  ty = r->y[target_idx];
  dx = tx - st->x;
  dy = ty - st->y;
  cos_y = cos(st->yaw);
  sin_y = sin(st->yaw);
  body_x = cos_y * dx + sin_y * dy;
  body_y = -sin_y * dx + cos_y * dy;
  if (body_x <= 1e-3) return 0.0;
  d2 = body_x * body_x + body_y * body_y;
  if (d2 < 1e-6) return 0.0;
  kappa = 2.0 * body_y / d2;
  return normalize_steer(p, atan(p->wheelbase * kappa));
}

/* ------------------------------------------- stanley.py */

static double seg_yaw(const ctrl_ref_t *r, int i) {
  return atan2(r->y[i + 1] - r->y[i], r->x[i + 1] - r->x[i]);
}
static double seg_len(const ctrl_ref_t *r, int i) {
  return hypot(r->x[i + 1] - r->x[i], r->y[i + 1] - r->y[i]);
}

static void project_onto_polyline(double x, double y, const ctrl_ref_t *r,
                                  double *offset, double *yaw_out,
                                  double *curvature) {
  const int n = r->n;
  int i, best_i = 0, lo, hi, k;
  double best_t = 0.0, best_d2 = INFINITY;
  double ax, ay, dx, dy, t, px, py, norm, yaw, ds, sum;

  for (i = 0; i < n - 1; ++i) {
    double l2, d2, ex, ey;
    ax = r->x[i];
    ay = r->y[i];
    dx = r->x[i + 1] - ax;
    dy = r->y[i + 1] - ay;
    l2 = dx * dx + dy * dy;
    t = l2 < 1e-12 ? 0.0 : ((x - ax) * dx + (y - ay) * dy) / l2;
    t = pmin(1.0, pmax(0.0, t));
    px = ax + t * dx;
    py = ay + t * dy;
    ex = x - px;
    ey = y - py;
    d2 = ex * ex + ey * ey;
    if (d2 < best_d2) {
      best_i = i;
      best_t = t;
      best_d2 = d2;
    }
  }

  i = best_i;
  t = best_t;
  ax = r->x[i];
  ay = r->y[i];
  dx = r->x[i + 1] - ax;
  dy = r->y[i + 1] - ay;
  px = ax + t * dx;
  py = ay + t * dy;
  norm = hypot(dx, dy);
  if (norm < 1e-9) {
    *offset = 0.0;
  } else {
    const double cross = (dx * (y - py) - dy * (x - px)) / norm;
    *offset = best_d2 > 0.0 ? -copysign(sqrt(best_d2), cross) : 0.0;
  }

  yaw = seg_yaw(r, i);
  if (t > 0.5 && i + 1 < n - 1) {
    yaw = seg_yaw(r, i) + (t - 0.5) * ctrl_wrap_pi(seg_yaw(r, i + 1) - seg_yaw(r, i));
  } else if (t < 0.5 && i > 0) {
    yaw = seg_yaw(r, i) - (0.5 - t) * ctrl_wrap_pi(seg_yaw(r, i) - seg_yaw(r, i - 1));
  }
  *yaw_out = ctrl_wrap_pi(yaw);

  lo = i - 1 > 0 ? i - 1 : 0;
  hi = i + 1 < n - 2 ? i + 1 : n - 2;
  sum = 0.0;
  for (k = lo + 1; k < hi; ++k) sum += seg_len(r, k);
  ds = 0.5 * seg_len(r, lo) + sum + 0.5 * seg_len(r, hi);
  *curvature = (hi > lo && ds > 1e-6)
                   ? ctrl_wrap_pi(seg_yaw(r, hi) - seg_yaw(r, lo)) / ds
                   : 0.0;
}

double ctrl_stanley(const ctrl_params_t *p, const ctrl_state_t *st,
                    const ctrl_ref_t *r, double *prev_steer_norm) {
  double fx, fy, off, pyaw, pcurv, yaw_error, speed, cte_term, omega_ref;
  double yaw_rate_term, steer_rad, prev_steer_rad, steer_norm, k_yr;
  if (ctrl_ref_empty(r)) return 0.0;

  fx = st->x + p->wheelbase * cos(st->yaw);
  fy = st->y + p->wheelbase * sin(st->yaw);
  project_onto_polyline(fx, fy, r, &off, &pyaw, &pcurv);
  yaw_error = ctrl_wrap_pi(pyaw - st->yaw);
  speed = state_speed(st);
  cte_term = atan2(p->stanley_k * off, p->stanley_k_soft + speed);
  omega_ref = speed * pcurv;
  k_yr = p->stanley_k_yaw_rate;
  if (p->stanley_yaw_rate_loop_gain > 0.0) {
    k_yr += p->stanley_yaw_rate_loop_gain * p->wheelbase / pmax(speed, 1.0);
  }
  yaw_rate_term = k_yr * (st->yaw_rate - omega_ref);
  steer_rad = yaw_error + cte_term - yaw_rate_term;

  prev_steer_rad = *prev_steer_norm * p->max_steer_rad;
  if (p->stanley_k_damp_steer > 0.0) {
    steer_rad -= p->stanley_k_damp_steer * (steer_rad - prev_steer_rad);
  }
  steer_rad = pmax(-p->max_steer_rad, pmin(p->max_steer_rad, steer_rad));
  steer_norm = normalize_steer(p, steer_rad);
  *prev_steer_norm = steer_norm;
  return steer_norm;
}

/* ------------------------------------------- pi_velocity.py */

#define PI_DT (1.0 / 40.0)

void ctrl_pi_reset(ctrl_pi_t *pi) {
  pi->integral = 0.0;
  pi->seeded = 0;
}

double ctrl_brake_entry_speed(const ctrl_params_t *p, double v_end, double s) {
  const double a = p->a_dec_max;
  double P, m, v_star, s_const;
  if (s <= 0.0 || a <= 0.0) return pmax(0.0, v_end);
  P = p->regen_power_w;
  m = p->vehicle_mass;
  if (P <= 0.0 || m <= 0.0) return sqrt(v_end * v_end + 2.0 * a * s);
  v_star = P / (m * a);
  if (v_end >= v_star) return pow(pow(v_end, 3.0) + 3.0 * P * s / m, 1.0 / 3.0);
  s_const = (v_star * v_star - v_end * v_end) / (2.0 * a);
  if (s <= s_const) return sqrt(v_end * v_end + 2.0 * a * s);
  return pow(pow(v_star, 3.0) + 3.0 * P * (s - s_const) / m, 1.0 / 3.0);
}

static void corner_speed_ahead(const ctrl_params_t *p, const ctrl_state_t *st,
                               const ctrl_ref_t *r, double lost,
                               double *kmax_out, double *v_allow_out) {
  int i, nearest;
  double s0, kmax = 0.0, v_allow = p->v_max;
  if (ctrl_ref_empty(r)) {
    *kmax_out = 0.0;
    *v_allow_out = p->v_max;
    return;
  }
  nearest = nearest_index(st->x, st->y, r);
  s0 = r->s[nearest];
  if (p->unseen_corner_radius > 0.0) {
    const double v_unseen = sqrt(p->a_lat_max * p->unseen_corner_radius);
    v_allow = pmin(v_allow,
                   ctrl_brake_entry_speed(p, v_unseen, r->s[r->n - 1] - s0 - lost));
  }
  for (i = nearest; i < r->n; ++i) {
    double v_c;
    const double k = fabs(r->curvature[i]);
    if (k > kmax) kmax = k;
    if (k < 1e-3) continue;
    v_c = sqrt(p->a_lat_max / k);
    if (v_c >= v_allow) continue;
    v_allow = pmin(v_allow, ctrl_brake_entry_speed(p, v_c, r->s[i] - s0 - lost));
  }
  *kmax_out = kmax;
  *v_allow_out = v_allow;
}

static double pi_setpoint(const ctrl_params_t *p, ctrl_pi_t *pi,
                          const ctrl_state_t *st, const ctrl_ref_t *r) {
  const double lost = pmax(0.0, st->vx) * p->brake_latency_s;
  double kappa_max, v_corner, v_stop;
  corner_speed_ahead(p, st, r, lost, &kappa_max, &v_corner);
  pi->last_kappa_max = kappa_max;
  if (r->stop_distance < INFINITY && r->stop_distance > 0.0) {
    v_stop = ctrl_brake_entry_speed(p, 0.0, r->stop_distance - lost);
  } else {
    v_stop = INFINITY;
  }
  if (r->stop_latched && r->stop_distance <= 0.0) v_stop = 0.0;
  return pmax(0.0, pmin(pmin(p->v_max, v_corner), v_stop));
}

static double condition_velocity(const ctrl_params_t *p, ctrl_pi_t *pi,
                                 double v_raw) {
  double max_dv, dv, alpha;
  if (!pi->seeded) {
    pi->seeded = 1;
    pi->v_filt = v_raw;
    pi->v_prev_raw = v_raw;
    return v_raw;
  }
  max_dv = p->v_meas_max_accel * PI_DT;
  dv = v_raw - pi->v_prev_raw;
  if (dv > max_dv) {
    v_raw = pi->v_prev_raw + max_dv;
  } else if (dv < -max_dv) {
    v_raw = pi->v_prev_raw - max_dv;
  }
  pi->v_prev_raw = v_raw;
  alpha = PI_DT / (p->v_meas_tau + PI_DT);
  pi->v_filt += alpha * (v_raw - pi->v_filt);
  return pi->v_filt;
}

void ctrl_pi_compute(const ctrl_params_t *p, ctrl_pi_t *pi,
                     const ctrl_state_t *st, const ctrl_ref_t *r,
                     double *throttle, double *regen) {
  double v_set, v_meas, err, u_unsat, u;
  v_set = pi_setpoint(p, pi, st, r);
  pi->last_v_set = v_set;
  v_meas = condition_velocity(p, pi, st->vx);
  v_meas = pmax(v_meas, pi->v_prev_raw);

  err = v_set - v_meas;
  u_unsat = p->kp * err + p->ki * pi->integral;
  u = pmax(-1.0, pmin(1.0, u_unsat));
  if (u_unsat == u) pi->integral += err * PI_DT;

  if (u >= p->deadband) {
    *throttle = pmin(u, p->throttle_max);
    *regen = 0.0;
  } else if (u <= -p->deadband) {
    *throttle = 0.0;
    *regen = -u;
  } else {
    *throttle = 0.0;
    *regen = 0.0;
  }
}

/* ------------------------------------------- odom_history.py */

#define HIDX(h, k) (((h)->head + (k)) % CTRL_ODOM_HIST)

void ctrl_hist_clear(ctrl_odom_hist_t *h) {
  h->head = 0;
  h->len = 0;
}

void ctrl_hist_add(ctrl_odom_hist_t *h, double t, double x, double y,
                   double yaw) {
  int j;
  if (h->len > 0) {
    const double newest = h->t[HIDX(h, h->len - 1)];
    if (t <= newest) {
      if (t < newest) {
        ctrl_hist_clear(h);
      } else {
        return;
      }
    }
  }
  if (h->len == CTRL_ODOM_HIST) { /* full: drop the oldest */
    h->head = (h->head + 1) % CTRL_ODOM_HIST;
    h->len--;
  }
  j = HIDX(h, h->len);
  h->t[j] = t;
  h->x[j] = x;
  h->y[j] = y;
  h->yaw[j] = yaw;
  h->len++;
  while (h->len > 0 && h->t[h->head] < t - HIST_HORIZON_S) {
    h->head = (h->head + 1) % CTRL_ODOM_HIST;
    h->len--;
  }
}

int ctrl_hist_at(const ctrl_odom_hist_t *h, double t, double out[3]) {
  int lo, hi, i, i0, i1;
  double t0, t1, a, dh;
  if (h->len == 0) return 0;
  i1 = HIDX(h, h->len - 1);
  if (t >= h->t[i1]) {
    if (t - h->t[i1] <= HIST_MAX_GAP_S) {
      out[0] = h->x[i1];
      out[1] = h->y[i1];
      out[2] = h->yaw[i1];
      return 1;
    }
    return 0;
  }
  if (t < h->t[h->head]) return 0;
  lo = 0; /* bisect_left */
  hi = h->len;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (h->t[HIDX(h, mid)] < t) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  i = lo;
  i1 = HIDX(h, i);
  if (h->t[i1] == t) {
    out[0] = h->x[i1];
    out[1] = h->y[i1];
    out[2] = h->yaw[i1];
    return 1;
  }
  i0 = HIDX(h, i - 1);
  t0 = h->t[i0];
  t1 = h->t[i1];
  if (t1 - t0 > HIST_MAX_GAP_S) return 0;
  a = (t - t0) / (t1 - t0);
  dh = py_mod(h->yaw[i1] - h->yaw[i0] + PI_D, TWO_PI_D) - PI_D;
  out[2] = py_mod(h->yaw[i0] + a * dh + PI_D, TWO_PI_D) - PI_D;
  out[0] = h->x[i0] + a * (h->x[i1] - h->x[i0]);
  out[1] = h->y[i0] + a * (h->y[i1] - h->y[i0]);
  return 1;
}

void ctrl_predict_pose(const double pose[3], double v, const double *hist,
                       int count, int n, double wheelbase, double max_steer_rad,
                       double dt, double out[3]) {
  int j;
  double x = pose[0], y = pose[1], yaw = pose[2];
  for (j = 0; j < n; ++j) {
    const int back = n - j;
    const double u = count >= back ? hist[count - back] : 0.0;
    const double r = v * tan(-u * max_steer_rad) / wheelbase;
    const double hm = yaw + 0.5 * r * dt;
    x += v * cos(hm) * dt;
    y += v * sin(hm) * dt;
    yaw += r * dt;
  }
  if (n > 0) yaw = py_mod(yaw + PI_D, TWO_PI_D) - PI_D;
  out[0] = x;
  out[1] = y;
  out[2] = yaw;
}

void ctrl_project_pose(const double pose[3], const double then[3],
                       const double now[3], double out[3]) {
  const double dh = pose[2] - then[2];
  const double c = cos(dh), s = sin(dh);
  const double ddx = now[0] - then[0], ddy = now[1] - then[1];
  double yaw1 = pose[2] + (now[2] - then[2]);
  yaw1 = py_mod(yaw1 + PI_D, TWO_PI_D) - PI_D;
  out[0] = pose[0] + c * ddx - s * ddy;
  out[1] = pose[1] + s * ddx + c * ddy;
  out[2] = yaw1;
}

double ctrl_quat_yaw(double w, double x, double y, double z) {
  /* transforms3d quat2mat, then mat2euler 'sxyz': az = atan2(M10, M00). */
  const double nq = w * w + x * x + y * y + z * z;
  double s, X, Y, Z, wZ, xY, yY, zZ, m00, m10, cy;
  if (nq < FLOAT_EPS) return 0.0; /* identity */
  s = 2.0 / nq;
  X = x * s;
  Y = y * s;
  Z = z * s;
  (void)X;
  wZ = w * Z;
  xY = x * Y;
  yY = y * Y;
  zZ = z * Z;
  m00 = 1.0 - (yY + zZ);
  m10 = xY + wZ;
  cy = sqrt(m00 * m00 + m10 * m10);
  return cy > FLOAT_EPS * 4.0 ? atan2(m10, m00) : 0.0;
}

double ctrl_quat_yaw_z(double w, double z) { return 2.0 * atan2(z, w); }

/* ------------------------------------------- control_node.py */

void ctrl_node_reset(ctrl_node_t *n) {
  n->have_pose = 0;
  n->have_odom = 0;
  ctrl_hist_clear(&n->hist);
  n->path_n = 0;
  n->stop_latched = 0;
  n->stop_tx = n->stop_ty = 0.0;
  n->stop_dx = n->stop_dy = 0.0;
  n->final_lap = 1;
  n->travelled = 0.0;
  n->final_lap_from = 0.0;
  n->have_last_xy = 0;
  n->last_x = n->last_y = 0.0;
  n->last_throttle = n->last_regen = n->last_steering = 0.0;
  n->tick_count = 0;
  n->steer_n = 0;
  ctrl_pi_reset(&n->pi);
  n->pi.last_v_set = 0.0;
  n->pi.last_kappa_max = 0.0;
  n->stanley_prev = 0.0;
}

void ctrl_node_init(ctrl_node_t *n, const ctrl_params_t *p) {
  n->p = *p;
  ctrl_node_reset(n);
}

void ctrl_node_on_pose(ctrl_node_t *n, double stamp, double x, double y,
                       double qw, double qx, double qy, double qz) {
  n->have_pose = 1;
  n->pose_stamp = stamp;
  n->pose_x = x;
  n->pose_y = y;
  n->pose_yaw = ctrl_quat_yaw(qw, qx, qy, qz);
}

void ctrl_node_on_odom(ctrl_node_t *n, double stamp, double x, double y,
                       double qw, double qz, double vx, double vy, double wz) {
  n->have_odom = 1;
  n->odom_x = x;
  n->odom_y = y;
  n->odom_yaw = ctrl_quat_yaw_z(qw, qz);
  n->odom_vx = vx;
  n->odom_vy = vy;
  n->odom_wz = wz;
  ctrl_hist_add(&n->hist, stamp, x, y, n->odom_yaw);
}

int ctrl_node_on_path(ctrl_node_t *n, const double *xs, const double *ys,
                      const double *zs, int count) {
  int i;
  if (count > CTRL_MAX_PATH) count = CTRL_MAX_PATH;
  if (count < 0) count = 0;
  for (i = 0; i < count; ++i) {
    n->path_x[i] = xs[i];
    n->path_y[i] = ys[i];
    n->path_k[i] = zs[i];
  }
  n->path_n = count;
  return count;
}

int ctrl_node_on_final_lap(ctrl_node_t *n, int value) {
  int changed;
  value = value ? 1 : 0;
  if (value && !n->final_lap) n->final_lap_from = n->travelled;
  changed = value != n->final_lap;
  n->final_lap = value;
  return changed;
}

int ctrl_node_on_orange(ctrl_node_t *n, const double *xs, const double *ys,
                        int count, double *ax_out, double *ay_out,
                        double *tx_out, double *ty_out) {
  int i;
  double sx = 0.0, sy = 0.0, c, s, ax, ay, tx, ty;
  const double min_travel = n->p.stop_latch_min_travel;
  if (n->stop_latched || !n->have_pose) return 0;
  if (count < 2) return 0;
  if (n->travelled < min_travel) return 0;
  if (!n->final_lap) return 0;
  if (n->travelled - n->final_lap_from < min_travel) return 0;
  for (i = 0; i < count; ++i) sx += xs[i];
  for (i = 0; i < count; ++i) sy += ys[i];
  sx /= (double)count;
  sy /= (double)count;
  c = cos(n->pose_yaw);
  s = sin(n->pose_yaw);
  ax = n->pose_x + c * sx - s * sy;
  ay = n->pose_y + s * sx + c * sy;
  if (n->p.stop_on_anchor) {
    tx = ax;
    ty = ay;
    n->stop_dx = n->stop_dy = 0.0;
  } else {
    const double dx = ax - n->pose_x, dy = ay - n->pose_y;
    const double d = hypot(dx, dy);
    if (d < 1e-6) {
      tx = ax;
      ty = ay;
      n->stop_dx = n->stop_dy = 0.0;
    } else {
      const double ux = dx / d, uy = dy / d;
      tx = ax + n->p.stop_past_anchor_m * ux;
      ty = ay + n->p.stop_past_anchor_m * uy;
      n->stop_dx = ux;
      n->stop_dy = uy;
    }
  }
  n->stop_tx = tx;
  n->stop_ty = ty;
  n->stop_latched = 1;
  if (ax_out) *ax_out = ax;
  if (ay_out) *ay_out = ay;
  if (tx_out) *tx_out = tx;
  if (ty_out) *ty_out = ty;
  return 1;
}

static void remember_steer(ctrl_node_t *n, double u) {
  int i;
  if (n->steer_n == CTRL_STEER_HIST) {
    for (i = 1; i < CTRL_STEER_HIST; ++i) n->steer_hist[i - 1] = n->steer_hist[i];
    n->steer_n--;
  }
  n->steer_hist[n->steer_n++] = u;
}

static double slew(double prev, double target, double max_step,
                   double max_down) {
  if (target > prev + max_step) return prev + max_step;
  if (target < prev - max_down) return prev - max_down;
  return target;
}

void ctrl_node_tick(ctrl_node_t *n, ctrl_out_t *out) {
  const ctrl_params_t *p = &n->p;
  ctrl_ref_t *ref = &n->ref;
  ctrl_state_t st;
  double steer_norm, thr, regen, cap, dt;

  memset(out, 0, sizeof(*out));
  out->stop_distance = INFINITY;

  ctrl_ref_build(ref, n->path_x, n->path_y, n->path_n > 0 ? n->path_k : NULL,
                 n->path_n);
  out->path_n = ref->n;
  if (!n->have_pose || !n->have_odom || ctrl_ref_empty(ref)) {
    remember_steer(n, 0.0); /* the fail-safe zero command is published */
    return;
  }

  /* _build_state: /slam/pose carried forward to the newest /odom. */
  {
    double pose[3], then[3], now[3], moved[3];
    pose[0] = n->pose_x;
    pose[1] = n->pose_y;
    pose[2] = n->pose_yaw;
    if (ctrl_hist_at(&n->hist, n->pose_stamp, then)) {
      now[0] = n->odom_x;
      now[1] = n->odom_y;
      now[2] = n->odom_yaw;
      ctrl_project_pose(pose, then, now, moved);
      st.x = moved[0];
      st.y = moved[1];
      st.yaw = moved[2];
    } else {
      st.x = pose[0];
      st.y = pose[1];
      st.yaw = pose[2];
    }
    st.vx = n->odom_vx;
    st.vy = n->odom_vy;
    st.yaw_rate = n->odom_wz;
    /* delay compensation: the pose when this command acts */
    {
      const int nt = (int)(p->latency_comp_s * CTRL_RATE_HZ + 0.5);
      if (nt > 0) {
        pose[0] = st.x;
        pose[1] = st.y;
        pose[2] = st.yaw;
        ctrl_predict_pose(pose, st.vx, n->steer_hist, n->steer_n, nt,
                          p->wheelbase, p->max_steer_rad, 1.0 / CTRL_RATE_HZ,
                          moved);
        st.x = moved[0];
        st.y = moved[1];
        st.yaw = moved[2];
      }
    }
  }

  if (!n->have_last_xy) {
    n->have_last_xy = 1;
  } else {
    n->travelled += hypot(st.x - n->last_x, st.y - n->last_y);
  }
  n->last_x = st.x;
  n->last_y = st.y;

  if (!n->stop_latched) {
    ref->stop_distance = INFINITY;
  } else {
    const double dx = n->stop_tx - st.x, dy = n->stop_ty - st.y;
    if (n->stop_dx == 0.0 && n->stop_dy == 0.0) {
      ref->stop_distance = hypot(dx, dy);
    } else {
      ref->stop_distance = dx * n->stop_dx + dy * n->stop_dy;
    }
  }
  ref->stop_latched = n->stop_latched;

  if (p->lateral == CTRL_LAT_STANLEY) {
    steer_norm = ctrl_stanley(p, &st, ref, &n->stanley_prev);
  } else {
    steer_norm = ctrl_pure_pursuit(p, &st, ref);
  }
  ctrl_pi_compute(p, &n->pi, &st, ref, &thr, &regen);

  out->throttle = pmax(0.0, pmin(1.0, thr));
  out->brake = pmax(0.0, pmin(1.0, regen));
  if (n->travelled < STARTUP_RAMP_DIST_M) {
    const double ramp = n->travelled / STARTUP_RAMP_DIST_M;
    cap = STARTUP_STEER_MIN + (1.0 - STARTUP_STEER_MIN) * ramp;
  } else {
    cap = 1.0;
  }
  out->steering = pmax(-cap, pmin(cap, -steer_norm));

  dt = 1.0 / CTRL_RATE_HZ;
  out->throttle = slew(n->last_throttle, out->throttle, p->throttle_rate * dt,
                       p->throttle_release_rate * dt);
  out->brake = slew(n->last_regen, out->brake, p->regen_rate * dt,
                    p->regen_rate * dt);
  out->steering = slew(n->last_steering, out->steering, p->steering_rate * dt,
                       p->steering_rate * dt);
  n->last_throttle = out->throttle;
  n->last_regen = out->brake;
  n->last_steering = out->steering;
  remember_steer(n, out->steering);

  out->computed = 1;
  out->v_set = n->pi.last_v_set;
  out->kappa_max = n->pi.last_kappa_max;
  out->speed = state_speed(&st);
  out->path_len = ref->s[ref->n - 1];
  out->stop_distance = ref->stop_distance;
  out->stop_latched = ref->stop_latched;
  n->tick_count++;
}
