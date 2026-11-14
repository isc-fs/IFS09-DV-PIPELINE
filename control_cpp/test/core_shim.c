/* Flat entry points over control_core for the ctypes equivalence tests
 * (control/test/test_control_core_c.py). Not part of the node build. */
#include <stdlib.h>
#include <string.h>

#include "control_core/control_core.h"

static void st_from(ctrl_state_t *st, const double *s6) {
  st->x = s6[0]; st->y = s6[1]; st->yaw = s6[2];
  st->vx = s6[3]; st->vy = s6[4]; st->yaw_rate = s6[5];
}

static ctrl_ref_t g_ref;

static ctrl_ref_t *ref_from(const double *xs, const double *ys,
                            const double *ks, int n, double stop_d,
                            int stop_latched) {
  ctrl_ref_build(&g_ref, xs, ys, ks, n);
  g_ref.stop_distance = stop_d;
  g_ref.stop_latched = stop_latched;
  return &g_ref;
}

int shim_ref(const double *xs, const double *ys, const double *ks, int n,
             double *yaw, double *curv, double *s) {
  int i;
  ref_from(xs, ys, ks, n, 0.0, 0);
  if (n < 2) return 0;
  for (i = 0; i < n; ++i) {
    yaw[i] = g_ref.yaw[i]; curv[i] = g_ref.curvature[i]; s[i] = g_ref.s[i];
  }
  return 1;
}

double shim_pp(const ctrl_params_t *p, const double *s6, const double *xs,
               const double *ys, const double *ks, int n) {
  ctrl_state_t st;
  st_from(&st, s6);
  return ctrl_pure_pursuit(p, &st, ref_from(xs, ys, ks, n, 0.0, 0));
}

double shim_stanley(const ctrl_params_t *p, const double *s6, const double *xs,
                    const double *ys, const double *ks, int n, double *prev) {
  ctrl_state_t st;
  st_from(&st, s6);
  return ctrl_stanley(p, &st, ref_from(xs, ys, ks, n, 0.0, 0), prev);
}

void *shim_pi_new(void) { ctrl_pi_t *pi = calloc(1, sizeof(*pi)); ctrl_pi_reset(pi); return pi; }

void shim_pi_step(const ctrl_params_t *p, void *pi, const double *s6,
                  const double *xs, const double *ys, const double *ks, int n,
                  double stop_d, int stop_latched, double *out4) {
  ctrl_state_t st;
  ctrl_pi_t *q = (ctrl_pi_t *)pi;
  st_from(&st, s6);
  ctrl_pi_compute(p, q, &st, ref_from(xs, ys, ks, n, stop_d, stop_latched),
                  &out4[0], &out4[1]);
  out4[2] = q->last_v_set;
  out4[3] = q->last_kappa_max;
}

void *shim_hist_new(void) { ctrl_odom_hist_t *h = malloc(sizeof(*h)); ctrl_hist_clear(h); return h; }

void *shim_node_new(const ctrl_params_t *p) {
  ctrl_node_t *n = malloc(sizeof(*n));
  ctrl_node_init(n, p);
  return n;
}

int shim_node_final_lap(void *node) { return ((ctrl_node_t *)node)->final_lap; }

void shim_node_tick(void *node, double *out8) {
  ctrl_out_t o;
  ctrl_node_tick((ctrl_node_t *)node, &o);
  out8[0] = o.throttle; out8[1] = o.brake; out8[2] = o.steering;
  out8[3] = o.computed; out8[4] = o.v_set; out8[5] = o.kappa_max;
  out8[6] = ((ctrl_node_t *)node)->travelled;
  out8[7] = ((ctrl_node_t *)node)->stop_latched;
}

void shim_free(void *x) { free(x); }
