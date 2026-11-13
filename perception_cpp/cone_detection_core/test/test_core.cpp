// Unit tests for the C++ core. Plain asserts, no framework: ctest runs the binary.
// Equivalence with the Numba kernels on real scans is cone_detection/test/test_golden.py
// and test_cpp_detect.py; these check each kernel on its own.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <vector>

#include "cone_detection_core/core.hpp"

using namespace cone_detection_core;

static int g_failed = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
      ++g_failed;                                                         \
    }                                                                     \
  } while (0)
#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

// Numba 0.68.1: np.random.seed(s); [np.random.randint(0, n) for n in NS]
static void test_rng_matches_numba() {
  const int64_t ns[8] = {2, 3, 10, 1000, 174000, (int64_t(1) << 33) + 7, 5, 7};
  const int64_t want12345[8] = {0, 1, 1, 420, 109993, 6580224546LL, 1, 6};
  const int64_t want0[8] = {0, 1, 0, 835, 152315, 6973152979LL, 2, 4};
  NumbaMt a(12345), b(0);
  for (int i = 0; i < 8; ++i) {
    CHECK(a.randint(0, ns[i]) == want12345[i]);
    CHECK(b.randint(0, ns[i]) == want0[i]);
  }
  NumbaMt c(7);
  CHECK(c.randint(5, 6) == 5);  // n == 1 draws nothing
}

static void test_sample_without_replacement() {
  NumbaMt rng(3);
  const auto s = sample_without_replacement(rng, 1000, 300);
  CHECK(s.size() == 300);
  std::set<int64_t> u(s.begin(), s.end());
  CHECK(u.size() == 300);
  CHECK(*u.begin() >= 0 && *u.rbegin() < 1000);
}

static void test_decode_packed_26_byte_points() {
  // x, y, z, intensity f32 + ring u16 + timestamp f64 = 26 bytes (Hesai ATX)
  const std::size_t step = 26, n = 3;
  std::vector<uint8_t> buf(n * step, 0xAB);
  for (std::size_t i = 0; i < n; ++i) {
    const float v[3] = {float(i) + 0.5f, -float(i), 2.0f * float(i)};
    std::memcpy(&buf[i * step + 0], &v[0], 4);
    std::memcpy(&buf[i * step + 4], &v[1], 4);
    std::memcpy(&buf[i * step + 8], &v[2], 4);
  }
  std::vector<float> out(3 * n);
  decode_xyz(buf.data(), n, step, 0, 4, 8, out.data());
  for (std::size_t i = 0; i < n; ++i) {
    CHECK(out[3 * i] == float(i) + 0.5f);
    CHECK(out[3 * i + 1] == -float(i));
    CHECK(out[3 * i + 2] == 2.0f * float(i));
  }
}

static void test_crop_promote() {
  const float pts[] = {1, 0, 0, 30, 0, 0, 0.01f, 0, 0, 3, 4, 1};
  const auto c = crop_promote<float>(pts, 4, 25.0f * 25.0f, 0.1f * 0.1f);
  CHECK(c.size() == 6);  // far point and the no-return point dropped
  CHECK(c[0] == 1.0 && c[3] == 3.0 && c[5] == 1.0);
}

static Cloud tilted_ground(NumbaMt& rng, int n, double slope) {
  Cloud c;
  for (int i = 0; i < n; ++i) {
    const double x = 1.0 + 20.0 * (rng.next32() / 4294967296.0);
    const double y = -5.0 + 10.0 * (rng.next32() / 4294967296.0);
    c.insert(c.end(), {x, y, slope * x});
  }
  return c;
}

static void test_ransac_recovers_plane_and_warm_start() {
  NumbaMt gen(1);
  Cloud c = tilted_ground(gen, 3000, 0.02);
  // a few obstacles above the ground
  for (int i = 0; i < 100; ++i) c.insert(c.end(), {5.0 + 0.001 * i, 1.0, 0.5 + 0.001 * i});
  NumbaMt rng(42);
  std::vector<uint8_t> outl;
  const Plane p = ransac_plane(c, 0.9999, 0.05, 200, 5000, {}, rng, outl);
  const double nn = std::sqrt(p.coefs[1] * p.coefs[1] + p.coefs[2] * p.coefs[2] + p.coefs[3] * p.coefs[3]);
  CHECK(p.coefs[3] > 0);
  CHECK_NEAR(p.coefs[1] / nn, -0.02 / std::sqrt(1 + 0.02 * 0.02), 1e-3);
  std::size_t n_out = 0;
  for (auto o : outl) n_out += o;
  CHECK(n_out == 100);
  // Same seed, same plane; warm start with the answer keeps it.
  NumbaMt rng2(42);
  std::vector<uint8_t> outl2;
  const Plane p2 = ransac_plane(c, 0.9999, 0.05, 200, 5000, {}, rng2, outl2);
  CHECK(p2.coefs == p.coefs);
  NumbaMt rng3(9);
  const std::vector<double> warm(p.coefs.begin(), p.coefs.end());
  std::vector<uint8_t> outl3;
  const Plane p3 = ransac_plane(c, 0.9999, 0.05, 200, 5000, warm, rng3, outl3);
  CHECK(p3.coefs == p.coefs);
}

static void test_rotation_maps_normal_to_z() {
  Plane p;
  p.coefs = {0.3, 0.1, -0.2, 0.97};
  const Mat3 R = rot_to_normal(p);
  const double nn = std::sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.97 * 0.97);
  // select_rotate computes v @ R: the unit normal must land on +z.
  const double v[3] = {0.1 / nn, -0.2 / nn, 0.97 / nn};
  CHECK_NEAR(v[0] * R[0] + v[1] * R[3] + v[2] * R[6], 0.0, 1e-12);
  CHECK_NEAR(v[0] * R[1] + v[1] * R[4] + v[2] * R[7], 0.0, 1e-12);
  CHECK_NEAR(v[0] * R[2] + v[1] * R[5] + v[2] * R[8], 1.0, 1e-12);
}

static void test_column_veto() {
  Cloud c = {0.05, 0.05, 1.0,   // tall
             0.40, 0.05, 0.2,   // next cell: vetoed
             2.00, 2.00, 0.2};  // far: kept
  CHECK(column_veto(c, 0.75, 0.3) == 2);
  CHECK(c.size() == 3 && c[0] == 2.0);
  Cloud low = {0, 0, 0.1, 1, 1, 0.2};
  CHECK(column_veto(low, 0.75, 0.3) == 0);
  CHECK(low.size() == 6);
}

static void test_voxel_first() {
  Cloud c = {0.001, 0.001, 0.001, 0.005, 0.005, 0.005, 0.5, 0.5, 0.5, -0.001, 0.0, 0.0};
  const auto k = voxel_first(c, 0.02, int64_t(1) << 20, 21);
  CHECK((k == std::vector<int64_t>{0, 2, 3}));
}

static void test_grid_dbscan_components() {
  // Chain of points 0.25 m apart (one component), a far pair, and a lone point.
  Cloud c = {0, 0, 0, 0.25, 0, 0, 0.5, 0, 0, 10, 10, 0, 10.1, 10, 0, 50, 50, 0};
  const auto l = grid_dbscan(c, 0.3);
  CHECK(l.size() == 6);
  CHECK(l[0] == l[1] && l[1] == l[2] && l[0] >= 0);
  CHECK(l[3] == l[4] && l[3] >= 0 && l[3] != l[0]);
  CHECK(l[5] == -1);
  CHECK(grid_dbscan({}, 0.3).empty());
}

static Cloud cone_face(double cx, double cy, double c, double d, int n) {
  // one-sided near-face points of a cone with apex height d and slope c
  Cloud pts;
  const double nr = std::hypot(cx, cy), ux = cx / nr, uy = cy / nr;
  for (int i = 0; i < n; ++i) {
    const double z = 0.06 + (d - 0.15) * i / (n - 1);  // h = d - 0.15
    const double g = (d - z) / c;
    const double psi = -1.0 + 2.0 * ((i * 7) % n) / double(n - 1);
    pts.insert(pts.end(), {cx - g * (std::cos(psi) * ux - std::sin(psi) * uy),
                           cy - g * (std::cos(psi) * uy + std::sin(psi) * ux), z});
  }
  return pts;
}

static void test_template_fit_gn_recovers_apex() {
  const Cloud p = cone_face(6.0, 1.5, 5.16, 0.43, 40);
  std::vector<double> x, y, z;
  for (std::size_t i = 0; i < p.size(); i += 3) {
    x.push_back(p[i]);
    y.push_back(p[i + 1]);
    z.push_back(p[i + 2]);
  }
  const auto f = template_fit_gn(x.data(), y.data(), z.data(), x.size(), 5.16, 0.43, 5.9, 1.45,
                                 5.0, 7.0, 0.5, 2.5, 50);
  CHECK_NEAR(f[0], 6.0, 1e-4);
  CHECK_NEAR(f[1], 1.5, 1e-4);
  CHECK(f[2] < 1e-8);
}

static void test_fit_clusters_gates_and_classes() {
  Cloud pts = cone_face(6.0, 1.5, 5.16, 0.43, 40);  // small cone, label 0
  const Cloud big = cone_face(9.0, -2.0, 6.45, 0.65, 40);  // big cone, label 1
  pts.insert(pts.end(), big.begin(), big.end());
  std::vector<int64_t> labels(80, 0);
  for (int i = 40; i < 80; ++i) labels[i] = 1;
  pts.insert(pts.end(), {30.0, 0.0, 0.3, 30.1, 0.0, 0.4, 30.0, 0.1, 0.2});  // too far, label 2
  labels.insert(labels.end(), {2, 2, 2});
  pts.insert(pts.end(), {1.0, 1.0, 0.1});  // noise, label -1 (below min points)
  labels.push_back(-1);
  // production values from bringup/config/params.yaml
  const std::array<double, 11> p = {3, 0.04, 0.02, 0.55, 20.0, 0.2, 1.5, 0.4, 0.5, 12, 1.0};
  const auto r = fit_clusters(pts, labels, p);
  CHECK(r.counters[N_CLUSTERS] == 4);
  CHECK(r.counters[AFTER_MIN_PTS] == 3);
  CHECK(r.counters[FAR_DROPPED] == 1);
  CHECK(r.counters[ACCEPTED] == 2);
  CHECK(r.counters[SMALL] == 1 && r.counters[BIG] == 1);
  CHECK(r.rows.size() == 10);
  CHECK_NEAR(r.rows[0], 6.0, 1e-3);   // label 0 first
  CHECK(r.rows[2] == 0.43);
  CHECK_NEAR(r.rows[5], 9.0, 1e-3);
  CHECK(r.rows[7] == 0.65);
}

int main() {
  test_rng_matches_numba();
  test_sample_without_replacement();
  test_decode_packed_26_byte_points();
  test_crop_promote();
  test_ransac_recovers_plane_and_warm_start();
  test_rotation_maps_normal_to_z();
  test_column_veto();
  test_voxel_first();
  test_grid_dbscan_components();
  test_template_fit_gn_recovers_apex();
  test_fit_clusters_gates_and_classes();
  if (g_failed) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("cone_detection_core: all tests passed\n");
  return 0;
}
