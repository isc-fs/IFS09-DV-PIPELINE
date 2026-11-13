// 1:1 port of cone_detection/fused.py (+ clustering._grid_components and
// cone_fit._template_fit_gn). Comments name the Python function each block ports;
// expression order follows the Python source on purpose (see core.hpp).
#include "cone_detection_core/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

namespace cone_detection_core {

namespace {
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// cone_fit.py constants
constexpr double kConeBaseRadius = 0.114;
constexpr double kNearFaceBiasFrac = 0.4;
constexpr double kLinearityThreshold = 0.10;
constexpr double kSmallC = 5.16;
constexpr double kSmallD = 0.43;
constexpr double kBigC = 6.45;
constexpr double kBigD = 0.65;
constexpr double kClassTolerance = 0.5;
constexpr double kAbBoundMargin = 0.05;

// Numba lowers float ** int by repeated squaring: r ** 3 == r * (r * r).
inline double cube(double r) { return r * (r * r); }

inline double mean(const double* v, std::size_t n) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += v[i];
  return s / static_cast<double>(n);
}
inline double vmin(const double* v, std::size_t n) {
  double m = v[0];
  for (std::size_t i = 1; i < n; ++i) m = v[i] < m ? v[i] : m;
  return m;
}
inline double vmax(const double* v, std::size_t n) {
  double m = v[0];
  for (std::size_t i = 1; i < n; ++i) m = v[i] > m ? v[i] : m;
  return m;
}
}  // namespace

// ---------------------------------------------------------------------------
// NumbaMt
// ---------------------------------------------------------------------------
NumbaMt::NumbaMt() {
  std::random_device rd;
  seed_(rd());
}

void NumbaMt::seed_(uint32_t seed) {
  for (uint32_t pos = 0; pos < 624; ++pos) {
    mt_[pos] = seed;
    seed = 1812433253U * (seed ^ (seed >> 30)) + pos + 1;
  }
  index_ = 624;
}

void NumbaMt::shuffle() {
  constexpr int N = 624, M = 397;
  constexpr uint32_t UPPER = 0x80000000U, LOWER = 0x7fffffffU, MATRIX_A = 0x9908b0dfU;
  int i;
  uint32_t y;
  for (i = 0; i < N - M; ++i) {
    y = (mt_[i] & UPPER) | (mt_[i + 1] & LOWER);
    mt_[i] = mt_[i + M] ^ (y >> 1) ^ ((0U - (y & 1U)) & MATRIX_A);
  }
  for (; i < N - 1; ++i) {
    y = (mt_[i] & UPPER) | (mt_[i + 1] & LOWER);
    mt_[i] = mt_[i + (M - N)] ^ (y >> 1) ^ ((0U - (y & 1U)) & MATRIX_A);
  }
  y = (mt_[N - 1] & UPPER) | (mt_[0] & LOWER);
  mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ ((0U - (y & 1U)) & MATRIX_A);
}

uint32_t NumbaMt::next32() {
  if (index_ == 624) {
    shuffle();
    index_ = 0;
  }
  uint32_t y = mt_[index_++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= (y >> 18);
  return y;
}

int64_t NumbaMt::randint(int64_t lo, int64_t hi) {
  const int64_t n = hi - lo;
  if (n == 1) return lo;  // Numba's numpy branch draws nothing for n == 1
  const uint64_t nm1 = static_cast<uint64_t>(n - 1);
  const int nbits = 64 - __builtin_clzll(nm1);
  int64_t r;
  do {
    uint64_t v;
    if (nbits <= 32) {
      v = next32() & (0xffffffffU >> (32 - nbits));  // last nbits, like np.random
    } else {
      const uint64_t high = next32() & (0xffffffffU >> (64 - nbits));
      const uint64_t low = next32();
      v = low + (high << 32);
    }
    r = static_cast<int64_t>(v);
  } while (r >= n);
  return lo + r;
}

std::vector<int64_t> sample_without_replacement(NumbaMt& rng, int64_t n, int64_t k) {
  std::vector<uint8_t> seen(static_cast<std::size_t>(n), 0);
  std::vector<int64_t> out(static_cast<std::size_t>(k));
  int64_t t = 0;
  for (int64_t j = n - k; j < n; ++j) {
    int64_t r = rng.randint(0, j + 1);
    if (seen[r]) r = j;
    seen[r] = 1;
    out[t++] = r;
  }
  return out;
}

// ---------------------------------------------------------------------------
// decode + front end
// ---------------------------------------------------------------------------
void decode_xyz(const uint8_t* data, std::size_t n_points, std::size_t point_step,
                std::size_t ox, std::size_t oy, std::size_t oz, float* out) {
  for (std::size_t i = 0; i < n_points; ++i) {
    const uint8_t* p = data + i * point_step;
    std::memcpy(out + 3 * i + 0, p + ox, 4);
    std::memcpy(out + 3 * i + 1, p + oy, 4);
    std::memcpy(out + 3 * i + 2, p + oz, 4);
  }
}

template <typename T>
Cloud crop_promote(const T* xyz, std::size_t n, T crop2, T min2) {
  Cloud out(3 * n);
  std::size_t m = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const T x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    const T r2 = x * x + y * y;
    if (crop2 > 0 && !(r2 <= crop2)) continue;
    if (min2 > 0 && !(r2 + z * z >= min2)) continue;
    out[3 * m] = x;
    out[3 * m + 1] = y;
    out[3 * m + 2] = z;
    ++m;
  }
  out.resize(3 * m);
  return out;
}
template Cloud crop_promote<float>(const float*, std::size_t, float, float);
template Cloud crop_promote<double>(const double*, std::size_t, double, double);

Plane ransac_plane(const Cloud& xyz, double prob, double threshold, int64_t max_iter,
                   int64_t n_sub_max, const std::vector<double>& warm, NumbaMt& rng,
                   std::vector<uint8_t>& outliers) {
  constexpr int m = 3;
  const int64_t n = static_cast<int64_t>(xyz.size() / 3);
  std::vector<double> sx, sy, sz;
  int64_t ns;
  if (n > n_sub_max) {
    const auto sub = sample_without_replacement(rng, n, n_sub_max);
    ns = n_sub_max;
    sx.resize(ns);
    sy.resize(ns);
    sz.resize(ns);
    for (int64_t i = 0; i < ns; ++i) {
      const int64_t j = sub[i];
      sx[i] = xyz[3 * j];
      sy[i] = xyz[3 * j + 1];
      sz[i] = xyz[3 * j + 2];
    }
  } else {
    ns = n;
    sx.resize(ns);
    sy.resize(ns);
    sz.resize(ns);
    for (int64_t i = 0; i < ns; ++i) {
      sx[i] = xyz[3 * i];
      sy[i] = xyz[3 * i + 1];
      sz[i] = xyz[3 * i + 2];
    }
  }
  auto count = [&](double c0, double c1, double c2) {
    int64_t s = 0;
    for (int64_t i = 0; i < ns; ++i)
      if (std::abs(c0 + c1 * sx[i] + c2 * sy[i] - sz[i]) < threshold) ++s;
    return s;
  };
  double k = static_cast<double>(max_iter);
  int64_t support = 0;
  Plane best;
  auto refine = [&]() {
    const double r = static_cast<double>(support) / static_cast<double>(ns);
    if (0.0 < r && r < 1.0) {
      const double kn = std::log(1 - prob) / std::log(1 - cube(r));
      if (kn < k) k = kn;
    }
  };
  if (warm.size() == 4 && std::abs(warm[3]) > 1e-12) {
    const double c0 = warm[0] / -warm[3], c1 = warm[1] / -warm[3], c2 = warm[2] / -warm[3];
    const int64_t s = count(c0, c1, c2) - m;
    if (s > support) {
      support = s;
      best.coefs = {warm[0], warm[1], warm[2], warm[3]};
      refine();
    }
  }
  int64_t idx[m];
  int64_t it = 0;
  while (static_cast<double>(it) < k && it < max_iter) {
    for (int jj = 0; jj < m; ++jj) {
      while (true) {
        const int64_t cand = rng.randint(0, n);
        bool dup = false;
        for (int kk = 0; kk < jj; ++kk)
          if (idx[kk] == cand) {
            dup = true;
            break;
          }
        if (!dup) {
          idx[jj] = cand;
          break;
        }
      }
    }
    const double p0x = xyz[3 * idx[0]], p0y = xyz[3 * idx[0] + 1], p0z = xyz[3 * idx[0] + 2];
    const double v1x = xyz[3 * idx[1]] - p0x, v1y = xyz[3 * idx[1] + 1] - p0y,
                 v1z = xyz[3 * idx[1] + 2] - p0z;
    const double v2x = xyz[3 * idx[2]] - p0x, v2y = xyz[3 * idx[2] + 1] - p0y,
                 v2z = xyz[3 * idx[2] + 2] - p0z;
    double nx = v1y * v2z - v1z * v2y;
    double ny = v1z * v2x - v1x * v2z;
    double nz = v1x * v2y - v1y * v2x;
    const double nn = std::sqrt(nx * nx + ny * ny + nz * nz);
    nx /= nn;
    ny /= nn;
    nz /= nn;
    const double bias = -(nx * p0x + ny * p0y + nz * p0z);
    const int64_t s = count(bias / -nz, nx / -nz, ny / -nz) - m;
    if (s > support) {
      support = s;
      best.coefs = {bias, nx, ny, nz};
      refine();
    }
    ++it;
  }
  if (best.coefs[3] < 0)
    for (auto& c : best.coefs) c = -c;
  outliers = plane_outliers(xyz, best, threshold);
  return best;
}

std::vector<uint8_t> plane_outliers(const Cloud& xyz, const Plane& p, double threshold) {
  const double c0 = p.coefs[0] / -p.coefs[3], c1 = p.coefs[1] / -p.coefs[3],
               c2 = p.coefs[2] / -p.coefs[3];
  const std::size_t n = xyz.size() / 3;
  std::vector<uint8_t> out(n);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = !(std::abs(c0 + c1 * xyz[3 * i] + c2 * xyz[3 * i + 1] - xyz[3 * i + 2]) < threshold);
  return out;
}

Mat3 rot_to_normal(const Plane& p) {
  Mat3 R{1, 0, 0, 0, 1, 0, 0, 0, 1};
  const double w0 = p.coefs[1], w1 = p.coefs[2], w2 = p.coefs[3];
  const double nn = std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
  if (nn < 1e-12) return R;
  const double vx = w0 / nn, vy = w1 / nn, vz = w2 / nn;
  double ax = -vy, ay = vx;  // cross((0, 0, 1), v)
  const double s = std::sqrt(ax * ax + ay * ay);
  if (s == 0.0) {
    if (vz < 0)
      for (auto& r : R) r = -r;
    return R;
  }
  const double ang = std::acos(std::min(1.0, std::max(-1.0, vz)));
  ax /= s;
  ay /= s;
  const double c = std::cos(ang), sn = std::sin(ang), t = 1 - c;
  R = {c + ax * ax * t, ax * ay * t,     ay * sn,
       ax * ay * t,     c + ay * ay * t, -ax * sn,
       -ay * sn,        ax * sn,         c};
  return R;
}

Cloud select_rotate(const Cloud& xyz, const std::vector<uint8_t>& outl, const Mat3& R,
                    double zshift) {
  std::size_t m = 0;
  for (auto o : outl) m += o ? 1 : 0;
  Cloud out(3 * m);
  std::size_t j = 0;
  for (std::size_t i = 0; i < outl.size(); ++i) {
    if (!outl[i]) continue;
    const double x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    out[3 * j] = x * R[0] + y * R[3] + z * R[6];
    out[3 * j + 1] = x * R[1] + y * R[4] + z * R[7];
    out[3 * j + 2] = (x * R[2] + y * R[5] + z * R[8]) - zshift;
    ++j;
  }
  return out;
}

int64_t column_veto(Cloud& pts, double h_veto, double cell) {
  const std::size_t n = pts.size() / 3;
  std::vector<int64_t> ci(n), cj(n);
  int64_t ntall = 0;
  int64_t imin = int64_t(1) << 62, jmin = imin, imax = -(int64_t(1) << 62), jmax = imax;
  for (std::size_t k = 0; k < n; ++k) {
    const int64_t i = static_cast<int64_t>(std::floor(pts[3 * k] / cell));
    const int64_t j = static_cast<int64_t>(std::floor(pts[3 * k + 1] / cell));
    ci[k] = i;
    cj[k] = j;
    imin = std::min(imin, i);
    imax = std::max(imax, i);
    jmin = std::min(jmin, j);
    jmax = std::max(jmax, j);
    if (pts[3 * k + 2] > h_veto) ++ntall;
  }
  if (ntall == 0) return 0;
  const int64_t W = jmax - jmin + 3;
  std::vector<uint8_t> veto(static_cast<std::size_t>((imax - imin + 3) * W), 0);  // 1-cell pad
  for (std::size_t k = 0; k < n; ++k) {
    if (!(pts[3 * k + 2] > h_veto)) continue;
    const int64_t a = ci[k] - imin + 1, b = cj[k] - jmin + 1;
    for (int da = -1; da <= 1; ++da)
      for (int db = -1; db <= 1; ++db) veto[(a + da) * W + (b + db)] = 1;
  }
  std::size_t m = 0;
  for (std::size_t k = 0; k < n; ++k) {
    if (veto[(ci[k] - imin + 1) * W + (cj[k] - jmin + 1)]) continue;
    if (m != k) {
      pts[3 * m] = pts[3 * k];
      pts[3 * m + 1] = pts[3 * k + 1];
      pts[3 * m + 2] = pts[3 * k + 2];
    }
    ++m;
  }
  pts.resize(3 * m);
  return static_cast<int64_t>(n - m);
}

std::vector<int64_t> voxel_first(const Cloud& pts, double voxel, int64_t offset, int bits) {
  const std::size_t n = pts.size() / 3;
  std::size_t cap = 1;
  int lg = 0;
  while (cap < 2 * n) {
    cap <<= 1;
    ++lg;
  }
  const unsigned sh = 64U - static_cast<unsigned>(std::max(lg, 1));
  std::vector<int64_t> table(cap, -1);  // every key is >= 0
  std::vector<int64_t> keep;
  keep.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    const int64_t a = static_cast<int64_t>(std::floor(pts[3 * k] / voxel)) + offset;
    const int64_t b = static_cast<int64_t>(std::floor(pts[3 * k + 1] / voxel)) + offset;
    const int64_t c = static_cast<int64_t>(std::floor(pts[3 * k + 2] / voxel)) + offset;
    const int64_t key = (a << (2 * bits)) | (b << bits) | c;
    std::size_t slot =
        static_cast<std::size_t>((static_cast<uint64_t>(key) * 0x9E3779B97F4A7C15ULL) >> sh) &
        (cap - 1);
    while (true) {
      if (table[slot] == -1) {
        table[slot] = key;
        keep.push_back(static_cast<int64_t>(k));
        break;
      }
      if (table[slot] == key) break;
      slot = (slot + 1) & (cap - 1);
    }
  }
  return keep;
}

void bound_cluster_input_voxel(Cloud& pts, int64_t cap, double voxel0, double voxel_max) {
  constexpr int64_t kOffset = int64_t(1) << 20;  // cone_detection._GUARD_KEY_OFFSET
  constexpr int kBits = 21;                      // cone_detection._GUARD_KEY_BITS
  if (cap <= 0 || static_cast<int64_t>(pts.size() / 3) <= cap) return;
  double voxel = voxel0;
  while (voxel > 0.0 && voxel <= voxel_max + 1e-9 && static_cast<int64_t>(pts.size() / 3) > cap) {
    const auto keep = voxel_first(pts, voxel, kOffset, kBits);
    for (std::size_t m = 0; m < keep.size(); ++m) {  // keep is ascending: in-place is safe
      const std::size_t k = static_cast<std::size_t>(keep[m]);
      pts[3 * m] = pts[3 * k];
      pts[3 * m + 1] = pts[3 * k + 1];
      pts[3 * m + 2] = pts[3 * k + 2];
    }
    pts.resize(3 * keep.size());
    voxel *= 2.0;
  }
}

// ---------------------------------------------------------------------------
// clustering._grid_components
// ---------------------------------------------------------------------------
namespace {
inline int64_t find(std::vector<int64_t>& parent, int64_t x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}
}  // namespace

std::vector<int64_t> grid_dbscan(const Cloud& pts, double eps) {
  const std::size_t n = pts.size() / 3;
  if (n == 0) return {};
  const double side = eps / 1.7320508075688772 * (1.0 - 1e-9);
  std::vector<int64_t> cell(3 * n);
  for (std::size_t i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) cell[3 * i + d] = static_cast<int64_t>(std::floor(pts[3 * i + d] / side));
  int64_t lo[3], dim[3];
  for (int d = 0; d < 3; ++d) {
    int64_t mn = cell[d], mx = cell[d];
    for (std::size_t i = 1; i < n; ++i) {
      mn = std::min(mn, cell[3 * i + d]);
      mx = std::max(mx, cell[3 * i + d]);
    }
    lo[d] = mn;
    dim[d] = mx - mn + 5;  // 2 cubes of padding each side
  }
  std::vector<int64_t> keys(n), order(n);
  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = ((cell[3 * i] - lo[0] + 2) * dim[1] + (cell[3 * i + 1] - lo[1] + 2)) * dim[2] +
              (cell[3 * i + 2] - lo[2] + 2);
    order[i] = static_cast<int64_t>(i);
  }
  // Order inside a cube does not change the labels; only the cube order does.
  std::sort(order.begin(), order.end(),
            [&](int64_t a, int64_t b) { return keys[a] < keys[b] || (keys[a] == keys[b] && a < b); });
  std::vector<int64_t> starts, ckeys;
  starts.reserve(n + 1);
  ckeys.reserve(n);
  for (std::size_t k = 0; k < n; ++k) {
    if (k == 0 || keys[order[k]] != keys[order[k - 1]]) {
      starts.push_back(static_cast<int64_t>(k));
      ckeys.push_back(keys[order[k]]);
    }
  }
  const int64_t m = static_cast<int64_t>(ckeys.size());
  starts.push_back(static_cast<int64_t>(n));
  int64_t deltas[124];
  int nd = 0;
  for (int dx = -2; dx <= 2; ++dx)
    for (int dy = -2; dy <= 2; ++dy)
      for (int dz = -2; dz <= 2; ++dz) {
        const int64_t delta = (dx * dim[1] + dy) * dim[2] + dz;
        if (delta > 0) deltas[nd++] = delta;
      }
  std::vector<int64_t> parent(m);
  for (int64_t a = 0; a < m; ++a) parent[a] = a;
  std::vector<uint8_t> has_nbr(m, 0);
  const double eps2 = eps * eps;
  for (int64_t a = 0; a < m; ++a) {
    for (int t = 0; t < nd; ++t) {
      const int64_t want = ckeys[a] + deltas[t];
      const int64_t b = std::lower_bound(ckeys.begin(), ckeys.end(), want) - ckeys.begin();
      if (b >= m || ckeys[b] != want) continue;
      const int64_t ra = find(parent, a), rb = find(parent, b);
      if (ra == rb) continue;
      bool found = false;
      for (int64_t ia = starts[a]; ia < starts[a + 1] && !found; ++ia) {
        const int64_t p = order[ia];
        for (int64_t ib = starts[b]; ib < starts[b + 1]; ++ib) {
          const int64_t q = order[ib];
          const double dx = pts[3 * p] - pts[3 * q];
          const double dy = pts[3 * p + 1] - pts[3 * q + 1];
          const double dz = pts[3 * p + 2] - pts[3 * q + 2];
          if (dx * dx + dy * dy + dz * dz <= eps2) {
            found = true;
            break;
          }
        }
      }
      if (found) {
        parent[rb] = ra;
        has_nbr[a] = 1;
        has_nbr[b] = 1;
      }
    }
  }
  std::vector<int64_t> labels(n, -1), cid(m, -1);
  int64_t nxt = 0;
  for (int64_t a = 0; a < m; ++a) {
    if (starts[a + 1] - starts[a] >= 2 || has_nbr[a]) {
      const int64_t r = find(parent, a);
      if (cid[r] < 0) cid[r] = nxt++;
      for (int64_t k = starts[a]; k < starts[a + 1]; ++k) labels[order[k]] = cid[r];
    }
  }
  return labels;
}

// ---------------------------------------------------------------------------
// back end: fused.fit_clusters and its helpers
// ---------------------------------------------------------------------------
namespace {
std::array<double, 2> geometric(const double* x, const double* y, std::size_t n) {
  double cx = 0.0, cy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    cx += x[i];
    cy += y[i];
  }
  cx /= static_cast<double>(n);
  cy /= static_cast<double>(n);
  const double nr = std::hypot(cx, cy);
  if (nr < 1e-6) return {cx, cy};
  const double off = kNearFaceBiasFrac * kConeBaseRadius;
  return {cx + off * cx / nr, cy + off * cy / nr};
}

double linearity(const double* x, const double* y, std::size_t n) {
  if (n < 2) return 0.0;
  const double mx = mean(x, n), my = mean(y, n);
  double sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - mx, dy = y[i] - my;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const double nn = static_cast<double>(n);
  sxx /= nn;
  sxy /= nn;
  syy /= nn;
  const double tr = 0.5 * (sxx + syy);
  const double dd = sxx - syy;
  const double disc = std::sqrt(std::max(0.0, 0.25 * (dd * dd) + sxy * sxy));
  return (tr - disc) / (tr + disc + 1e-12);
}

double mse(double a, double b, double c, double d, const double* x, const double* y,
           const double* z, std::size_t n) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - a, dy = y[i] - b;
    const double e = z[i] - (d - c * std::sqrt(dx * dx + dy * dy));
    s += e * e;
  }
  return s / static_cast<double>(n);
}

// {a, b, c, d}; c = NaN means "no collinear fit" (geometric a, b).
std::array<double, 4> collinear(const double* x, const double* y, const double* z,
                                std::size_t n, std::vector<double>& t) {
  auto fallback = [&]() {
    const auto g = geometric(x, y, n);
    return std::array<double, 4>{g[0], g[1], kNan, kNan};
  };
  const double cx = mean(x, n), cy = mean(y, n);
  const double nr = std::hypot(cx, cy);
  if (nr < 1e-6 || n < 3) return fallback();
  const double ux = cx / nr, uy = cy / nr;
  t.resize(n);
  for (std::size_t i = 0; i < n; ++i) t[i] = x[i] * ux + y[i] * uy;
  const double tm = mean(t.data(), n), zm = mean(z, n);
  double stt = 0.0, stz = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    stt += (t[i] - tm) * (t[i] - tm);
    stz += (t[i] - tm) * (z[i] - zm);
  }
  if (stt < 1e-6) return fallback();
  const double ce = stz / stt;
  if (ce <= 0.0) return fallback();
  const double es = std::abs(ce - kSmallC), eb = std::abs(ce - kBigC);
  double cu, dc;
  if (es <= eb) {
    if (es > kClassTolerance) return fallback();
    cu = kSmallC;
    dc = kSmallD;
  } else {
    if (eb > kClassTolerance) return fallback();
    cu = kBigC;
    dc = kBigD;
  }
  const double ta = tm + (dc - zm) / cu;
  const double off = ta - tm;
  if (off < 0.0 || off > 2.0 * kConeBaseRadius + 0.05) return fallback();
  return {ta * ux, ta * uy, cu, dc};
}

std::array<double, 3> template_fit(const double* x, const double* y, const double* z,
                                   std::size_t n, double c, double d, int64_t maxiter) {
  auto g = geometric(x, y, n);
  double a0 = g[0], b0 = g[1];
  if (!(std::isfinite(a0) && std::isfinite(b0))) {
    a0 = mean(x, n);
    b0 = mean(y, n);
  }
  const double alo = vmin(x, n) - kAbBoundMargin, ahi = vmax(x, n) + kAbBoundMargin;
  const double blo = vmin(y, n) - kAbBoundMargin, bhi = vmax(y, n) + kAbBoundMargin;
  a0 = std::min(std::max(a0, alo), ahi);
  b0 = std::min(std::max(b0, blo), bhi);
  return template_fit_gn(x, y, z, n, c, d, a0, b0, alo, ahi, blo, bhi, maxiter);
}
}  // namespace

std::array<double, 3> template_fit_gn(const double* x, const double* y, const double* z,
                                      std::size_t n, double c, double d, double a0, double b0,
                                      double a_lo, double a_hi, double b_lo, double b_hi,
                                      int64_t max_iter) {
  double a = a0, b = b0, loss = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - a, dy = y[i] - b;
    const double r = std::sqrt(dx * dx + dy * dy);
    const double e = z[i] - (d - c * r);
    loss += e * e;
  }
  double lam = 1e-8;
  for (int64_t it = 0; it < max_iter; ++it) {
    double h00 = 0.0, h01 = 0.0, h11 = 0.0, g0 = 0.0, g1 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double dx = x[i] - a, dy = y[i] - b;
      double r = std::sqrt(dx * dx + dy * dy);
      if (r < 1e-12) r = 1e-12;
      const double e = z[i] - (d - c * r);
      const double ja = -c * dx / r, jb = -c * dy / r;
      h00 += ja * ja;
      h01 += ja * jb;
      h11 += jb * jb;
      g0 += ja * e;
      g1 += jb * e;
    }
    double step_a = 0.0, step_b = 0.0;
    bool improved = false;
    for (int tr = 0; tr < 8; ++tr) {
      const double d00 = h00 * (1.0 + lam), d11 = h11 * (1.0 + lam);
      const double det = d00 * d11 - h01 * h01;
      if (det <= 1e-300) break;
      const double da = -(d11 * g0 - h01 * g1) / det;
      const double db = -(d00 * g1 - h01 * g0) / det;
      const double a_new = std::min(std::max(a + da, a_lo), a_hi);
      const double b_new = std::min(std::max(b + db, b_lo), b_hi);
      double loss_new = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double dx = x[i] - a_new, dy = y[i] - b_new;
        const double r = std::sqrt(dx * dx + dy * dy);
        const double e = z[i] - (d - c * r);
        loss_new += e * e;
      }
      if (loss_new < loss) {
        step_a = a_new - a;
        step_b = b_new - b;
        a = a_new;
        b = b_new;
        loss = loss_new;
        lam = std::max(lam / 10.0, 1e-12);
        improved = true;
        break;
      }
      lam *= 10.0;
    }
    if (!improved) break;
    if (std::abs(step_a) < 1e-7 && std::abs(step_b) < 1e-7) break;
  }
  return {a, b, loss / static_cast<double>(n)};
}

FitResult fit_clusters(const Cloud& pts, const std::vector<int64_t>& labels,
                       const std::array<double, 11>& p) {
  FitResult res;
  auto& cnt = res.counters;
  const std::size_t n = labels.size();
  if (n == 0) return res;
  // Stable counting sort by label (= np.argsort(labels, kind="mergesort")).
  int64_t lmin = labels[0], lmax = labels[0];
  for (auto l : labels) {
    lmin = std::min(lmin, l);
    lmax = std::max(lmax, l);
  }
  std::vector<std::size_t> bucket(static_cast<std::size_t>(lmax - lmin + 2), 0);
  for (auto l : labels) ++bucket[l - lmin + 1];
  for (std::size_t i = 1; i < bucket.size(); ++i) bucket[i] += bucket[i - 1];
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) order[bucket[labels[i] - lmin]++] = i;

  std::vector<double> x, y, z, tbuf;
  x.reserve(n);
  y.reserve(n);
  z.reserve(n);
  std::size_t s = 0;
  while (s < n) {
    const int64_t lab = labels[order[s]];
    std::size_t e = s;
    while (e < n && labels[order[e]] == lab) ++e;
    ++cnt[N_CLUSTERS];
    const std::size_t size = e - s, st = s;
    s = e;
    if (static_cast<double>(size) < p[0]) continue;
    ++cnt[AFTER_MIN_PTS];
    const double thr = p[1];
    x.clear();
    y.clear();
    z.clear();
    for (std::size_t q = st; q < st + size; ++q) {
      const std::size_t r = order[q];
      if (pts[3 * r + 2] > thr) {
        x.push_back(pts[3 * r]);
        y.push_back(pts[3 * r + 1]);
        z.push_back(pts[3 * r + 2]);
      }
    }
    const std::size_t m = x.size();
    if (m == 0) continue;
    const double* X = x.data();
    const double* Y = y.data();
    const double* Z = z.data();
    const double h = vmax(Z, m) - vmin(Z, m);
    const double rng = std::hypot(mean(X, m), mean(Y, m));
    if (!(p[2] <= h && h <= p[3] && rng <= p[4])) {
      ++cnt[SHAPE_REJECTED];
      if (rng > p[4]) ++cnt[FAR_DROPPED];
      continue;
    }
    ++cnt[AFTER_SHAPE_GATE];
    bool done = false;
    double a = 0.0, b = 0.0, d = 0.0, rs = 0.0, other = kInf;
    bool big = false;
    if (p[10] != 0.0 && linearity(X, Y, m) < kLinearityThreshold) {
      const auto col = collinear(X, Y, Z, m, tbuf);
      if (std::isfinite(col[2])) {
        a = col[0];
        b = col[1];
        rs = mse(a, b, col[2], col[3], X, Y, Z, m);
        other = kNan;
        big = col[3] >= kBigD - 1e-6;
        d = col[3];
        done = true;
      }
    }
    if (!done) {
      const int64_t mi = static_cast<int64_t>(p[9]);
      if (h < p[7]) {
        const auto f = template_fit(X, Y, Z, m, kSmallC, kSmallD, mi);
        a = f[0]; b = f[1]; rs = f[2]; d = kSmallD; big = false; other = kInf;
      } else if (h > p[8]) {
        const auto f = template_fit(X, Y, Z, m, kBigC, kBigD, mi);
        a = f[0]; b = f[1]; rs = f[2]; d = kBigD; big = true; other = kInf;
      } else {
        const auto fs = template_fit(X, Y, Z, m, kSmallC, kSmallD, mi);
        const auto fb = template_fit(X, Y, Z, m, kBigC, kBigD, mi);
        if (fs[2] <= fb[2]) {
          a = fs[0]; b = fs[1]; rs = fs[2]; d = kSmallD; big = false; other = fb[2];
        } else {
          a = fb[0]; b = fb[1]; rs = fb[2]; d = kBigD; big = true; other = fs[2];
        }
      }
    }
    if (!(std::isfinite(rs) && rs <= p[5])) {
      ++cnt[RESIDUAL_REJECTED];
      continue;
    }
    const bool amb = std::isfinite(other) && other / std::max(rs, 1e-12) < p[6];
    if (amb) ++cnt[AMBIGUOUS];
    if (big) ++cnt[BIG]; else ++cnt[SMALL];
    ++cnt[ACCEPTED];
    double sig = 1.5 * (0.05 + 0.005 * std::hypot(a, b)) /
                 std::sqrt(static_cast<double>(std::max<std::size_t>(1, m)) / 10.0);
    if (amb) sig *= 1.5;
    res.rows.insert(res.rows.end(), {a, b, d, sig, rs});
  }
  return res;
}

}  // namespace cone_detection_core
