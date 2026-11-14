// C++ port of the fused cone-detection kernels (cone_detection/fused.py).
//
// Every function is a 1:1 port of the Numba kernel of the same name, with the
// same loop order and the same floating-point expression order, so the output
// matches the fused backend to float rounding (the golden test checks the plane
// to 1e-9 and the cones to 1e-6). Build without -ffast-math and with
// -ffp-contract=off: fused multiply-adds change the rounding.
//
// RANSAC draws from NumbaMt, a copy of Numba's np.random state (MT19937 with
// Numba's seeding and its rejection-sampled randint), so seeding it with the same
// value replays the same samples as the Numba kernels.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cone_detection_core {

// ---------------------------------------------------------------------------
// RNG: Numba's np.random (numba/_random.c + randomimpl._randrange_impl)
// ---------------------------------------------------------------------------
class NumbaMt {
 public:
  NumbaMt();                         // seeded from std::random_device
  explicit NumbaMt(uint32_t seed) { seed_(seed); }
  void seed_(uint32_t seed);         // = numba_rnd_init (np.random.seed in njit)
  uint32_t next32();                 // = get_next_int32
  int64_t randint(int64_t lo, int64_t hi);  // = np.random.randint(lo, hi), int64

 private:
  void shuffle();
  std::array<uint32_t, 624> mt_{};
  int index_ = 624;
};

// k distinct indices in [0, n), Floyd's algorithm (= ransac.sample_without_replacement)
std::vector<int64_t> sample_without_replacement(NumbaMt& rng, int64_t n, int64_t k);

// ---------------------------------------------------------------------------
// Point clouds are row-major (n, 3) arrays of double unless stated otherwise.
// ---------------------------------------------------------------------------
using Cloud = std::vector<double>;  // size 3 * n

// PointCloud2 decode: x/y/z float32 at byte offsets ox/oy/oz of a point_step stride.
void decode_xyz(const uint8_t* data, std::size_t n_points, std::size_t point_step,
                std::size_t ox, std::size_t oy, std::size_t oz, float* out);

// Horizontal crop + min-range drop in the input dtype, then float64 (= crop_promote).
template <typename T>
Cloud crop_promote(const T* xyz, std::size_t n, T crop2, T min2);

struct Plane {
  std::array<double, 4> coefs{0.0, 0.0, 0.0, 1.0};  // [bias, nx, ny, nz]
};

// ransac.ransac2 (m = 3) without the augmented copy (= fused.ransac_plane).
// warm: empty or 4 coefs. outliers gets one flag per point.
Plane ransac_plane(const Cloud& xyz, double prob, double threshold, int64_t max_iter,
                   int64_t n_sub_max, const std::vector<double>& warm, NumbaMt& rng,
                   std::vector<uint8_t>& outliers);

// ground_methods.ray_slope_kernel: per-azimuth ray walk on height above a GPF
// prior, then a plane refit on the ground returns. outliers gets one flag per point.
Plane ray_slope_ground(const Cloud& xyz, double thr, double max_slope, double max_gap,
                       double max_height, int64_t n_sectors, int64_t fit_cap,
                       std::vector<uint8_t>& outliers);

std::vector<uint8_t> plane_outliers(const Cloud& xyz, const Plane& p, double threshold);

using Mat3 = std::array<double, 9>;  // row-major
Mat3 rot_to_normal(const Plane& p);

// xyz[outl] @ R, then z -= zshift (= select_rotate).
Cloud select_rotate(const Cloud& xyz, const std::vector<uint8_t>& outl, const Mat3& R,
                    double zshift);

// Tall-column veto on height above ground. Returns the number of points dropped.
int64_t column_veto(Cloud& pts, double h_veto, double cell);

// Ascending indices of the first point of every voxel (= voxel_first).
std::vector<int64_t> voxel_first(const Cloud& pts, double voxel, int64_t offset, int bits);

// Voxel stage of _bound_dbscan_input. The seeded random fallback (when the cloud is
// still above cap at the largest voxel) stays on the Python side.
void bound_cluster_input_voxel(Cloud& pts, int64_t cap, double voxel0, double voxel_max);

// DBSCAN(eps, min_samples=2) as connected components (= clustering._grid_components).
std::vector<int64_t> grid_dbscan(const Cloud& pts, double eps);

// fit_clusters counter slots (same as fused.py).
enum Counter : int {
  N_CLUSTERS = 0, AFTER_MIN_PTS, AFTER_SHAPE_GATE, ACCEPTED, FAR_DROPPED,
  SMALL, BIG, AMBIGUOUS, SHAPE_REJECTED, RESIDUAL_REJECTED, N_COUNTERS
};

// p = [min_pts, floor_margin, hmin, hmax, rmax, res_gate, amb_ratio,
//      h_small, h_big, maxiter, use_collinear]  (= fused.fit_params)
struct FitResult {
  std::vector<double> rows;  // (K, 5) [a, b, d, sigma_xy, residual]
  std::array<int64_t, N_COUNTERS> counters{};
};
FitResult fit_clusters(const Cloud& pts, const std::vector<int64_t>& labels,
                       const std::array<double, 11>& p);

// cone_fit._template_fit_gn: returns {a, b, mse}.
std::array<double, 3> template_fit_gn(const double* x, const double* y, const double* z,
                                      std::size_t n, double c, double d, double a0, double b0,
                                      double a_lo, double a_hi, double b_lo, double b_hi,
                                      int64_t max_iter);

}  // namespace cone_detection_core
