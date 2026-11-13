// pybind11 module `cone_detection_core`: the fused kernels for
// cone_detection's detector_backend="cpp". Every compute call releases the GIL.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

#include "cone_detection_core/core.hpp"

namespace py = pybind11;
namespace cdc = cone_detection_core;

namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Hand a vector to numpy without copying it.
template <typename T>
py::array_t<T> to_np(std::vector<T>&& v, std::vector<py::ssize_t> shape) {
  auto* owned = new std::vector<T>(std::move(v));
  py::capsule owner(owned, [](void* p) { delete static_cast<std::vector<T>*>(p); });
  return py::array_t<T>(shape, owned->data(), owner);
}

cdc::Cloud cloud_from(const py::array_t<double, py::array::c_style | py::array::forcecast>& a) {
  if (a.ndim() != 2 || a.shape(1) != 3) throw std::invalid_argument("expected an (N, 3) array");
  return cdc::Cloud(a.data(), a.data() + a.size());
}

std::vector<double> vec_from(const std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>& a) {
  if (!a) return {};
  return std::vector<double>(a->data(), a->data() + a->size());
}

struct FrontOut {
  std::size_t n_cropped = 0;
  cdc::Cloud cropped;
  cdc::Plane plane;
  cdc::Cloud clean;
  std::size_t n_outliers = 0;
  int64_t n_vetoed = -1;  // -1: the veto did not run
  std::size_t n_before_guard = 0;
  double crop_ms = 0, ransac_ms = 0, rotate_ms = 0, veto_ms = 0, guard_ms = 0;
};

// fused.run_scan up to the cluster input: crop, RANSAC, rotate, veto, voxel guard.
template <typename T>
FrontOut front_impl(const T* data, std::size_t n, T crop2, T min2, const std::vector<double>& warm,
                    double prob, double thr, int64_t n_sub_max, bool veto, double veto_h,
                    double veto_cell, int64_t cap, double voxel0, double voxel_max,
                    bool keep_cropped, cdc::NumbaMt& rng) {
  FrontOut o;
  auto t = Clock::now();
  cdc::Cloud pts = cdc::crop_promote<T>(data, n, crop2, min2);
  o.crop_ms = ms_since(t);
  o.n_cropped = pts.size() / 3;
  if (o.n_cropped == 0) return o;
  t = Clock::now();
  std::vector<uint8_t> outl;
  if (o.n_cropped < 3) {
    outl.assign(o.n_cropped, 1);
  } else {
    o.plane = cdc::ransac_plane(pts, prob, thr, 200, n_sub_max, warm, rng, outl);
  }
  o.ransac_ms = ms_since(t);
  t = Clock::now();
  const auto& c = o.plane.coefs;
  const double nrm = std::sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3]);
  const double floor = nrm < 1e-12 ? 0.0 : (-c[0] * c[3]) / nrm;  // dot([0, 0, -bias], w) / |w|
  o.clean = cdc::select_rotate(pts, outl, cdc::rot_to_normal(o.plane), floor);
  o.rotate_ms = ms_since(t);
  o.n_outliers = o.clean.size() / 3;
  if (keep_cropped) o.cropped = std::move(pts);
  if (o.n_outliers == 0) return o;
  if (veto) {
    t = Clock::now();
    o.n_vetoed = cdc::column_veto(o.clean, veto_h, veto_cell);
    o.veto_ms = ms_since(t);
    if (o.clean.empty()) return o;
  }
  o.n_before_guard = o.clean.size() / 3;
  t = Clock::now();
  cdc::bound_cluster_input_voxel(o.clean, cap, voxel0, voxel_max);
  o.guard_ms = ms_since(t);
  return o;
}

class Core {
 public:
  Core() = default;
  void seed(uint32_t s) { rng_.seed_(s); }
  cdc::NumbaMt& rng() { return rng_; }

  py::tuple front(py::array data, std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> warm,
                  double crop2, double min2, double prob, double thr, int64_t n_sub_max,
                  bool veto, double veto_h, double veto_cell, int64_t cap, double voxel0,
                  double voxel_max, bool keep_cropped) {
    if (data.ndim() != 2 || data.shape(1) != 3) throw std::invalid_argument("expected an (N, 3) array");
    const auto w = vec_from(warm);
    FrontOut o;
    if (py::isinstance<py::array_t<float>>(data)) {
      auto a = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(data);
      const float* p = a.data();
      const std::size_t n = static_cast<std::size_t>(a.shape(0));
      py::gil_scoped_release nogil;
      o = front_impl<float>(p, n, static_cast<float>(crop2), static_cast<float>(min2), w, prob,
                            thr, n_sub_max, veto, veto_h, veto_cell, cap, voxel0, voxel_max,
                            keep_cropped, rng_);
    } else if (py::isinstance<py::array_t<double>>(data)) {
      auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(data);
      const double* p = a.data();
      const std::size_t n = static_cast<std::size_t>(a.shape(0));
      py::gil_scoped_release nogil;
      o = front_impl<double>(p, n, crop2, min2, w, prob, thr, n_sub_max, veto, veto_h,
                             veto_cell, cap, voxel0, voxel_max, keep_cropped, rng_);
    } else {
      throw std::invalid_argument("expected float32 or float64 points");
    }
    py::dict timings;
    timings["crop_ms"] = o.crop_ms;
    timings["ransac_ms"] = o.ransac_ms;
    timings["rotate_ms"] = o.rotate_ms;
    timings["column_veto_ms"] = o.veto_ms;
    timings["guard_ms"] = o.guard_ms;
    const auto c = o.plane.coefs;
    py::object cropped = py::none();
    if (keep_cropped) {
      const auto nc = static_cast<py::ssize_t>(o.cropped.size() / 3);
      cropped = to_np(std::move(o.cropped), {nc, 3});
    }
    const auto m = static_cast<py::ssize_t>(o.clean.size() / 3);
    return py::make_tuple(o.n_cropped, cropped,
                          to_np(std::vector<double>{c[0], c[1], c[2], c[3]}, {4}),
                          to_np(std::move(o.clean), {m, 3}), o.n_outliers, o.n_vetoed,
                          o.n_before_guard, timings);
  }

  // Cluster (grid, unless labels are given) and fit.
  py::tuple back(const py::array_t<double, py::array::c_style | py::array::forcecast>& pts,
                 std::optional<py::array_t<int64_t, py::array::c_style | py::array::forcecast>> labels_in,
                 double eps, const std::array<double, 11>& params) {
    cdc::Cloud cloud = cloud_from(pts);
    std::vector<int64_t> labels;
    if (labels_in) labels.assign(labels_in->data(), labels_in->data() + labels_in->size());
    const bool have = labels_in.has_value();
    double dbscan_ms = 0.0, fit_ms = 0.0;
    cdc::FitResult fit;
    {
      py::gil_scoped_release nogil;
      auto t = Clock::now();
      if (!have) labels = cdc::grid_dbscan(cloud, eps);
      dbscan_ms = ms_since(t);
      t = Clock::now();
      fit = cdc::fit_clusters(cloud, labels, params);
      fit_ms = ms_since(t);
    }
    const auto nl = static_cast<py::ssize_t>(labels.size());
    const auto k = static_cast<py::ssize_t>(fit.rows.size() / 5);
    std::vector<int64_t> cnt(fit.counters.begin(), fit.counters.end());
    return py::make_tuple(to_np(std::move(labels), {nl}), to_np(std::move(fit.rows), {k, 5}),
                          to_np(std::move(cnt), {static_cast<py::ssize_t>(cdc::N_COUNTERS)}),
                          dbscan_ms, fit_ms);
  }

 private:
  cdc::NumbaMt rng_;
};

}  // namespace

PYBIND11_MODULE(cone_detection_core, m) {
  m.doc() = "C++ port of the fused cone-detection kernels (cone_detection/fused.py)";

  py::class_<Core>(m, "Core")
      .def(py::init<>())
      .def("seed", &Core::seed, py::arg("seed"))
      .def("front", &Core::front, py::arg("data"), py::arg("warm"), py::arg("crop2"),
           py::arg("min2"), py::arg("prob"), py::arg("threshold"), py::arg("n_sub_max"),
           py::arg("veto"), py::arg("veto_h"), py::arg("veto_cell"), py::arg("cap"),
           py::arg("voxel0"), py::arg("voxel_max"), py::arg("keep_cropped"))
      .def("back", &Core::back, py::arg("pts"), py::arg("labels"), py::arg("eps"),
           py::arg("params"))
      .def("randint", [](Core& c, int64_t lo, int64_t hi) { return c.rng().randint(lo, hi); });

  m.def("decode_xyz",
        [](py::buffer buf, std::size_t n_points, std::size_t point_step, std::size_t ox,
           std::size_t oy, std::size_t oz) {
          py::buffer_info info = buf.request();
          const std::size_t nbytes = static_cast<std::size_t>(info.size * info.itemsize);
          if (n_points && (point_step * (n_points - 1) + std::max({ox, oy, oz}) + 4 > nbytes))
            throw std::invalid_argument("PointCloud2 buffer shorter than width*height*point_step");
          py::array_t<float> out({static_cast<py::ssize_t>(n_points), py::ssize_t(3)});
          const auto* src = static_cast<const uint8_t*>(info.ptr);
          float* dst = out.mutable_data();
          {
            py::gil_scoped_release nogil;
            cdc::decode_xyz(src, n_points, point_step, ox, oy, oz, dst);
          }
          return out;
        },
        py::arg("data"), py::arg("n_points"), py::arg("point_step"), py::arg("ox"),
        py::arg("oy"), py::arg("oz"));

  m.def("grid_dbscan",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& pts, double eps) {
          auto c = cloud_from(pts);
          auto l = cdc::grid_dbscan(c, eps);
          const auto n = static_cast<py::ssize_t>(l.size());
          return to_np(std::move(l), {n});
        });
  m.def("voxel_first",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& pts, double voxel,
           int64_t offset, int bits) {
          auto k = cdc::voxel_first(cloud_from(pts), voxel, offset, bits);
          const auto n = static_cast<py::ssize_t>(k.size());
          return to_np(std::move(k), {n});
        });
  m.def("column_veto",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& pts, double h,
           double cell) {
          auto c = cloud_from(pts);
          const int64_t dropped = cdc::column_veto(c, h, cell);
          const auto n = static_cast<py::ssize_t>(c.size() / 3);
          return py::make_tuple(to_np(std::move(c), {n, 3}), dropped);
        });
  m.def("ransac_plane",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& pts, double prob,
           double thr, int64_t max_iter, int64_t n_sub_max,
           std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> warm,
           uint32_t seed) {
          cdc::NumbaMt rng(seed);
          std::vector<uint8_t> outl;
          auto p = cdc::ransac_plane(cloud_from(pts), prob, thr, max_iter, n_sub_max,
                                     vec_from(warm), rng, outl);
          std::vector<bool> o(outl.begin(), outl.end());
          return py::make_tuple(py::array(py::cast(o)),
                                to_np(std::vector<double>(p.coefs.begin(), p.coefs.end()), {4}));
        });
  m.def("sample_without_replacement", [](uint32_t seed, int64_t n, int64_t k) {
    cdc::NumbaMt rng(seed);
    auto v = cdc::sample_without_replacement(rng, n, k);
    const auto s = static_cast<py::ssize_t>(v.size());
    return to_np(std::move(v), {s});
  });
  m.def("fit_clusters",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& pts,
           const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& labels,
           const std::array<double, 11>& params) {
          std::vector<int64_t> l(labels.data(), labels.data() + labels.size());
          auto f = cdc::fit_clusters(cloud_from(pts), l, params);
          const auto k = static_cast<py::ssize_t>(f.rows.size() / 5);
          std::vector<int64_t> cnt(f.counters.begin(), f.counters.end());
          return py::make_tuple(to_np(std::move(f.rows), {k, 5}),
                                to_np(std::move(cnt), {static_cast<py::ssize_t>(cdc::N_COUNTERS)}));
        });
}
