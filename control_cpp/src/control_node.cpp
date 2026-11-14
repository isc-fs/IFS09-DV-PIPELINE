// control_node (C++): the ROS 2 shell around control_core (plain C).
//
// Same node name, topics, QoS, parameters and lifecycle as the Python
// control/control_node.py. All control logic lives in control_core.c, which
// has no ROS, heap or OS dependency so it can move to the uDV STM32H7 as is;
// this file only turns messages into core calls and core output into
// messages.
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <dv_msgs/msg/cone_array.hpp>
#include <fs_msgs/msg/control_command.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>

#include "control_core/control_core.h"
#include "node_base_cpp/base_lifecycle_node.hpp"

namespace {

double stamp_s(const builtin_interfaces::msg::Time & t)
{
  return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

bool contains(const std::vector<std::string> & v, const std::string & s)
{
  for (const auto & e : v) {
    if (e == s) {return true;}
  }
  return false;
}

}  // namespace

class ControlNode : public node_base_cpp::BaseLifecycleNode
{
public:
  explicit ControlNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : BaseLifecycleNode("control_node", options), core_(std::make_unique<ctrl_node_t>())
  {
    declare_params();
    ctrl_params_t p;
    ctrl_params_default(&p);
    ctrl_node_init(core_.get(), &p);
  }

protected:
  CallbackReturn on_configure_impl(const rclcpp_lifecycle::State &) override
  {
    RCLCPP_INFO(get_logger(), "on_configure: strategies + publishers");
    ctrl_params_t p;
    if (!build_params(p)) {return CallbackReturn::FAILURE;}
    ctrl_node_init(core_.get(), &p);

    rclcpp::PublisherOptions po;
    po.use_default_callbacks = false;  // no incompatible-QoS waitable per entity
    const auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    cmd_pub_ = create_publisher<fs_msgs::msg::ControlCommand>("/ctrl/cmd_internal", 10, po);
    emergency_pub_ = create_publisher<std_msgs::msg::Bool>("/ctrl/emergency", latched, po);
    v_set_pub_ = create_publisher<std_msgs::msg::Float32>("/control/v_set_mps", 10, po);
    kappa_pub_ = create_publisher<std_msgs::msg::Float32>("/control/kappa_max_per_m", 10, po);
    latency_pub_ = create_publisher<std_msgs::msg::Float32>("/control/latency_ms", 10, po);
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override
  {
    RCLCPP_INFO(
      get_logger(), "on_activate: subs + 40 Hz timer (lateral=%s)",
      core_->p.lateral == CTRL_LAT_STANLEY ? "stanley" : "pure_pursuit");
    // Fresh run: per-run state and controller state (PI integral, filters).
    refresh_node_params();
    ctrl_node_reset(core_.get());

    rclcpp::SubscriptionOptions so;
    so.use_default_callbacks = false;
    using std::placeholders::_1;
    sub_path_ = create_subscription<nav_msgs::msg::Path>(
      "Path", 10, std::bind(&ControlNode::on_path, this, _1), so);
    sub_pose_ = create_subscription<nav_msgs::msg::Odometry>(
      "/slam/pose", 10, std::bind(&ControlNode::on_pose, this, _1), so);
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&ControlNode::on_odom, this, _1), so);
    sub_orange_ = create_subscription<dv_msgs::msg::ConeArray>(
      "/cones_orange", 10, std::bind(&ControlNode::on_orange, this, _1), so);
    // SLAM latches /slam/final_lap: match TRANSIENT_LOCAL or trackdrive
    // would sit on the safe default and stop after lap 1.
    sub_final_lap_ = create_subscription<std_msgs::msg::Bool>(
      "/slam/final_lap", rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
      std::bind(&ControlNode::on_final_lap, this, _1), so);

    // ROS clock, so the tick follows /clock under use_sim_time (as rclpy's
    // create_timer does).
    tick_timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0 / CTRL_RATE_HZ),
      std::bind(&ControlNode::tick, this));

    const auto ret = BaseLifecycleNode::on_activate(state);
    if (ret != CallbackReturn::SUCCESS) {return ret;}
    std_msgs::msg::Bool b;
    b.data = false;
    emergency_pub_->publish(b);
    return ret;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override
  {
    RCLCPP_INFO(get_logger(), "on_deactivate: dropping timers + subs");
    drop_inputs();
    return BaseLifecycleNode::on_deactivate(state);
  }

  CallbackReturn on_cleanup_impl(const rclcpp_lifecycle::State &) override
  {
    RCLCPP_INFO(get_logger(), "on_cleanup: destroying publishers");
    drop_inputs();
    cmd_pub_.reset();
    emergency_pub_.reset();
    v_set_pub_.reset();
    kappa_pub_.reset();
    latency_pub_.reset();
    return CallbackReturn::SUCCESS;
  }

private:
  // ------------------------------------------------------------ parameters
  void declare_num(const std::string & name, double def)
  {
    rcl_interfaces::msg::ParameterDescriptor d;
    d.dynamic_typing = true;  // the YAML or mode_manager may send 5 for 5.0
    declare_parameter(name, rclcpp::ParameterValue(def), d);
  }

  double num(const std::string & name)
  {
    const auto v = get_parameter(name).get_parameter_value();
    if (v.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
      return static_cast<double>(v.get<int64_t>());
    }
    return v.get<double>();
  }

  // `<mode>.<name>` for the running mode when declared, else `<name>`.
  double mode_num(const std::string & name)
  {
    const std::string key = mode_name_ + "." + name;
    if (!mode_name_.empty() && has_parameter(key)) {
      const double v = num(key);
      RCLCPP_INFO(get_logger(), "mode=%s: %s=%g", mode_name_.c_str(), name.c_str(), v);
      return v;
    }
    return num(name);
  }

  std::vector<std::string> strings(const std::string & name)
  {
    const auto v = get_parameter(name).get_parameter_value();
    if (v.get_type() == rclcpp::ParameterType::PARAMETER_STRING_ARRAY) {
      return v.get<std::vector<std::string>>();
    }
    return {};
  }

  void declare_params()
  {
    declare_parameter<std::string>("drive_controller", "composite");
    declare_parameter<std::string>("lateral_controller", "pure_pursuit");
    declare_parameter<std::string>("longitudinal_controller", "pi_velocity");
    for (const auto & [name, def] : std::vector<std::pair<std::string, double>>{
        {"v_max", 5.0}, {"a_lat_max", 3.0}, {"a_dec_max", 4.0},
        {"regen_power_w", 11000.0}, {"vehicle_mass", 275.0},
        {"unseen_corner_radius", 4.5}, {"brake_latency_s", 0.3},
        {"lookahead_min", 1.0}, {"lookahead_k", 0.5},
        {"stanley_k", 2.0}, {"stanley_k_soft", 1.5},
        {"stanley_k_yaw_rate", 0.0}, {"stanley_yaw_rate_loop_gain", 0.2},
        {"stanley_k_damp_steer", 0.2}, {"latency_comp_s", 0.0},
        {"wheelbase", 1.627}, {"max_steer_deg", 18.2},
        {"kp_v", 0.5}, {"ki_v", 0.05}, {"deadband_v", 0.05}, {"throttle_max", 1.0},
        {"accel.v_max", 12.0}, {"v_meas_tau", 0.15}, {"v_meas_max_accel", 20.0},
        {"throttle_rate", 2.0}, {"throttle_release_rate", 8.0},
        {"regen_rate", 3.33}, {"steering_rate", 5.0},
        {"stop_latch_min_travel", 30.0}, {"stop_past_anchor_m", 8.0}})
    {
      declare_num(name, def);
    }
    declare_parameter<std::vector<std::string>>("sight_cap_off_modes", {"accel"});
    declare_parameter<std::vector<std::string>>("stop_on_anchor_modes", {"skidpad"});
  }

  bool build_params(ctrl_params_t & p)
  {
    ctrl_params_default(&p);
    const auto dc = get_parameter("drive_controller").as_string();
    if (dc != "composite") {
      RCLCPP_ERROR(get_logger(), "unknown drive_controller='%s'; supported: 'composite'", dc.c_str());
      return false;
    }
    std::string lat = behavior_;
    if (lat != "stanley" && lat != "pure_pursuit") {
      lat = get_parameter("lateral_controller").as_string();
    }
    if (lat == "pure_pursuit") {
      p.lateral = CTRL_LAT_PURE_PURSUIT;
    } else if (lat == "stanley") {
      p.lateral = CTRL_LAT_STANLEY;
    } else {
      RCLCPP_ERROR(get_logger(), "unknown lateral_controller='%s'", lat.c_str());
      return false;
    }
    const auto lon = get_parameter("longitudinal_controller").as_string();
    if (lon != "pi_velocity") {
      RCLCPP_ERROR(get_logger(), "unknown longitudinal_controller='%s'", lon.c_str());
      return false;
    }
    p.wheelbase = num("wheelbase");
    p.max_steer_rad = num("max_steer_deg") * (M_PI / 180.0);  // math.radians
    p.lookahead_min = num("lookahead_min");
    p.lookahead_k = num("lookahead_k");
    p.stanley_k = num("stanley_k");
    p.stanley_k_soft = num("stanley_k_soft");
    p.stanley_k_yaw_rate = num("stanley_k_yaw_rate");
    p.stanley_k_damp_steer = num("stanley_k_damp_steer");
    p.stanley_yaw_rate_loop_gain = num("stanley_yaw_rate_loop_gain");
    p.v_max = mode_num("v_max");
    p.a_lat_max = num("a_lat_max");
    p.a_dec_max = num("a_dec_max");
    p.kp = num("kp_v");
    p.ki = num("ki_v");
    p.deadband = num("deadband_v");
    p.throttle_max = mode_num("throttle_max");
    p.regen_power_w = num("regen_power_w");
    p.vehicle_mass = num("vehicle_mass");
    if (contains(strings("sight_cap_off_modes"), mode_name_)) {
      RCLCPP_INFO(get_logger(), "mode=%s: straight track, sight-distance cap off", mode_name_.c_str());
      p.unseen_corner_radius = 0.0;
    } else {
      p.unseen_corner_radius = num("unseen_corner_radius");
    }
    p.brake_latency_s = num("brake_latency_s");
    p.v_meas_tau = num("v_meas_tau");
    p.v_meas_max_accel = num("v_meas_max_accel");
    node_params(p);
    return true;
  }

  // Parameters the Python node read on every tick / orange message; read at
  // configure and again at every activate.
  void node_params(ctrl_params_t & p)
  {
    p.throttle_rate = num("throttle_rate");
    p.throttle_release_rate = num("throttle_release_rate");
    p.regen_rate = num("regen_rate");
    p.steering_rate = num("steering_rate");
    p.stop_latch_min_travel = num("stop_latch_min_travel");
    p.stop_past_anchor_m = num("stop_past_anchor_m");
    p.stop_on_anchor = contains(strings("stop_on_anchor_modes"), mode_name_) ? 1 : 0;
    p.latency_comp_s = num("latency_comp_s");
  }

  void refresh_node_params() {node_params(core_->p);}

  // ------------------------------------------------------------- inputs
  void drop_inputs()
  {
    sub_path_.reset();
    sub_pose_.reset();
    sub_odom_.reset();
    sub_orange_.reset();
    sub_final_lap_.reset();
    tick_timer_.reset();
  }

  void on_pose(const nav_msgs::msg::Odometry::SharedPtr m)
  {
    const auto & p = m->pose.pose;
    ctrl_node_on_pose(
      core_.get(), stamp_s(m->header.stamp), p.position.x, p.position.y,
      p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
  }

  void on_odom(const nav_msgs::msg::Odometry::SharedPtr m)
  {
    const auto & p = m->pose.pose;
    const auto & t = m->twist.twist;
    ctrl_node_on_odom(
      core_.get(), stamp_s(m->header.stamp), p.position.x, p.position.y,
      p.orientation.w, p.orientation.z, t.linear.x, t.linear.y, t.angular.z);
  }

  void on_path(const nav_msgs::msg::Path::SharedPtr m)
  {
    const size_t n = m->poses.size();
    px_.resize(n);
    py_.resize(n);
    pz_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const auto & q = m->poses[i].pose.position;
      px_[i] = q.x;
      py_[i] = q.y;
      pz_[i] = q.z;  // planner curvature side channel
    }
    const int kept = ctrl_node_on_path(
      core_.get(), px_.data(), py_.data(), pz_.data(), static_cast<int>(n));
    if (static_cast<size_t>(kept) < n) {
      RCLCPP_WARN_ONCE(
        get_logger(), "/Path has %zu points; control keeps the first %d", n, kept);
    }
  }

  void on_final_lap(const std_msgs::msg::Bool::SharedPtr m)
  {
    const bool v = m->data;
    if (ctrl_node_on_final_lap(core_.get(), v ? 1 : 0)) {
      RCLCPP_INFO(
        get_logger(), "/slam/final_lap -> %s%s", v ? "True" : "False",
        v ? "" : " - holding stop anchor until the closing lap");
    }
  }

  void on_orange(const dv_msgs::msg::ConeArray::SharedPtr m)
  {
    const size_t n = std::min(m->x.size(), m->y.size());
    ox_.assign(m->x.begin(), m->x.begin() + n);
    oy_.assign(m->y.begin(), m->y.begin() + n);
    double ax, ay, tx, ty;
    if (ctrl_node_on_orange(
        core_.get(), ox_.data(), oy_.data(), static_cast<int>(n), &ax, &ay, &tx, &ty))
    {
      RCLCPP_INFO(
        get_logger(),
        "stop latched at odom=(%.2f, %.2f) from %zu big-orange cones; "
        "stopping %.1f m past it at (%.2f, %.2f)",
        ax, ay, n, core_->p.stop_on_anchor ? 0.0 : core_->p.stop_past_anchor_m, tx, ty);
    }
  }

  // --------------------------------------------------------------- tick
  void tick()
  {
    const auto t0 = std::chrono::steady_clock::now();
    ctrl_out_t o;
    ctrl_node_tick(core_.get(), &o);

    fs_msgs::msg::ControlCommand cmd;
    cmd.throttle = o.throttle;
    cmd.brake = o.brake;
    cmd.steering = o.steering;
    cmd_pub_->publish(cmd);

    if (o.computed) {
      std_msgs::msg::Float32 f;
      f.data = static_cast<float>(o.v_set);
      v_set_pub_->publish(f);
      f.data = static_cast<float>(o.kappa_max);
      kappa_pub_->publish(f);
      if (core_->tick_count % 20 == 0) {
        RCLCPP_INFO(
          get_logger(),
          "v=%.2f travelled=%.1fm -> thr=%+.3f regen=%+.3f steer=%+.3f | "
          "path_n=%d path_len=%.1fm stop_d=%.1f latched=%s",
          o.speed, core_->travelled, o.throttle, o.brake, o.steering, o.path_n,
          o.path_len, o.stop_distance, o.stop_latched ? "True" : "False");
      }
    }

    if (latency_pub_ && latency_pub_->get_subscription_count() > 0) {
      std_msgs::msg::Float32 f;
      f.data = static_cast<float>(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      latency_pub_->publish(f);
    }
  }

  std::unique_ptr<ctrl_node_t> core_;  // 66 KB of fixed arrays: on the heap here
  std::vector<double> px_, py_, pz_, ox_, oy_;

  rclcpp_lifecycle::LifecyclePublisher<fs_msgs::msg::ControlCommand>::SharedPtr cmd_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Bool>::SharedPtr emergency_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr v_set_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr kappa_pub_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::Float32>::SharedPtr latency_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pose_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<dv_msgs::msg::ConeArray>::SharedPtr sub_orange_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_final_lap_;
  rclcpp::TimerBase::SharedPtr tick_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ControlNode>();
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
