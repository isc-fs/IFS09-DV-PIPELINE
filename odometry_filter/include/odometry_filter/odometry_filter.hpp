// Copyright 2026 IFSSIM contributors.
//
// 9-state EKF for body-frame odometry on the IFS-08 DV pipeline.
//
// Replaces the prior complementary filter. The complementary version
// integrated IMU body-y directly into vy, which during cornering reads
// centripetal acceleration (= ω·vx) — so /odom.vy drifted to >4 m/s
// after ~68 s of steady-state cornering and the inflated state.speed
// fed back into PurePursuit was the trackdrive ceiling. The fix is to
// carry the Coriolis cross-terms in a coupled state, which the
// complementary filter cannot do. Hence the EKF.
//
// State vector (9-d):
//   x[X]      world-frame x  [m]
//   x[Y]      world-frame y  [m]
//   x[THETA]  world-frame yaw [rad], wrapped to (-π, π]
//   x[VX]     body-frame longitudinal velocity [m/s]
//   x[VY]     body-frame lateral velocity     [m/s]  (left positive, REP-103)
//   x[OMEGA]  body-frame yaw rate              [rad/s]
//   x[BA_X]   accelerometer bias, body-x      [m/s²]
//   x[BA_Y]   accelerometer bias, body-y      [m/s²]
//   x[BG_Z]   gyro bias, body-z                [rad/s]
//
// Process model (continuous, body-frame; ã = raw IMU, b_a = accel bias,
//   ω̃ = raw gyro, b_g = gyro bias):
//   ẋ      = vx·cosθ − vy·sinθ
//   ẏ      = vx·sinθ + vy·cosθ
//   θ̇      = ω
//   v̇x     = (ã_x − b_a,x) + ω·vy        ← Coriolis term
//   v̇y     = (ã_y − b_a,y) − ω·vx        ← Coriolis term
//   ω̇      = 0           (ω is set from the bias-corrected gyro each tick)
//   ḃ_a,*  = 0           (random walk via Q)
//   ḃ_g,z  = 0           (random walk via Q)
//
// Measurement models:
//   * RPM:        z = vx,     h(x) = vx
//   * Steering:   z = (vx/L)·tan(δ),  h(x) = ω.   GATED — only applied
//     when |residual| < slip_yaw_residual_threshold (above which the
//     kinematic-bicycle model is wrong because the tires are sliding,
//     so the measurement would corrupt the estimate). Outside the gate
//     we still publish /odom_diag/yaw_residual + slip_flag for tuning.
//
// Inputs: same real-car parity set as before MINUS brake pressure
// (which added noise, never signal — drop confirmed by user during
// the rewrite scoping).
//   * IMU (BMI088-class) — accel + gyro at native ~400 Hz. Used as the
//     predict-step input AND for stationary bias calibration on
//     activate.
//   * Motor RPM (~80 Hz) — primary longitudinal velocity correction.
//     Multiplied by rpm_to_ms to yield body-frame longitudinal speed
//     at base_link (rear axle midpoint per URDF).
//   * Steering angle (rad) — kinematic-bicycle yaw cross-check.
//
// Output state plus a 9×9 covariance (exposed for the lifecycle node
// to populate nav_msgs/Odometry covariance fields meaningfully).
//
// The pure-math class is ROS-agnostic — drivers / nodes wrap it.
// The real-car uDV firmware links the same library.

#ifndef ODOMETRY_FILTER__ODOMETRY_FILTER_HPP_
#define ODOMETRY_FILTER__ODOMETRY_FILTER_HPP_

#include <Eigen/Core>
#include <optional>

namespace odometry_filter {

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

// Motor-RPM → body-frame longitudinal velocity.
// v = rpm × (2π × WheelRadius / GearRatio) / 60.
//
// 2026-07-12 (Option B): WheelRadius corrected 0.228 → 0.202 m to match
// the real IFS-08 tyre. The sim *generates* rpm from ground speed with the
// same radius (settings.json WheelRadius → FSDSVehiclePawn), so sim
// /odom.vx = ground truth for any consistent radius — this change is
// GT-neutral in sim and makes the real car (which reports true rotor rpm)
// geometrically correct. UE settings.json + Chaos wheels move to 0.202 in
// lockstep.
//     RPM_TO_MS = (2π × 0.202 / 2.909) / 60 = 0.00727
//
// History (issue #380): the prior 0.00821 was validated in sim as
// mean(|GT.vx|)/mean(|/odom.vx|) = 0.9140 over 3324 samples on
// test_submodule.csv — but that only confirmed the sim's self-consistent
// 0.228 loop, not the real tyre. Superseded by the geometric 0.202 above.
constexpr double kRpmToMs = 0.00727;

// Stationary-calibration window. While collecting IMU samples the
// filter does not publish — the autonomy stack tolerates brief
// startup delay (it's already inside SetMission's "configuring" stage).
constexpr double kCalibrationSeconds = 3.0;

// Wheel speed (m/s) below which the car counts as stationary for bias
// calibration. The window's "mean gyro = bias" assumption only holds at a
// true standstill: if the car is already turning during the window (bag
// recorder opens late / mid-motion launch), that real rotation gets soaked
// into the gyro bias and drifts heading until SLAM loses lock. Gate bias
// accumulation on rpm ≈ 0 instead of trusting a fixed time window.
constexpr double kStationarySpeedMs = 0.1;

// Wheelbase used in the kinematic-bicycle yaw cross-check:
//     ω_pred = (v_x / wheelbase) · tan(δ)
// 1.570 m is the authoritative IFS-08 spec — confirmed in
// docs/MODEL_IFS_08/DYNAMIC_MOD/Cooling/BR_regenerativeBR.m
// (`L = 1.570`) and the MONO sheet of docs/MODEL_IFS_08/ISC_IFS_08.xlsx
// (BQ64 `L = 1570 mm`). The URDF currently lists 1.55 (approximate);
// inputs_vehicle_dynamics.m lists 1.627 (4 % disagreement between
// authors). 1.570 is what the BR/Sandra and master MONO sheet agree on.
constexpr double kWheelbaseM = 1.570;

// |yaw_residual| > this → slip_flag goes true AND the steering update
// is rejected (the kinematic-bicycle model is no longer applicable;
// folding it in would corrupt yaw).
constexpr double kSlipYawResidualThreshold = 0.3;

// vx threshold below which `correct_steering` is gated off entirely.
// The kinematic bicycle model assumes steady-state slip equilibrium
// (ω = (vx/L)·tan(δ)). During launch + low-speed transients this is
// wrong: the front wheels are turning but the chassis hasn't built up
// the slip angle / lateral force to actually rotate yet. Applying the
// correction in that window pulls the EKF's ω state away from the
// correct gyro reading and bakes in spurious yaw. Empirical: at
// vx ≈ 1.8 m/s with δ=5° the model predicts +5.7°/s while truth is
// ~0°/s — that 5.7°/s gain stays below the slip threshold and the
// EKF integrates it for ≥1 s before slip_flag finally fires. 3 m/s
// puts the gate above the launch transient on FS-DV. (Bag analysis
// from autocross_track_20260404_013721_20260517_230727 post-#518
// /odom-handshake fix; see commit message for the trace.)
constexpr double kMinVxForSteeringCorrect = 3.0;

// Gravity magnitude (sim's BMI088 model uses standard gravity).
constexpr double kG = 9.81;


// ---------------------------------------------------------------------
// State indexing
// ---------------------------------------------------------------------
enum StateIdx : int {
  X       = 0,
  Y       = 1,
  THETA   = 2,
  VX      = 3,
  VY      = 4,
  OMEGA   = 5,
  BA_X    = 6,
  BA_Y    = 7,
  BG_Z    = 8,
};
constexpr int kStateDim = 9;


// ---------------------------------------------------------------------
// Public state — the slice of the EKF state vector that downstream
// consumers (the lifecycle node, /odom message, tests) care about.
// ---------------------------------------------------------------------
struct OdometryState {
  double x{0.0};
  double y{0.0};
  double yaw{0.0};
  double vx{0.0};
  double vy{0.0};
  double yaw_rate{0.0};
};


// ---------------------------------------------------------------------
// Diagnostics — surfaced on /odom_diag/* by the lifecycle node.
// ---------------------------------------------------------------------
struct FilterDiagnostics {
  // ω_pred − ω, where ω_pred = (vx/L)·tan(δ). Persistent non-zero
  // residual means the car is sliding (real ω diverges from kinematic
  // prediction). Published every IMU tick.
  double yaw_residual_rad_s{0.0};

  // True iff |yaw_residual| > slip_yaw_residual_threshold on the
  // latest tick. Raw per-tick truth; downstream consumers debounce.
  // When true, the steering kinematic measurement is REJECTED (not
  // folded into the EKF update).
  bool slip_flag{false};

  // True iff the steering correction was gated off this tick by the
  // low-vx guard (vx < min_vx_for_steering_correct). At launch the
  // kinematic-bicycle prediction is wrong because slip equilibrium
  // hasn't built up; this flag tells the operator when the EKF is
  // relying on the gyro alone for ω.
  bool low_vx_gate_on{false};
};


// ---------------------------------------------------------------------
// Tunable parameters. All defaults are starting points; the
// production-tuning knobs live here so YAML overrides at the node
// level pass through cleanly.
// ---------------------------------------------------------------------
struct EkfParams {
  // Geometry
  double wheelbase_m = kWheelbaseM;

  // RPM scaling (see kRpmToMs derivation)
  double rpm_to_ms = kRpmToMs;

  // Calibration window length [s]
  double calibration_seconds = kCalibrationSeconds;

  // Wheel speed (m/s) below which the car counts as stationary for bias
  // calibration. See kStationarySpeedMs.
  double stationary_speed_ms = kStationarySpeedMs;

  // Process noise std-devs (continuous-time densities; Q is built
  // as diag(σ²) · dt at each predict step).
  double sigma_ax = 0.05;       // accel-x density [m/s² / √Hz]
  double sigma_ay = 0.05;       // accel-y density
  double sigma_gz = 0.01;       // gyro-z  density [rad/s / √Hz]
  double sigma_ba_walk = 1.0e-4;  // accel bias random-walk
  // Gyro-bias random-walk density. Was 1e-5 (frozen-bias semantics:
  // expects bg_z to drift < 0.004° over a 60 s run, which is the
  // BMI088 datasheet stability), but in practice real bias drift
  // from vibration / temperature is bigger mid-run. With the tight
  // 1e-5 value, slip_flag gates the steering correction during
  // cornering and bg_z gets no measurement update for seconds at a
  // time → drift accumulates into θ as phantom yaw.
  //
  // 1e-4 lets the EKF allocate P[BG_Z] over those windows so the
  // next steering correction (post-corner straight) can pull bg_z
  // back to truth, but stays tight enough that bg_z doesn't track
  // gyro noise during cornering and overcorrect on the next
  // straight. (Tried 1e-3 first — killed the +44° first-turn jump
  // on bag _235642 but introduced -100° overcorrection mid-run on
  // bag _001700. 1e-4 splits the difference.)
  //
  // NOTE (post-#555): the paragraph above describes the pre-#555
  // filter. #555 zeroes K[BG_Z] in *every* correction (RPM, NHC,
  // steering — Schmidt-Kalman partition), so after calibration NO
  // measurement moves bg_z any more: the state holds the stationary-
  // calibration value for the whole run. This density therefore only
  // inflates P[BG_Z] (and, through F[THETA, BG_Z] = -dt, P[THETA]) —
  // it is a covariance-honesty knob for /odom.pose.covariance, not a
  // tracking-speed knob. Re-enabling bg_z tracking would mean keeping
  // K[BG_Z] in correct_steering() (the only measurement that actually
  // observes ω), which needs bag validation before it is turned on.
  double sigma_bg_walk = 1.0e-4;  // gyro bias random-walk

  // Measurement noise std-devs.
  // sigma_rpm is set tight (≈ 2 cm/s) so RPM dominates over the IMU
  // accel integration in vx tracking — without this, predict-step
  // accel-noise inflates P[VX,VX] slowly and RPM corrections become
  // weak (low Kalman gain), letting vx drift by tenths of a m/s during
  // cornering. With vx loose, the Coriolis cancellation
  // (v̇y = ay − ω·vx) leaves residual that integrates into vy drift.
  double sigma_rpm = 0.02;      // [m/s]
  double sigma_steer = 0.30;    // [rad/s] — deliberately loose; gated under slip
  // sigma_vy_nhc — non-holonomic-constraint pseudo-measurement. The
  // tight default (0.10 m/s) lets the rolling-tire assumption pull
  // vy → 0 on ~1 s timescales while tolerating small real lateral
  // motion within sideslip tolerance.
  //
  // Previously: NHC was gated OFF entirely when slip_flag fired. The
  // gate is "correct" in the sense that real tire slip means vy ≠ 0,
  // but bag analysis (#447) shows slip_flag fires for ~half of every
  // autocross lap, and during those windows vy runs unbounded to
  // ±1 m/s → integrates into 56 m of /odom XY drift mid-lap → SLAM
  // cone-DA cascades because the /odom prior is far from truth.
  //
  // New behaviour (2026-05-24, fix/nhc-loose-during-slip): NHC fires
  // every IMU tick. Its sigma adapts to slip state:
  //   * !slip_flag  →  sigma_vy_nhc          (the tight 0.10 m/s)
  //   * slip_flag   →  sigma_vy_nhc_slip     (loose, default 0.5 m/s)
  // Loose enough to tolerate real sideslip up to ~0.5 m/s 1-σ without
  // fighting the measurement, but still tight enough to bound the
  // integration-error mode that produced the 56 m drift. If the car
  // genuinely sideslips harder than that (heavy autocross), this will
  // bias vy toward 0 — accept that trade because the alternative
  // (slip_flag = T → no vy observation) is what kills the lap.
  double sigma_vy_nhc      = 0.10;   // [m/s] — applied when !slip_flag
  double sigma_vy_nhc_slip = 0.50;   // [m/s] — applied when slip_flag

  // Steering gating
  double slip_yaw_residual_threshold = kSlipYawResidualThreshold;
  // Below this vx (m/s), `correct_steering` returns early without
  // folding the kinematic-bicycle prediction in. See
  // kMinVxForSteeringCorrect for the rationale.
  double min_vx_for_steering_correct = kMinVxForSteeringCorrect;

  // dt clamps (s). dt outside [dt_min, dt_max] → predict step skipped.
  double dt_min = 1.0e-5;
  double dt_max = 0.1;
};


// ---------------------------------------------------------------------
// 9-state EKF.
//
// Lifecycle:
//   1. Construction: state and covariance hold their initial values;
//      calibration is not yet started.
//   2. push_imu() during first calibration_seconds: accumulate bias
//      estimates from the stationary readings. Output undefined;
//      is_calibrated() returns false.
//   3. After calibration: push_imu() drives the predict step (with
//      Coriolis-correct body-frame mechanics). push_rpm() applies the
//      vx measurement update. push_steering() caches δ; the steering
//      kinematic update fires on the next IMU tick (when vx is fresh).
//   4. state(), diagnostics(), and covariance() return the current
//      estimate.
// ---------------------------------------------------------------------
class OdometryFilter {
 public:
  OdometryFilter() : OdometryFilter(EkfParams{}) {}
  explicit OdometryFilter(const EkfParams & params);

  // ----- Public read-only accessors -----
  const OdometryState & state() const noexcept { return state_; }
  const FilterDiagnostics & diagnostics() const noexcept { return diag_; }
  bool is_calibrated() const noexcept { return calib_.completed; }
  const EkfParams & params() const noexcept { return params_; }

  // Seed the post-calibration forward velocity. The accel-bias
  // calibration zeroes the whole state, which is only correct when the
  // car is truly stationary at calibration. When replay (or a real
  // start) begins mid-motion, an unaided accel integrator started from
  // v=0 can never recover absolute speed (constant cruise has no
  // forward accel to integrate). One wheel/GT speed sample fixes the
  // initial condition. No-op before calibration completes.
  void seed_forward_velocity(double vx);

  // 9×9 covariance copy. Cheap (576 bytes); called once per /odom
  // publish (100 Hz) to populate nav_msgs/Odometry covariance fields.
  Eigen::Matrix<double, kStateDim, kStateDim> covariance() const;

  // Full state vector (for tests + tuning logs).
  Eigen::Matrix<double, kStateDim, 1> state_vector() const;

  // Tear down all state. Call on lifecycle on_activate so a
  // deactivate→activate cycle starts a fresh stationary calibration.
  void reset();

  // ----- Ingestion API (called from the ROS node) -----

  // One IMU sample. Called from the IMU subscription callback at the
  // BMI088 native rate (~400 Hz). accel and gyro are body-frame; accel
  // includes gravity (a stationary upright IMU reads (0, 0, +9.81)).
  void push_imu(
    double t,
    const Eigen::Vector3d & accel,
    const Eigen::Vector3d & gyro);

  // One motor-RPM sample. Called from the RPM subscription (~80 Hz).
  // Triggers the vx EKF update step immediately.
  void push_rpm(double t, double rpm);

  // One steering-angle sample (rad). Cached; the EKF update fires from
  // push_imu so vx, ω, and δ are time-aligned at the same tick.
  void push_steering(double t, double angle_rad);

  // ----- Internal-state access for unit tests only -----
  const Eigen::Vector3d & accel_bias_for_tests() const noexcept {
    return calib_.accel_bias;
  }
  const Eigen::Vector3d & gyro_bias_for_tests() const noexcept {
    return calib_.gyro_bias;
  }
  int n_calibration_samples_for_tests() const noexcept {
    return calib_.n_samples;
  }

 private:
  // ----- Calibration accumulators -----
  struct Calibration {
    int n_samples{0};
    Eigen::Vector3d accel_sum{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_sum{Eigen::Vector3d::Zero()};
    std::optional<double> t_first{};
    bool completed{false};

    Eigen::Vector3d accel_bias{Eigen::Vector3d::Zero()};
    Eigen::Vector3d gyro_bias{Eigen::Vector3d::Zero()};
  };

  // Predict step: integrate the process model by dt (Euler), update
  // covariance via F·P·Fᵀ + Q·dt. Uses the latest bias-corrected
  // IMU input. Run on each post-calibration push_imu() tick.
  void predict_step(
    double dt,
    const Eigen::Vector3d & accel,
    const Eigen::Vector3d & gyro);

  // Measurement update: z is the bias-corrected vx from RPM.
  void correct_rpm(double z_vx);

  // Measurement update: z is the kinematic-bicycle yaw-rate
  // prediction from the latest steering angle. Gated: rejects when
  // |z − ω| > slip_yaw_residual_threshold and raises slip_flag.
  // Always updates diagnostics (yaw_residual + slip_flag).
  //
  // Effect on the estimate (be aware when tuning sigma_steer): the
  // update touches x[OMEGA] only (K is partitioned to OMEGA, #555),
  // and predict_step() re-assigns x[OMEGA] = ω̃_z − bg_z from the gyro
  // on the very next IMU tick (F[OMEGA, *] = 0 except F[OMEGA, BG_Z]),
  // which also rebuilds the OMEGA row/column of P. So the correction
  // persists for at most one IMU interval (~2.5 ms) — it nudges the
  // yaw_rate that /odom happens to publish in that window and nothing
  // else. Its lasting outputs are the diagnostics and, via slip_flag,
  // the NHC sigma selection in correct_nhc(). θ and bg_z are NOT
  // corrected by steering in the current partition.
  void correct_steering();

  // Non-holonomic-constraint pseudo-measurement: z = 0, h = vy. Applied
  // on every post-calibration IMU tick; slip-aware — uses sigma_vy_nhc
  // normally and the looser sigma_vy_nhc_slip while slip_flag is set
  // (it is no longer gated off under slip, see push_imu). This is the
  // only direct observation of vy — without it, bias-noise integration
  // drives vy unbounded over seconds (see correct_nhc impl comment).
  void correct_nhc();

  // Stationary-calibration accumulator.
  void accumulate_calibration(
    double t,
    const Eigen::Vector3d & accel,
    const Eigen::Vector3d & gyro);

  // Mirror the OdometryState slice of x_ into state_. Called after
  // every predict/correct step.
  void publish_state_view();

  EkfParams params_;
  OdometryState state_;
  FilterDiagnostics diag_;
  Calibration calib_;

  // Full EKF state vector + covariance.
  Eigen::Matrix<double, kStateDim, 1> x_{Eigen::Matrix<double, kStateDim, 1>::Zero()};
  Eigen::Matrix<double, kStateDim, kStateDim> P_{
    Eigen::Matrix<double, kStateDim, kStateDim>::Zero()};

  std::optional<double> t_imu_last_{};
  double latest_steering_rad_{0.0};
  bool have_steering_{false};

  // Latest wheel speed [m/s] and whether any RPM sample has arrived.
  // Tracked even before calibration completes so accumulate_calibration
  // can gate the bias estimate on a genuine standstill (rpm ≈ 0).
  double latest_rpm_ms_{0.0};
  bool have_rpm_{false};
};

}  // namespace odometry_filter

#endif  // ODOMETRY_FILTER__ODOMETRY_FILTER_HPP_
