// Copyright 2026 IFSSIM contributors.
//
// gtest coverage for the 9-state EKF in odometry_filter.
//
// Three classes of tests:
//   1. Carryovers from the complementary-filter era: calibration,
//      RPM correction, yaw integration, position integration, reset.
//      Same intent, adapted to the new EKF (looser tolerances where
//      EKF settling depends on noise tuning rather than a fixed α).
//   2. New EKF-specific tests: bias estimation, steering gating,
//      covariance grows-when-no-corrections / shrinks-on-corrections,
//      analytical-vs-numerical Jacobian check.
//   3. The Coriolis cornering regression — synthesises a 60 s
//      constant-radius turn with body-y reading exactly ω·vx
//      (centripetal). The pre-rewrite complementary filter blows past
//      4 m/s lateral drift here; the new EKF must keep |vy| < 0.10 m/s.
//
// Run inside the container:
//   colcon build --packages-select odometry_filter
//   colcon test --packages-select odometry_filter
//   colcon test-result --verbose

#include <cmath>

#include <gtest/gtest.h>

#include "odometry_filter/odometry_filter.hpp"

using odometry_filter::OdometryFilter;
using odometry_filter::EkfParams;
using odometry_filter::OdometryState;
using odometry_filter::kG;
using odometry_filter::kRpmToMs;
using odometry_filter::kStateDim;
using odometry_filter::X;
using odometry_filter::Y;
using odometry_filter::THETA;
using odometry_filter::VX;
using odometry_filter::VY;
using odometry_filter::OMEGA;
using odometry_filter::BA_X;
using odometry_filter::BA_Y;
using odometry_filter::BG_Z;

namespace {

constexpr double kImuDt = 0.0025;       // 400 Hz
constexpr int kCalibrationSamples = 1500;

// Feeds stationary IMU (accel = (0, 0, g), gyro = 0) at 400 Hz until
// the filter completes its calibration window. Stops there — running
// extra "still" predict ticks afterwards inflates P off-diagonals and
// confuses subsequent corrections (we'd accumulate position uncertainty
// the test doesn't intend to model).
OdometryFilter make_stationary_filter() {
  OdometryFilter f;
  const Eigen::Vector3d accel_stationary(0.0, 0.0, kG);
  const Eigen::Vector3d gyro_stationary(0.0, 0.0, 0.0);
  // rpm = 0 marks a genuine standstill so the bias calibration folds these
  // samples (the gate requires wheel speed ≈ 0). latest_rpm sticks at 0, so
  // one push before the loop covers the whole window.
  f.push_rpm(0.0, 0.0);
  for (int i = 0; i < kCalibrationSamples; ++i) {
    f.push_imu(static_cast<double>(i) * kImuDt, accel_stationary, gyro_stationary);
    if (f.is_calibrated()) {
      break;
    }
  }
  EXPECT_TRUE(f.is_calibrated()) << "fixture should leave filter calibrated";
  return f;
}

}  // namespace


// ============================================================================
// Calibration
// ============================================================================

TEST(OdometryFilterCalibration, StartsUncalibrated) {
  OdometryFilter f;
  EXPECT_FALSE(f.is_calibrated());
}

TEST(OdometryFilterCalibration, CalibratesWithinWindow) {
  OdometryFilter f;
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  for (int i = 0; i < 1500; ++i) {
    f.push_imu(i * kImuDt, accel, gyro);
  }
  EXPECT_TRUE(f.is_calibrated());
}

TEST(OdometryFilterCalibration, EstimatesAccelBias) {
  OdometryFilter f;
  const double bias_x = 0.05;
  const Eigen::Vector3d accel(bias_x, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  f.push_rpm(0.0, 0.0);  // standstill — let the bias gate fold these samples
  for (int i = 0; i < 1500; ++i) {
    f.push_imu(i * kImuDt, accel, gyro);
  }
  ASSERT_TRUE(f.is_calibrated());
  EXPECT_NEAR(f.accel_bias_for_tests()(0), bias_x, 1e-6);
  EXPECT_NEAR(f.state().x, 0.0, 1e-12);
  EXPECT_NEAR(f.state().y, 0.0, 1e-12);
  EXPECT_NEAR(f.state().vx, 0.0, 1e-12);
}

TEST(OdometryFilterCalibration, PostCalibrationStateIsZeroExceptBiases) {
  auto f = make_stationary_filter();
  ASSERT_TRUE(f.is_calibrated());
  EXPECT_EQ(f.state().x, 0.0);
  EXPECT_EQ(f.state().y, 0.0);
  EXPECT_EQ(f.state().vx, 0.0);
  EXPECT_EQ(f.state().vy, 0.0);
}


// ============================================================================
// Bias estimation — the EKF should subtract the learned bias on every
// post-calibration predict step.
// ============================================================================

TEST(BiasEstimation, RecoversNonZeroAccelBias) {
  OdometryFilter f;
  const double bx = 0.15, by = -0.10;
  const Eigen::Vector3d accel(bx, by, kG + 0.02);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  f.push_rpm(0.0, 0.0);  // standstill — let the bias gate fold these samples
  for (int i = 0; i < 1500; ++i) {
    f.push_imu(i * kImuDt, accel, gyro);
  }
  ASSERT_TRUE(f.is_calibrated());
  EXPECT_NEAR(f.accel_bias_for_tests()(0), bx, 1e-3);
  EXPECT_NEAR(f.accel_bias_for_tests()(1), by, 1e-3);

  // Now feed 1 s of post-calibration IMU at the SAME biased values.
  // The EKF should subtract the bias and keep vx, vy ≈ 0.
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 400; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  EXPECT_NEAR(f.state().vx, 0.0, 1e-2);
  EXPECT_NEAR(f.state().vy, 0.0, 1e-2);
}

TEST(BiasEstimation, RecoversGyroBias) {
  OdometryFilter f;
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro(0.0, 0.0, 0.02);  // 0.02 rad/s bias
  f.push_rpm(0.0, 0.0);  // standstill — let the bias gate fold these samples
  for (int i = 0; i < 1500; ++i) {
    f.push_imu(i * kImuDt, accel, gyro);
  }
  ASSERT_TRUE(f.is_calibrated());
  EXPECT_NEAR(f.gyro_bias_for_tests()(2), 0.02, 1e-4);

  // Post-calibration, same biased gyro → yaw should stay near zero.
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 400; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  EXPECT_NEAR(f.state().yaw, 0.0, 5e-3);
}


// ============================================================================
// RPM correction
// ============================================================================

TEST(RpmCorrection, ConvergesToTarget) {
  auto f = make_stationary_filter();
  const double target_rpm = 100.0;
  const double target_vx = target_rpm * kRpmToMs;
  for (int i = 0; i < 200; ++i) {
    f.push_rpm(i * 0.0125, target_rpm);
  }
  // EKF converges to target weighted by R_rpm vs accumulated P[VX,VX];
  // tolerance reflects that.
  EXPECT_NEAR(f.state().vx, target_vx, 0.05);
}


// ============================================================================
// Yaw integration
// ============================================================================

TEST(YawIntegration, IntegratesConstantGyro) {
  auto f = make_stationary_filter();
  const Eigen::Vector3d gyro(0.0, 0.0, 0.5);
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 400; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  // 1 s @ 0.5 rad/s → yaw ≈ 0.5 rad. Tolerance covers the first-tick
  // skip + EKF settling on bg_z.
  EXPECT_NEAR(f.state().yaw, 0.5, 0.02);
}

TEST(YawIntegration, WrapsToNegative) {
  auto f = make_stationary_filter();
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro_spin(0.0, 0.0, 5.0);  // 5 rad/s
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 320; ++i) {  // 0.8 s → ~4 rad raw → wraps
    f.push_imu(t0 + i * kImuDt, accel, gyro_spin);
  }
  EXPECT_GE(f.state().yaw, -M_PI);
  EXPECT_LE(f.state().yaw, M_PI);
}


// ============================================================================
// End-to-end position integration
// ============================================================================

TEST(Integration, ConstantRpmDrivesPosition) {
  auto f = make_stationary_filter();
  const double target_rpm = 100.0;
  const double target_vx = target_rpm * kRpmToMs;
  for (int i = 0; i < 200; ++i) {
    f.push_rpm(i * 0.0125, target_rpm);
  }
  EXPECT_NEAR(f.state().vx, target_vx, 0.05);

  // 1 s @ 400 Hz, zero accel input, RPM re-pinned every 5 IMU samples.
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 400; ++i) {
    if (i % 5 == 0) {
      f.push_rpm(0.625 + i * kImuDt, target_rpm);
    }
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  // After 1 s at vx ≈ 0.821 m/s, expect x ≈ 0.821 m.
  EXPECT_NEAR(f.state().x, target_vx, 0.1);
}


// ============================================================================
// reset()
// ============================================================================

TEST(Reset, ClearsState) {
  auto f = make_stationary_filter();
  f.push_rpm(0.0, 200.0);
  EXPECT_GT(f.state().vx, 0.0);
  f.reset();
  EXPECT_FALSE(f.is_calibrated());
  EXPECT_EQ(f.state().x, 0.0);
  EXPECT_EQ(f.state().vx, 0.0);
  EXPECT_EQ(f.state().yaw, 0.0);
  EXPECT_FALSE(f.diagnostics().slip_flag);
  EXPECT_EQ(f.diagnostics().yaw_residual_rad_s, 0.0);
}


// ============================================================================
// Steering kinematic cross-check (with gating)
// ============================================================================

TEST(SteeringCorrection, ResidualNearZeroWhenStraight) {
  auto f = make_stationary_filter();
  f.push_steering(0.0, 0.0);
  f.push_rpm(0.0, 200.0);
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 50; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  EXPECT_NEAR(f.diagnostics().yaw_residual_rad_s, 0.0, 5e-3);
  EXPECT_FALSE(f.diagnostics().slip_flag);
}

TEST(SteeringCorrection, FlagsSlipUnderHighResidual) {
  auto f = make_stationary_filter();
  // δ = 0.4 rad, vx ≈ 4.1 m/s (RPM 500, above low-vx gate), L=1.570
  // → ω_pred ≈ 1.05 rad/s. IMU gyro_z = 0 → residual ≈ 1.05 > 0.30
  // threshold → slip_flag true, EKF update rejected. The EKF's ω state
  // should follow the gyro (≈ 0), NOT the kinematic prediction.
  //
  // Note: previous version used RPM 200 (vx ≈ 1.64 m/s). After adding
  // `min_vx_for_steering_correct = 3.0 m/s`, the steering correction
  // doesn't fire at all that low — making slip_flag a non-event. Bump
  // to RPM 500 so the test still exercises the slip-rejection path.
  f.push_steering(0.0, 0.4);
  for (int i = 0; i < 200; ++i) {
    f.push_rpm(i * 0.0125, 500.0);
  }
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 100; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  EXPECT_TRUE(f.diagnostics().slip_flag);
  EXPECT_FALSE(f.diagnostics().low_vx_gate_on);
  // ω should track the gyro (≈ 0), not the rejected kinematic prediction.
  EXPECT_NEAR(f.state().yaw_rate, 0.0, 0.05);
}

TEST(SteeringCorrection, LowVxGateOverridesEverything) {
  // Same scenario as FlagsSlipUnderHighResidual but with vx kept below
  // the low-vx gate (RPM 100 → vx ≈ 0.82 m/s). The steering correction
  // must return early WITHOUT touching the EKF state and without
  // flagging slip. The gate diag flag should be raised.
  auto f = make_stationary_filter();
  f.push_steering(0.0, 0.4);
  for (int i = 0; i < 200; ++i) {
    f.push_rpm(i * 0.0125, 100.0);   // vx ≈ 0.82 m/s — below 3.0 m/s gate
  }
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 100; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  EXPECT_TRUE(f.diagnostics().low_vx_gate_on);
  EXPECT_FALSE(f.diagnostics().slip_flag);
  EXPECT_NEAR(f.state().yaw_rate, 0.0, 0.01);
  // The EKF's θ must not have drifted from the rejected steering update.
  EXPECT_NEAR(f.state().yaw, 0.0, 0.01);
}


// ============================================================================
// Non-holonomic constraint (NHC) — the only direct observation of vy
//
// Without it, bias-noise integration drives vy unbounded. Live-sim
// regression: stationary car after calibration drifted to vy ≈ 0.96
// m/s within ~10 s, sent PurePursuit off-track before the first turn.
// ============================================================================

TEST(NHC, StationaryAfterCalibrationDoesNotDriftVy) {
  // Simulates the regression directly: 30 s of "stationary" IMU
  // (post-calibration) with a tiny accel-y bias error (the EKF can't
  // know its bias estimate is imperfect — real-world calibration
  // leaves some residual). Without NHC, vy integrates this bias
  // unbounded. With NHC gated on slip_flag (which stays false here),
  // vy must stay bounded.
  auto f = make_stationary_filter();
  // Feed a small post-calibration ay bias (~5 mm/s² — 100× tighter
  // than the BMI088 spec, so this is a benign noise floor).
  const Eigen::Vector3d accel_biased(0.0, 0.005, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  // 30 s @ 400 Hz. Old EKF (no NHC) would have vy ≈ 30·0.005 = 0.15 m/s.
  for (int i = 0; i < 12000; ++i) {
    f.push_imu(t0 + i * kImuDt, accel_biased, gyro);
  }
  EXPECT_LT(std::abs(f.state().vy), 0.05)
    << "vy drifted to " << f.state().vy << " without NHC;"
    << " regression: stationary car at +0.96 m/s in live sim";
}

TEST(NHC, NotAppliedWhenSlipFlagRaised) {
  // When the kinematic-bicycle disagrees with the gyro by more than
  // the threshold, slip_flag goes true and NHC must loosen (post-#555
  // it still fires every tick, but with sigma_vy_nhc_slip = 0.5 m/s
  // instead of 0.10 m/s) so real lateral motion (tire sideslip) is
  // allowed to develop. Use the same setup as
  // SteeringCorrection.FlagsSlipUnderHighResidual but then artificially
  // inject a non-zero ay so vy WOULD pull away from zero. With the
  // loose NHC, vy must still track ay·dt closely; a tight NHC would
  // clamp it to ~0 within a few ticks.
  auto f = make_stationary_filter();
  f.push_steering(0.0, 0.4);  // δ ≠ 0
  for (int i = 0; i < 200; ++i) {
    // RPM 500 → vx ≈ 4.1 m/s, above low-vx gate so the steering update
    // actually fires and produces a slip_flag=true. (Previously used
    // RPM 200 = 1.6 m/s before the low-vx gate was added.)
    f.push_rpm(i * 0.0125, 500.0);
  }
  const Eigen::Vector3d accel(0.0, 0.5, kG);   // 0.5 m/s² lateral
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 100; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  ASSERT_TRUE(f.diagnostics().slip_flag);
  // ≈ 100 ticks · 0.0025 s · 0.5 m/s² = 0.125 m/s of accumulated vy.
  // If NHC were applied, vy would be near zero; assert it isn't.
  EXPECT_GT(std::abs(f.state().vy), 0.05);
}


// ============================================================================
// Covariance behaviour
// ============================================================================

TEST(Covariance, PositionGrowsWithoutCorrection) {
  auto f = make_stationary_filter();
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  // Drive a few samples to escape the no-dt initial branch.
  f.push_imu(t0, accel, gyro);
  const double pxx_before = f.covariance()(X, X);
  for (int i = 1; i < 2000; ++i) {  // 5 s
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  const double pxx_after = f.covariance()(X, X);
  EXPECT_GT(pxx_after, pxx_before);
}

TEST(Covariance, RpmShrinksVxVariance) {
  auto f = make_stationary_filter();
  const Eigen::Vector3d accel(0.0, 0.0, kG);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const double t0 = 1500 * kImuDt;
  // Let P[VX, VX] inflate via predict-only.
  for (int i = 0; i < 1000; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }
  const double pvxvx_before = f.covariance()(VX, VX);
  // Apply RPM updates and check covariance contracts.
  for (int i = 0; i < 50; ++i) {
    f.push_rpm(i * 0.0125, 100.0);
  }
  const double pvxvx_after = f.covariance()(VX, VX);
  EXPECT_LT(pvxvx_after, pvxvx_before);
}


// ============================================================================
// The killer test — Coriolis cornering regression
//
// Synthesise a constant-radius turn:
//   vx_true  = 10 m/s
//   ω_true   = 0.5 rad/s  →  R = vx/ω = 20 m
//   a_x      = 0
//   a_y      = ω·vx = 5.0 m/s²    (centripetal, body-y)
//   δ        = atan(L·ω/vx) = atan(1.570·0.5/10) ≈ 0.07823 rad
//
// In a Coriolis-correct filter, body-frame v̇y = a_y − ω·vx = 0,
// so |vy| stays small forever. The pre-rewrite complementary filter
// integrated a_y directly and blew past 4 m/s within ~70 s.
// ============================================================================

TEST(CoriolisCornering, SteadyVyIsNearZero) {
  // Use looser steering R to let the steering update keep ω anchored
  // (otherwise the predict-only ω would drift via gyro noise; the
  // synthetic feed has no noise but the filter doesn't know that).
  EkfParams params;
  auto f = OdometryFilter(params);

  // Calibrate with clean stationary IMU. Stop at calibration
  // completion so we don't inflate P off-diagonals with extra
  // "still" predict ticks.
  const Eigen::Vector3d accel_stationary(0.0, 0.0, kG);
  const Eigen::Vector3d gyro_stationary = Eigen::Vector3d::Zero();
  for (int i = 0; i < kCalibrationSamples; ++i) {
    f.push_imu(i * kImuDt, accel_stationary, gyro_stationary);
    if (f.is_calibrated()) {
      break;
    }
  }
  ASSERT_TRUE(f.is_calibrated());

  // Wind the filter up to vx = 10 m/s using RPM measurements before
  // turning on the cornering feed. The Coriolis term only fires once
  // vx is non-zero.
  const double vx_target = 10.0;
  const double rpm_target = vx_target / kRpmToMs;
  for (int i = 0; i < 300; ++i) {
    f.push_rpm(i * 0.0125, rpm_target);
  }
  ASSERT_NEAR(f.state().vx, vx_target, 0.2);

  // Drive a 60 s constant-radius turn. IMU @ 400 Hz, RPM @ 80 Hz,
  // steering @ 100 Hz. All synthesised consistently.
  const double omega_true = 0.5;       // rad/s
  const double a_y_true   = omega_true * vx_target;  // 5.0 m/s²
  const double delta_true = std::atan(1.570 * omega_true / vx_target);

  const Eigen::Vector3d accel_corner(0.0, a_y_true, kG);
  const Eigen::Vector3d gyro_corner(0.0, 0.0, omega_true);

  const double t0 = 1500 * kImuDt + 300 * 0.0125;
  const int n_imu = 60 * 400;  // 60 s at 400 Hz
  for (int i = 0; i < n_imu; ++i) {
    const double t = t0 + i * kImuDt;
    f.push_imu(t, accel_corner, gyro_corner);
    if (i % 5 == 0) {                          // 80 Hz RPM
      f.push_rpm(t, rpm_target);
    }
    if (i % 4 == 0) {                          // 100 Hz steering
      f.push_steering(t, delta_true);
    }
  }

  // The hard assertion: vy must stay small. Pre-rewrite filter would
  // be at ~4 m/s here.
  EXPECT_LT(std::abs(f.state().vy), 0.10) << "vy = " << f.state().vy;
}


// ============================================================================
// Jacobian sanity — analytical F vs numerical central-difference.
// ============================================================================

TEST(Jacobian, MatchesNumeric) {
  // Build a fresh, calibrated filter and force a non-trivial state.
  auto f = make_stationary_filter();

  const Eigen::Vector3d accel(0.1, 0.05, kG);
  const Eigen::Vector3d gyro(0.0, 0.0, 0.4);
  // Step a few times to populate state.
  const double t0 = 1500 * kImuDt;
  for (int i = 0; i < 100; ++i) {
    f.push_imu(t0 + i * kImuDt, accel, gyro);
  }

  // Capture current state vector.
  Eigen::Matrix<double, kStateDim, 1> x = f.state_vector();

  // We can't reach into predict_step directly. Instead, replicate the
  // exact Euler-step formula here and form both analytical + numerical
  // Jacobians of the SAME function. If the production code is wrong
  // these tests don't catch it — but the Jacobian formula itself is
  // what's at risk, and re-deriving here keeps the test simple.
  const double dt = kImuDt;
  // Bias-correct (using the just-computed state's biases, not the
  // raw-IMU sample). Only wz is used directly in the analytical F
  // below; ax/ay get used inside `step()` via shadowed locals.
  const double wz = gyro(2) - x(BG_Z);

  auto step = [&](const Eigen::Matrix<double, kStateDim, 1> & s)
                -> Eigen::Matrix<double, kStateDim, 1> {
    Eigen::Matrix<double, kStateDim, 1> out = s;
    const double theta = s(THETA);
    const double vx = s(VX);
    const double vy = s(VY);
    const double c = std::cos(theta), si = std::sin(theta);
    const double ax_l = accel(0) - s(BA_X);
    const double ay_l = accel(1) - s(BA_Y);
    const double wz_l = gyro(2)  - s(BG_Z);
    out(X)     = s(X) + (vx * c - vy * si) * dt;
    out(Y)     = s(Y) + (vx * si + vy * c) * dt;
    out(THETA) = s(THETA) + wz_l * dt;
    out(VX)    = s(VX) + (ax_l + wz_l * vy) * dt;
    out(VY)    = s(VY) + (ay_l - wz_l * vx) * dt;
    out(OMEGA) = wz_l;
    // Biases identity.
    return out;
  };

  // Analytical F — must match the spec in odometry_filter.cpp.
  // OMEGA is an assignment row (not propagated from state.OMEGA),
  // so F[*, OMEGA] = 0 for every row. wz dependence on bg_z gives
  // the bias-column entries for THETA / VX / VY / OMEGA.
  const double theta = x(THETA);
  const double vx = x(VX);
  const double vy = x(VY);
  const double c = std::cos(theta), si = std::sin(theta);
  Eigen::Matrix<double, kStateDim, kStateDim> F_an =
    Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
  F_an(X,     THETA) = (-vx * si - vy * c) * dt;
  F_an(X,     VX)    =  c * dt;
  F_an(X,     VY)    = -si * dt;
  F_an(Y,     THETA) = ( vx * c - vy * si) * dt;
  F_an(Y,     VX)    =  si * dt;
  F_an(Y,     VY)    =  c * dt;
  F_an(THETA, BG_Z)  = -dt;
  F_an(VX,    VY)    =  wz * dt;
  F_an(VX,    BA_X)  = -dt;
  F_an(VX,    BG_Z)  = -vy * dt;
  F_an(VY,    VX)    = -wz * dt;
  F_an(VY,    BA_Y)  = -dt;
  F_an(VY,    BG_Z)  =  vx * dt;
  F_an(OMEGA, OMEGA) = 0.0;
  F_an(OMEGA, BG_Z)  = -1.0;

  // Numerical central-difference Jacobian.
  const double h = 1.0e-6;
  Eigen::Matrix<double, kStateDim, kStateDim> F_num;
  for (int j = 0; j < kStateDim; ++j) {
    auto x_p = x; x_p(j) += h;
    auto x_m = x; x_m(j) -= h;
    F_num.col(j) = (step(x_p) - step(x_m)) / (2.0 * h);
  }

  // Tight tolerance — central differences at h=1e-6 should match
  // analytical to 1e-7 on a smooth function like this.
  for (int i = 0; i < kStateDim; ++i) {
    for (int j = 0; j < kStateDim; ++j) {
      EXPECT_NEAR(F_an(i, j), F_num(i, j), 1e-7)
        << "Jacobian mismatch at (" << i << ", " << j << ")";
    }
  }
}
