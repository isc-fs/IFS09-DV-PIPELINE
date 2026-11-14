"""Shape constants of the two FSAE cone classes the fits recognise (small, big orange).

Cones are modelled as ``z = d - c * ||(x, y) - (a, b)||`` with z the height above the
fitted ground: ``c`` is the apparent side slope and ``d`` where the sides meet.
"""

# FSAE traffic cones (small / "regular") have base diameter ~228 mm. The big
# orange marker is ~285 mm. We default to the small-cone radius for the
# centroid correction; the per-class size mismatch is absorbed by the empirical
# tuning of ``closed_form_fits.NEAR_FACE_CENTROID_BIAS_FRAC``.
CONE_BASE_RADIUS_M = 0.114

# Known FSAE cone classes: (apparent slope c, apex height d). These are the
# *empirical* values measured from free 4-parameter VARPRO fits of simulator
# clusters, with z measured above the RANSAC ground plane (IFSSIM
# tools/sim_benchmark/diagnose_cone_shape.py;
# median over 612 small-cone fits on a 1.74 M pts/s bag and 1858 big-cone fits
# on a 95 s bag that faces the start gate; fit RMSE ~7 mm):
#   small cones   : c ≈ 5.16, d ≈ 0.43 m
#   big orange    : c ≈ 6.45, d ≈ 0.65 m
# d is where the fitted sides meet, not the cone's height: the sim cones have
# flat tops (highest returns ~0.30 m / ~0.48 m), and z = d - c * r only
# describes the sides. The previous 5.0 / 0.35 and 5.5 / 0.55 sat 8-10 cm
# low, masked while z was measured from the sensor instead of the ground.
CONE_SMALL_C = 5.16
CONE_SMALL_D = 0.43
CONE_BIG_C = 6.45
CONE_BIG_D = 0.65

# Tolerance for matching the estimated slope to one of the known cone classes.
# Sources of noise on the slope estimate: LiDAR z noise (~1 cm) divided by the
# cluster's along-ray span (typically 5–15 cm), giving σ_c ~ 0.1–0.3. With the
# c values 5.16 and 6.45 about 1.3 apart, 0.5 keeps the two acceptance bands
# [4.66, 5.66] and [5.95, 6.95] disjoint while still absorbing slope noise;
# blatantly off slopes (e.g. ground residuals, walls) fall through to the
# geometric fallback.
CONE_CLASS_TOLERANCE = 0.5
