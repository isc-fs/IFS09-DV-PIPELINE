# Ground Removal — Overview

How one LiDAR scan goes from raw points to "above-ground points, upright, with
z = height above the floor", ready for cone clustering.

Code:
- `cone_detection/cone_detection/detector.py` — `RealtimeConeDetector.detect`
- `cone_detection/cone_detection/ground/separation.py` — `separate_ground`, the stage entry point
- `cone_detection/cone_detection/ground/alignment.py` — `ground_rotation_matrix`,
  `plane_floor_z`
- `cone_detection/cone_detection/ground/methods/` — the pluggable ground
  methods (`remove_ground` and the `@register(...)` functions in the
  `*_methods.py` modules)
- `cone_detection/cone_detection/ground/ransac.py` — `ransac2`, the RANSAC method (the default before `mrf`)
- `cone_detection/cone_detection/ground/alignment.py` — `vectors2matrix`

Deeper dives:
- [`total_least_squares_plane_fit.md`](total_least_squares_plane_fit.md) — the plane fit
- [`gpf_ground_plane_fitting.md`](gpf_ground_plane_fitting.md) — GPF in full

---

## 1. The big picture

```
 raw scan (sensor frame, LiDAR tilted ~9° to the floor)
   │
   ├─ 0. input crop         keep 0.1 m < range, horizontal range ≤ 25 m
   │
   ├─ 1. remove_ground      method = cfg.ground_method (ransac | gpf | mrf | …)
   │        → outlier mask  (True = not ground)
   │        → ONE plane     [bias, n_x, n_y, n_z]
   │
   ├─ 2. rotation correct   outliers @ R, where R takes the plane normal to +z
   │
   ├─ 3. height shift       subtract the sensor's height → floor at z ≈ 0
   │
   ├─ 4. pre-DBSCAN filters tall-column veto (on), flat-cell filter (off),
   │                        memory guard
   │
   ├─ 5. DBSCAN             clusters of above-ground points
   │
   └─ 6. per cluster        drop points < floor_margin_m (4 cm) above the floor,
                            then shape gates and cone template fit
        │
        └─ stateful methods keep what they need on the remover (RANSAC: warm-start plane)
```

The **contract** that makes methods interchangeable is in step 1: every method
returns a point mask **and a single global plane**. Steps 2–6 only see the mask
and the plane, never the method.

---

## 2. Coordinate frames

| Frame | z means | Where it is used |
|---|---|---|
| **Sensor frame** | along the LiDAR's own up axis; the floor is a ramp (~15.6 cm per metre at 9°) | input; RANSAC and GPF work here |
| **Levelled frame** (`_level`) | height above a coarse GPF plane; `uv` = position in that plane | inside grid / ray methods (MRF, height_jump, …) |
| **Ground frame** (after steps 2–3) | height above the fitted floor | everything after ground removal |

Why level at all: a 30 cm grid cell or a "height jump" threshold is meaningless
on a 9° ramp — the floor alone would rise ~5 cm across a cell.

---

## 3. Step 0 — input crop (`detect`)

```python
r2 = x² + y²
keep = (r2 <= input_range_crop_m²)            # 25 m
     & (r2 + z² >= input_min_range_m²)        # 0.1 m: no-return / self-hit points
```

Points past 25 m would be discarded later by the 20 m range gate anyway; cutting
them here removes most of the ~174k-point scan before the expensive stages.

---

## 4. Step 1 — ground removal (`GroundRemover.remove`)

```python
ground = make_ground_remover(cfg)        # once, owned by the detector
non_ground_mask, plane = ground.remove(data)  # every scan
```

- Builds the method registered under `ground_method`: a plain function for a
  single-scan method, a class for one that keeps state between scans.
  `remove_ground(method, data, cfg)` is the one-shot form (fresh state).
- Fewer than 3 points → everything is an outlier, plane = `[0, 0, 0, 1]`.
- Every method uses `cfg.ground_threshold_m` (5 cm) as its "on the ground" distance.
- Non-planar methods still return one plane: a least-squares (TLS) refit on the
  points they labelled ground (`_refit_on_ground`), falling back to their coarse
  plane if fewer than 3.

### 4.1 RANSAC (`ransac`, the default before `mrf`)

`ground/ransac.py:ransac2`, Numba-compiled.

1. **Warm start.** The previous scan's plane is scored first. At 10 Hz the floor
   barely moves, so it usually wins immediately and the adaptive iteration bound
   collapses to a handful of iterations. If the scene changed, it simply loses.
2. **Random search.** Pick 3 random points → plane through them (cross product)
   → count points within 5 cm. Keep the best. Scoring uses a random subset of
   ≤ 5000 points; the inlier ratio, not the count, is what matters.
3. **Adaptive stop.** After each improvement the iteration budget is updated to
   the classic bound $k = \log(1-p)/\log(1-w^3)$ ($p = 0.9999$, $w$ = inlier
   ratio), capped at 200.
4. **Labels** on the full cloud: ground iff within 5 cm of the best plane.

Notes: the plane is the 3-point hypothesis itself (no least-squares refit), and
the 5 cm is measured **vertically** in the sensor frame, not perpendicular to the
plane (a factor $n_z \approx \cos 9° = 0.988$, negligible).

Weakness: maximises *count*, so a large wall in view can beat a partly hidden
floor. Single plane: slopes and crests are not representable.

### 4.2 GPF — Ground Plane Fitting (`gpf`)

Deterministic; also the coarse plane that many other methods start from.

1. Subsample to ≤ 20 000 points (stride).
2. **Seeds:** every point below (2nd percentile of z) + 12 cm. The percentile,
   not the minimum, so multipath returns below the floor don't drag it down.
3. **Fit/grow ≤ 10×:** TLS plane on the current set → new set = all points
   within ±5 cm → repeat until the set size stops changing.
4. **Labels** on the full cloud: ground iff $|$distance$|$ < 5 cm.

Why it works: the loop is alternating minimisation of the robust cost
$\sum \min(r_i^2, t^2)$ — it never gets worse, far points can't pull it, and
starting from the lowest points puts it in the floor's basin (it can't jump to
a wall like RANSAC can). Full derivation in
[`gpf_ground_plane_fitting.md`](gpf_ground_plane_fitting.md).

### 4.3 MRF — Markov random field on a height grid (`mrf`, the default)

Not a plane fit: labels **cells** of a grid as ground or object, so it handles
uneven ground.

1. **Level** the cloud on a coarse GPF plane → `(uv, h)`.
2. **Grid** of 30 cm cells. Per cell: lowest return `mn` (empty cells copy from
   the nearest filled cell) and height spread `span = max − min`.
3. **Local floor** `ref` = greyscale opening (min then max filter) of `mn` over
   9×9 cells (2.7 m): removes anything narrower than that, keeps slow terrain
   changes.
4. **Unary costs** (evidence of each cell on its own):
   - ground: $\min\big((\text{span}/0.10)^2 + ((\text{mn} - \text{ref})/0.10)^2,\ 10\big)$
   - object: $1$
   - empty cell: $0$ for both (neighbours decide)
5. **Pairwise costs** (neighbours agree): 4-connected neighbours pay
   $0.5 \cdot e^{-(\Delta h / 0.05)^2}$ for different labels, where $\Delta h$ is
   the difference of their `mn`. Expensive on flat floor, nearly free across a
   real height step (cone base, curb).
6. **Solve exactly** by min-cut / max-flow (two labels + non-negative pairwise
   weights → submodular → the cut *is* the global optimum). Backends: scipy
   Dinic (default) or PyMaxflow.
7. **Points:** floor height for every cell from the nearest ground cell. A point
   is ground if it is < 10 cm above it (in a ground cell) or < 5 cm (in an
   object cell — this catches the floor hits around a cone's base).
8. **Plane:** TLS refit on the ground points.

### 4.4 The other methods (benchmarking)

All in `ground/methods/` (the `*_methods.py` modules), same contract:

| Method | Idea |
|---|---|
| `ransac_zones`, `ransac_rings` | RANSAC per zone / per range ring |
| `czm` | GPF per zone of a concentric zone model |
| `grid_resid` | GPF plane, then per-cell residual check |
| `ray_slope` | walk each azimuth ray, break on steep slope |
| `height_jump` | flat cells near the floor are ground (incl. sidewalks) |
| `linefit` | line segments per azimuth sector (Himmelsbach 2010) |
| `range_image` | angle labelling on a range image (Bogoslavskyi 2016) |
| `groundgrid` | elevation grid from low-variance cells (Steinke 2023) |
| `pmf` | progressive morphological filter (Zhang 2003) |
| `gp_insac` | Gaussian-process INSAC per sector (Douillard 2011) |
| `csf` | cloth simulation (optional package) |
| `patchworkpp` | Patchwork++ (optional package) |

### 4.5 Shared building blocks

```
fit_plane  (TLS)  ─┬─ GPF ─┬─ gpf
                   │       ├─ _level ─┬─ mrf, height_jump, groundgrid, pmf, linefit, …
                   │       ├─ czm, grid_resid, ray_slope
                   └─ _refit_on_ground  (final plane of every non-planar method)
```

A bad GPF plane on a frame therefore affects most methods, not just `gpf`.

---

## 5. Step 2 — rotation correction

```python
data = data[outliers] @ ground_rotation_matrix(plane)
```

**What the matrix is.** `ground_rotation_matrix` builds $R$ = the rotation that
takes $\hat z = (0,0,1)$ onto the unit plane normal $n$, via `vectors2matrix`:

- axis $= \hat z \times n$ (normalised), angle $= \arccos(\hat z \cdot n)$
- converted to a matrix with scipy's `Rotation.from_rotvec` (Rodrigues' formula)

**Why right-multiplying works.** Points are rows, so `p_row @ R` is
$(R^\top p)^\top$. $R^\top$ is the inverse rotation: it takes $n$ to $\hat z$. So
after the multiplication the floor's normal points straight up, and for every
point

$$
z_{\text{rotated}} = n \cdot p .
$$

(Verified numerically: rotated z equals $n\cdot p$ to machine precision.)

**Why only outliers.** Selecting rows and right-multiplying commute
(`(A @ R)[mask] == A[mask] @ R`), so rotating just the ~1% above-ground points
gives the same result for a fraction of the cost. The full cloud is only rotated
(`rotate_xyz_to_ground`, decimated) when the `/lidar_points/ground` visualisation
has a subscriber.

---

## 6. Step 3 — height shift

```python
data[:, 2] -= plane_floor_z(plane)
```

**Goal:** after this, z = height above the floor. Everything downstream relies
on it: the 4 cm floor cull, the 0.75 m tall-column veto, and the cone templates,
whose apex is fixed at 0.35 / 0.55 m *above ground*. (On the car the LiDAR is
~1.07 m up; without the shift every cone would be fitted ~1 m too high.)

**What it computes.** With unit normal, the floor satisfies $n\cdot p + \text{bias} = 0$,
so after rotation the floor sits at $z = -\text{bias}$. `plane_floor_z` returns
$-\text{bias}\cdot n_z$, so the floor ends up at

$$
z_{\text{floor}} = -\text{bias} + \text{bias}\cdot n_z = -\text{bias}\,(1 - n_z).
$$

> **Known offset.** This is 0 only when the plane is level ($n_z = 1$). For the
> real car (bias ≈ 1.07 m, tilt ≈ 9°, $n_z \approx 0.988$) the floor lands at
> **≈ −1.3 cm** (−0.15 cm at 3°, −3.7 cm at 15°; checked numerically). The exact
> shift is $z' = n\cdot p + \text{bias}$, i.e. subtract $-\text{bias}$, which is
> exactly `plane_residual`. Effect today: the 4 cm floor cull effectively acts at
> ~5.3 cm, and template heights are ~1.3 cm off. The simulator mounts the LiDAR
> at ground level (bias ≈ 0), which hides it.

---

## 7. Step 4 — filters before clustering

Run on the rotated, shifted outliers (`apply_prefilters`, `clustering/prefilters.py`):

| Filter | Default | What it does |
|---|---|---|
| **Flat-cell filter** | off | drop every 30 cm cell with ≥ 6 points spanning < 8 cm: sidewalks and road bumps left over by a planar method |
| **Tall-column veto** | on | if any point in a 30 cm cell is > 0.75 m up, drop that cell and its 8 neighbours: tire walls, fences, people (no cone is taller than 0.55 m), including their cone-height lower parts |
| **DBSCAN memory guard** | cap 20 000 pts | voxel-dedupe (2 → 4 → 8 cm cells), then random subsample to the cap — protects against a mis-fitted plane turning the whole floor into "outliers" |

---

## 8. Steps 5–6 — clustering and the final floor cull

- DBSCAN (`eps` 0.3 m, `min_samples` 2) groups the remaining points.
- Clusters with fewer than 3 points are dropped.
- **Floor cull:** inside each cluster, points with $z \le$ `floor_margin_m`
  (4 cm) are removed. This strips the floor returns around a cone's base that
  survived ground removal (every method labels the bottom few cm of objects
  ambiguously).
- Shape gates (height 0.02–0.55 m, centroid range ≤ 20 m) and the cone template
  fit follow.

---

## 9. Feedback across scans

Any state a method carries from one scan to the next lives on its
`GroundRemover` instance, which the detector owns; the detector itself knows
nothing about it. `ransac`, `ransac_zones` and `ransac_rings` keep the previous
scan's global RANSAC plane as a warm start (§4.1); `patchworkpp` keeps its
estimator, which learns across scans. Every other method is purely single-scan.
`RealtimeConeDetector.reset()` (→ `GroundRemover.reset()`) clears it, e.g. when
a bag loops.

---

## 10. Parameters

| Parameter | Default | Used by |
|---|---|---|
| `input_range_crop_m` | 25 m | step 0 |
| `input_min_range_m` | 0.1 m | step 0 |
| `ground_method` | `mrf` | step 1 |
| `ground_threshold_m` | 0.05 m | step 1, every method |
| `ransac_prob` | 0.9999 | RANSAC |
| GPF `seed_pct` / `seed_band_m` / `n_iter` | 2% / 0.12 m / 10 | `_gpf_plane` defaults |
| MRF `_MRF_*` | cell 0.30 m, span/step 0.10 m, window 9, smooth 0.5, Δh 0.05 m, band 0.10 m | module constants |
| `tall_column_veto` / height / cell | on / 0.75 m / 0.30 m | step 4 |
| `flat_cell_filter` / size / range / min pts | off / 0.30 m / 0.08 m / 6 | step 4 |
| `dbscan_max_points` | 20 000 | step 4 |
| `floor_margin_m` | 0.04 m | step 6 |

`ground_method`, `ground_threshold_m` and the filter settings are ROS parameters
(`bringup/config/params.yaml`); the GPF and MRF constants are code-only.

---

## 11. Choosing a method

| | RANSAC | GPF | MRF |
|---|---|---|---|
| Model | one plane | one plane | per-cell labels, any shape |
| Deterministic | no | yes | yes |
| Uses previous scan | yes (warm start) | no | no |
| Can lock onto a wall | yes | no | no (labels cells, not a plane) |
| Uneven ground / slopes | no | no | yes |
| Main failure | big non-floor plane wins | lowest 2% isn't floor | bad levelling plane; far cones with 1–2 returns per cell look flat |
| Cost | few iterations when warm | ≤ 10 TLS fits | grid stats + one max-flow |
