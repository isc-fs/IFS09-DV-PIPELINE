# Perception pipeline: performance changes

Branch `feat/5-ground-removal-algorithms`, 2026-11-12. All changes are uncommitted in the working tree.

This document covers every performance change made to the per-scan perception path
(decode → ground removal → rotate → filters → DBSCAN → cluster split → cone fits), what each one
does, how it works, and what it measured. Ground removal is covered for **`ransac`** and
**`mrf`** only; the other ground methods are mentioned only where a shared change touches them.

Two sets of changes are described:

* **Part A, "mine"**: made in the session that wrote this document (decode, cluster split,
  plane-fit kernels, MRF graph reduction, MRF warm-up).
* **Part B, "theirs"**: made by a second session working in the same tree afterwards (referred to as
  "Fable" by the user). I did **not** write these and did not write their tests; I read their diffs, measured
  their effect, and ran the full suite on the final tree. Descriptions of them come from that code.

---

## 1. Headline results

End-to-end per-scan compute, **decode through cone fits** (decode + `detect()`), milliseconds. "pre" is
the code at the start of this work, "mine" adds Part A, "theirs" adds Part B on top.

| Ground method | Workload | Stat | pre | mine | theirs | mine % | theirs % | cumulative % |
|---|---|---|---|---|---|---|---|---|
| RANSAC | sim scans (~8.2k pts) | p10 | 0.893 | 0.817 | 0.395 | −9% | −52% | −56% |
| RANSAC | sim scans (~8.2k pts) | median | 1.002 | 0.924 | 0.445 | −8% | −52% | −56% |
| RANSAC | sim scans (~8.2k pts) | p95 | 1.330 | 1.140 | 0.519 | −14% | −55% | −61% |
| RANSAC | dense synthetic (~98.6k pts) | p10 | 7.828 | 6.875 | 2.085 | −12% | −70% | −73% |
| RANSAC | dense synthetic (~98.6k pts) | median | 8.608 | 7.505 | 2.217 | −13% | −70% | −74% |
| RANSAC | dense synthetic (~98.6k pts) | p95 | 11.030 | 9.365 | 2.585 | −15% | −72% | −77% |
| MRF | sim scans (~8.2k pts) | p10 | 5.162 | 2.000 | 1.565 | −61% | −22% | −70% |
| MRF | sim scans (~8.2k pts) | median | 5.557 | 2.171 | 1.742 | −61% | −20% | −69% |
| MRF | sim scans (~8.2k pts) | p95 | 7.301 | 2.847 | 2.094 | −61% | −26% | −71% |
| MRF | dense synthetic (~98.6k pts) | p10 | 15.206 | 7.099 | 4.254 | −53% | −40% | −72% |
| MRF | dense synthetic (~98.6k pts) | median | 16.787 | 7.601 | 4.620 | −55% | −39% | −72% |
| MRF | dense synthetic (~98.6k pts) | p95 | 20.487 | 8.905 | 5.934 | −57% | −33% | −71% |

In one line each:

* **RANSAC:** about −56% on real sim scans (−73% to −77% on the dense cloud). Almost all of it is Part B
  (`ransac_ground_mask` and the fused scan kernels). Part A's share on RANSAC is decode copies and, on
  dense clouds, the cluster split.
* **MRF:** about −70% on sim scans and −72% on the dense cloud. Most of it is Part A (plane-fit kernels and
  the reduced graph); Part B adds a further 22% (sim) to 40% (dense).
* **Tail latency:** p95 follows the typical time (MRF sim p95 7.3 → 2.8 → 2.1 ms). The p95/p10 spread is
  about 1.3–1.45× in every state on a quiet machine, so there is no tail-specific gain beyond the average one
  (see section 3).
* All variants are far under the 100 ms scan period.

Excluded from these totals: marker building, ROS publishing, and the optional viz clouds. Only the
numeric pipeline is timed.

---

## 2. How this was measured

### Three code states

| State | Contents |
|---|---|
| **pre** | The branch's uncommitted work at the start of the session (the pluggable ground methods, including `mrf`), with the files the second session later changed (`clustering.py`, `cone_fit.py`, `ransac.py`) at git `HEAD`. |
| **mine** | pre + Part A. |
| **theirs** | The working tree as snapshotted 2026-11-12 ~11:57: mine + Part B. The other session may have edited files since. |

Each state was copied to its own directory and mounted separately, so measuring never touched the shared
working tree.

### Harness

A script run inside the pipeline's own container image (`ifssim-dv_pipeline_stack:latest`), so Python,
numba, scipy and ROS are the same as the benchmark's:

* **Workloads.**
  * *sim scans*: 40 real LiDAR scans from the sim bag
    `bags/trackdrive_track_20260404_013721_20260620_103753` (~8.2k points each).
  * *dense synthetic*: ~98.6k points, built by overlaying 12 jittered scans. It is a scaling stress
    test, **not** a real car scan, and produces ~70–77 detections per scan.
* **Timed:** `ConeDetectionNode.pointcloud2_to_xyz` (the real decode function) on a constructed
  `PointCloud2`-shaped message, then `RealtimeConeDetector.detect()` with `stage_timings`.
* **Statistics:** per-stage p10, median, and p95 over pooled scans and passes; for each state/method the
  lower of two interleaved repetitions.
* **Configs:** `ground_method` = `ransac` and `mrf`, each in all three states.

### Caveats on the numbers

* **These numbers were re-measured on a quiet machine** (load average 1–3 on 24 cores, 2026-11-12 ~22:38,
  harness: 10 passes over the 40 sim scans, 8 over the 6 dense clouds, two interleaved repetitions, the
  lower of the two reported). An earlier set measured while another session's jobs held the machine at
  load 40–70 gave about twice the absolute times and misleading tails; it has been replaced. The cumulative
  reductions were similar in both sets except MRF (about −64% then vs −70% now).
* **Noise floor:** about ±5% between runs, and ±20% for stages below 0.1 ms. `*` in the tables marks
  changes inside that band.
* **"theirs" is a snapshot** of the shared working tree taken 2026-11-12 22:37:56, including the other
  session's later edits to `ransac.py`; the full suite was run on that same tree (section 6).
* The container benchmark `run_perception_benchmark.py` was also used (section 5), mainly for detection
  accuracy against ground truth.

---

## 3. Per-stage results

p10 ms per scan. "mine %" is mine vs pre; "theirs %" is theirs vs mine; "cumul. %" is theirs vs pre.
`*` marks a change within the noise floor.

"Range crop + misc" is `detect()` total minus the listed stages; it holds the range crop, the per-cluster
gate loop and bookkeeping.

### RANSAC, sim scans (~8.2k points)

| Stage | pre | mine | theirs | mine % | theirs % | cumul. % | Cause (mine) | Cause (theirs) |
|---|---|---|---|---|---|---|---|---|
| decode | 0.082 | 0.021 | 0.016 | −75% | −24% | −81% | **no copies** | none |
| range crop + misc | 0.069 | 0.068 | 0.018 | −1%* | −73% | −73% | none | **numba fused pass** |
| ground removal | 0.261 | 0.249 | 0.095 | −5%* | −62% | −64% | none (noise) | **numba kernel, in-place** |
| rotate + shift | 0.060 | 0.060 | 0.036 | +1%* | −41% | −40% | none | **numba** fused select+rotate |
| tall-column veto | 0.004 | 0.003 | 0.002 | −2%* | −50% | −52% | none | dense grid, no sort |
| DBSCAN | 0.126 | 0.125 | 0.106 | −1%* | −15% | −15% | none | merge order |
| cluster split | 0.022 | 0.020 | 0.004 | −9%* | −81% | −83% | none (noise) | **numba counting sort** |
| cone fits | 0.254 | 0.254 | 0.108 | −0%* | −57% | −57% | none | **numba fit kernels** |
| **detect() total** | 0.808 | 0.795 | 0.377 | −2%* | −53% | −53% |  |  |

### RANSAC, dense synthetic (~98.6k points)

| Stage | pre | mine | theirs | mine % | theirs % | cumul. % |
|---|---|---|---|---|---|---|
| decode | 0.888 | 0.171 | 0.129 | −81% | −25% | −85% |
| range crop + misc | 0.669 | 0.668 | 0.160 | −0%* | −76% | −76% |
| ground removal | 2.065 | 2.058 | 0.140 | −0%* | −93% | −93% |
| rotate + shift | 0.197 | 0.198 | 0.078 | +1%* | −61% | −61% |
| tall-column veto | 0.005 | 0.005 | 0.003 | +0%* | −46% | −46% |
| DBSCAN | 0.266 | 0.260 | 0.186 | −2%* | −29% | −30% |
| cluster split | 0.281 | 0.076 | 0.008 | −73% | −89% | −97% |
| cone fits | 3.285 | 3.274 | 1.324 | −0%* | −60% | −60% |
| **detect() total** | 6.905 | 6.684 | 1.923 | −3%* | −71% | −72% |

### MRF, sim scans (~8.2k points)

| Stage | pre | mine | theirs | mine % | theirs % | cumul. % | Cause (mine) | Cause (theirs) |
|---|---|---|---|---|---|---|---|---|
| decode | 0.123 | 0.023 | 0.023 | −81% | −3%* | −82% | **no copies** | none (noise) |
| range crop + misc | 0.105 | 0.071 | 0.022 | −32% | −70% | −79% | none (indirect, see note below) | **numba fused pass** |
| ground removal | 4.123 | 1.284 | 1.142 | −69% | −11% | −72% | **numba plane fit + smaller graph** | numba fusion, compiled opening |
| rotate + shift | 0.101 | 0.066 | 0.049 | −34% | −26% | −51% | none (indirect, see note below) | **numba** fused select+rotate |
| tall-column veto | 0.006 | 0.004 | 0.003 | −38% | −29% | −57% | none (indirect, see note below) | dense grid, no sort |
| DBSCAN | 0.196 | 0.133 | 0.132 | −32% | −1%* | −32% | none (indirect, see note below) | none (noise) |
| cluster split | 0.035 | 0.020 | 0.005 | −41% | −76% | −86% | sort once | **numba counting sort** |
| cone fits | 0.347 | 0.235 | 0.106 | −32% | −55% | −69% | none (indirect, see note below) | **numba fit kernels** |
| **detect() total** | 5.030 | 1.970 | 1.539 | −61% | −22% | −69% |  |  |

### MRF, dense synthetic (~98.6k points)

| Stage | pre | mine | theirs | mine % | theirs % | cumul. % |
|---|---|---|---|---|---|---|
| decode | 1.363 | 0.170 | 0.167 | −87% | −2%* | −88% |
| range crop + misc | 1.059 | 0.662 | 0.161 | −38% | −76% | −85% |
| ground removal | 6.770 | 2.708 | 2.343 | −60% | −13% | −65% |
| rotate + shift | 0.302 | 0.198 | 0.086 | −35% | −57% | −72% |
| tall-column veto | 0.008 | 0.005 | 0.003 | −36% | −37% | −60% |
| DBSCAN | 0.391 | 0.250 | 0.191 | −36% | −24% | −51% |
| cluster split | 0.381 | 0.069 | 0.009 | −82% | −87% | −98% |
| cone fits | 4.587 | 2.876 | 1.199 | −37% | −58% | −74% |
| **detect() total** | 13.683 | 6.900 | 4.064 | −50% | −41% | −70% |

### p95 per stage (appendix)

RANSAC, sim scans:

| Stage | pre | mine | theirs | mine % | theirs % | cumul. % |
|---|---|---|---|---|---|---|
| decode | 0.104 | 0.029 | 0.028 | −72% | −4% | −73% |
| range crop + misc | 0.086 | 0.080 | 0.021 | −6% | −73% | −75% |
| ground removal | 0.334 | 0.282 | 0.108 | −16% | −62% | −68% |
| rotate + shift | 0.070 | 0.071 | 0.041 | +1% | −41% | −41% |
| tall-column veto | 0.004 | 0.004 | 0.002 | −2% | −50% | −51% |
| DBSCAN | 0.147 | 0.145 | 0.124 | −1% | −15% | −16% |
| cluster split | 0.035 | 0.026 | 0.005 | −23% | −82% | −86% |
| cone fits | 0.469 | 0.461 | 0.204 | −2% | −56% | −56% |
| **detect() total** | 1.172 | 1.103 | 0.487 | −6% | −56% | −58% |

MRF, sim scans:

| Stage | pre | mine | theirs | mine % | theirs % | cumul. % |
|---|---|---|---|---|---|---|
| decode | 0.232 | 0.085 | 0.032 | −64% | −62% | −86% |
| range crop + misc | 0.166 | 0.110 | 0.036 | −34% | −68% | −79% |
| ground removal | 6.043 | 1.896 | 1.549 | −69% | −18% | −74% |
| rotate + shift | 0.134 | 0.099 | 0.061 | −26% | −39% | −55% |
| tall-column veto | 0.008 | 0.005 | 0.003 | −44% | −29% | −60% |
| DBSCAN | 0.302 | 0.208 | 0.155 | −31% | −25% | −49% |
| cluster split | 0.054 | 0.029 | 0.006 | −46% | −81% | −90% |
| cone fits | 0.714 | 0.472 | 0.228 | −34% | −52% | −68% |
| **detect() total** | 7.170 | 2.800 | 2.069 | −61% | −26% | −71% |

Dense-cloud p95 `detect()` totals follow the same pattern: RANSAC 9.93 → 8.84 → 2.44 ms, MRF 19.07 → 8.71 → 5.56 ms.

**Tails.** On the quiet machine the p95/p10 ratio of the `detect()` total is 1.3–1.45× in every state (RANSAC
sim: 1.45 → 1.39 → 1.29; MRF sim: 1.43 → 1.42 → 1.34).
An earlier measurement under heavy load showed MRF's tail shrinking from 2.2× to 1.2×; that was
contention from other jobs, not a property of the code, and has been dropped. The single worst frame
did fall (MRF sim: 10.2 → 4.0 → 2.8 ms; RANSAC dense:
10.7 → 10.1 → 3.0 ms), but a single maximum is a noisy statistic.

**MRF side effect (not explained).** In MRF, stages whose code Part A did not touch also run 30–40% faster in
"mine" than in "pre" (sim scans: range crop 0.105 → 0.071, rotate 0.101 → 0.066, DBSCAN 0.196 → 0.133, cone
fits 0.347 → 0.235 ms). The effect is reproducible across runs and absent for RANSAC, so it is not noise. Its
cause was not investigated. A plausible but **unverified** explanation is that the old MRF path's
multi-threaded NumPy/BLAS calls interfere with the single-threaded numba stages that run after it.
Because of this, the per-stage "mine" percentages for MRF stages other than ground removal should not be read
as the effect of any specific change.


---

## 4. Part A: changes made in this session

### A1. Point-cloud decode (`cone_detection_node.py`, `pointcloud2_to_xyz`)

**Before.** Each of x, y, z was copied out of the raw byte buffer with `.copy()` (to make the 4-byte columns
contiguous for `.view(float32)`), the three columns were stacked into a new array, and
`ascontiguousarray` ran on top: about four full-size allocations per scan.

**After.** x, y and z are read through strided `np.ndarray` views over the message buffer
(stride = `point_step`, offset = the field's byte offset) and each is written once into a single
preallocated `(N, 3)` float32 result.

**Why it is correct for every layout.** The strided view works for any `point_step`, including the packed
26-byte Hesai layout (which is not a multiple of 4) that an earlier bug fix exists for; unaligned reads are
fine for NumPy.

**Effect.** Decode 0.082 → 0.021 ms (−75%) on sim scans and 0.888 → 0.171 ms (−81%) on ~98k points.
In the container benchmark, decode was −71%. It scales with point count, so the real car cloud benefits
more than the sim.

**Tests.** The existing `test_pointcloud_decode.py` (26-byte, 32-byte and 16-byte layouts) passes when the
ROS modules are stubbed. It could not be run unstubbed here because `rclpy` is not available on the host.

### A2. Cluster split (`cone_detection.py`)

**Before.** For every cluster label, `np.array(clean_data[labels == label])` built a full-length boolean
mask and copied the matching rows: O(clusters × points), plus a redundant `np.array` copy.

**After.** One stable `argsort` on the labels, one gather, then `np.split` at the label boundaries, so each
cluster is a view of a single sorted copy. Group order is unchanged (ascending label, noise `-1` first,
exactly what `np.unique` gave), so downstream handling is identical.

**Effect.** Cluster split is within noise on RANSAC sim scans (0.022 → 0.020 ms) and 0.281 → 0.076 ms (−73%) on the dense
cloud, where cluster counts are large. This was **superseded in cost** by Part B's counting sort (B5), which
replaces it.

### A3. Above-ground viz cloud (`cone_detection.py`)

`last_outlier_xyz` used to be converted to a contiguous float32 copy on every scan, but it is only ever
published (and `xyz_to_pointcloud2` already casts to float32, and only when someone subscribes). The
eager copy was removed; `last_outlier_xyz` is now the float64 array as it comes out of the rotation. This
saves one copy per scan.

### A4. RANSAC design-matrix build (`ground_methods.py`, `ground_ransac`)

`np.c_[np.ones(n), data]` allocated the ones column and then the result. It was replaced by a single
`np.empty` plus two assignments. Effect: within noise on the quiet machine (−5% of the RANSAC ground stage; an earlier loaded run showed −7%). **Superseded** by Part B's
`ransac_ground_mask` (B2), which avoids the augmented matrix altogether.

### A5. Plane-fit kernels (`ground_methods.py`)

These replace the NumPy implementations that dominated MRF's setup. They are used by `mrf` (the coarse
GPF plane that levels the cloud, and the final refit) and by every other method that calls `fit_plane` /
`_gpf_plane`. They are not used by `ransac`.

**`fit_plane` → `_fit_plane_nb`.** The old code did `points.mean(axis=0)`, then `u = points - c`
(allocating and filling a float64 `(N,3)` array from float32 input), then `u.T @ u`, then `eigh`. The new
numba kernel makes two passes over the points with float64 accumulators: one for the centroid, one for the
centred covariance moments (centring is done per point inside the loop, nothing is materialised). It then
takes the eigenvector of the 3×3 covariance with the smallest eigenvalue. Centring per point matters:
summing raw products would cancel catastrophically, because the centimetre-level height spread sits next
to metres of x/y range. A test with a 25 m x/y extent guards this.

**`_gpf_plane` → `_gpf_grow`.** Zermas ground-plane fitting seeded on the lowest points and grown for up to
ten iterations. The old loop built a boolean mask, gathered `pts[mask]`, fitted, computed residuals and
gathered again. The new kernel keeps the set as a boolean mask over the original array and does the
moment accumulation, the plane fit, the residual test and the re-mask in place. The low-percentile seed is
computed with `np.partition` at the two interpolation ranks (same linear interpolation as
`np.percentile`, without the full sort).

**Effect (microbenchmark, 40 real scans):** `fit_plane` 0.251 → 0.042 ms, `_gpf_plane` 1.171 → 0.113 ms.
Agreement with the old code: planes within ~1e-6 for `fit_plane`; `_gpf_plane` normals within 0.013° and
bias within 2.5e-7 m (the residual difference comes from partition-based percentile ranking).
In the container benchmark (an earlier, moderately loaded run, not repeated), MRF's ground stage fell from 5.9 to 3.57 ms with **identical detections**
(recall 0.5691, precision 1.0, 6.67 cones/frame), so this change is behaviour-neutral.

### A6. MRF graph reduction (`ground_methods.py`)

MRF labels ground/object on a cell grid and solves the labelling exactly with a graph cut (scipy
`maximum_flow`, Dinic). On a real scan the grid is about 75×139 = 10.4k cells but only ~3.3k contain
points, and the graph had ~45k edges. Two reductions were added.

**Decided-cell elimination (`_persistent_states`) — exact.** Some cells have the same label in *every*
optimal labelling and can be fixed instead of cut. A set G is ground in every optimal labelling if each of
its cells `c` satisfies `cost_ground[c] − cost_object[c] + Σ w(c,n) over neighbours n not in G < 0`.
*Proof sketch:* flip the object cells T ⊆ G of an optimal labelling to ground. Unaries change by
`Σ_T (cost_ground − cost_object)`, edges inside T are unchanged, edges between T and the rest of G drop
(they were cut), and each edge from T to a cell outside G adds at most `w`. The energy changes by at most
`Σ_T (cost_ground − cost_object + Σ_{n∉G} w) < 0`, a contradiction unless T is empty. The largest such set is
found by starting from every cell with `cost_ground < cost_object` and removing violators until stable
(removing only raises the others' boundary sums, so order does not matter); object cells use the same
argument with the roles swapped. Needs `w ≥ 0`. Fixed cells' edges are folded into their neighbours'
unary costs, and a compact CSR graph over only the undecided cells is cut (`_reduce_graph`, `_sub_csr`,
`_grid_mincut_reduced`).

A simple *per-cell* test (a cell's unary difference beats the sum of its neighbour weights) decides only
**0.1%** of cells (9 of ~10.5k) because ground cells have `cost_ground ≈ 0`, `cost_object = 1` and four
neighbours of weight 0.5. The *set-based* test above fixes 1,552 of the 3,327 occupied cells (47%),
removing 15–23% of all graph nodes.

**Reduced graph (`_MRF_GRAPH = "reduced"`) — approximate.** Empty cells have no points and zero unary cost;
they only chain their neighbours together. Dropping them and cutting only the occupied cells (with the
persistency step run inside that smaller graph) shrinks the max-flow problem from ~10.5k nodes to ~1.8k
(and edges from ~45k to ~4–6k).

**Modes** (module constant `_MRF_GRAPH`):

| Mode | Graph cut | Labels vs full | Ground-stage time (sim scan) |
|---|---|---|---|
| `"full"` | every cell, no reduction (original) | reference | 3.83 ms |
| `"exact"` | every cell, minus proven cells | identical (0 differences) | 4.25 ms (slower: proof pass costs more than it saves) |
| `"reduced"` (**default**) | occupied cells only, minus proven cells | 0.4 of ~3,300 occupied cells differ on average (max 2); ≤4 of ~8,200 points | 3.03 ms |

The ground-stage times above are medians over 40 real scans from a local numba/scipy environment (not the
container image), so compare the three rows with each other, not with section 3.

PyMaxflow (the optional GPL backend) always cuts the full grid. `"exact"` is slower than `"full"` on its
own; the gain only appears combined with dropping empty cells.

**Behaviour change to be aware of.** `"reduced"` is not bit-identical to the original. On the sim bag it
raised recall from 0.569 to 0.587 (+61 true positives per 300 frames), precision stayed 1.0, and the mean
match error rose slightly (26.9 → 27.7 mm); cones per frame went 6.67 → 6.87. It looks beneficial here
but is one bag. To revert: set `_MRF_GRAPH = "full"` in `ground_methods.py`.

**Tests added:** brute-force optimality of the reduced cut on grids up to 3×4 (including empty cells and
zero weights); every proven cell takes its label in all minimum-energy labellings (brute force);
`exact` equals `full` on random 40×55 grids; fully-decided, empty and all-object grids; the three modes
agree on a scene; unknown mode raises.

### A7. MRF JIT warm-up (`ground_methods.py` `warm_mrf_graph`, `cone_detection.py`)

The configure-time warm-up replays `detect()` on a synthetic scene (flat floor + two cones). With
decided-cell elimination that scene is *fully decided*, so the flow kernels (`_sub_csr`, `_residual_reach`)
never ran during warm-up and compiled on the first real scan: **a ~1 s stall (608 ms in a cold-cache test;
~1,030 ms in the container benchmark's first frame)**. `warm_mrf_graph()` runs the cut on a small grid
with an object and gaps, which always leaves undecided cells, and `warmup_numba_functions` calls it when
`ground_method == "mrf"`. First scan after warm-up: 608 → 5 ms (cold cache); benchmark first frame
~1,030 → ~58 ms. A test checks that a real scan compiles no new signature after warm-up.

---

## 5. Part B: changes made by the other session

Read from the working-tree diffs; effects are the measured Part B columns in section 3. Not authored by
me; the full suite, including their added tests, passes on the final tree (section 6).

### B1. `range_crop` (`scan_kernels.py`, used in `cone_detection.py`)

One pass and one allocation replace the chain of squared columns, two boolean masks and a gather. Rows
beyond the crop radius (25 m) or inside the minimum range are dropped; non-finite rows fail the comparisons
and are dropped. Accounts for most of the drop in "range crop + misc" (−73% on sim scans, −76% on the dense cloud).

### B2. `ransac_ground_mask` (`ransac.py`, used in `ground_ransac`)

The numba RANSAC plane fit restructured around what dominated its time on 100–170k point scans:

* reads the raw float32 `(N,3)` cloud in place, with no float64 bias-augmented copy and no
  `A[:, :-1]` / `A[:, -1]` temporaries;
* the scoring subsample is `iter_subsample_max` independent random draws (O(n_sub)); the old
  `np.random.choice(..., replace=False)` is a full O(N) permutation in numba. Sampling with replacement is
  acceptable because the subsample only estimates an inlier *ratio*;
* candidate planes are scored in a scalar loop over the subsample (no per-iteration `dot` result array, no
  BLAS threads waking for 5,000 points);
* the final pass writes the outlier mask directly instead of building an index array to invert.

Same plane test (vertical distance below the threshold), same warm-start-as-candidate-zero and
adaptive-iteration logic. A positive `ransac_threshold` is required; a threshold of 0 (the MAD fallback)
or non-3-D data falls back to the original `ransac2` path. Collinear/vertical candidate planes are
rejected; non-finite points count as outliers. **Effect:** ground removal 0.249 → 0.095 ms (−62%) on sim
scans and 2.058 → 0.140 ms (−93%) on the dense cloud. RANSAC cone counts were unchanged within noise
(7.33 → 7.32 → 7.32 per sim scan).

### B3. `rotate_outliers`

Select-outliers, rotate-to-ground and subtract-floor now happen in one pass over the cloud, allocating the
result once at its final size (no gathered float32 copy, no upcast copy for the matrix product).
Rotate stage −41% (sim) / −61% (dense).

### B4. `tall_column_keep`

The tall-column veto marks the xy cells containing a return above the height threshold (plus a 3×3
dilation) and drops points in marked cells. The old version built int64 cell keys, `unique`d them and used
`isin`. The new one marks a small dense boolean grid over the tall returns' bounding box and does three
passes; it falls back to the sort path if the grid would exceed 4M cells. Veto −50% (sim); absolute cost
is tiny either way (~0.007 ms).

### B5. `group_by_label` and `cluster_stats`

Replaces my argsort+split (A2). A counting sort groups points by label (ascending, original order inside a
group); `cluster_stats` reduces each group to the scalars the shape gates need (point count above the floor
margin, top and bottom height, centroid xy). Clusters rejected by the gates, which are most of them and the
largest, are never masked or copied; only survivors are sliced for the fit. Cluster split −81% (sim) /
−89% (dense) beyond A2.

### B6. DBSCAN grid kernel (`clustering.py`)

For `cluster_backend = grid` (the fallback in these runs; the `pypi` backend was not installed in the
image): cubes that touch are merged for every cube before any distant pair is examined (two sweeps), so
most distant pairs already share a component and skip the expensive point-by-point test. One cursor per
neighbour offset walks the sorted cube keys once instead of a binary search per (cube, offset). Components
do not depend on merge order. DBSCAN −15% (RANSAC sim) / −29% (RANSAC dense) / −24% (MRF dense); no visible effect on sim MRF scans (−1%).

### B7. Cone fit kernels (`cone_fit.py`)

* `_linearity_nb`: the cluster-linearity test (eigenvalue ratio of the 2×2 xy covariance) in closed form in
  numba, instead of mean/centre/matmul/`eigvalsh` on a handful of points where call overhead was the cost.
* `_template_fit_gn_cluster`: for the Gauss-Newton (`solver == "gn"`) path, builds the warm start (centroid
  biased toward the near face), the xy bounds and runs `_template_fit_gn` in a single compiled call per
  cluster, instead of two means and four min/max calls in Python before it.

Cone fits −57% (RANSAC sim) / −55% (MRF sim) / −60% (RANSAC dense) / −58% (MRF dense). These were the
largest remaining item for RANSAC (31% of its pre-change total).

### B8. MRF-specific and shared ground-method changes (`ground_methods.py`)

* **`_level_cells_nb` / `_level_cells`:** fuses the projection into the levelled frame with the cell-index
  computation. `uv` is never stored and each point's cell coordinates are computed once (the grid origin
  is known only after the first pass, so they are kept and offset in the second). Used by `mrf`, and also by
  `height_jump`, `groundgrid` and `pmf`.
* **`_grey_opening`:** a compiled separable min-then-max window filter replacing
  `scipy.ndimage.grey_opening` on these few-thousand-cell grids, where the scipy call was mostly overhead.
  Valid for odd window sizes; scipy's default reflect border gives the same result as a window clipped at
  the border.
* **`_fit_plane_ground`:** the final refit (`_refit_on_ground`) reads the ground rows in place inside one
  kernel (thinned to at most 20,000 points by stride), with no index array and no gathered copy.
* **`_gpf_grow` pass merging:** the pass that rebuilds the mask also sums its points, so an iteration is
  two passes instead of three. (A refinement of my A5 kernel; same sums in the same order.)

MRF ground stage 1.284 → 1.142 ms on top of Part A (−11%).

### Tests the other session added

By name in `test_ground_methods.py`: `test_ransac_ground_mask_matches_ransac2_plane_test`,
`test_refit_on_ground_matches_gathered_fit`, `test_level_cells_matches_level_then_cell_index`,
`test_grey_opening_matches_scipy`. Their pass status on the final tree is **unconfirmed** (see below).

---

## 6. Test status and risks

* **After Part A only:** the full suite passed (107 passed, 9 skipped; the ROS-dependent decode test passed
  under stubs). This is the last state I verified end-to-end.
* **On the combined tree (A + B), re-run on a quiet machine (load ~1) after the measurements:**
  111 passed, 9 skipped, identical over four consecutive runs (25–35 s each), and the ROS-dependent
  decode test passed under stubs (3 passed). The 111 includes the other session's four added tests. The
  skips are optional packages not installed in this environment (`pypatchworkpp`, `CSF`, `maxflow` /
  PyMaxflow, `dbscan` / the `pypi` DBSCAN backend) and `range_image`, which is covered by its raycast test.
  Unverified by tests: the PyMaxflow-equivalence test and the `pypi` backend.
* **Timings and tests share one tree:** the timings in section 3 were taken from a snapshot of the working tree at 22:37:56 on 2026-11-12 and the suite ran on the same files (`ransac.py` included) at ~22:30–22:40.
* **Known intermittent test:** `test_elevated_sensor_cone_fit_uses_height_above_ground` fails ~1.5% of the
  time (33/2000 after, 30/2000 before my changes), from unseeded RANSAC randomness: ground points just
  below the fitted plane can end up as outliers with negative height. Pre-existing; seeding RANSAC in the
  test would fix it.
* **Behaviour differences to review:**
  * MRF `"reduced"` graph changes a few labels (A6); recall up, match error +0.8 mm on one bag.
  * RANSAC subsample is now drawn with replacement (B2) and its random stream differs; cone counts matched
    within noise.
  * `last_outlier_xyz` is float64 now (A3), not float32.
* **Unmeasured:** a real car point cloud (the sim clouds are ~8k points, the car's ~100k+); the other
  ground methods; the `pypi` DBSCAN backend (not installed in the benchmark image).

---

## 7. What is left

Profile of `ground_mrf` taken **before** the later changes (sum of medians 4.5 ms on 8.2k-point scans):
plane fit (`_gpf_plane`) 1.07 ms, max-flow 1.03 ms, two nearest-cell fills 0.51 ms each (scipy
`distance_transform_edt`), grey opening 0.42 ms, final refit 0.40 ms. Parts A and B addressed the plane
fits, the graph size and the opening; what remains is unmeasured since but should be dominated by:

* **Max-flow** on the reduced graph (~0.4 ms), scipy Dinic. PyMaxflow (Boykov–Kolmogorov) would likely be
  faster but is GPL, so it is not a drop-in for a published image.
* **The two `_fill_nearest` calls** (~1 ms together), scipy's EDT on a mostly empty grid. A proper numba
  exact EDT (Felzenszwalb) should beat it; a naive numba attempt was slower and was not adopted.

For RANSAC, after Part B the remaining time is dominated by cone fits and DBSCAN.

---

## 8. Reproducing

* **Detection accuracy / ground truth:** from the repo root,
  `python run_perception_benchmark.py <bag> --profile --profile-frames 300 --max-frames 300
  --sanity-scan -1 [--pipeline-overrides overrides.json]` where `overrides.json` is
  `{"cone_detection": {"ground_method": "mrf"}}`. The sim bag used is
  `bags/trackdrive_track_20260404_013721_20260620_103753`. The benchmark's own stage statistics are
  mean/median/p95/max per stage, which are inflated on a loaded machine.
* **Per-stage tail timing (this document's tables):** an in-container harness that runs
  `detect()` over the 40 extracted scans plus the dense synthetic workload and records p10, median, p95
  and max per stage for each code state. It lives in the session scratchpad, not in the repo; it can be
  added to `tools/` if wanted.
* **To switch MRF back to the original graph:** set `_MRF_GRAPH = "full"` in
  `cone_detection/ground_methods.py`.
