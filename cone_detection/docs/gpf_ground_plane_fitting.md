# Ground Plane Fitting (GPF) — Full Derivation

This document explains, step by step and with proofs where they exist, the GPF
ground-removal method in `cone_detection/cone_detection/ground/methods/` (`plane_fitting.py`, `plane_methods.py`):

- `ground_gpf` (line 153) — the registered method (`ground_method: gpf`)
- `_gpf_plane` (line 91) — the algorithm itself
- `_subsample` (line 66), `fit_plane` (line 72), `plane_residual` (line 83) — helpers

It builds on the total-least-squares derivation in
[`total_least_squares_plane_fit.md`](total_least_squares_plane_fit.md); results
from there are cited as "TLS §k".

```python
def _gpf_plane(pts, thr, n_iter=10, seed_band_m=0.12, seed_pct=2.0):
    if len(pts) < 3:
        return _DEFAULT_PLANE.copy()
    lpr = np.percentile(pts[:, 2], seed_pct)
    seeds = pts[pts[:, 2] < lpr + seed_band_m]
    plane = _DEFAULT_PLANE.copy()
    for _ in range(n_iter):
        if len(seeds) < 3:
            break
        plane = fit_plane(seeds)
        grown = pts[np.abs(plane_residual(pts, plane)) < thr]
        converged = len(grown) == len(seeds)
        seeds = grown
        if converged:
            break
    return plane


def ground_gpf(data, cfg):
    thr = cfg.ground_threshold_m                       # 0.05 m default
    plane = _gpf_plane(_subsample(data), thr)
    outlier = np.abs(plane_residual(data, plane)) >= thr
    return outlier, plane
```

---

## 0. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $p_1,\dots,p_N$ | points in the **sensor** frame, $p_i = (x_i, y_i, z_i)$ | `pts` |
| $P = (n, d)$ | a plane, $\lVert n\rVert = 1$, $n_z > 0$ | `plane` |
| $r_i(P) = n^\top p_i + d$ | signed distance of $p_i$ to $P$ (TLS §1) | `plane_residual` |
| $t$ | inlier threshold | `thr` = `cfg.ground_threshold_m` = 0.05 m |
| $q$ | seed percentile | `seed_pct` = 2 |
| $b$ | seed band height | `seed_band_m` = 0.12 m |
| $A \subseteq \{1..N\}$ | a set of points treated as ground | `seeds` / `grown` |
| $\operatorname{TLS}(A)$ | total-least-squares plane through the points of $A$ | `fit_plane` |

---

## 1. The algorithm at a glance

1. **Subsample** the cloud to at most 20 000 points (fit only).
2. **Seed:** $\text{LPR} = $ the $q$-th percentile of $z$; $A_0 = \{\, i : z_i < \text{LPR} + b \,\}$.
3. **Iterate** for $k = 0, 1, \dots$ (at most 10 times):
   $$
   P_{k+1} = \operatorname{TLS}(A_k), \qquad
   A_{k+1} = \{\, i : |r_i(P_{k+1})| < t \,\}
   $$
   stop when $|A_{k+1}| = |A_k|$.
4. **Label** every point of the *full* cloud: ground iff $|r_i(P)| < t$.

The rest of the document justifies each of these.

---

## 2. Subsampling (`_subsample`)

```python
return data if len(data) <= cap else data[:: int(np.ceil(len(data) / cap))]
```

Keeps every $k$-th point with $k = \lceil N / 20000 \rceil$.

**Why it's safe.** A plane has 3 degrees of freedom. For a TLS fit on $m$
inliers with noise $\sigma$ spread over in-plane extent with variance
$s^2$, the normal's angular error scales like

$$
\delta\theta \sim \frac{\sigma}{\sqrt{m}\; s}.
$$

With $m = 20\,000$, $\sigma = 2$ cm, $s \approx 5$ m this is ~$3\times10^{-5}$ rad
— far below anything that matters. More points would only cost time.

**Why stride and not random.** LiDAR points arrive ordered by ring and azimuth,
so a fixed stride spreads samples evenly over the scan, and the result is
deterministic (same frame → same plane), which a random sample is not.

**Note.** The fit uses the subsample; the final labelling (§8) uses every point.

---

## 3. Seeding

### 3.1 The lowest-point representative (LPR)

$$
\text{LPR} = Q_q(z_1, \dots, z_N), \qquad q = 2\%
$$

the 2nd percentile of the $z$ coordinates.

**Why not the minimum?** LiDAR produces *multipath* returns: the beam reflects
off a surface and reports a point **below** the true floor. Compare the
robustness of three estimators of "the floor's lowest height" against $k$ bad
points at depth $D$ below the floor:

| Estimator | Effect of the bad points |
|---|---|
| $\min_i z_i$ | one point ($k=1$) drags it down by $D$ — unbounded |
| mean of the $N_{\text{LPR}}$ lowest (Zermas et al.) | shifts by $\approx kD/N_{\text{LPR}}$ — grows with $D$ |
| $q$-th percentile (this code) | **independent of $D$** while $k < qN/100$ |

*Why the percentile is independent of $D$:* the $q$-th percentile is the value
with $qN/100$ points below it. If $k < qN/100$ points are moved arbitrarily far
down, the order statistic at rank $qN/100$ is still a floor point; it moves only
from the floor's $q$-th percentile to its $(q - 100k/N)$-th percentile. How far
down the bad points are no longer matters. Its **breakdown point** is $q$: up to
2% of the cloud can be junk below the floor before the LPR leaves the floor.

### 3.2 The seed set

$$
A_0 = \{\, i : z_i < \text{LPR} + b \,\}, \qquad b = 0.12 \text{ m}
$$

This is a **half-space in the sensor frame** — everything below a horizontal
(in sensor coordinates) cut. It contains the lowest 2% of the cloud plus
everything up to 12 cm above that level.

The band $b$ must be larger than the floor's noise thickness (so the seeds are
mostly floor rather than a sliver of the lowest noise), and smaller than an
obstacle's height (so few non-floor points get in). 12 cm is a compromise: a
cone (~33 cm) is mostly excluded, but its bottom ~12 cm can enter (§5.1 shows
why that is harmless).

### 3.3 Geometry: what the seeds look like with a tilted sensor

The sensor is pitched ~9° relative to the floor. If $s$ is distance along the
floor in the direction of the tilt, the floor's sensor-frame height is
approximately

$$
z(s) \approx z_0 - s \sin\theta, \qquad \sin 9^\circ \approx 0.156 .
$$

So the floor is a **ramp** in sensor $z$, dropping 15.6 cm per metre. The seed
half-space $z < \text{LPR} + b$ cuts this ramp, keeping

- the part of the floor below the 2% level (the lowest end of the ramp), plus
- a further strip of depth $b / \sin\theta = 0.12/0.156 \approx 0.77$ m.

Which end of the floor is "low" depends on the direction of the tilt:

- **Low end far away** (floor points sparse there): the lowest 2% is a deep
  stretch, and the seeds cover a wide range.
- **Low end near the sensor** (points dense there): the lowest 2% is a thin
  stretch, and the seeds form a strip only ~1 m deep.

The simulation in §9 shows both cases (seeds at 14–20 m vs 1.3–2.3 m).

### 3.4 The first fit is biased (truncation bias)

The seed set is selected by a cut in **sensor $z$**, which is *oblique* to the
real floor. Near the boundary of the cut, a floor point is kept only if its
noise happens to push it low; points whose noise pushed them up are dropped.
So along the strip, the kept points' noise is not zero-mean: it is zero-mean in
the interior and negative near the cut edge. That shears the strip, and the TLS
fit of the seeds is tilted — a **systematic** error, not a random one.

It grows with the floor noise $\sigma$ (roughly as $\sigma^2$) and shrinks as the
strip gets deeper. Measured on the simulated scene of §9 (thin, near-field strip,
first fit only):

| floor noise $\sigma$ | normal error after fit 1 |
|---|---|
| 0 | 0.000° |
| 5 mm | 0.06° |
| 15 mm | 0.63° |
| 30 mm | 2.55° |

0.6° extrapolated to 10 m is ~10 cm — twice the threshold $t$. This is why GPF
must *iterate*: the grow step (§4) re-selects with a band that is **symmetric
about the current plane** ($|r| < t$), which truncates noise equally on both
sides and therefore does not have this bias.

---

## 4. The iteration is alternating minimisation of a truncated loss

This is the core result: the fit/grow loop is not a heuristic, it is a descent
method for a specific robust cost.

### 4.1 The cost

Define the **truncated quadratic** loss of a plane:

$$
F(P) = \sum_{i=1}^{N} \min\!\big(r_i(P)^2,\ t^2\big).
$$

Each point costs its squared distance if it is within $t$ of the plane, and a
fixed $t^2$ otherwise. Far points (cones, walls, multipath) all cost the same
$t^2$, so they cannot pull the plane — unlike plain TLS, where a point's pull
grows with $r^2$ (TLS §11.1).

Introduce the inlier set as an explicit variable:

$$
G(P, A) = \sum_{i \in A} r_i(P)^2 \;+\; (N - |A|)\, t^2 .
$$

**Lemma 1 (best set for a fixed plane).** For fixed $P$,
$\min_A G(P, A) = F(P)$, attained at $A = \{ i : |r_i(P)| < t \}$.

*Proof.* $G$ is a sum over points; each point independently contributes $r_i^2$
if $i \in A$ or $t^2$ if not. Choosing the smaller of the two for every point
gives $\min(r_i^2, t^2)$; it is $r_i^2$ exactly when $|r_i| < t$
(ties $|r_i| = t$ cost the same either way). $\blacksquare$

That is precisely the **grow** step:
`grown = pts[np.abs(plane_residual(pts, plane)) < thr]`.

**Lemma 2 (best plane for a fixed set).** For fixed $A$,
$\min_P G(P, A)$ is attained at $P = \operatorname{TLS}(A)$.

*Proof.* The term $(N-|A|)t^2$ does not depend on $P$, and the remaining term
$\sum_{i\in A} r_i(P)^2$ is exactly the TLS objective on $A$, whose global
minimiser is $\operatorname{TLS}(A)$ (TLS §3, §5.2). $\blacksquare$

That is precisely the **fit** step: `plane = fit_plane(seeds)`.

### 4.2 Monotone descent

**Theorem.** For $k \ge 1$, $F(P_{k+1}) \le F(P_k)$.

*Proof.* For $k \ge 1$, $A_k$ is the grow set of $P_k$, so by Lemma 1
$G(P_k, A_k) = F(P_k)$. Then

$$
F(P_{k+1})
\overset{\text{L1}}{=} \min_A G(P_{k+1}, A)
\;\le\; G(P_{k+1}, A_k)
\overset{\text{L2}}{\le} G(P_k, A_k)
\overset{\text{L1}}{=} F(P_k). \qquad\blacksquare
$$

The very first set $A_0$ (the seeds) does **not** come from a plane, so the
guarantee starts from $P_1$. The seeds are only an initialisation.

### 4.3 Termination and fixed points

- $G$ depends on $A$, and there are finitely many subsets, while $F$ never
  increases. If ever $A_{k+1} = A_k$, then $P_{k+2} = \operatorname{TLS}(A_{k+1})
  = P_{k+1}$: the iteration has reached a **fixed point** and nothing changes
  afterwards. Away from exact ties, it must reach one in finitely many steps.
  The cap `n_iter = 10` guarantees termination regardless.
- A fixed point $P^\star$ satisfies
  $$
  P^\star = \operatorname{TLS}\big(\{ i : |r_i(P^\star)| < t \}\big):
  $$
  **the plane is the TLS fit of its own inlier band.** That is the natural
  definition of "the floor plane": refitting on the points it explains gives it
  back.
- A fixed point is a **local** minimum of $F$. $F$ is non-convex: a wall, a
  sidewalk top or a large flat object face is also a fixed point. **Which one GPF
  reaches is decided entirely by the initialisation** — this is the real job of
  the lowest-point seeds: they start the descent inside the floor's basin of
  attraction.

### 4.4 Relation to other methods

| Method | Cost | Search |
|---|---|---|
| RANSAC | number of outliers, $\sum_i [\,\lvert r_i\rvert \ge t\,]$ | random hypotheses, global |
| MSAC (Torr & Zisserman) | truncated quadratic $\sum_i \min(r_i^2, t^2)$ — **same $F$ as GPF** | random hypotheses, global |
| LTS C-steps (Rousseeuw & Van Driessen) | sum of the $h$ smallest $r_i^2$ | same fit/reselect alternation, fixed **count** instead of fixed threshold |
| **GPF** | truncated quadratic $F$ | deterministic local descent from lowest-point seeds |

So GPF is a **deterministic local optimiser of the MSAC cost**, initialised
with a physical prior (the floor is at the bottom) instead of random samples.
It cannot lock onto a wall the way RANSAC sometimes does (the wall is never in
the initial basin), but it has no way to escape if the initialisation is wrong.

---

## 5. Why the descent lands on the floor (quantitatively)

### 5.1 Contamination is bounded

Any non-floor point inside an inlier set has $|r_i| < t$ by construction. If
$k$ of the $m$ points in $A$ are contaminants, the shift they cause in the TLS
centroid along the normal is at most

$$
\frac{k}{m}\, t .
$$

Example (§9): ~115 cone-base points among ~20 115 inliers, $t = 5$ cm →
$\le 0.3$ mm offset. Contaminants are not just few; their leverage is capped at
$t$ — this is the truncation in $F$ at work.

### 5.2 The support grows outward

After the first (biased) fit, the plane is accurate near the seeds and drifts
away from the true floor with distance (§3.4). The grow step accepts floor
points wherever the drift is still below $t$, so $A_1$ is wider than $A_0$. A
wider support means a longer lever arm, so $P_2$'s tilt is more accurate, so
$A_2$ reaches further, and so on until the band covers the entire floor. In the
near-field case of §9 the reach goes 2.3 m → 10.2 m → 20 m in three fits.

---

## 6. The stopping test

```python
converged = len(grown) == len(seeds)
```

This compares **sizes**, not sets.

- A true fixed point ($A_{k+1} = A_k$) implies equal sizes, so a fixed point is
  always detected.
- The converse is not guaranteed: two different sets of the same size also stop
  the loop. In practice this only happens once the plane has settled (points
  near $|r| = t$ swapping in and out); by §4.2 $F$ has already stopped improving
  meaningfully.
- In the simulation (§9), the count went 20 115 → 20 116 → 20 115 → 20 115: the
  loop ran one round past the effective convergence, then stopped.
- Since $|A_k| \le N$, comparing sets exactly (e.g. boolean masks) would cost the
  same $O(N)$; sizes are just simpler.

---

## 7. Edge cases

| Situation | What happens |
|---|---|
| $N < 3$ | returns `_DEFAULT_PLANE` = $[0,0,0,1]$, i.e. the plane $z = 0$ **through the sensor**; almost everything becomes an outlier |
| fewer than 3 seeds on the first round | loop breaks before any fit → same default plane |
| fewer than 3 inliers on a later round | loop breaks, **last good plane** is returned |
| 10 rounds without equal sizes | returns $P_{10}$; by §4.2 it is no worse than any earlier $P_k$ ($k\ge1$) |

`remove_ground` already returns "all outliers" for clouds with fewer than 3
points, so the first row only matters if `_gpf_plane` is called directly (it is,
by `_level`, `czm`, etc., but always on full clouds or zones).

---

## 8. Labelling (`ground_gpf`)

$$
\text{outlier}_i = \big[\, |r_i(P)| \ge t \,\big]
$$

on **every** point of the original cloud.

- It is the same rule as the grow step, so the labels are exactly the final
  inlier band — evaluated on the full cloud instead of the subsample.
- It uses $|r|$, so points far **below** the plane (multipath) are outliers too,
  not ground. The downstream clustering then has to tolerate a few such points;
  in practice they are sparse and do not form clusters.
- It is symmetric: a cone's bottom $t = 5$ cm is labelled ground. No
  threshold-based method can avoid this; it is why `thr` should stay small.
- The returned plane $P$ is the one the rest of the pipeline uses to rotate the
  outliers upright and to cull the floor margin.

---

## 9. Simulation

Synthetic scene: floor at $z = -1.5$ m with noise $\sigma = 1.5$ cm, 20 000 points
with density decreasing with range (1.5–20 m, ±34° azimuth), five 33 cm cones
(150 points each), rotated by ±9° about the sensor's $y$-axis. $t = 5$ cm. "Normal
error" is the angle between the fitted and true floor normals; $F$ is the
truncated loss of §4.1; "reach" is the floor-distance range covered by the grown
set.

**Low end far (seeds 14.0–20.0 m, 535 points):**

| fit | $\lvert A_k \rvert$ → $\lvert A_{k+1} \rvert$ | $F$ | normal error | reach |
|---|---|---|---|---|
| 1 | 535 → 20 115 | 6.145 | 0.005° | 1.3–20 m |
| 2 | 20 115 → 20 116 | 6.136 | 0.003° | 1.3–20 m |
| 3 | 20 116 → 20 115 | 6.136 | 0.003° | 1.3–20 m |
| 4 | 20 115 → 20 115 | 6.136 | 0.004° | stop |

**Low end near (seeds 1.3–2.3 m, 7 236 points):**

| fit | $\lvert A_k \rvert$ → $\lvert A_{k+1} \rvert$ | $F$ | normal error | reach |
|---|---|---|---|---|
| 1 | 7 236 → 16 484 | 17.907 | 0.611° | 1.3–10.2 m |
| 2 | 16 484 → 20 002 | 7.314 | 0.115° | 1.3–20 m |
| 3 | 20 002 → 20 116 | 6.142 | 0.005° | 1.3–20 m |
| 4 | 20 116 → 20 115 | 6.136 | 0.003° | 1.3–20 m |
| 5 | 20 115 → 20 115 | 6.136 | 0.004° | stop |

Observations:

- $F$ never increases (§4.2), and both runs reach the **same** fixed point.
- The near-field strip starts 0.6° off (truncation bias, §3.4) and needs 3 fits
  to reach full support (§5.2). With Zermas' typical 3 fixed iterations it would
  barely make it; the cap of 10 here leaves margin.
- Roughly 120 cone points (of 750) end up as ground: their bottom ~5 cm (§8).
  (The inlier count exceeds the 20 000 floor points by 115, minus the few floor
  points whose noise exceeds $t$.)

---

## 10. Cost

| Step | Cost |
|---|---|
| subsample | $O(N)$ (a view with a stride) |
| percentile | $O(m)$, $m \le 20\,000$ (`np.percentile` uses partial sorting) |
| per iteration | $O(m)$ residuals + $O(\lvert A\rvert)$ scatter matrix + $3\times3$ eigensolve |
| labelling | $O(N)$ residuals on the full cloud |

At most 10 iterations, so the total is a small constant times $N$.

---

## 11. Differences from Zermas et al. (ICRA 2017)

| | Paper | This code | Why |
|---|---|---|---|
| Seed level | mean of the $N_{\text{LPR}}$ lowest points | 2nd percentile | bounded influence of multipath (§3.1) |
| Planes | one per segment along $x$ (typically 3) | **one global plane** | the pipeline needs one plane; `czm` covers the segmented variant |
| Iterations | fixed $N_{\text{iter}}$ (~3) | up to 10, early stop on equal size | tilt-induced seed bias can need more rounds (§9) |
| Inlier test | $r < Th_{\text{dist}}$ | $\lvert r\rvert < t$ (symmetric) | rejects below-floor noise, unbiased band |

---

## 12. Limitations and failure modes

1. **One plane.** Slopes, crests, dips and curbs violate the model. The fixed
   point becomes the dominant flat region; floor outside it is labelled
   obstacle (or, on a crest, objects get absorbed). This is what the grid,
   ray, MRF and zone methods address.
2. **Wrong basin.** If the lowest 2% of the scan is not floor — a ditch, a pit,
   a lower road section in view, or more than 2% below-floor noise — the descent
   starts in the wrong basin and §4.3 says it will stay there.
3. **Noise-dependent seed bias.** Higher floor noise means a more tilted first
   fit (§3.4) and more rounds needed. With heavy noise and a thin near-field
   strip, 10 rounds might not be enough to reach the far floor.
4. **Cone bases.** The bottom $t$ of every object is ground (§8).
5. **Near-field dominance.** As in TLS §11.2, dense near returns dominate the
   fit; the far floor is the first to drift off the plane on a gentle slope.

---

## 13. Code mapping

| Code | Math | Section |
|---|---|---|
| `_subsample(data)` | stride subsample to $\le 20\,000$ | §2 |
| `np.percentile(pts[:, 2], seed_pct)` | $\text{LPR} = Q_{2\%}(z)$ | §3.1 |
| `pts[pts[:, 2] < lpr + seed_band_m]` | $A_0 = \{z_i < \text{LPR} + b\}$ | §3.2 |
| `plane = fit_plane(seeds)` | $P_{k+1} = \operatorname{TLS}(A_k) = \arg\min_P G(P, A_k)$ | §4.1, Lemma 2 |
| `np.abs(plane_residual(pts, plane)) < thr` | $A_{k+1} = \arg\min_A G(P_{k+1}, A)$ | §4.1, Lemma 1 |
| `len(grown) == len(seeds)` | size-based fixed-point test | §6 |
| `range(n_iter)` with `n_iter=10` | hard termination | §4.3 |
| `len(seeds) < 3: break` | degenerate set guard | §7 |
| `np.abs(plane_residual(data, plane)) >= thr` | final labels on the full cloud | §8 |

---

## 14. Summary

$$
A_0 = \{ z_i < Q_{2\%}(z) + b \}
\;\longrightarrow\;
\Big( P_{k+1} = \operatorname{TLS}(A_k),\;\; A_{k+1} = \{ |r_i(P_{k+1})| < t \} \Big)^{\le 10}
\;\longrightarrow\;
\text{ground} = \{ |r_i(P)| < t \}
$$

1. The **percentile** seed level is immune to how deep multipath noise goes, up
   to 2% of the cloud.
2. The seeds only **initialise**; they are biased when the sensor is tilted and
   the floor is noisy.
3. The fit/grow loop is **alternating minimisation** of the truncated quadratic
   (MSAC) cost $F$: it never increases $F$ and stops at a plane that is the TLS
   fit of its own inlier band.
4. $F$ is non-convex; the lowest-point initialisation is what selects the floor
   among its local minima.
5. Contaminating points have leverage capped at $t$, and the support grows
   outward each round, which is why a handful of rounds is enough.
