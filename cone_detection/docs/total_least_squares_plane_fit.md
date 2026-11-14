# Total Least Squares Plane Fitting — Full Derivation

This document derives, from first principles, the plane fit used in
`cone_detection/cone_detection/ground/methods/plane_fitting.py` (`fit_plane`):

```python
def fit_plane(points: np.ndarray) -> np.ndarray:
    """Total-least-squares plane through ``points`` -> ``[bias, n_x, n_y, n_z]``."""
    c = points.mean(axis=0)
    u = points - c
    _, vecs = np.linalg.eigh(u.T @ u)
    n = vecs[:, 0]
    if n[2] < 0:
        n = -n
    return np.array([-float(n @ c), n[0], n[1], n[2]], dtype=np.float64)
```

Every line of that function corresponds to one step below. A mapping table is at
the end (§12).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $p_1, \dots, p_N \in \mathbb{R}^3$ | the input points (column vectors) |
| $n \in \mathbb{R}^3$ | plane normal |
| $d \in \mathbb{R}$ | plane offset ("bias" in the code) |
| $c = \frac{1}{N}\sum_i p_i$ | centroid |
| $u_i = p_i - c$ | centred points; note $\sum_i u_i = 0$ |
| $S = \sum_i u_i u_i^\top$ | $3\times 3$ scatter matrix (= $N\cdot$covariance) |
| $\lambda_1 \le \lambda_2 \le \lambda_3$, $e_1, e_2, e_3$ | eigenvalues / orthonormal eigenvectors of $S$ |

A plane is the set $\{\,p : n^\top p + d = 0\,\}$.

---

## 1. Point-to-plane distance

**Claim.** If $\lVert n \rVert = 1$, the signed perpendicular distance from a point
$p$ to the plane is $n^\top p + d$.

**Proof.** Take any point $q$ on the plane, so $n^\top q = -d$. The perpendicular
distance from $p$ is the length of the projection of $p - q$ onto the unit
normal direction $n/\lVert n\rVert$:

$$
\operatorname{dist}(p) = \frac{n^\top (p - q)}{\lVert n \rVert}
= \frac{n^\top p - n^\top q}{\lVert n \rVert}
= \frac{n^\top p + d}{\lVert n \rVert}.
$$

With $\lVert n\rVert = 1$ this is $n^\top p + d$. Its sign tells which side of the
plane $p$ lies on (positive on the side $n$ points to). $\blacksquare$

This is exactly `plane_residual(data, coefs) = data @ coefs[1:] + coefs[0]`.

---

## 2. The optimisation problem

Total least squares (TLS) chooses the plane minimising the sum of **squared
perpendicular** distances:

$$
\boxed{\;\min_{n,\,d}\; E(n,d) = \sum_{i=1}^{N} \left(n^\top p_i + d\right)^2
\quad \text{subject to} \quad n^\top n = 1\;}
$$

**Why the constraint is necessary.** Without it, $(n,d) = (0,0)$ gives $E = 0$,
which is not a plane. Also, $(n,d)$ and $(\alpha n, \alpha d)$ describe the same
plane for any $\alpha \neq 0$, so some normalisation is required; fixing
$\lVert n\rVert = 1$ is the one that makes the residuals equal to true distances
(§1).

---

## 3. Step A — eliminating $d$: the plane passes through the centroid

### 3.1 Proof by completing the square

Write each point as $p_i = c + u_i$:

$$
n^\top p_i + d = n^\top u_i + \underbrace{(n^\top c + d)}_{=:\,\delta}.
$$

Then

$$
E(n,d) = \sum_i \left(n^\top u_i + \delta\right)^2
= \sum_i (n^\top u_i)^2 \;+\; 2\delta\, n^\top \underbrace{\sum_i u_i}_{=\,0}
\;+\; N\delta^2 .
$$

The cross term vanishes because centred points sum to zero. Hence

$$
\boxed{\;E(n,d) = \underbrace{\sum_i (n^\top u_i)^2}_{\text{depends only on } n}
\;+\; \underbrace{N\,(n^\top c + d)^2}_{\ge 0}\;}
$$

For any fixed $n$, the second term is minimised (to exactly zero) by

$$
\boxed{\;d^\star(n) = -\,n^\top c\;}
$$

which means $n^\top c + d^\star = 0$: **the centroid lies on the optimal plane,
whatever the orientation.**

### 3.2 Same result by calculus

For fixed $n$, $E$ is a quadratic in $d$ with
$\partial^2 E / \partial d^2 = 2N > 0$ (strictly convex), so its unique
minimiser solves

$$
\frac{\partial E}{\partial d} = 2\sum_i (n^\top p_i + d) = 0
\;\Longrightarrow\; N d = -\,n^\top \sum_i p_i
\;\Longrightarrow\; d = -\,n^\top c .
$$

### 3.3 Why this step matters

1. **Separation.** Position and orientation decouple. The 4-unknown constrained
   problem becomes a 3-unknown problem on the unit sphere (this is called
   *profiling out* a nuisance parameter — the same reason a 2D regression
   intercept is $b = \bar y - m\bar x$).
2. **It enables §4–5.** What remains is a pure quadratic form, which is an
   eigenproblem.
3. **Skipping it is wrong, not just suboptimal.** Taking the smallest
   eigenvector of the *uncentred* matrix $\sum_i p_i p_i^\top$ is the same as
   setting $d = 0$, i.e. forcing the plane through the origin — which in the
   sensor frame is the LiDAR itself. Indeed, using $\sum_i p_i p_i^\top = S + N c c^\top$:

   $$
   n^\top\!\Big(\sum_i p_i p_i^\top\Big) n = n^\top S n + N (n^\top c)^2 = E(n, 0).
   $$

   See the worked counterexample in §8.2.

---

## 4. Step B — the quadratic form

Substituting $d^\star$ leaves

$$
E(n) = \sum_i (n^\top u_i)^2 = \sum_i n^\top u_i\, u_i^\top n
= n^\top \Big(\sum_i u_i u_i^\top\Big) n = n^\top S\, n .
$$

### Properties of $S$

- **Symmetric:** $(u_i u_i^\top)^\top = u_i u_i^\top$, so $S^\top = S$.
- **Positive semi-definite:** for any $v$,
  $v^\top S v = \sum_i (v^\top u_i)^2 \ge 0$.

In matrix form, with $U \in \mathbb{R}^{N\times 3}$ the matrix whose rows are
$u_i^\top$, we have $S = U^\top U$ — exactly `u.T @ u` in the code.

Entry-wise, with $u_i = (x_i, y_i, z_i)$ (centred coordinates):

$$
S = \begin{pmatrix}
\sum x_i^2 & \sum x_i y_i & \sum x_i z_i \\
\sum x_i y_i & \sum y_i^2 & \sum y_i z_i \\
\sum x_i z_i & \sum y_i z_i & \sum z_i^2
\end{pmatrix}.
$$

The problem is now:

$$
\min_{\lVert n \rVert = 1} \; n^\top S\, n .
$$

---

## 5. Step C — minimising over the unit sphere

### 5.1 Necessary condition (Lagrange multipliers)

$$
\mathcal{L}(n,\lambda) = n^\top S n - \lambda\,(n^\top n - 1),
\qquad
\nabla_n \mathcal{L} = 2 S n - 2\lambda n = 0
\;\Longrightarrow\;
\boxed{\,S n = \lambda n\,}
$$

Every stationary point is an **eigenvector** of $S$, and at such a point

$$
E(n) = n^\top S n = n^\top (\lambda n) = \lambda\, n^\top n = \lambda .
$$

**The cost equals the eigenvalue.** So among the stationary points, the smallest
eigenvalue gives the smallest cost.

### 5.2 Global optimality (spectral theorem / Rayleigh quotient)

Lagrange only identifies stationary points; here is a direct proof that the
smallest eigenvector is the **global** minimum.

Because $S$ is real symmetric, the spectral theorem gives real eigenvalues
$0 \le \lambda_1 \le \lambda_2 \le \lambda_3$ (non-negative by PSD) and an
orthonormal eigenbasis $e_1, e_2, e_3$. Expand any unit vector in that basis:

$$
n = a_1 e_1 + a_2 e_2 + a_3 e_3, \qquad a_1^2 + a_2^2 + a_3^2 = 1 .
$$

Then

$$
n^\top S n = \sum_{k} \lambda_k a_k^2 \;\ge\; \lambda_1 \sum_k a_k^2 = \lambda_1 ,
$$

with equality iff all the weight sits on eigenvalues equal to $\lambda_1$. Hence

$$
\boxed{\;n^\star = e_1, \qquad E_{\min} = \lambda_1\;}
$$

**Uniqueness.** If $\lambda_1 < \lambda_2$, the minimiser is unique up to sign
($\pm e_1$). If $\lambda_1 = \lambda_2$, every unit vector in
$\operatorname{span}(e_1, e_2)$ is optimal — the normal is undetermined (see §6.2).

In code: `np.linalg.eigh` returns eigenvalues in **ascending** order with
orthonormal eigenvectors as columns, so `vecs[:, 0]` is $e_1$.

### 5.3 Step D — sign

$e_1$ and $-e_1$ give the same plane (with $d$ flipping sign too). The pipeline
fixes the convention $n_z > 0$ (normal points up, so residuals are positive
above the floor):

$$
n \leftarrow \operatorname{sign}(n_z)\, n, \qquad d = -\,n^\top c .
$$

---

## 6. What the eigenvalues mean

### 6.1 Spread along each axis

For each eigenvector,

$$
\lambda_k = e_k^\top S e_k = \sum_i (e_k^\top u_i)^2 ,
$$

the total squared spread of the points along $e_k$. Geometrically, the point
cloud is summarised by an ellipsoid with axes $e_k$ and RMS half-widths
$\sqrt{\lambda_k / N}$. Also $\lambda_1 + \lambda_2 + \lambda_3 = \operatorname{tr} S
= \sum_i \lVert u_i \rVert^2$ (total spread is conserved under rotation).

For a floor patch:

| | |
|---|---|
| $\lambda_3, e_3$ | largest in-plane spread (e.g. along the track) |
| $\lambda_2, e_2$ | the other in-plane direction |
| $\lambda_1, e_1$ | out-of-plane spread = **sum of squared residuals**; $e_1$ = normal |

Free diagnostics:

$$
\text{RMS residual} = \sqrt{\lambda_1 / N},
\qquad
\text{planarity} \sim \lambda_1 / \lambda_2 \;(\ll 1 \text{ for a clean plane}).
$$

### 6.2 Stability of the normal: the eigen-gap

If the data (hence $S$) is perturbed by $\Delta S$ (noise, a few outliers), the
estimated normal rotates by an angle $\theta$ that is bounded, to first order,
by the Davis–Kahan theorem:

$$
\sin\theta \;\lesssim\; \frac{\lVert \Delta S \rVert}{\lambda_2 - \lambda_1}.
$$

So the **gap** $\lambda_2 - \lambda_1$ controls how trustworthy the normal is:

- Points spread well in two directions → $\lambda_2$ large → stable normal.
- Points nearly **collinear** (a thin strip) → $\lambda_2 \approx \lambda_1 \approx 0$
  → the plane can rotate about the strip's axis almost for free; the normal is
  poorly determined.

In GPF, a near-field seed strip ~1 m deep still has a comfortable gap, so noise
alone tilts its normal only by hundredths of a degree. The larger first-fit
error GPF actually shows comes from *truncation bias* in how the seeds are
selected, not from a small gap — see
[`gpf_ground_plane_fitting.md`](gpf_ground_plane_fitting.md) §3.4.

---

## 7. Statistical justification: TLS is maximum likelihood

Model: each observed point is a true point on the plane plus isotropic Gaussian
noise,

$$
p_i = x_i + \varepsilon_i, \qquad n^\top x_i + d = 0, \qquad
\varepsilon_i \sim \mathcal{N}(0, \sigma^2 I_3).
$$

The unknowns are the plane and the latent $x_i$. The negative log-likelihood is

$$
-\log L = \frac{1}{2\sigma^2}\sum_i \lVert p_i - x_i \rVert^2 + \text{const}.
$$

For a fixed plane, each term is minimised by taking $x_i$ as the **orthogonal
projection** of $p_i$ onto the plane, giving
$\lVert p_i - x_i\rVert = |n^\top p_i + d|$ (§1). So

$$
\min_{x_i} (-\log L) = \frac{1}{2\sigma^2} \sum_i (n^\top p_i + d)^2 + \text{const}
= \frac{E(n,d)}{2\sigma^2} + \text{const},
$$

and maximising likelihood is exactly TLS.

### Contrast with ordinary least squares (OLS)

OLS fits $z = a x + b y + c_0$ by minimising **vertical** residuals
$\sum_i (z_i - a x_i - b y_i - c_0)^2$. It is the MLE only if $x_i, y_i$ are
**exact** and all noise is in $z$. Consequences:

| | OLS | TLS |
|---|---|---|
| Residual | vertical | perpendicular |
| Noise model | $z$ only | all coordinates (isotropic) |
| Rotate the data | fit changes non-trivially | fit rotates with the data |
| Vertical plane (wall) | slope $\to\infty$, breaks | no special case |
| Noisy $x,y$ | slope biased toward flat (regression dilution) | consistent |

LiDAR noise is mainly along the beam (range), which mixes $x, y, z$, so TLS's
model fits better. At a 9° tilt the two residuals differ only by a factor
$\cos 9^\circ \approx 0.988$, so on a clean floor the two planes are close; TLS is
the principled choice at no extra cost.

---

## 8. Worked examples

### 8.1 Four points, by hand

Points: $(1,0,0),\ (-1,0,0),\ (0,1,h),\ (0,-1,h)$.

- Centroid: $c = (0, 0, h/2)$.
- Centred: $(\pm 1, 0, -h/2)$ and $(0, \pm 1, +h/2)$.
- Scatter entries:
  - $\sum x^2 = 2,\ \sum y^2 = 2,\ \sum z^2 = 4\cdot (h/2)^2 = h^2$
  - $\sum xy = 0$; $\sum xz = (1)(-\tfrac h2) + (-1)(-\tfrac h2) = 0$; $\sum yz = 0$

$$
S = \operatorname{diag}(2,\ 2,\ h^2).
$$

With $h = 0.1$: eigenvalues $0.01,\ 2,\ 2$. Smallest → $e_1 = (0,0,1)$.
$d = -n^\top c = -h/2$, so the plane is $z = 0.05$.

Check: residuals are $\pm 0.05$, and
$\sqrt{\lambda_1/N} = \sqrt{0.01/4} = 0.05$. ✓

### 8.2 Why centring is mandatory (2D side view)

Floor points at $z = -1.5$, spread uniformly over $x \in [0, 20]$.

**Centred.** $c = (10, -1.5)$, $u_i = (x_i - 10,\ 0)$, so
$S \propto \begin{pmatrix} 33.3 & 0 \\ 0 & 0 \end{pmatrix}$.
Smallest eigenvector $(0, 1)$ → line $z = -1.5$, residuals all $0$. ✓

**Uncentred.** Per-point moments: $\mathbb{E}[x^2] = 133.3$,
$\mathbb{E}[xz] = -1.5 \cdot 10 = -15$, $\mathbb{E}[z^2] = 2.25$:

$$
M \propto \begin{pmatrix} 133.3 & -15 \\ -15 & 2.25 \end{pmatrix},
\quad \operatorname{tr} = 135.58,\ \det = 300 - 225 = 75 .
$$

$$
\lambda_{\min} = \tfrac{1}{2}\left(135.58 - \sqrt{135.58^2 - 4\cdot 75}\right) \approx 0.555,
\qquad e_{\min} \approx (0.112,\ 0.994).
$$

Line through the origin: $0.112\,x + 0.994\,z = 0 \Rightarrow z \approx -0.113\,x$
— a ray from the sensor sloping down ~6.4°, crossing the real floor near
$x \approx 13.3$ m, with RMS error $\approx 0.75$ m. Completely wrong.

---

## 9. Equivalent formulation: SVD

With $U$ the centred $N\times 3$ data matrix, its SVD $U = W \Sigma V^\top$ gives

$$
S = U^\top U = V \Sigma^2 V^\top ,
$$

so the right singular vectors of $U$ are the eigenvectors of $S$ and
$\lambda_k = \sigma_k^2$. The normal is the right singular vector for the
smallest singular value. The SVD route avoids forming $S$; the eigen route
(used in the code) is cheaper. §10 shows why the eigen route is accurate enough
here.

---

## 10. Numerical considerations

1. **Centre before forming $S$.** Uncentred, the matrix is $S + N c c^\top$; with
   the sensor ~1.5 m above the floor the $N c c^\top$ term dwarfs the small
   out-of-plane spread, and subtracting it afterwards loses precision
   (catastrophic cancellation). The code centres first (`u = points - c`).

2. **Conditioning of $U^\top U$.** Forming $S$ squares the condition number;
   the relative error on $\lambda_1$ is roughly $\epsilon \cdot \lambda_3 / \lambda_1$.
   Typical numbers: $\lambda_3/\lambda_1 \approx (10\,\text{m})^2/(0.02\,\text{m})^2
   = 2.5\times 10^5$, $\epsilon_{\text{float64}} \approx 2\times 10^{-16}$
   → error $\sim 10^{-11}$. Negligible.

3. **Cost.** Building $S$: $O(N)$ (6 unique sums). Eigensolve: constant
   ($3\times 3$). This is why GPF can refit up to 10 times per frame.

4. **`eigh` vs `eig`.** `eigh` exploits symmetry: guaranteed real, sorted
   eigenvalues and orthonormal eigenvectors; faster and more stable.

---

## 11. Limitations

### 11.1 Quadratic influence of outliers

A point at distance $r$ contributes $r^2$. Relative to inliers at the 5 cm
threshold, a point 1 m away weighs

$$
\left(\frac{1\ \text{m}}{0.05\ \text{m}}\right)^2 = 400
$$

inlier points. TLS is **not robust**. The pipeline compensates by only ever
fitting pre-filtered inliers:

- GPF: seed band, then the $\pm$`thr` band each iteration.
- RANSAC: inliers of the best hypothesis.
- `_refit_on_ground`: points a method has already labelled ground.

### 11.2 Density weighting

Every point counts equally, and LiDAR returns are much denser near the sensor.
The fitted plane is therefore dominated by the near-field floor. Stride
subsampling (`_subsample`) preserves this ratio. Harmless on a flat track; on a
gentle slope the far floor drifts off the plane first.

### 11.3 Degeneracy

Nearly collinear points ($\lambda_2 \approx 0$) leave the normal undetermined
(§6.2). Fewer than 3 points cannot define a plane at all; callers guard this
(`len(seeds) < 3` in `_gpf_plane`, `len(ground) < 3` in `_refit_on_ground`).

---

## 12. Mapping to the code

| Line | Math | Section |
|---|---|---|
| `c = points.mean(axis=0)` | $c = \frac1N \sum_i p_i$ | §3 |
| `u = points - c` | $u_i = p_i - c$ | §3 |
| `u.T @ u` | $S = U^\top U = \sum_i u_i u_i^\top$ | §4 |
| `np.linalg.eigh(...)` | spectral decomposition of symmetric $S$ | §5.2 |
| `n = vecs[:, 0]` | $n^\star = e_1$ (smallest eigenvalue) | §5 |
| `if n[2] < 0: n = -n` | sign convention $n_z > 0$ | §5.3 |
| `-float(n @ c)` | $d^\star = -n^\top c$ | §3 |
| `plane_residual` | $n^\top p + d$ = signed distance | §1 |

---

## 13. Summary

$$
\min_{\lVert n\rVert=1,\,d} \sum_i (n^\top p_i + d)^2
\;\xrightarrow{\ d^\star = -n^\top c\ }\;
\min_{\lVert n\rVert=1} n^\top S n
\;\xrightarrow{\ \text{Rayleigh}\ }\;
n^\star = e_1(S),\quad E_{\min} = \lambda_1 .
$$

1. The optimal plane passes through the centroid (position separates from
   orientation).
2. The optimal normal is the direction of least spread of the centred points —
   the smallest eigenvector of the scatter matrix.
3. The minimum cost is the smallest eigenvalue; the eigen-gap
   $\lambda_2 - \lambda_1$ measures how well-determined the normal is.
4. Under isotropic Gaussian noise this is the maximum-likelihood plane.
5. It is not robust to outliers, so it is always applied to pre-filtered inliers.
