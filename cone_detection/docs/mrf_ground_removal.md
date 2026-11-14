# MRF Ground Removal — End to End

What happens to a LiDAR scan when `ground_method: mrf`: from the raw, tilted
point cloud to the upright "above-ground" points that DBSCAN turns into cones.

Every step below has an **In short** box first, then the details.

Code (`cone_detection/cone_detection/`):

| Stage | Function | File |
|---|---|---|
| coarse plane | `_gpf_plane`, `fit_plane` | `ground/methods/plane_fitting.py` |
| levelling | `_level` | `ground/methods/scan_geometry.py` |
| grid | `_level_cells`, `_cell_stats`, `_fill_nearest` | `ground/methods/scan_geometry.py`, `ground/methods/height_grid.py` |
| MRF + labels | `ground_mrf`, `_grid_mincut` | `ground/methods/mrf_methods.py` |
| rotation + shift | `ground_rotation_matrix`, `plane_floor_z` | `ground/alignment.py` |

Background reading: [`total_least_squares_plane_fit.md`](total_least_squares_plane_fit.md)
(how a plane is fitted) and [`gpf_ground_plane_fitting.md`](gpf_ground_plane_fitting.md)
(GPF).

---

## 0. The whole thing in one picture

```
raw scan, sensor frame (LiDAR ~1.07 m up, tilted ~9°)
  │
  │  inside ground_mrf ──────────────────────────────────────────────────┐
  │  1. rough floor        GPF fits one plane P₀                         │
  │  2. level              re-express every point as (u, v, h),          │
  │                        h = height above P₀                           │
  │  3. grid               chop the floor into 30 cm squares;            │
  │                        per square: lowest point, height spread       │
  │  4. local floor        smooth the "lowest point" map so cones vanish │
  │  5. cell costs         how un-floor-like does each square look?      │
  │  6. neighbour costs    how much should neighbours agree?             │
  │  7. graph cut          pick the best ground/object label per square  │
  │  8. point labels       each point: close to the local floor → ground │
  │  9. final plane        fit one plane P₁ to the ground points         │
  │ ─────────────────────────────────────────────────────────────────────┘
  │     returns (outlier mask, P₁)     ← (u, v, h) and the local floor are dropped here
  │
  10. rotate + shift     outliers rotated so P₁ is horizontal, floor moved to z ≈ 0
  11. tall-column veto, DBSCAN, 4 cm floor cull, cone fit
```

**The core idea.** Look at the floor from above as a chessboard of 30 cm squares.
Each square is either *ground* or *object*. A square "votes" based on its own
points (flat and low → ground; tall spread → object), and neighbouring squares
at the same height prefer to agree. An exact optimiser finds the
labelling that best satisfies both at once.

### Where exactly is "the MRF"?

There is no single function called MRF. **The MRF is the model that a few
lines of `ground_mrf` set up, and `_grid_mincut` solves it.** An MRF (Markov
random field) has three parts, plus a solver:

| MRF part | Here | Code |
|---|---|---|
| **Variables:** one unknown label per site | one label per 30 cm tile: ground or object | the grid from `_level_cells` (§3) |
| **Unary terms:** what each variable's own data says | the tile's cost for "ground" vs "object" | `_mrf_costs` in `mrf_methods.py` (`cost_ground`, `cost_object`) — §5 |
| **Pairwise terms:** links between neighbouring variables | the penalty for neighbours that disagree | `_mrf_costs` in `mrf_methods.py` (`w_down`, `w_right`) — §6 |
| **Solving it** | find the labelling with the lowest total cost | `ground_mrf` → `_grid_mincut` / `_grid_mincut_reduced` in `mrf_methods.py` — §7 |

Why the name:

- **"Random field":** a grid of unknowns, each with a probability. The total
  cost $E$ maps to a probability, $P(\text{labels}) \propto e^{-E}$, so the
  lowest-cost labelling is the most likely one.
- **"Markov":** a tile's label depends only on its direct neighbours. Once the 4
  neighbours are known, nothing further away matters directly.

---

## 1. Rough floor plane (GPF)

> **In short:** find one flat plane that roughly matches the floor. It only
> needs to be roughly right.

```python
plane = _gpf_plane(_subsample(data), thr)
```

GPF starts from the lowest points in the scan (almost certainly floor), fits a
plane to them, gathers every point within ±5 cm of it, fits again, and repeats
until nothing changes.

The fit uses `np.linalg.eigh`, which finds the plane's **normal** — the arrow
pointing straight out of the floor. It does **not** move or rotate any points.
GPF works entirely in the tilted sensor frame; it just outputs a plane.

Why "roughly right" is enough: step 4 builds a local floor that follows the real
ground, so a plane that is a bit tilted or offset gets corrected there.

---

## 2. Levelling (`_level`)

> **In short:** turn the point cloud so the rough floor is horizontal. Then
> "height" means height above the floor, everywhere.

### Why

The LiDAR is tilted ~9°. In its own coordinates the floor is a ramp that rises
~16 cm per metre:

```
sensor z
   ▲          ___----   ← floor in sensor coordinates (a ramp)
   │   ___----
   │---
   └──────────────────▶ forward
```

A 30 cm square on that ramp already spans ~5 cm of height from the floor alone,
so "height spread" and "10 cm above the floor" would be meaningless. We need a
frame where the floor is flat.

### How

From the plane's normal $n$, build three perpendicular unit arrows:

```python
n  = plane[1:]                          # "up" (out of the floor)
e1 = x̂ − (x̂·n) n ;  e1 /= |e1|          # "forward", tipped into the floor
e2 = n × e1                             # "left", also in the floor
```

- $e_1$: take the sensor's forward axis $\hat x$ and remove the part that points
  along $n$ (this is Gram–Schmidt). What remains lies in the floor.
- $e_2$: the cross product of two perpendicular unit vectors is a third one
  perpendicular to both.

Each point $p$ is then described by how far it goes along each arrow:

```python
u, v = p·e1, p·e2          # position on the floor
h    = p·n + d             # height above the plane (plane_residual)
```

These three arrows are perpendicular and of length 1, so this is just a
**rotation** of the cloud (plus a shift that puts the plane at $h = 0$). Nothing
is stretched or bent. Checked numerically: `h` equals the exact
point-to-plane distance.

(If $n$ were almost parallel to $\hat x$, the projection would be nearly zero, so
the code uses $\hat y$ instead when $|n_x| \ge 0.9$. For a floor, $n$ points
roughly up, so $\hat x$ is always used.)

---

## 3. Height grid (`_level_cells`, `_cell_stats`)

> **In short:** chop the floor into 30 cm squares and, for each square, record
> its lowest point and how tall its points spread.

```python
ij   = floor(uv / 0.30)          # which square each point falls in (integer row, col)
ij  -= ij.min(axis=0)            # start numbering at 0
cell = i * ny + j                # one number per square
cnt, mn, mx = per-square count, min h, max h     # one fast pass (Numba)
```

A 25 m crop gives about 170 × 170 ≈ 29 000 squares. Most are empty or hold just
a few points.

Two numbers per square:

| Name | What it is | On floor | On a cone |
|---|---|---|---|
| `mn_g` | the **lowest** point's height | ≈ floor | ≈ floor if the low laser rows hit the cone base, otherwise higher |
| `span` | **max − min** height in the square | 1–3 cm (noise) | 10–30 cm |

Empty squares have no points. For `mn_g` they copy the value of the nearest
non-empty square (`_fill_nearest`), so later steps can compare heights across
gaps. Their `span` is 0. They are never labelled ground cells themselves (step 7).

---

## 4. Local floor (`grey_opening`)

> **In short:** estimate "where the floor would be if no cones were there" by
> erasing every bump narrower than 2.7 m from the lowest-point map.

```python
ref = ndimage.grey_opening(mn_g, size=(9, 9))     # 9 squares = 2.7 m
```

An **opening** is two passes over the map with a 9 × 9 window:

1. **Min pass** (erosion): every square takes the *lowest* value in its window.
   Any bump narrower than the window is swallowed by the floor around it.
2. **Max pass** (dilation): every square takes the *highest* value in its
   window. Wide shapes that survived pass 1 grow back to their original size.

Side view, window of 3 squares:

```
input           ▁▁▁█▁▁▁        a cone: one tall square
after min       ▁▁▁▁▁▁▁        the cone is gone (its neighbours are lower)
after max       ▁▁▁▁▁▁▁        nothing to restore → local floor is flat

input           ▁▂▃▄▅▆▇        a slope
after min       ▁▁▂▃▄▅▆        shifted one square downhill
after max       ▁▂▃▄▅▆▇        shifted back → slope kept
```

Useful properties:

- Narrow things (cones) disappear; wide things (slopes, wide sidewalks, a tilted
  reference plane) stay.
- The result is **never above** the input, so "lowest point − local floor" is
  always ≥ 0.

---

## 5. Cell costs — what each square says by itself

> **In short:** each square gets a price for being called "ground". Flat and
> sitting on the local floor → cheap. Tall or raised → expensive. Calling it
> "object" always costs 1.

$$
\text{cost}_{\text{ground}} = \min\!\left( \Big(\frac{\text{span}}{10\text{ cm}}\Big)^2
+ \Big(\frac{\text{mn\_g} - \text{ref}}{10\text{ cm}}\Big)^2,\ 10 \right)
\qquad
\text{cost}_{\text{object}} = 1
$$

Read it as: "10 cm of height spread, or 10 cm above the local floor, makes
*ground* exactly as expensive as *object*." Squaring makes small values very
cheap and large values very expensive. The cap at 10 stops one extreme square
from dominating everything. **Empty squares cost 0 for both labels** — they have
no opinion.

On its own, a square would pick whichever label is cheaper, i.e. *ground* when
its cost is below 1:

| Square | span | above local floor | ground cost | alone it picks |
|---|---|---|---|---|
| clean floor | 2 cm | 1 cm | 0.05 | ground |
| noisy floor | 8 cm | 4 cm | 0.80 | ground (barely) |
| very noisy floor | 11 cm | 0 | 1.21 | **object** ✗ |
| cone body | 30 cm | 0 | 9 | object |
| cone flank, low rows missed | 15 cm | 10 cm | 3.25 | object |
| narrow raised strip (< 2.7 m wide, 12 cm up) | 2 cm | 12 cm | 1.48 | object |
| wide sidewalk (> 2.7 m) | 2 cm | ~0 (part of the local floor) | ~0.04 | ground |

The "very noisy floor" row is the problem the next step solves.

---

## 6. Neighbour costs — squares should agree

> **In short:** two side-by-side squares at the same height pay a penalty (up to
> 0.5) if they get different labels. If there is a real height step between
> them, the penalty almost vanishes.

### Which squares are neighbours?

**Only up, down, left and right (4-connected). Diagonals are not linked.**

```
        ┌───┐
        │ ↑ │
    ┌───┼───┼───┐
    │ ← │ X │ → │        X is linked to these 4 only
    └───┼───┼───┘
        │ ↓ │
        └───┘
```

In the code there are exactly two sets of links (`_grid_mincut`):

- `w_down[i, j]` joins square (i, j) with (i+1, j)
- `w_right[i, j]` joins square (i, j) with (i, j+1)

The PyMaxflow branch uses the same pattern (its `structure=` arrays). Diagonal
squares still influence each other, but only indirectly, through the neighbours
they share.

Two other steps do look at diagonals, but they are not MRF links:

- the local floor (§4) uses a full 9 × 9 square window;
- "borrow the floor from the nearest ground square" (§8) uses straight-line
  distance in any direction.

### The penalty

For each pair of neighbouring squares:

$$
w = 0.5 \cdot e^{-(\Delta h / 5\text{ cm})^2}, \qquad
\Delta h = \text{difference of their lowest points}
$$

| Height difference | Penalty $w$ | Meaning |
|---|---|---|
| 0 | 0.50 | same surface: please agree |
| 2.5 cm | 0.39 | |
| 5 cm | 0.18 | |
| 10 cm | 0.009 | real edge (curb, raised object): free to disagree |

### Why the penalty is *low* when the height difference is *high*

First: **the penalty is only paid by neighbours that disagree.** Two neighbours
with the same label pay nothing, whatever their height difference.

The easiest way to see what it does is to think of the boundary between ground
and object as a **fence**:

1. **Every disagreeing pair is one piece of fence.** Wherever a ground square
   touches an object square, a fence segment runs between them, and its price is
   that pair's penalty:

   ```
   total = Σ each square's own cost  +  Σ price of every fence segment
   ```

2. **The optimiser chooses where to build the fence.** Making a square "object"
   means fencing it off from its ground neighbours. That is only worth it if the
   fence costs less than what the square saves by being object.

3. **The penalty sets the price of fencing in each place:**
   - **Flat edge (high penalty, 0.5):** fence there is expensive. A noisy floor
     square isn't worth fencing off, so it stays ground. *This is the smoothing.*
   - **Height-step edge (low penalty, ~0):** fence there is almost free. A real
     raised object can be fenced off cheaply, so the fence ends up along the
     step.

```
side view:   floor  floor  floor │ plank  plank
                                 │
             ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │▆▆▆▆▆▆▆▆▆▆▆▆
                                 ↑
                    fence HERE: big jump → almost free
                    not in the middle of the floor: no jump → expensive
```

So the high penalty doesn't *pull* the boundary toward the steps. It keeps the
boundary **out of flat areas**, and the low penalty keeps the steps **cheap**.
The boundary lands on the steps because that is the cheapest place left. What
matters is the **contrast** between the two prices.

**Proof by experiment.** The 5 × 5 example of §10b (noisy corner square + a
12 cm plank column), run through the real solver with three penalty schemes:

```
 height-dependent        same 0.5 everywhere     same 0.002 everywhere
 (what the code does)

 G G G G O               G G G G G               O G G G O
 G G G G O               G G G G G               G G G G O
 G G G G O               G G G G G               G G G G O
 G G G G O               G G G G G               G G G G O
 G G G G O               G G G G G               G G G G O

 both right              plank missed            noisy square → object
```

- **Same 0.5 everywhere:** fencing off the plank costs 5 edges × 0.5 = 2.5, but
  its 5 squares together only save 5 × (1.48 − 1) = 2.4. Not worth it — **the
  plank vanishes into the floor.**
- **Same 0.002 everywhere:** fences are free everywhere, so nothing is smoothed
  — **the noisy square becomes an object.**
- **Height-dependent:** fences are expensive on flat floor (noisy square stays
  ground) and nearly free along the step (plank fenced off for 5 × 0.002 = 0.01).
  **Both right.**

Image segmentation uses the same trick: cutting along a strong colour edge is
cheap, cutting through a uniform region is expensive.

Two clarifications:

- **A low penalty doesn't push neighbours to disagree.** It only stops
  discouraging it. Whether a square actually becomes an object still depends on
  its own cost (§5).
- **The jump is measured between the squares' *lowest* points.** A cone whose
  base is visible has its lowest point on the floor, so the jump to its
  neighbours is ~0 and the penalty stays high. The cone is detected by its
  **spread** (cost 9), not by the jump. The jump matters most for raised flat
  things: a plank, a curb, a sidewalk edge.

(Code detail: `np.diff(..., append=last)` pads the arrays to full size; the
extra last row/column has no real neighbour and is ignored.)

---

## 7. Choosing the labels — graph cut

> **In short:** add up all square costs and all neighbour penalties for a
> labelling. The graph cut finds the labelling with the smallest total —
> exactly, not approximately.

### 7.1 What is being minimised

$$
E = \underbrace{\sum_{\text{squares}} \text{cost of its label}}_{\text{each square's own opinion}}
\;+\; \underbrace{\sum_{\text{neighbour pairs}} w \cdot [\text{labels differ}]}_{\text{disagreement penalties}}
$$

### 7.2 A two-square example (by hand)

Square X is clean floor, square Y is very noisy floor:

- X: ground 0.2, object 1
- Y: ground 1.3, object 1
- They are at the same height → $w = 0.5$

All four labellings:

| X | Y | X cost | Y cost | penalty | **total** |
|---|---|---|---|---|---|
| ground | ground | 0.2 | 1.3 | 0 | **1.5** ← best |
| ground | object | 0.2 | 1 | 0.5 | 1.7 |
| object | ground | 1 | 1.3 | 0.5 | 2.8 |
| object | object | 1 | 1 | 0 | 2.0 |

Alone, Y would have chosen *object* (1 < 1.3). Its neighbour pulls it to
*ground*.

Now suppose Y sits 10 cm higher than X (a real step), so $w = 0.009$: the
*ground/object* row costs $0.2 + 1 + 0.009 = 1.21$, which beats 1.5. Y becomes
*object*. Same square costs, different answer, purely because of the height
step between them.

### 7.3 Why you can't just try every labelling

With 29 000 squares there are $2^{29\,000}$ labellings. The graph cut solves it
exactly anyway.

### 7.4 How the graph cut works

Build a network of pipes:

```
             SOURCE
        ┌──────┼──────┐        pipe width = cost of being GROUND
        ▼      ▼      ▼
       [X] ◀─▶ [Y] ◀─▶ [Z]     side pipes (both ways) = neighbour penalty w
        │      │      │
        ▼      ▼      ▼        pipe width = cost of being OBJECT
             SINK
```

A **cut** splits the squares into two groups by cutting pipes: squares still
connected to the source, and squares connected to the sink. Rule: **sink side =
ground**. The cut's price is the total width of the pipes cut:

- a square on the sink side had its source pipe cut → pays its *ground* cost
- a square on the source side had its sink pipe cut → pays its *object* cost
- two neighbours on different sides had the pipe between them cut → pay $w$

So **the price of every cut equals the total cost $E$ of the matching
labelling.** The cheapest cut is the best labelling.

Finding the cheapest cut: pump as much "water" as possible from source to sink.
When the flow is at its maximum, the pipes that are completely full form the
cheapest cut (the **max-flow / min-cut theorem**). Max-flow algorithms do this
in polynomial time.

This trick works for this energy because the neighbour penalty is never
negative — disagreeing never *rewards* you. That condition (called
*submodularity*) is exactly what makes a binary labelling problem solvable by a
cut.

Back to the two-square example: the cheapest cut keeps both X and Y on the sink
side (cuts both source pipes, 0.2 + 1.3 = 1.5), matching the best row of the
table.

### 7.5 The solvers

- **scipy (default).** `scipy.sparse.csgraph.maximum_flow` (Dinic's algorithm).
  It only accepts whole numbers, so all costs are multiplied by up to 100 000
  and rounded; the factor is lowered automatically if the total could overflow
  a 32-bit integer. After the flow, a search from the source along pipes that
  still have room finds the source side; every other square is ground.
- **PyMaxflow (optional, GPL licence).** Boykov–Kolmogorov algorithm, built for
  image grids. Same answer; `test_mrf_scipy_cut_matches_pymaxflow` checks that
  both reach the same total cost.

Finally `is_ground &= has`: empty squares take part in the cut (they pass
"agreement" between their neighbours across gaps) but are never ground squares
themselves.

---

## 8. From squares back to points

> **In short:** every point is compared with the floor height of its square. Close
> to it → ground. In object squares the "close" band is tighter, so the cone
> body survives but the floor hits around its base are removed.

```python
floor   = _fill_nearest(mn_g, is_ground)       # floor height for every square
outlier = h − floor[cell] >= (0.10 if cell is ground else 0.05)
```

It happens in **two stages**:

1. **The whole square gets a label first:** ground or object (§7).
2. **Then each point is compared with the floor height of its square.**

**Stage 2a — a floor height for every square.** This gives a bumpy, local floor,
not a plane. All heights are $h$, the height above the rough plane $P_0$ from
step 2.

- **Ground square:** its floor is **its own lowest point** (`mn_g`). The MRF has
  just decided this square is floor, so its lowest return is taken as a direct
  measurement of the floor right there.
- **Object square:** it has no trustworthy floor of its own (its lowest point
  may be on the cone, the plank top, …), so it **borrows the lowest point of the
  nearest ground square**. "Nearest" is straight-line distance on the grid, in
  any direction (a Euclidean distance transform, `_fill_nearest`).

Top view, numbers = each square's lowest point in cm:

```
 lowest point      label        floor used
  0   1   0         G  G  G      0   1   0
  1  12   0   →     G  O  G  →   1   1*  0      * borrowed from a nearest
  0   0   1         G  G  G      0   0   1        ground square (ties: any one)
```

Two consequences:

- The lowest point of a ground square is always ground (it is 0 cm above its own
  floor).
- The floor is **measured**, square by square, not smoothed or modelled — unlike
  the local floor `ref` of step 4, which only decides the costs.

**Stage 2b — how far above that floor is the point?**

| Square label | Point is ground if it is less than… | Why |
|---|---|---|
| ground | **10 cm** above the floor | tolerate floor noise and small bumps |
| object | **5 cm** above the floor | keep the cone, but drop the floor hits around its base |

The square's label decides **which band to use**; the band decides each point.

An object square with a cone in it (side view):

```
 height
 30 cm ┤         •            ← kept (cone)
       ┤        • •           ← kept
 15 cm ┤       •   •          ← kept
       ┤      •     •         ← kept
  5 cm ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← the 5 cm line
       ┤  •  •       •  •     ← ground (cone base + floor hits)
  0 cm ┴━━━━━━━━━━━━━━━━━━━━  ← floor borrowed from the neighbouring ground square
```

A ground square:

```
 10 cm ┼ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ← the 10 cm line
       ┤                      ← (a point here would be kept as an obstacle)
       ┤   •    •   •   •     ← all ground
  0 cm ┴━━━━━━━━━━━━━━━━━━━━  ← this square's own lowest point
```

---

## 8b. Why not skip the MRF?

> **In short:** "compare each point with its square's floor" only works if you
> know *which squares contain floor*. Treating every square as floor fails
> wherever the floor is hidden. The MRF is the step that decides which squares
> to trust and lets the others borrow.

### Option A: treat every square as ground

Use each square's own lowest point as its floor, for every square. It works when
a square's lowest point really is floor — e.g. a cone whose base is visible: the
lowest point is the base, and the cone body sits above it.

It breaks when the lowest point is **not** floor:

```
 plank / box top (floor hidden underneath)      cone split across two squares
                                                 
   ┌───────────┐ ← lowest point of these          ┌──┬──┐
   │ • • • • • │   squares is the TOP            •│• │  │  ← right square's lowest point
   │           │   → whole top = "floor"        • │ •│  │    is halfway up the cone
 ▁▁┘           └▁▁                             ▁▁▁│▁▁│•▁│▁   → that part = "floor"
```

### Option B: use the local floor `ref` (step 4) everywhere

Compare every point with the opening-based local floor. This catches raised
objects narrower than 2.7 m. But `ref` starts with a *minimum* over 81 squares,
so it picks up the lowest noise in the area and sits slightly **below** the real
floor. Then either:

- a tight band (5 cm) leaves floor points above it → floor clutter for DBSCAN, or
- a loose band (10 cm) is safe for floor but erases more of every cone.

### What the MRF adds

- It **chooses which squares' lowest points to trust** as floor, and object
  squares borrow from them — so hidden-floor squares get a real floor height.
- Because it knows which squares are object, it can use **two bands**: 10 cm in
  ground squares (floor noise never survives) and 5 cm in object squares (more of
  the cone survives).

### Measured

Synthetic scan, 9° tilt, real `_level` / `ground_mrf` code: floor with 1 cm
noise, 6 cones (4–16 m), a 12 cm plank (3 × 1.2 m) and a 40 cm box (1 × 1 m),
with the floor hidden under both.

| | MRF | A: every square ground, 5 cm | A: 10 cm | B: local floor, 5 cm | B: 10 cm |
|---|---|---|---|---|---|
| floor points wrongly kept | **0.0 %** | 2.0 % | 0.0 % | 4.9 % | 0.0 % |
| cone points kept | **90.0 %** | 89.7 % | 74.0 % | 92.7 % | 77.8 % |
| plank points kept | **100 %** | 0.9 % | 0.0 % | 100 % | 100 % |
| box points kept | **95.4 %** | 67.4 % | 61.3 % | 96.7 % | 91.0 % |

- **A** loses the plank and the box top completely.
- **B** keeps the objects but has to choose between floor clutter and losing
  cone points.
- The **MRF** is the only column with no floor clutter *and* ~90 % of cone points.

This is one synthetic scene; option B is close to the `pmf` method and option A
is close to a plain min-height grid, which is why those are benchmarked too.

---

## 9. Final plane

> **In short:** fit one flat plane through all points labelled ground, because the
> rest of the pipeline expects one plane.

```python
return outlier, _refit_on_ground(data, outlier, coarse)
```

Same plane fit as GPF (TLS, on up to 20 000 ground points); falls back to the
rough plane $P_0$ if fewer than 3 ground points exist.

**What gets thrown away here:** the levelled coordinates $(u, v, h)$ from step 2
and the local floor from step 8. Only the mask and $P_1$ leave the function.

---

## 10. Worked example: one row of six squares

A strip of floor with a cone in square C and a noisy patch in square E. Heights
in cm, in the levelled frame.

| | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| what is there | floor | floor | **cone** | floor | **noisy floor** | floor |
| lowest point `mn_g` | 0 | 0 | 1 | 0 | 0 | 0 |
| spread `span` | 2 | 1 | 30 | 3 | 11 | 2 |

**Step 4, local floor.** The cone's square is only 1 cm higher than its
neighbours and narrower than the window, so the opening gives `ref` = 0
everywhere.

**Step 5, square costs** (object = 1 for all):

| | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| ground cost | 0.04 | 0.01 | 9.01 | 0.09 | **1.21** | 0.04 |
| alone it picks | G | G | O | G | **O** ✗ | G |

**Step 6, neighbour penalties.** All squares are within 1 cm of each other:
A–B, D–E, E–F: 0.50 (0 cm); B–C, C–D: 0.48 (1 cm).

**Step 7, best labelling.** Compare the two squares in question, keeping the rest
as they would pick alone:

- C as object: $1 + 0.48 + 0.48 = 1.96$; as ground: $9.01$ → **object** ✓
- E as object: $1 + 0.5 + 0.5 = 2.0$; as ground: $1.21$ → **ground** ✓

The graph cut finds this labelling exactly: G G O G G G. The cone is so un-floor-like
that paying both neighbour penalties is worth it; the noisy patch isn't.

**Step 8, points.** In C (object), the cone's points up to 5 cm above the
borrowed floor (from B or D, height 0) are ground — that's the bottom of the cone
and the floor returns next to it. Everything higher is kept.

(Checked with the real `grey_opening` and `_grid_mincut`: same costs, same
labels.)

---

## 10b. Worked example: a 5 × 5 patch, seen from above

A 5 × 5 patch of 30 cm squares containing:

- **C**: a cone (spread 30 cm)
- **N₁**: a noisy floor square in the corner (spread 11 cm)
- **N₂**: a noisy floor square in the middle (spread 13 cm)
- **P**: the right-hand column, a plank lying on the floor, 12 cm higher than
  everything else

```
 What's there:          Ground cost                 Final labels
                        (object always costs 1)     (from the real min-cut)

 N₁ .  .  .  P          1.21 .04  .04  .04  1.48     G  G  G  G  O
 .  .  .  N₂ P          .04  .04  .04  1.69 1.48     G  G  G  G  O
 .  .  C  .  P          .04  .04  9.01 .04  1.48     G  G  O  G  O
 .  .  .  .  P          .04  .04  .04  .04  1.48     G  G  G  G  O
 .  .  .  .  P          .04  .04  .04  .04  1.48     G  G  G  G  O
```

Neighbour penalties: 0.5 between same-height squares (0.48 around the cone,
whose lowest point is 1 cm up), but only **0.002** across the 12 cm step into the
plank column.

Square by square:

| Square | As ground | As object | Result |
|---|---|---|---|
| plain floor | 0.04 | 1 + penalties | **ground** |
| cone C | 9.01 | 1 + 4 × 0.48 = 2.92 | **object**: paying all four penalties is still cheaper |
| N₁ (corner, 2 neighbours) | 1.21 | 1 + 0.5 + 0.5 = 2.0 | **ground**: alone it would pick object; its neighbours pull it back |
| N₂ (3 floor neighbours + plank) | 1.69 | 1 + 3 × 0.5 + 0.002 = 2.50 | **ground**: disagreeing with the plank is almost free, so the plank doesn't pull it |
| plank P | 1.48 (12 cm above local floor) | 1 + 0.002 to the floor side; agrees with the other plank squares | **object**: the whole column |

Then the points (§8):

- **Cone square:** the bottom 5 cm become ground; the rest is kept.
- **Plank squares:** the floor is borrowed from the ground column next to them
  (height 0). The plank's points are 12 cm up, above the 5 cm line, so they are
  all kept as an obstacle.

### How tall must a lone object be to survive?

Take a single object square surrounded by floor, with its lowest point on the
floor. It stays an object only if its ground cost beats $1 + 4 \times 0.5 = 3$:

$$
\Big(\frac{\text{span}}{10\text{ cm}}\Big)^2 > 3
\quad\Longleftrightarrow\quad
\text{span} > \sqrt{3}\times 10\text{ cm} \approx 17\text{ cm}
$$

Checked with the solver: a 17 cm spread became ground, 18 cm became object. A
full cone (~32 cm) passes easily; a far cone that only gets a couple of returns
per square might not.

---

## 11. Rotation and shift (`separate_ground`, `ground/separation.py`)

> **In short:** after ground removal, the pipeline turns the remaining points so
> $P_1$ is horizontal and moves the floor to $z = 0$, so that z means "height
> above the ground" for the cone fitting.

```python
data = data[outliers] @ ground_rotation_matrix(P₁)   # make P₁'s normal point straight up
data[:, 2] -= plane_floor_z(P₁)                      # move the floor to z ≈ 0
```

### Why rotate again, if step 2 already levelled the cloud?

1. **The eigen step never rotated anything.** `eigh` only found the normal.
   The MRF path levels the cloud exactly once, inside `_level` (step 2), and
   that result stays private to `ground_mrf`.
2. **The interface drops it.** Every ground method returns only
   `(non_ground_mask, plane)`, so RANSAC, GPF, MRF, … can be swapped freely.
   RANSAC never levels anything, so the pipeline levels the outliers itself,
   using whatever plane comes back.
3. **It is a different plane.** Step 2 used the rough $P_0$; this step uses the
   refit $P_1$. On flat ground they agree to ~0.004° (simulation below), so the
   second rotation repeats work already done — but cheaply, because only the
   ~1 % of points that are outliers get rotated. On a slope they differ (1.8° in
   the simulation).

### The real cost: the local floor is thrown away

After this step, z is "height above the single plane $P_1$". The MRF had a
much better answer — a local floor for every square (step 8), which follows
slopes — but the pipeline never sees it. On a slope, the 4 cm floor cull, the
0.75 m tall-column veto and the cone templates (apex fixed at 0.35 / 0.55 m
above ground) all measure from the wrong floor.

Simulation (§12), 5 % uphill starting at 10 m. The MRF labels are still right,
but heights after this step are off:

| Cone at | True height of its points | z after rotation + shift |
|---|---|---|
| 4 m | 0 – 0.32 m | −0.01 – 0.30 m |
| 8 m | 0 – 0.32 m | −0.04 – 0.25 m |
| 12 m | 0 – 0.32 m | 0.01 – 0.29 m |
| 16 m | 0 – 0.32 m | **0.17 – 0.45 m** |

The 16 m cone appears to float 17 cm up (taller than a small-cone template); the
8 m cone sinks partly below the 4 cm cull.

Possible fix: let local methods also return each point's height above its local
floor (`h − floor[cell]`) and use that as z downstream, keeping the global
rotation only for x and y.

### A small offset in the shift

After the rotation the floor sits at $z = -d$ ($d$ = sensor height above the
plane). `plane_floor_z` subtracts $-d \cdot n_z$ instead of $-d$, so the floor
ends at $-d\,(1 - n_z)$, not 0:

- car (d ≈ 1.07 m, 9° tilt, $n_z ≈ 0.988$): **≈ −1.3 cm** (measured −1.32 cm)
- simulator (LiDAR at ground level, d ≈ 0): ≈ 0, which is why it went unnoticed

The exact height is $n\cdot p + d$ — the same `h` that `_level` already computes.

---

## 12. Simulation

A synthetic scan run through the real `_level` and `ground_mrf`, then the
pipeline's rotation and shift: floor 1.07 m below the sensor with 1 cm noise,
40 000 floor points (denser near the car), 8 cones of 0.325 m at 4–16 m
(±1.5 m to each side), sensor tilted 9°.

| | Flat | 5 % uphill from 10 m |
|---|---|---|
| angle between $P_0$ and $P_1$ | 0.004° | 1.79° |
| floor points labelled ground | 99.995 % | 99.987 % |
| cone points kept | 88.2 % | 88.0 % |
| floor height after the shift | −1.32 cm | −1.15 cm |

The ~12 % of cone points labelled ground are their bottom ~5 cm (step 8), which
the 4 cm floor cull would mostly drop anyway.

---

## 13. Parameters

Module constants in `ground/methods/mrf_methods.py`, except `thr`.

| Constant | Value | What it controls | Turn it up → |
|---|---|---|---|
| `_MRF_CELL_M` | 0.30 m | square size | coarser grid, fewer squares, cones blur with floor |
| `_MRF_SPAN_M` | 0.10 m | spread at which ground cost = 1 | more tolerant of tall/noisy squares |
| `_MRF_STEP_M` | 0.10 m | height above local floor at which ground cost = 1 | raised areas count as ground more easily |
| `_MRF_MAX_COST` | 10 | cap on a square's ground cost | strong squares resist smoothing more |
| `_MRF_FLOOR_WINDOW` | 9 squares (2.7 m) | widest thing treated as an "object" by the local floor | bigger objects removed from the floor, slopes followed less closely |
| `_MRF_SMOOTH` | 0.5 | max neighbour penalty | smoother labels, but small objects can be swallowed |
| `_MRF_SMOOTH_DH_M` | 0.05 m | height difference at which neighbours stop caring | labels stick together across bigger steps |
| `_MRF_GROUND_BAND_M` | 0.10 m | point band above floor in ground squares | more points near the floor counted as ground |
| `_MRF_SOLVER` | `"scipy"` | max-flow backend | `"pymaxflow"` needs the optional package |
| `cfg.ground_threshold_m` | 0.05 m | GPF band; point band in object squares | more of each cone's base labelled ground |

---

## 14. Limitations

- **The single-plane interface throws away the local floor** (§11).
- **Sparse far cones.** At range a cone square may hold only 1–2 points, so its
  spread is ~0 and it looks like floor unless its lowest point is clearly raised.
  A lone object square needs ~17 cm of spread to survive smoothing (§10b).
- **Big objects.** Anything wider than 2.7 m (a car, a wall seen edge-on) becomes
  part of the local floor and looks like terrain. The tall-column veto
  downstream catches most of these.
- **Square borders.** A cone sitting on a border splits its points between two
  squares, so each has a smaller spread.
- **A very wrong rough plane** (10°+) would put a real height gradient across each
  square again; normal GPF errors are far smaller.

---

## 15. Code map

| Code | Section |
|---|---|
| `_gpf_plane(_subsample(data), thr)` | 1 |
| `_level`: `e1`, `e2`, `uv`, `plane_residual` | 2 |
| `_level_cells(data, thr, _MRF_CELL_M)`, `_cell_stats` | 3 |
| `_fill_nearest(..., has)`, `span` | 3 |
| `ndimage.grey_opening(mn_g, size=(9, 9))` | 4 |
| `cost_ground`, `cost_object`, `[~has] = 0` | 5 |
| `w.append(_MRF_SMOOTH * np.exp(...))` | 6 |
| `_grid_mincut(...) & has` | 7 |
| `floor = _fill_nearest(mn_g, is_ground)`, `outlier = ...` | 8 |
| `_refit_on_ground(data, outlier, coarse)` | 9 |
| `ground_rotation_matrix`, `plane_floor_z` | 11 |
