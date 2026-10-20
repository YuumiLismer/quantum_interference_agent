# Science Background: Quantum Interference GPT Rank Experiment

This document provides a self-contained scientific background for the `quantum_interference_agent` codebase. It is intended for readers with basic quantum mechanics but no prior knowledge of Generalised Probabilistic Theories (GPTs).

---

## 1. Physical Setup

### 1.1 The Interferometer

We model a **4-slit single-photon interferometer**. A collimated beam (405 nm wavelength in the IQC implementation) illuminates a mask with four slits, labelled k = 0, 1, 2, 3. Each slit can be independently opened or closed (controlled by a spatial light modulator or physical shutters) and can have an independently addressable phase shift. The diffraction pattern is recorded on a 1D pixel array (emICCD camera in the IQC experiment; simulated 500-pixel array here).

### 1.2 Gaussian Beam Field Model

The complex field amplitude at pixel position x contributed by slit k is:

```
ψ_k(x) = A_k · exp[ −2(x − x_k)² / r² ] · exp[ i(φ_k + κ_k · x) ]
```

where:
- `A_k ∈ {0, 1}` — shutter open (1) or closed (0)
- `x_k` — slit centre position (see table below)
- `r` — beam radius (BEAM_RADIUS = 1.0)
- `φ_k` — applied phase shift
- `κ_k` — transverse wave vector (tilt, see table below)

The total intensity (proportional to probability) at pixel x is:

```
I(x) = |ψ_0(x) + ψ_1(x) + ψ_2(x) + ψ_3(x)|²
```

After normalising so that ∑_x I(x) = 1, we obtain the probability distribution p(x | setting s).

### 1.3 Physical Parameters

| Parameter | Value |
|---|---|
| BEAM_RADIUS r | 1.0 (arbitrary units) |
| SLIT_X (centres) | [−0.15, −0.05, +0.05, +0.15] |
| KX_LIST (tilts) | [−20, −10, +10, +20] |
| NUM_PIXELS | 500 |
| X_GRID | linspace(−1, 1, 500) |

### 1.4 Measurement Settings

Each setting s is specified by:
1. A **shutter pattern** — which of the 4 slits are open (2⁴ = 16 patterns)
2. A **phase pattern** — the phase applied to each open slit, drawn from {0, π/4, π/2}⁴ = 81 combinations

Settings are grouped by **n_open** (number of open slits):

| n_open | Shutter patterns | Phase patterns | Total rows |
|---|---|---|---|
| 1 | C(4,1) = 4 | 3¹ = 3 | 12 |
| 2 | C(4,2) = 6 | 3² = 9 | 54 |
| 3 | C(4,3) = 4 | 3³ = 27 | 108 |
| 4 | C(4,4) = 1 | 3⁴ = 81 | 81 |
| **Total** | **16** | | **255** |

Note: not all 16 × 81 = 1296 combinations are used — only phases for *open* slits vary; closed slits contribute nothing.

---

## 2. GPT Framework

### 2.1 State Space

A Generalised Probabilistic Theory (GPT) specifies:
- A **state space** Ω (convex set of normalised states ω)
- A set of **effects** e_x with 0 ≤ e_x ≤ 1 such that ∑_x e_x = 1 for each measurement

The fundamental observable is the **probability matrix**:
```
D[s, x] = p(pixel x | setting s) = ⟨e_x, ω_s⟩
```

where ω_s is the state prepared by setting s and ⟨·,·⟩ is the pairing between effects and states.

### 2.2 GPT Rank

The **GPT rank** is the minimum number K such that D admits a rank-K factorisation:
```
D = U @ V.T,   U ∈ ℝ^{S×K},  V ∈ ℝ^{X×K}
```

Equivalently, it is the dimension of the minimal state space that can reproduce the observed statistics. Experimentally, the GPT rank is estimated by finding the smallest K for which the cross-validated chi-squared per degree of freedom (χ²/dof) drops to ≈ 1.

---

## 3. QM Prediction: Rank = n²−1

### 3.1 Why the Density Matrix Lives in su(n)

For an n-slit experiment, quantum mechanics describes the transverse mode of the photon by a density matrix ρ ∈ M_n(ℂ), Hermitian, positive semi-definite, with Tr(ρ) = 1. The trace condition removes one real parameter, so ρ lives in the **real affine subspace** of traceless Hermitian matrices plus the identity:

```
ρ = (I/n) + Σ_{j=1}^{n²−1} r_j · λ_j
```

where {λ_j} are the n²−1 generators of su(n) (generalised Gell-Mann matrices). The **coherence vector** r = (r_1, …, r_{n²−1}) ∈ ℝ^{n²−1} fully parametrises the state.

### 3.2 Born Rule Linearity

The Born rule probability for effect E_x = |x⟩⟨x| is:
```
p(x | ρ) = Tr(E_x ρ) = (1/n) + Σ_j r_j · Tr(E_x λ_j)
```

This is **linear** in the coherence vector r. Therefore the map s → ω_s (from experimental settings to quantum states) lives in an (n²−1)-dimensional space, and the matrix D has rank ≤ n²−1.

### 3.3 Achievability

For a generic 4-slit experiment with sufficient phase diversity, all n²−1 = 15 dimensions of su(4) are accessible. This means the GPT rank is *exactly* 15 (not less) when the full phase space is probed.

### 3.4 For n = 4 Slits

```
n = 4
n² − 1 = 16 − 1 = 15
```

So **QM predicts GPT rank = 15** for a 4-slit interferometer.

---

## 4. Competing Theories

| Theory | GPT Rank | Physical Interpretation |
|---|---|---|
| Classical (ray optics) | 4 | One real number per slit (intensity) |
| Classical wave optics | 4 | Complex amplitudes, but rank=4 in intensity basis |
| 2nd-order interference only | ≈ 10 | Pairs of slits; no genuine 3- or 4-slit terms |
| Rebit (real QM) | 9 | Real amplitudes; su(4) over ℝ has dim=9 |
| **Quantum Mechanics** | **15** | Full complex amplitudes, su(4), dim=15 |
| Post-quantum (hypothetical) | > 15 | Beyond standard QM |

Note: the "≈ 10" for 2nd-order interference is an upper bound based on pairwise correlations; the exact value depends on the observable basis.

---

## 5. Bilinear Model

### 5.1 The Factorisation

We model:
```
D ≈ U @ V.T
```
- **U** ∈ ℝ^{S×K}: one row per measurement setting, encodes the *state* prepared
- **V** ∈ ℝ^{X×K}: one row per pixel, encodes the *effect* (detector response)

### 5.2 Shared V Across Groups

A key physical insight: **V is shared across all n_open groups**. This is because V encodes the detector response functions (pixel sensitivities, diffraction kernel), which do not depend on which slits are open. The states U differ between groups (different n_open → different rank structure), but the measurement apparatus is identical.

This **joint fit** is more powerful than fitting each group separately:
- It enforces physical consistency
- It provides more data to constrain V
- It prevents overfitting in the U matrices

### 5.3 Non-negativity and Normalisation

In the physical model, D[s,x] ≥ 0 and ∑_x D[s,x] = 1. The ALS fit does not enforce these constraints explicitly (they are implicit in the Gaussian beam model), but in practice the fitted D̃ = U @ V.T closely respects them for well-chosen K.

---

## 6. ALS Fitting

### 6.1 Loss Function

We minimise a **Poisson-weighted chi-squared** loss:
```
L(U, V) = Σ_{s,x} w[s,x] · (D[s,x] − (U @ V.T)[s,x])²
```

where the weight is the inverse variance:
```
w[s,x] = 1 / σ[s,x]²
σ[s,x] = max( sqrt(|D[s,x]| / N_eff),  σ_floor )
```

This is equivalent to a Poisson chi-squared when N_eff is the effective photon count. The floor σ_floor = 1e-4 prevents infinite weights at zero-probability pixels.

### 6.2 ALS Normal Equations

Alternating Least Squares alternates between:

**Update U** (fix V, solve for each row u_s):
```
[ Σ_x w[s,x] V[x,:].T V[x,:] + reg·I ] · u_s = Σ_x w[s,x] D[s,x] V[x,:]
```

In einsum notation:
```python
A_skl = einsum('sx,xk,xl->skl', w, V, V) + reg * I_K
b_sk  = einsum('sx,sx,xk->sk',  w, D, V)
u_s   = solve(A_s, b_s)  # for each s
```

**Update V** (fix U, solve for each row v_x):
```
[ Σ_s w[s,x] U[s,:].T U[s,:] + reg·I ] · v_x = Σ_s w[s,x] D[s,x] U[s,:]
```

```python
A_xkl = einsum('sx,sk,sl->xkl', w, U, U) + reg * I_K
b_xk  = einsum('sx,sx,sk->xk',  w, D, U)
v_x   = solve(A_x, b_x)  # for each x
```

### 6.3 SVD Warm-Start

To avoid poor local minima, the first restart initialises:
```python
Us, ss, Vts = svd(D, full_matrices=False)
U_init = Us[:, :K] * ss[:K]   # absorb singular values into U
V_init = Vts[:K].T
```

Subsequent restarts perturb this initialisation with Gaussian noise scaled by `0.1 * U_init.std()`.

---

## 7. Cross-Validation Strategy

### 7.1 Fold Assignment

We use **10-fold cross-validation** on random (row, pixel) pairs:
1. Flatten the index: `flat_idx = rng.permutation(n_rows * n_pix)`
2. Assign folds: `fold_ids = (flat_idx % n_folds).reshape(n_rows, n_pix)`
3. For fold f: training mask = `fold_ids != f`; test mask = `fold_ids == f`

This assigns ~10% of all (setting, pixel) pairs to the test set, distributed randomly across the entire matrix — not block rows or columns.

### 7.2 Test Chi-Squared

The test chi-squared per degree of freedom is:
```
χ²_test/dof = (1 / n_test) · Σ_{(s,x) in test} [ (D[s,x] − D̃[s,x]) / σ[s,x] ]²
```

where D̃ = U @ V.T is the fitted model. A well-fitted model gives χ²/dof ≈ 1.

### 7.3 Model Selection

The optimal rank is:
```
K* = argmin_K  mean_{folds}[ χ²_test(K) / dof ]
```

We report K* separately for each n_open group (by slicing the appropriate rows of U) and for the joint 'all' matrix.

---

## 8. Statistical Interpretation

### 8.1 Chi-Squared per Degree of Freedom

| χ²/dof | Interpretation |
|---|---|
| >> 1 | Model underfits — K too small |
| ≈ 1 | Model fits correctly — K is correct |
| < 1 | Model overfits — K too large |

### 8.2 Delta-Chi-Squared Test

To quantify the improvement of QM over the classical prediction, we compute:
```
Δχ² = χ²(K=14) − χ²(K=15)
```

Under the null hypothesis that K=14 is sufficient, Δχ² follows a chi-squared distribution with degrees of freedom equal to the number of extra parameters. A large Δχ² (>> 1 per extra parameter) indicates that the additional rank-1 component is statistically significant.

### 8.3 Noise Dependence

The signal-to-noise for detecting the rank-15 component scales as:
```
SNR ∝ sqrt(N_eff) · ||D_15 − D_14||_F / σ_floor
```

where D_k is the best rank-k approximation. At low N_eff (≲ 1000), noise dominates and the optimal K may be less than 15 even for QM data.

---

## 9. Connection to the IQC Experiment

This simulation is inspired by an ongoing experiment at the **Institute for Quantum Computing (IQC)**, University of Waterloo:

| Component | Details |
|---|---|
| Light source | 405 nm Topmode diode laser (single-mode, coherent) |
| Nonlinear crystal | PPKTP — Type-II phase-matched SPDC for single-photon pairs |
| Detection | emICCD (electron-multiplied intensified CCD) camera |
| Interferometer | 4-slit mask, motorised phase plates |
| Dataset | 2026-04-09 (coincidence histogram, ~50k effective pairs) |

The PPKTP crystal produces entangled photon pairs via spontaneous parametric down-conversion (SPDC). The signal photon passes through the 4-slit interferometer; the idler photon is used as a herald. This heralded single-photon source eliminates multi-photon artifacts.

Key experimental challenges not modelled here:
- **Dark counts** and detector noise floor
- **Pixel crosstalk** in the emICCD
- **Phase drift** of the phase plates over the measurement run
- **Alignment** of the slit mask relative to the beam waist
- **Systematic errors** in the shutter open/close calibration

The simulation presented here uses a clean Gaussian beam model (no aberrations, no detector noise beyond Poisson) and should be regarded as a best-case scenario for the GPT rank test.
