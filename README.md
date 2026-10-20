# Quantum Interference GPT Rank — Research Scientist Agent

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An autonomous research-scientist agent that simulates a 4-slit quantum interferometer, fits a bilinear (GPT rank) model via Alternating Least Squares (ALS), and determines whether the data is consistent with the quantum-mechanical prediction of **rank = 15** (= n²−1 for n=4 slits).

---

## Quick Start

```bash
pip install -r requirements.txt
python run_agent.py
```

This runs the full 6-step workflow with default settings (N_eff = 50 000 photons, noisy simulation) and writes `results/REPORT.md` and `results/results.json`.

---

## What Is Being Tested

A **Generalised Probabilistic Theory (GPT)** assigns a *state space* to a physical system. The minimum rank needed to factorise the probability matrix

```
D[s, x] = p(pixel x | measurement setting s)
```

is called the **GPT rank** of the theory. For an n-slit experiment:

| Theory | GPT Rank | Reasoning |
|---|---|---|
| Classical (wave optics) | 4 | One real amplitude per slit |
| 2nd-order interference | ≈ 10 | Pairs of slits contribute |
| **Quantum Mechanics** | **15** | dim su(4) = n²−1 = 15 |

The QM prediction comes from the fact that the density matrix ρ lives in su(4) (traceless Hermitian 4×4 matrices), which has dimension 15. Any larger rank would signal *post-quantum* physics; any smaller rank would rule out standard QM.

---

## The Agent

The `ResearchScientist` class orchestrates six steps:

1. **`formulate_hypothesis`** — states H₀ (QM, rank=15), H₁ (classical, rank=4), H₂ (2nd-order, rank≈10) and the selection criterion (minimum cross-validated χ²/dof).
2. **`design_experiment`** — documents the 4-slit setup: 16 shutter patterns × 81 phase patterns = 1 296 measurement settings, grouped by n_open ∈ {1,2,3,4}.
3. **`simulate_data`** — generates the probability matrix D using the Gaussian beam model with optional Poisson noise.
4. **`train_and_evaluate`** — sweeps K ∈ [1, k_max] using 10-fold cross-validation (CV); for each K it fits a joint bilinear model D ≈ U @ V.T with ALS.
5. **`analyse_results`** — finds the best K per group, computes Δχ²(K=14→15), and delivers a verdict.
6. **`write_report`** — emits `REPORT.md` (detailed markdown) and `results.json` (machine-readable).

---

## Repository Structure

```
quantum_interference_agent/
├── README.md
├── SCIENCE.md
├── requirements.txt
├── .gitignore
├── run_agent.py              # CLI entry point
├── simulator/
│   ├── __init__.py
│   └── interferometer.py     # Gaussian beam model, settings generation
├── model/
│   ├── __init__.py
│   ├── als.py                # ALS bilinear factorisation
│   └── rank_sweep.py         # K-sweep + cross-validation
├── agent/
│   ├── __init__.py
│   ├── scientist.py          # ResearchScientist orchestrator
│   └── reporter.py           # REPORT.md writer
└── results/
    └── .gitkeep
```

---

## Running Options

```bash
# Default: N_eff=50000, noisy
python run_agent.py

# Sweep multiple noise levels
python run_agent.py --n_eff 1000 10000 50000

# Noiseless theory data
python run_agent.py --no-noise

# Custom sweep parameters
python run_agent.py --k_max 20 --n_folds 10

# Custom output directory
python run_agent.py --output results/

# Full example
python run_agent.py --n_eff 5000 50000 --k_max 20 --n_folds 10 --output results/ --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--n_eff` | 50000 | Effective photon count(s); accepts multiple values |
| `--no-noise` | False | Use noiseless theory probabilities |
| `--k_max` | 20 | Maximum rank K to sweep |
| `--n_folds` | 10 | Cross-validation folds |
| `--n_jobs` | -1 | Parallel workers (-1 = all cores) |
| `--output` | results | Output directory |
| `--seed` | 42 | Random seed |

---

## Physics

See [SCIENCE.md](SCIENCE.md) for a full derivation of:
- The Gaussian beam field model
- The GPT framework and rank definition
- Why QM predicts rank = n²−1 = 15
- The ALS loss function and normal equations
- The 10-fold cross-validation chi-squared criterion
- Connection to the IQC (Institute for Quantum Computing) emICCD experiment

---

## Expected Output

After a successful run:

- **`results/REPORT.md`** — a structured scientific report containing: abstract, introduction, theory, methods, per-N_eff results tables, discussion, conclusions, and the full agent reasoning log.
- **`results/results.json`** — machine-readable dictionary with keys `hypothesis`, `design`, `analysis`, `sweep_results` (chi²/dof values for every K, fold, and group).

Example console output:
```
[Agent] Step 1/6 — Formulating hypothesis...
[Agent] Step 2/6 — Designing experiment...
[Agent] Step 3/6 — Simulating data (N_eff=50000, noise=True)...
[Agent] Step 4/6 — Training and evaluating models...
...
╔══════════════════════════════════════╗
║         FINAL VERDICT TABLE         ║
╠══════════════════════════════════════╣
║  N_eff=50000 | Best K=15 | QM: ✓    ║
╚══════════════════════════════════════╝
Agent complete. See results/REPORT.md
```

---

## License

MIT © 2026
