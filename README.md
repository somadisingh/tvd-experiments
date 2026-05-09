# TV distance experiments

This repository holds empirical code for total variation (TV) distance estimation and approximation: a **deterministic FPTAS** with sparsification and a suite of Bernoulli-product **Monte Carlo** simulations with adaptive stopping.

---

## Contents

| Directory | Topic |
|-----------|--------|
| [`deterministic_approx_experiments/`](deterministic_approx_experiments/) | Deterministic approximation (sparsification bounds, support growth, ratio clustering, deterministic vs randomized comparison). |
| [`adaptive-monte-carlo_experiments/`](adaptive-monte-carlo_experiments/) | Data-adaptive vs fixed-batch Monte Carlo for TV on product Bernoullis; sanity checks and criterion isolation. |

---

## Requirements

- **Python** 3.x  
- **NumPy** and **matplotlib** (scripts use non-interactive `Agg` backend for figures).

Install manually, for example:

```bash
pip install numpy matplotlib
```

There is no `requirements.txt` in this repo; the dependency surface is intentionally small.

---

## Deterministic approximation experiments

**Location:** `deterministic_approx_experiments/`

Implements Algorithm 2 (deterministic FPTAS for TV) with sparsification, ratio distributions, brute-force verification on small discrete supports, randomized sampling baseline, and synthetic distribution families from `distribution_families.py`.

**Entry point:** from `deterministic_approx_experiments/`, run:

```bash
python run_paper_experiments.py
```

This invokes four bundles aligned with paper “findings”:

1. **Sparsification bound** — empirical tightness of the sparsification error bound (proxies vs theory).  
2. **Support size growth** — how support evolves through the recurrence.  
3. **Ratio clustering** — behavior near ratio \(r = 1\).  
4. **Algorithm comparison** — deterministic vs randomized runtime / accuracy regimes.

Plots and summaries are written to the **`deterministic_approx_experiments/`** directory (same folder as `run_paper_experiments.py`; each module accepts `out_dir`).

Notable modules under [`deterministic_approx_experiments/src/`](deterministic_approx_experiments/src/):

| File | Role |
|------|------|
| `deterministic_algorithm.py` | Deterministic TV FPTAS. |
| `randomized_algorithm.py` | Monte Carlo baseline. |
| `sparsify_full.py`, `ratio_distribution.py` | Sparsification and ratio coupling structure. |
| `brute_force.py` | Exact TV for small Cartesian products. |
| `distribution_families.py` | Test instances and plotting styles. |

**Note:** Some modules mix imports (`distribution_families` vs `src.distribution_families`). If you hit `ModuleNotFoundError`, run from `deterministic_approx_experiments` with:

```bash
PYTHONPATH=. python run_paper_experiments.py
```

or normalize imports locally so every module resolves `distribution_families` consistently.

Captured numerical outputs live in [`deterministic_approx_experiments/results/`](deterministic_approx_experiments/results/) (`*_results.txt`).

---

## Adaptive Monte Carlo experiments

**Location:** `adaptive-monte-carlo_experiments/`

Code studies **Bernoulli product distributions** \(P = \prod_i \mathrm{Bernoulli}(p_i)\), \(Q = \prod_i \mathrm{Bernoulli}(q_i)\), exploiting exact marginal TVs \(|p_i - q_i|\). Implements importance sampling under \(\pi\), the adaptive stopping rule, fixed-batch comparisons, and diagnostics described in detail in **[`experiments_analysis.md`](adaptive-monte-carlo_experiments/experiments_analysis.md)**.

**Scripts** (typically run from `adaptive-monte-carlo_experiments/` with `src` on `PYTHONPATH` or `cd adaptive-monte-carlo_experiments/src` depending on invocation):

| Script | Purpose |
|--------|---------|
| [`src/sanity_check.py`](adaptive-monte-carlo_experiments/src/sanity_check.py) | Analytical correctness and adaptive stopping checks at \(\varepsilon = 0.5\), small \(n\). |
| [`src/tv_experiment.py`](adaptive-monte-carlo_experiments/src/tv_experiment.py) | Baseline speedup sweep (fixed vs adaptive). |
| [`src/criterion_isolation_test.py`](adaptive-monte-carlo_experiments/src/criterion_isolation_test.py) | Isolates adaptive criterion vs burn-in at tighter \(\varepsilon\). |

Example:

```bash
cd adaptive-monte-carlo_experiments/src
python sanity_check.py
python tv_experiment.py
python criterion_isolation_test.py
```

**Figure output path:** [`tv_experiment.py`](adaptive-monte-carlo_experiments/src/tv_experiment.py) currently saves one figure to a hard-coded path (`/mnt/user-data/outputs/...`). On a typical laptop that directory will not exist; change `out = ...` near the bottom of that file to a local path if you want the PNG. Logged runs in [`adaptive-monte-carlo_experiments/results/`](adaptive-monte-carlo_experiments/results/) still contain valid numbers even when saving fails.

The narrative synthesis of all three adaptive experiments (what they prove separately, implications for \(\varepsilon = 0.5\) vs \(0.1\), suggested paper tweaks) is in **`experiments_analysis.md`** — read that alongside the `.txt` artifacts in [`results/`](adaptive-monte-carlo_experiments/results/).

