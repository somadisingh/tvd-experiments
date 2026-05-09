# Adaptive Monte Carlo Stopping Rule — Consolidated Empirical Analysis

This document consolidates three experimental validations of the adaptive stopping rule:

1. **Sanity check** (`sanity_check.py`) — analytical correctness of every component at ε=0.5, n=8.
2. **Baseline experiment** (`tv_experiment.py`) — demonstration of speedup at ε=0.5 across gap and n.
3. **Criterion isolation** (`criterion_isolation_test.py`) — new experiment at ε=0.1, n=12 designed to separate the adaptive-criterion contribution from the burn-in floor.

The headline finding: the ~10× speedup reported in the baseline experiment is real, but at those parameters it is driven almost entirely by the burn-in floor rather than by the adaptive criterion itself. The criterion-isolation experiment confirms that the criterion independently tracks the true analytical sample requirement m₀ within ~1% across a wide range, and at well-separated regimes produces speedups exceeding 100×.

---

## 1. Experiment A — Sanity Check (ε=0.5, n ≤ 8)

Source: `sanity_check.py`. Validates all seven core correctness properties against brute-force enumeration.

### 1.1 Checks 1–5: analytical components

| Case | π sums to 1 | E_π[f] = d_TV/γ | f ∈ [0,1] | Var ≤ E[f] | KL(sampler) |
|------|:-----------:|:---------------:|:---------:|:----------:|:-----------:|
| n=2 (manual)       | ✓ 1.00000000 | ✓ (diff 0.00e+00) | ✓ | ✓ 0.122 ≤ 0.769 | ✓ 0.00001 |
| n=8, gap=0.05      | ✓ 1.00000000 | ✓ (diff 2.22e-16) | ✓ | ✓ 0.097 ≤ 0.338 | ✓ 0.00145 |
| n=8, gap=0.40      | ✓ 1.00000000 | ✓ (diff 1.11e-16) | ✓ | ✓ 0.108 ≤ 0.831 | ✓ 0.00155 |

All analytical foundations are sound. The identity E_π[f] = d_TV/γ (the heart of the estimator) holds to machine precision. The importance sampler's KL divergence from the true π is well below the 0.005 acceptance threshold.

### 1.2 Check 6: adaptive stopping time at ε=0.5, n=8

This is where the first ambiguity surfaces.

| gap | true TV | CV² | m₀ (analytical) | Fixed m | Adaptive T̄ | Theor. speedup | Obs. speedup |
|----:|--------:|----:|----------------:|--------:|------------:|---------------:|-------------:|
| 0.05 | 0.1137 | 0.85 | 35 | 320 | **36.0** (≈ true) | 9.1× | 8.9× |
| 0.20 | 0.4426 | 0.50 | 21 | 320 | **30.8** (floor) | 15.2× | 10.4× |
| 0.40 | 0.8174 | 0.16 | 7  | 320 | **30.0** (floor) | 45.7× | 10.7× |

For gap=0.05, the adaptive T̄=36 matches analytical m₀=35. For gap=0.20 and gap=0.40, T̄ pins at the burn-in floor t₀=30 even though the analytical minimum is much smaller. The observed speedups cluster tightly around 10.7× = m/t₀, hinting that the burn-in is doing most of the work.

### 1.3 Check 7: accuracy over 30 independent runs

| gap | true TV | Fixed max err | Fixed failures | Adaptive max err | Adaptive failures |
|----:|--------:|--------------:|---------------:|-----------------:|------------------:|
| 0.05 | 0.1137 | 0.0285 | 0/30 | 0.1010 | 0/30 |
| 0.20 | 0.4426 | 0.0211 | 0/30 | 0.0797 | 0/30 |
| 0.40 | 0.8174 | 0.0108 | 0/30 | 0.0619 | 0/30 |

Both algorithms deliver relative error well below ε=0.5 on every run. No correctness failures.

---

## 2. Experiment B — Baseline Benchmark (ε=0.5, vary gap and n)

Source: `tv_experiment.py`. Two sub-experiments at ε=0.5, δ=0.1, averaged over 4 trials.

### 2.1 Experiment B.1 — Fixed n=8, varying gap

| gap | true TV | Fixed est | Adaptive est | Fixed samples | Adaptive samples | Speedup |
|----:|--------:|----------:|-------------:|--------------:|-----------------:|--------:|
| 0.05 | 0.1137 | 0.1126 | 0.1142 | 7,680 | **897**  | 8.56× |
| 0.10 | 0.2308 | 0.2324 | 0.2337 | 7,680 | **760**  | 10.10× |
| 0.15 | 0.3431 | 0.3425 | 0.3480 | 7,680 | **754**  | 10.19× |
| 0.20 | 0.4426 | 0.4418 | 0.4419 | 7,680 | **729**  | 10.53× |
| 0.25 | 0.5340 | 0.5308 | 0.5281 | 7,680 | **724**  | 10.60× |
| 0.30 | 0.6524 | 0.6547 | 0.6601 | 7,680 | **720**  | 10.67× |
| 0.35 | 0.7503 | 0.7518 | 0.7626 | 7,680 | **720**  | 10.67× |
| 0.40 | 0.8174 | 0.8171 | 0.8099 | 7,680 | **720**  | 10.67× |

Observation: from gap=0.30 onward, the adaptive total saturates at exactly 720 samples. This is s × t₀ = 24 × 30 = 720, meaning every batch terminates immediately after burn-in. The speedup ceiling is 7680/720 ≈ 10.67×, which is *exactly* what we observe.

### 2.2 Experiment B.2 — Fixed gap=0.30, varying n

| n | Fixed samples | Adaptive samples | Speedup | true TV | Fixed est | Adaptive est |
|--:|--------------:|-----------------:|--------:|--------:|----------:|-------------:|
| 2  | 1,920  | 720 | 2.67×  | 0.3900 | 0.3886 | 0.3876 |
| 3  | 2,880  | 753 | 3.82×  | 0.3960 | 0.3908 | 0.4015 |
| 4  | 3,840  | 720 | 5.33×  | 0.5067 | 0.5099 | 0.5036 |
| 5  | 4,800  | 720 | 6.67×  | 0.5498 | 0.5513 | 0.5531 |
| 6  | 5,760  | 726 | 7.93×  | 0.5574 | 0.5562 | 0.5591 |
| 7  | 6,720  | 720 | 9.33×  | 0.6254 | 0.6256 | 0.6202 |
| 8  | 7,680  | 720 | 10.67× | 0.6524 | 0.6498 | 0.6513 |
| 9  | 8,640  | 720 | 12.00× | 0.6605 | 0.6579 | 0.6569 |
| 10 | 9,600  | 720 | 13.33× | 0.7073 | 0.7089 | 0.7094 |
| 11 | 10,560 | 720 | 14.67× | 0.7256 | 0.7270 | 0.7360 |
| 12 | 11,520 | 720 | 16.00× | 0.7336 | 0.7338 | 0.7311 |
| 13 | 12,480 | 720 | 17.33× | 0.7674 | 0.7686 | 0.7729 |

The adaptive sample count flat-lines at 720 across nearly all n, while the fixed count scales linearly in n. The resulting speedup grows proportionally with n — a clean O(n) trend. All estimates are within ε=0.5 of the true TV.

### 2.3 What these two experiments do and don't show

What they **do** show:
- The algorithm is correct: accuracy holds across every configuration.
- The speedup is genuinely O(n) in the scaling sense: at fixed gap, doubling n doubles the speedup.

What they **don't** show:
- Whether the adaptive criterion itself is contributing. At ε=0.5, t₀ defaults to 30 (since ⌈3/ε²⌉=12 < 30), and saturation of adaptive samples at s·t₀ = 720 in nearly every cell strongly suggests the criterion fires immediately after burn-in. This means the observed speedup is explainable purely by "use a small fixed budget of 30 per batch instead of 10n/ε²" — the adaptive logic is never tested.

This motivates Experiment C.

---

## 3. Experiment C — Criterion Isolation (ε=0.1, n=12)

Source: `criterion_isolation_test.py`. Designed specifically to expose the adaptive criterion's behavior by moving into a regime where the analytical requirement m₀ is much larger than the burn-in floor.

### 3.1 Design

**Parameters:** ε=0.1, δ=0.1, n=12.

- Burn-in: t₀_std = max(30, ⌈3/ε²⌉) = 300 (now non-trivial).
- Fixed budget: m = 10n/ε² = 12,000 per batch.
- Window for criterion to act: m / t₀ = 40×.

**Three variants:**

| Variant | Burn-in t₀ | What it isolates |
|---------|------------|------------------|
| (A) **FIXED** | — | Worst-case baseline (always m = 12,000 per batch) |
| (B) **ADAPTIVE standard** | 300 | Paper's recipe |
| (C) **ADAPTIVE minimal** | 30 | Burn-in stripped — exposes pure criterion |

Variant C reveals what the criterion decides on its own.

### 3.2 Main results

Parameters: ε=0.1, n=12, δ=0.1. Fixed batch size m=12,000; s=24 batches.

| gap | CV² | m₀ | FIX total | ADA_std T̄ | ADA_min T̄ | % fired | sp_std | sp_min |
|----:|----:|---:|----------:|-----------:|-----------:|--------:|-------:|-------:|
| 0.02 | 1.11 | **1110** | 288,000 | 1115.1 | 1113.6 | 100.0% | 10.8× | 10.8× |
| 0.05 | 0.91 | **909**  | 288,000 | 908.2  | 910.0  | 100.0% | 13.2× | 13.2× |
| 0.10 | 0.71 | **713**  | 288,000 | 709.2  | 716.4  | 100.0% | 16.9× | 16.8× |
| 0.20 | 0.42 | **420**  | 288,000 | 415.0  | 418.9  | 100.0% | 28.9× | 28.6× |
| 0.30 | 0.24 | **245**  | 288,000 | 301.0  | 242.0  | 100.0% | 39.9× | 49.6× |
| 0.40 | 0.08 | **81**   | 288,000 | 300.0  | 70.9   |  91.7% | 40.0× | **169.2×** |

**Column key**
- *m₀* — exact Chebyshev requirement: 10·CV²/ε² samples/batch.
- *ADA_std T̄* — mean per-batch stop time with t₀=300.
- *ADA_min T̄* — mean per-batch stop time with t₀=30 (pure criterion signal).
- *% fired* — fraction of minimal-burn-in batches where T>30 (criterion kept going).
- *sp_std / sp_min* — total speedup over FIX.

### 3.3 Speedup decomposition

| gap | m₀ | Cost floor@30 | Cost floor@300 | Criterion extra |
|----:|---:|--------------:|---------------:|----------------:|
| 0.02 | 1110 |   720 |  7,200 | 26,007 |
| 0.05 |  909 |   720 |  7,200 | 21,120 |
| 0.10 |  713 |   720 |  7,200 | 16,473 |
| 0.20 |  420 |   720 |  7,200 |  9,335 |
| 0.30 |  245 |   720 |  7,200 |  5,089 |
| 0.40 |   81 |   720 |  7,200 |    982 |

*Criterion extra* = samples drawn by variant C *above* the minimal 30-sample floor, across all 24 batches. This is the portion of work demanded directly by the criterion.

### 3.4 Accuracy check

All three variants pass ε=0.1 tolerance on every gap:

| gap | true TV | err_fix | err_std | err_min |
|----:|--------:|--------:|--------:|--------:|
| 0.02 | 0.0551 | 0.0017 | 0.0062 | 0.0058 |
| 0.05 | 0.1397 | 0.0018 | 0.0070 | 0.0050 |
| 0.10 | 0.2780 | 0.0015 | 0.0087 | 0.0050 |
| 0.20 | 0.5298 | 0.0021 | 0.0078 | 0.0045 |
| 0.30 | 0.7336 | 0.0005 | 0.0026 | 0.0081 |
| 0.40 | 0.9014 | 0.0003 | 0.0061 | 0.0161 |

Zero failures across 90 runs (3 variants × 6 gaps × 5 trials).

---

## 4. Interpretation

### 4.1 The criterion is genuinely tracking CV²

The minimal-burn-in variant (t₀=30) lands within ~1% of the analytical m₀ for every gap where the criterion can act freely:

- gap 0.02: 1114 observed vs 1110 predicted (+0.4%)
- gap 0.05: 910 vs 909 (+0.1%)
- gap 0.10: 716 vs 713 (+0.4%)
- gap 0.20: 419 vs 420 (−0.2%)
- gap 0.30: 242 vs 245 (−1.2%)

This is the cleanest possible evidence that the stopping criterion is not a coincidence or a burn-in artefact — it actually converges on the true Chebyshev requirement once it has enough data to estimate CV² accurately.

### 4.2 Where each mechanism dominates

Comparing sp_std and sp_min in Experiment C reveals where the speedup is actually coming from:

- **gaps 0.02 – 0.20 (pure criterion work).** sp_std ≈ sp_min because m₀ > t₀_std = 300. Neither burn-in value is binding, so both variants track the same criterion-driven stopping time. The observed ~11× to ~29× speedup here is entirely attributable to the adaptive criterion.

- **gap 0.30 (transitional).** sp_std = 40×, sp_min = 50×. Analytical m₀ = 245 is just below the 300 floor, so the standard variant pays a small 55-sample-per-batch tax. Most of the speedup is still criterion-driven.

- **gap 0.40 (burn-in hides further speedup).** sp_std = 40×, sp_min = 169×. Here m₀ = 81 is well below both floors. The standard burn-in of 300 is over-sampling by nearly 4× relative to the true requirement. The minimal variant reveals this by dropping to 71 samples per batch.

The "% fired" column corroborates this: for all gaps ≤ 0.20 the criterion kept sampling past burn-in in 100% of batches; only at gap=0.40 did it trigger stop immediately in about 8% of batches.

### 4.3 Why Experiments A and B were unable to see the criterion

At ε=0.5, n=8:
- t₀ = max(30, ⌈3/0.25⌉) = max(30, 12) = **30** (the 30 floor, not the ⌈3/ε²⌉ formula, binds)
- m = 10·8/0.25 = 320
- Window = m/t₀ = 10.67×

The criterion had no room to act below t₀=30, so the algorithm was effectively operating as "sample exactly 30 per batch and stop" — which by itself yields the ~10.7× speedup visible in Experiment B. At ε=0.1, the floor becomes 300 while the fixed budget becomes 12,000, so the operational window expands 4-fold and the criterion has genuine room to decide.

### 4.4 Reinterpretation of Experiment B.2

The O(n) scaling in Experiment B.2 (speedup grows linearly with n while adaptive samples flat-line at 720) is consistent with both mechanisms:
- Burn-in-dominated view: "adaptive always uses s·t₀ = 720 samples, so speedup = 10n/(ε² · t₀ / s) scales linearly in n" — this is the mechanism actually in play at ε=0.5.
- Criterion-dominated view: "adaptive uses O(CV²/ε²) samples which is n-independent for fixed gap, so speedup is linear in n" — this would be the mechanism in play at smaller ε.

Both views predict the same shape. Experiment C resolves which one is operating at ε=0.5.

---

## 5. Implications for the Paper

### 5.1 Validated claims

- **Theorem 5.1 (Correctness).** 90 runs in Experiment C + 180 runs across A and B, all with zero failures. Correctness holds even for the minimal-burn-in variant, which is technically outside the "t₀ ≥ ⌈3/ε²⌉" prescription.

- **Theorem 5.2.1 (Worst-case runtime).** Trivially holds — algorithm never exceeded t_max in any experiment.

- **Theorem 5.2.2 (Best-case CV² scaling).** Now empirically validated by Experiment C: ADA_min T̄ ≈ m₀ = O(CV²/ε²) across the full range. The ε=0.5 experiments are simply in a regime where this claim was invisible.

- **Theorem 5.2.3 (O(n) maximum speedup).** Corroborated by all three experiments. At ε=0.1 the observed speedups exceed 100× for well-separated distributions, trending toward the asymptotic O(n/α) bound.

### 5.2 Suggested revisions

1. **Include Experiment C as a secondary validation.** The ε=0.5 experiments show the algorithm is *safe* (correctness, non-trivial speedup). The ε=0.1 experiment shows the criterion itself is *effective* (tracks the analytical optimum). Both are needed for a complete empirical story.

2. **Clarify the regime under which the best-case result is visible.** The paper's best-case claim $\mathbb{E}[T] = O(CV^2/\epsilon^2)$ is observable only once $CV^2/\epsilon^2 \gtrsim t_0$. Otherwise the burn-in dominates and the improvement is capped at $m/t_0$. Both regimes still yield the O(n) asymptotic speedup claim, but via different mechanisms.

3. **Report "% fired" as a diagnostic.** A useful indicator of whether the criterion is genuinely operating in any given experiment — if it's 0%, the observed speedup is pure burn-in savings.

---

## 6. Files

- `sanity_check.py` — Experiment A source.
- `sanity_check_output.txt` — captured output from the most recent run.
- `tv_experiment.py` — Experiment B source.
- `tv_experiment_output.txt` — captured output (figure save fails on this machine due to hard-coded `/mnt/user-data/outputs/` path; numbers are valid).
- `criterion_isolation_test.py` — Experiment C source.
- `criterion_isolation.png` — Experiment C figure (per-batch stopping time vs m₀, speedup comparison).
- `adaptive_stopping_revised.md` — the paper.
