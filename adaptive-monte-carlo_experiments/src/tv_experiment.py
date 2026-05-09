#!/usr/bin/env python3
"""
Empirical validation: Data-Adaptive vs Fixed Monte Carlo for TV Distance.

Distributions: Bernoulli product distributions P = Bernoulli(p_i)^n, Q = Bernoulli(q_i)^n
Key property: d_TV(Bernoulli(p), Bernoulli(q)) = |p - q|, so marginal TVs are exact.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng(2024)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CORE COMPONENTS
# ──────────────────────────────────────────────────────────────────────────────

def compute_gamma(ps, qs):
    """γ = Pr_C[X≠Y] = 1 − ∏(1 − |p_i − q_i|)"""
    return 1.0 - float(np.prod(1.0 - np.abs(ps - qs)))


def compute_suffix_B(ps, qs):
    """
    B[k] = ∏_{i=k}^{n-1} (1 − d_i)  where d_i = |p_i − q_i|.
    B[n] = 1 (empty product).  B[0] = 1 − γ.
    """
    n = len(ps)
    d = np.abs(ps - qs)
    B = np.ones(n + 1)
    for k in range(n - 1, -1, -1):
        B[k] = (1.0 - d[k]) * B[k + 1]
    return B


def sample_pi(ps, qs, B, rng):
    """
    Draw one sample x ~ π  using sequential conditional sampling.

    At step k, given previous choices x_0,...,x_{k-1}:
      - c_k = A_{k-1} * B[k+1]    (A = running prefix product of α_i(x_i))
      - Unnormalized weight for value v ∈ {0,1}:
            w(v) = p_k(v) − c_k * min{p_k(v), q_k(v)}
      - Sample v with probability w(1)/(w(0)+w(1))
      - Update A *= α_k(v) = min{p_k(v), q_k(v)} / p_k(v)
    """
    n = len(ps)
    x = np.zeros(n, dtype=int)
    A = 1.0

    for k in range(n):
        c_k = A * B[k + 1]
        p1, p0 = ps[k], 1.0 - ps[k]
        q1, q0 = qs[k], 1.0 - qs[k]

        w1 = p1 - c_k * min(p1, q1)
        w0 = p0 - c_k * min(p0, q0)
        w_sum = w1 + w0

        if w_sum < 1e-14:
            v = int(rng.random() < ps[k])
        else:
            v = int(rng.random() < w1 / w_sum)

        x[k] = v
        pv = p1 if v == 1 else p0
        qv = q1 if v == 1 else q0
        A *= (min(pv, qv) / pv) if pv > 1e-14 else 1.0

    return x


def f_val(x, ps, qs):
    """
    f(x) = max{0, 1 − ∏_i(q_i(x_i)/p_i(x_i))} / (1 − ∏_i(min{p_i,q_i}(x_i)/p_i(x_i)))
    Returns 0 when denominator ≈ 0 (π(x) = 0 at those points).
    """
    r_qp = 1.0
    r_mp = 1.0
    for i in range(len(ps)):
        pv = ps[i] if x[i] == 1 else (1.0 - ps[i])
        qv = qs[i] if x[i] == 1 else (1.0 - qs[i])
        if pv < 1e-14:
            return 0.0
        r_qp *= qv / pv
        r_mp *= min(pv, qv) / pv
    num = max(0.0, 1.0 - r_qp)
    den = 1.0 - r_mp
    return num / den if den > 1e-14 else 0.0


def exact_tv(ps, qs):
    """Brute-force exact TV distance (Bernoulli product, n ≤ 20)."""
    n = len(ps)
    tv = 0.0
    for mask in range(1 << n):
        pv = qv = 1.0
        for i in range(n):
            b = (mask >> i) & 1
            pv *= ps[i] if b else (1.0 - ps[i])
            qv *= qs[i] if b else (1.0 - qs[i])
        tv += max(0.0, pv - qv)
    return tv


# ──────────────────────────────────────────────────────────────────────────────
# 2. FIXED ALGORITHM  (Feng et al., verbatim)
# ──────────────────────────────────────────────────────────────────────────────

def run_fixed(ps, qs, eps, delta, rng):
    n = len(ps)
    gamma = compute_gamma(ps, qs)
    if gamma < 1e-12:
        return 0.0, 0

    B = compute_suffix_B(ps, qs)
    m = int(np.ceil(10.0 * n / eps ** 2))        # fixed samples per batch
    s = int(np.ceil(10.0 * np.log(1.0 / delta))) # number of batches

    total = 0
    batch_means = []
    for _ in range(s):
        vals = [f_val(sample_pi(ps, qs, B, rng), ps, qs) for _ in range(m)]
        batch_means.append(float(np.mean(vals)))
        total += m

    return gamma * float(np.median(batch_means)), total


# ──────────────────────────────────────────────────────────────────────────────
# 3. ADAPTIVE ALGORITHM  (this paper)
# ──────────────────────────────────────────────────────────────────────────────

def run_adaptive(ps, qs, eps, delta, rng):
    n = len(ps)
    gamma = compute_gamma(ps, qs)
    if gamma < 1e-12:
        return 0.0, 0

    B     = compute_suffix_B(ps, qs)
    t_max = int(np.ceil(10.0 * n / eps ** 2))
    t0    = max(30, int(np.ceil(3.0 / eps ** 2)))
    s     = int(np.ceil(10.0 * np.log(1.0 / delta)))

    total = 0
    batch_means = []

    for _ in range(s):
        # ── burn-in ──────────────────────────────────────────────────────────
        vals = [f_val(sample_pi(ps, qs, B, rng), ps, qs) for _ in range(t0)]
        total += t0
        t = t0

        # Welford initialise from burn-in samples
        F = float(np.mean(vals))
        M = sum((v - F) ** 2 for v in vals)   # accumulator for Σ(f_i − F)²
        S2 = M / (t - 1) if t > 1 else 0.0

        # ── adaptive loop ─────────────────────────────────────────────────────
        while t < t_max:
            # Stopping criterion (★★): S²_t / (F²_t · ε² · t) ≤ 1/10
            if F > 1e-12 and S2 / (F * F * eps * eps * t) <= 0.1:
                break

            fv = f_val(sample_pi(ps, qs, B, rng), ps, qs)
            total += 1
            t += 1

            # Welford one-step update  ─ O(1), numerically stable
            d1  = fv - F
            F  += d1 / t
            d2  = fv - F
            M  += d1 * d2
            S2  = M / (t - 1) if t > 1 else 0.0

        batch_means.append(F)

    return gamma * float(np.median(batch_means)), total


# ──────────────────────────────────────────────────────────────────────────────
# 4. EXPERIMENTS
# ──────────────────────────────────────────────────────────────────────────────

EPS    = 0.5
DELTA  = 0.1
TRIALS = 4    # independent runs per condition (averaged)

print("=" * 65)
print("EXPERIMENT 1 — Fixed n=8, vary marginal TV gap")
print(f"  ε={EPS}, δ={DELTA}, {TRIALS} trials per condition")
print("=" * 65)

N1   = 8
GAPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

res1 = {}
for gap in GAPS:
    ps = np.full(N1, 0.5)
    qs = np.full(N1, 0.5 + gap)
    true = exact_tv(ps, qs)

    fs_list, as_list, fe_list, ae_list = [], [], [], []
    for _ in range(TRIALS):
        fe, fs = run_fixed(ps, qs, EPS, DELTA, rng)
        ae, as_ = run_adaptive(ps, qs, EPS, DELTA, rng)
        fs_list.append(fs); as_list.append(as_)
        fe_list.append(fe); ae_list.append(ae)

    avg_fs = np.mean(fs_list)
    avg_as = np.mean(as_list)
    res1[gap] = dict(true=true, fs=avg_fs, as_=avg_as,
                     fe=np.mean(fe_list), ae=np.mean(ae_list),
                     speedup=avg_fs / avg_as)

    print(f"  gap={gap:.2f} | true_TV={true:.4f} | "
          f"fixed_est={np.mean(fe_list):.4f}  adapt_est={np.mean(ae_list):.4f} | "
          f"fixed={int(avg_fs):6d}  adaptive={int(avg_as):5d}  speedup={avg_fs/avg_as:.2f}×")

print()
print("=" * 65)
print("EXPERIMENT 2 — Vary n, fixed marginal TV gap = 0.3")
print(f"  ε={EPS}, δ={DELTA}, {TRIALS} trials per condition")
print("=" * 65)

NS   = list(range(2, 14))
GAP2 = 0.30

res2 = {}
for n in NS:
    ps = np.full(n, 0.5)
    qs = np.full(n, 0.5 + GAP2)
    true = exact_tv(ps, qs) if n <= 15 else None

    fs_list, as_list, fe_list, ae_list = [], [], [], []
    for _ in range(TRIALS):
        fe, fs = run_fixed(ps, qs, EPS, DELTA, rng)
        ae, as_ = run_adaptive(ps, qs, EPS, DELTA, rng)
        fs_list.append(fs); as_list.append(as_)
        fe_list.append(fe); ae_list.append(ae)

    avg_fs = np.mean(fs_list)
    avg_as = np.mean(as_list)
    res2[n] = dict(true=true, fs=avg_fs, as_=avg_as,
                   fe=np.mean(fe_list), ae=np.mean(ae_list),
                   speedup=avg_fs / avg_as)

    print(f"  n={n:2d} | fixed={int(avg_fs):6d}  adaptive={int(avg_as):5d}  "
          f"speedup={avg_fs/avg_as:.2f}×  | "
          f"true_TV={true:.4f}  fixed_est={np.mean(fe_list):.4f}  adapt_est={np.mean(ae_list):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

BLUE   = '#2563EB'
ORANGE = '#EA580C'
GREEN  = '#15803D'
GREY   = '#6B7280'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Data-Adaptive vs Fixed Monte Carlo for TV Distance Approximation\n'
    f'(Bernoulli Product Distributions, ε={EPS}, δ={DELTA})',
    fontsize=13, fontweight='bold')

# ── Plot A: Sample count vs separation ────────────────────────────────────────
ax = axes[0, 0]
xs    = list(res1.keys())
f_smp = [res1[g]['fs']  for g in xs]
a_smp = [res1[g]['as_'] for g in xs]

ax.plot(xs, f_smp, 'o-', color=BLUE,   lw=2.2, ms=8, label='Fixed  $m = \\lceil 10n/\\varepsilon^2 \\rceil$ per batch')
ax.plot(xs, a_smp, 's-', color=ORANGE, lw=2.2, ms=8, label='Adaptive  (stops when $S_t^2 / F_t^2 \\varepsilon^2 t \\leq 0.1$)')
ax.set_xlabel('Marginal separation: gap $= |p_i - q_i|$', fontsize=11)
ax.set_ylabel('Average total samples used', fontsize=11)
ax.set_title(f'(A)  Sample Count vs. Separation  [n = {N1}]', fontsize=11, fontweight='bold')
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.25)
ax.set_xlim(0.03, 0.42)

# ── Plot B: Speedup factor bar chart ─────────────────────────────────────────
ax = axes[0, 1]
speedups = [res1[g]['speedup'] for g in xs]
bars = ax.bar(range(len(xs)), speedups, color=GREEN, alpha=0.85, edgecolor='white', width=0.6)
ax.axhline(y=1.0, color=GREY, linestyle='--', lw=1.8, label='Baseline (no speedup)')
ax.set_xticks(range(len(xs)))
ax.set_xticklabels([f'{g:.2f}' for g in xs])
ax.set_xlabel('Marginal separation gap', fontsize=11)
ax.set_ylabel('Speedup factor   (fixed / adaptive)', fontsize=11)
ax.set_title(f'(B)  Speedup of Adaptive over Fixed  [n = {N1}]', fontsize=11, fontweight='bold')
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.25, axis='y')
for i, (bar, sp) in enumerate(zip(bars, speedups)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f'{sp:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── Plot C: Accuracy verification ────────────────────────────────────────────
ax = axes[1, 0]
true_tv  = [res1[g]['true'] for g in xs]
f_est    = [res1[g]['fe']   for g in xs]
a_est    = [res1[g]['ae']   for g in xs]
lower    = [(1 - EPS) * t for t in true_tv]
upper    = [(1 + EPS) * t for t in true_tv]

ax.plot(xs, true_tv, 'k--', lw=2.0, label='Exact TV distance', zorder=5)
ax.plot(xs, f_est,   'o-',  color=BLUE,   lw=2.0, ms=7, label='Fixed estimate')
ax.plot(xs, a_est,   's-',  color=ORANGE, lw=2.0, ms=7, label='Adaptive estimate')
ax.fill_between(xs, lower, upper, alpha=0.10, color='black',
                label=f'±{int(EPS*100)}% relative error band')
ax.set_xlabel('Marginal separation gap', fontsize=11)
ax.set_ylabel('TV distance estimate', fontsize=11)
ax.set_title(f'(C)  Accuracy: Both Estimates Stay Within ε={EPS} Relative Error', fontsize=11, fontweight='bold')
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.25)

# ── Plot D: Sample scaling with n ────────────────────────────────────────────
ax = axes[1, 1]
ns_plot = list(res2.keys())
f_n     = [res2[n]['fs']  for n in ns_plot]
a_n     = [res2[n]['as_'] for n in ns_plot]
sp_n    = [res2[n]['speedup'] for n in ns_plot]

ax.plot(ns_plot, f_n, 'o-', color=BLUE,   lw=2.2, ms=8, label='Fixed  (linear in $n$)')
ax.plot(ns_plot, a_n, 's-', color=ORANGE, lw=2.2, ms=8, label='Adaptive  (near-flat)')
ax.set_xlabel('Number of coordinates  $n$', fontsize=11)
ax.set_ylabel('Average total samples used', fontsize=11)
ax.set_title(f'(D)  Sample Scaling with $n$  [gap = {GAP2}]', fontsize=11, fontweight='bold')
ax.legend(fontsize=9.5)
ax.grid(True, alpha=0.25)

# Annotate the speedup on the right
ax2 = ax.twinx()
ax2.plot(ns_plot, sp_n, 'D--', color=GREEN, lw=1.5, ms=6, alpha=0.85, label='Speedup (right axis)')
ax2.set_ylabel('Speedup factor', fontsize=10, color=GREEN)
ax2.tick_params(axis='y', labelcolor=GREEN)
ax2.legend(loc='upper left', fontsize=9)

plt.tight_layout()
out = '/mnt/user-data/outputs/adaptive_vs_fixed.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nFigure saved → {out}')
