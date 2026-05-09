#!/usr/bin/env python3
"""
CRITERION ISOLATION TEST
========================
Goal: Separate the speedup contribution from the adaptive criterion
      from the speedup contribution of the burn-in floor.

Design rationale
----------------
At ε=0.5 (sanity_check.py default), burn-in t₀ = max(30, ⌈3/ε²⌉) = 30.
For n=8, fixed m = 320.  Hence speedup is capped at 320/30 ≈ 10.7× just
by having any floor — regardless of whether the criterion ever fires.

To probe the CRITERION specifically, we need regimes where the true
analytical m₀ = 10·CV²/ε² significantly EXCEEDS t₀, because then the
criterion must keep sampling beyond burn-in to reach its target.

Parameters chosen for this purpose
----------------------------------
  ε = 0.1   ⇒  t₀_standard = max(30, 300) = 300   (non-trivial burn-in)
  n = 12    ⇒  t_max = 10n/ε² = 12000             (wide window above t₀)
  Ratio t_max/t₀ ≈ 40×, so criterion has 40× of room to manoeuvre.

Three variants compared
-----------------------
  (A) FIXED               — always uses m = 12000 samples per batch
  (B) ADAPTIVE std t₀=300 — paper's recipe
  (C) ADAPTIVE min t₀=30  — strips burn-in to expose the criterion alone

Plus analytical m₀ from brute-force enumeration as ground truth.

We then report, per gap:
  • analytical m₀
  • observed T for variant (B) and (C)
  • fraction of variant-(C) batches where T>30 (criterion genuinely fired)
  • decomposition: how much of the speedup comes from burn-in vs criterion
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(2024)

EPS   = 0.1
DELTA = 0.1
N     = 12

# ──────────────────────────────────────────────────────────────────────────────
# Core functions (verbatim from tv_experiment.py / sanity_check.py)
# ──────────────────────────────────────────────────────────────────────────────

def compute_gamma(ps, qs):
    return 1.0 - float(np.prod(1.0 - np.abs(ps - qs)))

def compute_suffix_B(ps, qs):
    n = len(ps); d = np.abs(ps - qs); B = np.ones(n + 1)
    for k in range(n - 1, -1, -1):
        B[k] = (1.0 - d[k]) * B[k + 1]
    return B

def sample_pi(ps, qs, B, rng):
    n = len(ps); x = np.zeros(n, dtype=int); A = 1.0
    for k in range(n):
        c_k = A * B[k + 1]
        p1, p0 = ps[k], 1.0 - ps[k]
        q1, q0 = qs[k], 1.0 - qs[k]
        w1 = p1 - c_k * min(p1, q1)
        w0 = p0 - c_k * min(p0, q0)
        w_sum = w1 + w0
        v = int(rng.random() < w1 / w_sum) if w_sum > 1e-14 else int(rng.random() < ps[k])
        x[k] = v
        pv = p1 if v == 1 else p0
        qv = q1 if v == 1 else q0
        A *= (min(pv, qv) / pv) if pv > 1e-14 else 1.0
    return x

def f_val(x, ps, qs):
    r_qp = 1.0; r_mp = 1.0
    for i in range(len(ps)):
        pv = ps[i] if x[i] == 1 else (1.0 - ps[i])
        qv = qs[i] if x[i] == 1 else (1.0 - qs[i])
        if pv < 1e-14: return 0.0
        r_qp *= qv / pv
        r_mp *= min(pv, qv) / pv
    num = max(0.0, 1.0 - r_qp)
    den = 1.0 - r_mp
    return num / den if den > 1e-14 else 0.0

def run_fixed(ps, qs, eps, delta, rng):
    n = len(ps); gamma = compute_gamma(ps, qs)
    if gamma < 1e-12: return 0.0, 0
    B = compute_suffix_B(ps, qs)
    m = int(np.ceil(10.0 * n / eps**2))
    s = int(np.ceil(10.0 * np.log(1.0 / delta)))
    total = 0; batches = []
    for _ in range(s):
        vals = [f_val(sample_pi(ps, qs, B, rng), ps, qs) for _ in range(m)]
        batches.append(float(np.mean(vals))); total += m
    return gamma * float(np.median(batches)), total

def run_adaptive(ps, qs, eps, delta, t0, rng):
    """Adaptive with configurable burn-in t0.  Returns per-batch stopping times."""
    n = len(ps); gamma = compute_gamma(ps, qs)
    if gamma < 1e-12: return 0.0, 0, []
    B = compute_suffix_B(ps, qs)
    t_max = int(np.ceil(10.0 * n / eps**2))
    s = int(np.ceil(10.0 * np.log(1.0 / delta)))
    total = 0; batches = []; per_batch_T = []
    for _ in range(s):
        vals = [f_val(sample_pi(ps, qs, B, rng), ps, qs) for _ in range(t0)]
        total += t0; t = t0
        F = float(np.mean(vals)); M = sum((v - F)**2 for v in vals)
        S2 = M / (t - 1) if t > 1 else 0.0
        while t < t_max:
            if F > 1e-12 and S2 / (F * F * eps * eps * t) <= 0.1: break
            fv = f_val(sample_pi(ps, qs, B, rng), ps, qs)
            total += 1; t += 1
            d1 = fv - F; F += d1 / t; d2 = fv - F; M += d1 * d2
            S2 = M / (t - 1) if t > 1 else 0.0
        per_batch_T.append(t)
        batches.append(F)
    return gamma * float(np.median(batches)), total, per_batch_T

# ──────────────────────────────────────────────────────────────────────────────
# Brute-force analytical reference
# ──────────────────────────────────────────────────────────────────────────────

def brute_force(ps, qs, eps):
    n = len(ps); gamma = compute_gamma(ps, qs)
    true_tv = 0.0; E_f = 0.0; E_f2 = 0.0
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)])
        pv = qv = 1.0
        for i in range(n):
            b = x[i]
            pv *= ps[i] if b else (1-ps[i])
            qv *= qs[i] if b else (1-qs[i])
        true_tv += max(0.0, pv - qv)
        prod_alpha = 1.0
        for i in range(n):
            p_i = ps[i] if x[i] == 1 else (1-ps[i])
            q_i = qs[i] if x[i] == 1 else (1-qs[i])
            if p_i > 1e-14: prod_alpha *= min(p_i, q_i) / p_i
        pi_x = pv * (1 - prod_alpha) / gamma if gamma > 1e-14 else 0.0
        fv = f_val(x, ps, qs)
        E_f += pi_x * fv; E_f2 += pi_x * fv**2
    Var_f = E_f2 - E_f**2
    CV_sq = Var_f / E_f**2 if E_f > 1e-14 else float('inf')
    m_true = int(np.ceil(10 * CV_sq / eps**2))
    return dict(true_tv=true_tv, gamma=gamma, E_f=E_f, Var_f=Var_f,
                CV_sq=CV_sq, m_true=m_true)


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

t0_std = max(30, int(np.ceil(3.0 / EPS**2)))     # 300
t0_min = 30                                      # strip burn-in floor
m_fix  = int(np.ceil(10.0 * N / EPS**2))         # 12000
s_cnt  = int(np.ceil(10.0 * np.log(1.0 / DELTA)))

print("=" * 78)
print("CRITERION ISOLATION TEST")
print("=" * 78)
print(f"Parameters:  ε={EPS}   δ={DELTA}   n={N}")
print(f"             t₀_standard = max(30, ⌈3/ε²⌉) = {t0_std}")
print(f"             t₀_minimal  = 30  (strips burn-in floor)")
print(f"             fixed m      = ⌈10n/ε²⌉      = {m_fix}")
print(f"             batches s    = ⌈10 ln(1/δ)⌉ = {s_cnt}")
print(f"             Room for criterion to act: t_max/t₀ = {m_fix/t0_std:.1f}×")
print()

GAPS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]   # range from hard to easy problems
TRIALS = 5

# Store results for each gap
results = {}

header = (f"{'gap':>5} | {'CV²':>6} {'m_true':>7} | "
          f"{'FIX':>6} | {'ADA_std T̄':>10} {'err_std':>8} | "
          f"{'ADA_min T̄':>10} {'err_min':>8} | "
          f"{'%fired':>7} | {'sp_std':>7} {'sp_min':>7}")
print(header); print("-" * len(header))

for gap in GAPS:
    ps = np.full(N, 0.5); qs = np.full(N, 0.5 + gap)
    bf = brute_force(ps, qs, EPS)

    fixed_ns, std_Ts, min_Ts = [], [], []
    std_errs, min_errs, fix_errs = [], [], []
    std_fired_counts = []
    min_fired_counts = []

    for _ in range(TRIALS):
        fe, fs = run_fixed(ps, qs, EPS, DELTA, rng)
        ae_s, as_s, Ts_s = run_adaptive(ps, qs, EPS, DELTA, t0_std, rng)
        ae_m, as_m, Ts_m = run_adaptive(ps, qs, EPS, DELTA, t0_min, rng)

        fixed_ns.append(fs)
        std_Ts.extend(Ts_s); min_Ts.extend(Ts_m)
        fix_errs.append(abs(fe  - bf['true_tv']) / bf['true_tv'])
        std_errs.append(abs(ae_s - bf['true_tv']) / bf['true_tv'])
        min_errs.append(abs(ae_m - bf['true_tv']) / bf['true_tv'])
        std_fired_counts.append(sum(1 for t in Ts_s if t > t0_std))
        min_fired_counts.append(sum(1 for t in Ts_m if t > t0_min))

    mean_fix = np.mean(fixed_ns)
    mean_T_std = np.mean(std_Ts); mean_T_min = np.mean(min_Ts)
    frac_fired_min = np.sum(min_fired_counts) / (TRIALS * s_cnt)
    frac_fired_std = np.sum(std_fired_counts) / (TRIALS * s_cnt)

    speedup_std = mean_fix / (mean_T_std * s_cnt)
    speedup_min = mean_fix / (mean_T_min * s_cnt)

    results[gap] = dict(
        bf=bf, mean_fix=mean_fix, mean_T_std=mean_T_std, mean_T_min=mean_T_min,
        frac_fired_std=frac_fired_std, frac_fired_min=frac_fired_min,
        speedup_std=speedup_std, speedup_min=speedup_min,
        err_fix=np.mean(fix_errs), err_std=np.mean(std_errs),
        err_min=np.mean(min_errs),
    )

    print(f"{gap:>5.2f} | {bf['CV_sq']:>6.2f} {bf['m_true']:>7d} | "
          f"{int(mean_fix):>6d} | {mean_T_std:>10.1f} {np.mean(std_errs):>8.4f} | "
          f"{mean_T_min:>10.1f} {np.mean(min_errs):>8.4f} | "
          f"{frac_fired_min*100:>6.1f}% | "
          f"{speedup_std:>6.1f}× {speedup_min:>6.1f}×")

print()
print("Legend:")
print("  m_true    = 10·CV²/ε²   (exact analytical Chebyshev requirement)")
print("  FIX       = total samples used by fixed algorithm (= s · m)")
print("  ADA_std T̄ = mean per-batch stop time, std burn-in (t₀=300)")
print("  ADA_min T̄ = mean per-batch stop time, stripped burn-in (t₀=30)")
print("  %fired    = fraction of ADA_min batches where T > 30 (criterion kept going)")
print("  sp_std    = total speedup with standard burn-in")
print("  sp_min    = total speedup with minimal burn-in")
print()

# ──────────────────────────────────────────────────────────────────────────────
# Decomposition of speedup
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 78)
print("SPEEDUP DECOMPOSITION")
print("=" * 78)
print()
print(f"{'gap':>5} | {'m_true':>7} | {'floor@30':>9} | {'floor@300':>10} | "
      f"{'criterion contrib':>18}")
print("-" * 72)

for gap in GAPS:
    r = results[gap]; bf = r['bf']
    # What does the observed T_min tell us?
    # If criterion never fires, T_min = 30.
    # If criterion always fires to level m_true, T_min ≈ m_true.
    # Contribution of the criterion = amount above floor30
    floor30_cost   = 30 * s_cnt
    floor300_cost  = 300 * s_cnt
    ada_min_cost   = r['mean_T_min'] * s_cnt
    criterion_extra = ada_min_cost - floor30_cost
    print(f"{gap:>5.2f} | {bf['m_true']:>7d} | "
          f"{floor30_cost:>9d} | {floor300_cost:>10d} | "
          f"{criterion_extra:>18,.0f}")

print()
print("Interpretation:")
print("  • When m_true > t₀_std=300: the STANDARD adaptive variant is genuinely")
print("    criterion-limited (can't stop early just because of burn-in).")
print("  • When m_true < t₀_std=300: the standard variant is burn-in-floored.")
print("  • The MINIMAL-burn-in variant is always criterion-limited (except t=30).")
print("    So its observed T is the truthful criterion-driven stop time.")
print("  • If ADA_min T̄ ≈ m_true, the criterion is working correctly.")

# ──────────────────────────────────────────────────────────────────────────────
# Accuracy sanity — must stay < ε relative error
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 78)
print("ACCURACY CHECK — all three methods vs true TV")
print("=" * 78)
print(f"{'gap':>5} | {'true_TV':>7} | {'err_fix':>8} {'err_std':>8} {'err_min':>8} | ε = " + str(EPS))
print("-" * 60)
for gap in GAPS:
    r = results[gap]; bf = r['bf']
    tv = bf['true_tv']
    flags_fix = '✓' if r['err_fix'] <= EPS else '✗'
    flags_std = '✓' if r['err_std'] <= EPS else '✗'
    flags_min = '✓' if r['err_min'] <= EPS else '✗'
    print(f"{gap:>5.2f} | {tv:>7.4f} | "
          f"{r['err_fix']:>7.4f}{flags_fix} {r['err_std']:>7.4f}{flags_std} {r['err_min']:>7.4f}{flags_min}")

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

gaps_arr = np.array(GAPS)
m_trues  = np.array([results[g]['bf']['m_true']  for g in GAPS])
T_stds   = np.array([results[g]['mean_T_std']    for g in GAPS])
T_mins   = np.array([results[g]['mean_T_min']    for g in GAPS])
sp_stds  = np.array([results[g]['speedup_std']   for g in GAPS])
sp_mins  = np.array([results[g]['speedup_min']   for g in GAPS])

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.axhline(t0_std, color='#6B7280', ls=':', lw=1.5, label=f'burn-in floor t₀={t0_std}')
ax.axhline(30,     color='#9CA3AF', ls=':', lw=1.2, label='t₀=30 (minimal)')
ax.axhline(m_fix,  color='#2563EB', ls='--', lw=1.8, label=f'fixed m={m_fix}')
ax.plot(gaps_arr, m_trues, 'k-',  lw=2.5, label='analytical m₀=10·CV²/ε²', zorder=5)
ax.plot(gaps_arr, T_stds,  'o-', color='#EA580C', lw=2.2, ms=9,
        label='ADA std (t₀=300)')
ax.plot(gaps_arr, T_mins,  's-', color='#15803D', lw=2.2, ms=9,
        label='ADA min (t₀=30)')
ax.set_yscale('log')
ax.set_xlabel('Marginal gap  |p − q|', fontsize=11)
ax.set_ylabel('Per-batch samples (log scale)', fontsize=11)
ax.set_title(f'Per-batch stopping time vs analytical minimum  [n={N}, ε={EPS}]',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3, which='both')

ax = axes[1]
x_idx = np.arange(len(GAPS)); w = 0.38
b1 = ax.bar(x_idx - w/2, sp_stds, w, color='#EA580C', alpha=0.85,
            edgecolor='white', label='Standard (t₀=300)')
b2 = ax.bar(x_idx + w/2, sp_mins, w, color='#15803D', alpha=0.85,
            edgecolor='white', label='Minimal (t₀=30)')
ax.axhline(1.0, color='#6B7280', ls='--', lw=1.2)
ax.set_xticks(x_idx); ax.set_xticklabels([f'{g:.2f}' for g in GAPS])
ax.set_xlabel('Marginal gap', fontsize=11)
ax.set_ylabel('Speedup factor (fixed / adaptive)', fontsize=11)
ax.set_title(f'Speedup: burn-in vs no burn-in floor', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + max(sp_stds.max(), sp_mins.max())*0.01,
                f'{h:.1f}×', ha='center', va='bottom', fontsize=8.5)

plt.tight_layout()
outpath = 'criterion_isolation.png'
plt.savefig(outpath, dpi=140, bbox_inches='tight')
print(f"\nFigure saved → {outpath}")
