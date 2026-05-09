#!/usr/bin/env python3
"""
SANITY CHECK — verifies every component against brute-force analytical values.

Checks:
  1. π is a valid probability distribution (sums to 1)
  2. E_π[f] = d_TV / γ  (the key identity — if this fails everything is wrong)
  3. f(x) ∈ [0,1] for all x in support
  4. Var_π[f] ≤ E_π[f]  (the theoretical bound)
  5. sample_pi produces the correct distribution (KL divergence test)
  6. Stopping time T aligns with the analytical true-minimum sample count
  7. Both algorithm estimates land within ε of true TV distance
"""

import numpy as np
rng = np.random.default_rng(0)

EPS   = 0.5
DELTA = 0.1

# ─── paste the core functions ─────────────────────────────────────────────────

def compute_gamma(ps, qs):
    return 1.0 - float(np.prod(1.0 - np.abs(ps - qs)))

def compute_suffix_B(ps, qs):
    n = len(ps)
    d = np.abs(ps - qs)
    B = np.ones(n + 1)
    for k in range(n - 1, -1, -1):
        B[k] = (1.0 - d[k]) * B[k + 1]
    return B

def sample_pi(ps, qs, B, rng):
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
    n = len(ps)
    gamma = compute_gamma(ps, qs)
    if gamma < 1e-12: return 0.0, 0
    B = compute_suffix_B(ps, qs)
    m = int(np.ceil(10.0 * n / eps**2))
    s = int(np.ceil(10.0 * np.log(1.0 / delta)))
    total = 0; batches = []
    for _ in range(s):
        vals = [f_val(sample_pi(ps, qs, B, rng), ps, qs) for _ in range(m)]
        batches.append(float(np.mean(vals))); total += m
    return gamma * float(np.median(batches)), total

def run_adaptive(ps, qs, eps, delta, rng):
    n = len(ps)
    gamma = compute_gamma(ps, qs)
    if gamma < 1e-12: return 0.0, 0
    B = compute_suffix_B(ps, qs)
    t_max = int(np.ceil(10.0 * n / eps**2))
    t0    = max(30, int(np.ceil(3.0 / eps**2)))
    s     = int(np.ceil(10.0 * np.log(1.0 / delta)))
    total = 0; batches = []
    for _ in range(s):
        vals = [f_val(sample_pi(ps, qs, B, rng), ps, qs) for _ in range(t0)]
        total += t0; t = t0
        F = float(np.mean(vals))
        M = sum((v - F)**2 for v in vals)
        S2 = M / (t - 1) if t > 1 else 0.0
        while t < t_max:
            if F > 1e-12 and S2 / (F * F * eps * eps * t) <= 0.1: break
            fv = f_val(sample_pi(ps, qs, B, rng), ps, qs)
            total += 1; t += 1
            d1 = fv - F; F += d1 / t; d2 = fv - F; M += d1 * d2
            S2 = M / (t - 1) if t > 1 else 0.0
        batches.append(F)
    return gamma * float(np.median(batches)), total

# ─── ANALYTICAL BRUTE-FORCE (n ≤ 15) ─────────────────────────────────────────

def brute_force_stats(ps, qs):
    """
    Enumerate all 2^n outcomes.  Returns:
      - true_tv, gamma
      - E_f, Var_f (exact under π)
      - pi_mass per outcome (dict: tuple(x) -> float)
      - f_values per outcome (dict: tuple(x) -> float)
      - f_in_01: True if every f(x) with π(x)>0 is in [0,1]
    """
    n = len(ps); assert n <= 15
    gamma = compute_gamma(ps, qs)
    pi_mass = {}; f_values = {}
    true_tv = 0.0; E_f = 0.0; E_f2 = 0.0; pi_sum = 0.0; f_in_01 = True

    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)])
        pv = qv = 1.0
        for i in range(n):
            b = x[i]
            pv *= ps[i] if b else (1-ps[i])
            qv *= qs[i] if b else (1-qs[i])
        true_tv += max(0.0, pv - qv)

        # π(x)
        prod_alpha = 1.0
        for i in range(n):
            p_i = ps[i] if x[i] == 1 else (1-ps[i])
            q_i = qs[i] if x[i] == 1 else (1-qs[i])
            if p_i > 1e-14: prod_alpha *= min(p_i, q_i) / p_i
        pi_x = pv * (1 - prod_alpha) / gamma if gamma > 1e-14 else 0.0
        pi_sum += pi_x

        fv = f_val(x, ps, qs)
        if pi_x > 1e-12 and not (0.0 - 1e-9 <= fv <= 1.0 + 1e-9):
            f_in_01 = False

        key = tuple(x.tolist())
        pi_mass[key] = pi_x
        f_values[key] = fv
        E_f  += pi_x * fv
        E_f2 += pi_x * fv**2

    Var_f = E_f2 - E_f**2
    CV_sq = Var_f / E_f**2 if E_f > 1e-14 else float('inf')
    m_true = int(np.ceil(10 * CV_sq / EPS**2))

    return dict(
        true_tv=true_tv, gamma=gamma,
        E_f=E_f, E_f_theory=true_tv/gamma if gamma>1e-14 else 0.0,
        Var_f=Var_f, CV_sq=CV_sq,
        pi_sum=pi_sum, f_in_01=f_in_01,
        var_bound_holds=(Var_f <= E_f + 1e-9),
        m_true=m_true,
        pi_mass=pi_mass, f_values=f_values
    )

def empirical_pi_distribution(ps, qs, n_samples=50000):
    """Draw n_samples from sample_pi, record empirical frequencies."""
    B   = compute_suffix_B(ps, qs)
    freq = {}
    for _ in range(n_samples):
        key = tuple(sample_pi(ps, qs, B, rng).tolist())
        freq[key] = freq.get(key, 0) + 1
    return {k: v / n_samples for k, v in freq.items()}

def kl_divergence(p_dict, q_dict):
    """KL(true π ‖ empirical π).  A value < 0.005 means the sampler is correct."""
    kl = 0.0
    for key, p in p_dict.items():
        if p < 1e-12: continue
        q = q_dict.get(key, 1e-9)
        kl += p * np.log(p / q)
    return kl


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1-5  on three concrete cases
# ══════════════════════════════════════════════════════════════════════════════

CASES = [
    # (label,  ps,              qs,               comment)
    ("n=2 (manual)",   np.array([0.8, 0.7]), np.array([0.3, 0.4]),   "Simple 2-coord case"),
    ("n=8 gap=0.05",   np.full(8, 0.5),      np.full(8, 0.55),       "Low separation — suspicious speedup"),
    ("n=8 gap=0.40",   np.full(8, 0.5),      np.full(8, 0.90),       "High separation"),
]

print("=" * 72)
print("CHECKS 1–5: ANALYTICAL VERIFICATION")
print("=" * 72)

for label, ps, qs, comment in CASES:
    print(f"\n── {label}  ({comment})")
    s = brute_force_stats(ps, qs)

    # CHECK 1: π sums to 1
    c1 = abs(s['pi_sum'] - 1.0) < 1e-8
    print(f"  [{'PASS' if c1 else 'FAIL'}]  1. π sums to {s['pi_sum']:.8f}  (want 1.0)")

    # CHECK 2: E_π[f] = d_TV / γ
    err2 = abs(s['E_f'] - s['E_f_theory'])
    c2 = err2 < 1e-8
    print(f"  [{'PASS' if c2 else 'FAIL'}]  2. E_π[f] = {s['E_f']:.6f}  vs  d_TV/γ = {s['E_f_theory']:.6f}  (diff {err2:.2e})")

    # CHECK 3: f(x) ∈ [0,1]
    print(f"  [{'PASS' if s['f_in_01'] else 'FAIL'}]  3. f(x) ∈ [0,1] for all x with π(x)>0")

    # CHECK 4: Var_π[f] ≤ E_π[f]
    print(f"  [{'PASS' if s['var_bound_holds'] else 'FAIL'}]  4. Var_π[f] = {s['Var_f']:.6f}  ≤  E_π[f] = {s['E_f']:.6f}")

    # Key diagnostic: how tight is the bound?
    ratio = s['Var_f'] / s['E_f'] if s['E_f'] > 1e-12 else 0.0
    print(f"         >>> Var/E[f] = {ratio:.4f}  (worst-case bound is 1.0)")
    print(f"         >>> CV²      = {s['CV_sq']:.4f}")
    print(f"         >>> True m₀  = {s['m_true']} samples/batch  "
          f"vs fixed m = {int(np.ceil(10*len(ps)/EPS**2))}")

    # CHECK 5: sampler KL divergence (only for n≤8 with not-too-many outcomes)
    if len(ps) <= 8:
        emp = empirical_pi_distribution(ps, qs, n_samples=80000)
        kl  = kl_divergence(s['pi_mass'], emp)
        c5  = kl < 0.005
        print(f"  [{'PASS' if c5 else 'FAIL'}]  5. KL(true π ‖ sampled π) = {kl:.5f}  (want < 0.005)")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6: Does the adaptive algorithm stop at ≈ m_true samples per batch?
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CHECK 6: ADAPTIVE STOPPING TIME vs ANALYTICAL TRUE MINIMUM")
print("=" * 72)

ps8 = np.full(8, 0.5)

for gap in [0.05, 0.20, 0.40]:
    qs8 = ps8 + gap
    s   = brute_force_stats(ps8, qs8)
    B   = compute_suffix_B(ps8, qs8)
    gamma = s['gamma']
    n   = len(ps8)

    t_max = int(np.ceil(10.0 * n / EPS**2))
    t0    = max(30, int(np.ceil(3.0 / EPS**2)))
    num_s = int(np.ceil(10.0 * np.log(1.0 / DELTA)))

    stop_times = []
    for _ in range(num_s * 10):          # 10× more runs for stable average
        vals = [f_val(sample_pi(ps8, qs8, B, rng), ps8, qs8) for _ in range(t0)]
        t = t0
        F = float(np.mean(vals))
        M = sum((v - F)**2 for v in vals)
        S2 = M / (t - 1) if t > 1 else 0.0
        while t < t_max:
            if F > 1e-12 and S2 / (F * F * EPS * EPS * t) <= 0.1: break
            fv = f_val(sample_pi(ps8, qs8, B, rng), ps8, qs8)
            t += 1
            d1 = fv - F; F += d1 / t; d2 = fv - F; M += d1 * d2
            S2 = M / (t - 1) if t > 1 else 0.0
        stop_times.append(t)

    avg_T = np.mean(stop_times)
    print(f"\n  gap={gap:.2f}  |  true_TV={s['true_tv']:.4f}  γ={s['gamma']:.4f}")
    print(f"    Analytical:  E_π[f]={s['E_f']:.4f}  Var_π[f]={s['Var_f']:.4f}  "
          f"CV²={s['CV_sq']:.4f}")
    print(f"    True m₀ (Chebyshev exact):  {s['m_true']} samples/batch")
    print(f"    Fixed m (worst-case bound): {t_max} samples/batch")
    print(f"    Adaptive avg stop time T:   {avg_T:.1f} samples/batch  "
          f"({'≈ TRUE' if abs(avg_T - s['m_true']) < s['m_true'] * 0.5 else 'MISMATCH'})")
    speedup_theoretical = t_max / s['m_true']
    speedup_observed    = t_max / avg_T
    print(f"    Theoretical speedup: {speedup_theoretical:.1f}×   "
          f"Observed speedup: {speedup_observed:.1f}×")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7: Both algorithms give correct answers (within ε of true TV)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("CHECK 7: ACCURACY — both estimates within ε of true TV over 30 runs")
print("=" * 72)

RUNS = 30
for gap in [0.05, 0.20, 0.40]:
    ps8 = np.full(8, 0.5)
    qs8 = ps8 + gap
    true_tv = brute_force_stats(ps8, qs8)['true_tv']

    f_errors = []; a_errors = []
    for _ in range(RUNS):
        fe, _ = run_fixed(ps8, qs8, EPS, DELTA, rng)
        ae, _ = run_adaptive(ps8, qs8, EPS, DELTA, rng)
        f_errors.append(abs(fe - true_tv) / true_tv)
        a_errors.append(abs(ae - true_tv) / true_tv)

    f_fail = sum(1 for e in f_errors if e > EPS)
    a_fail = sum(1 for e in a_errors if e > EPS)

    print(f"\n  gap={gap:.2f}  true_TV={true_tv:.4f}  (ε={EPS}, {RUNS} independent runs)")
    print(f"    Fixed   — mean rel. error: {np.mean(f_errors):.4f}  "
          f"max: {max(f_errors):.4f}  failures (>ε): {f_fail}/{RUNS}")
    print(f"    Adaptive— mean rel. error: {np.mean(a_errors):.4f}  "
          f"max: {max(a_errors):.4f}  failures (>ε): {a_fail}/{RUNS}")

print("\n" + "=" * 72)
print("SANITY CHECK COMPLETE")
print("=" * 72)
