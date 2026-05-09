"""
Implements the complete Sparsify subroutine, which uses a geometric partition of [0, 1] and (1, ∞).

"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ratio_distribution import RatioDistribution


def build_geometric_partition(eps_s, delta_s):
    """
    Build the geometric interval partition of [0, ∞) used by Sparsify.

    Returns:
    intervals_below : list of (lo, hi, type) for [0, 1)
                      type='geo' means geometric (error ε_s·TV contribution)
                      type='tail' means tail interval (error δ_s)
    intervals_above : list of (lo, hi, type) for (1, ∞)
    """
    # Number of geometric intervals below 1
    if delta_s >= 1.0:
        m = 1
    elif delta_s < 1e-300:
        # delta_s effectively 0 — use a large but finite m
        m = int(np.ceil(700 / np.log(1.0 + eps_s)))
    else:
        m = int(np.ceil(-np.log(delta_s) / np.log(1.0 + eps_s)))
    m = max(m, 1)

    # a_t = 1 - (1 + eps_s)^{-t}
    a = [1.0 - (1.0 + eps_s) ** (-t) for t in range(m + 1)]

    intervals_below = []
    for t in range(m):
        intervals_below.append((a[t], a[t + 1], 'geo'))
    # Last interval up to (but not including) 1
    intervals_below.append((a[m], 1.0, 'tail'))

    # Symmetric intervals above 1
    # J_t = (1/a_{t+1}, 1/a_t] for t < m
    # J_m = (1/a_m, ∞)
    intervals_above = []
    for t in range(m - 1, -1, -1):
        lo = 1.0 / a[t + 1] if a[t + 1] > 1e-15 else 1e15
        hi = 1.0 / a[t] if a[t] > 1e-15 else 1e15
        intervals_above.append((lo, hi, 'geo'))
    # Last interval: (1/a_m, ∞)
    lo_tail = 1.0 / a[m] if a[m] > 1e-15 else 1e15
    intervals_above.append((lo_tail, np.inf, 'tail'))

    return intervals_below, intervals_above, m


def sparsify(R, eps_s, delta_s):
    """
    Sparsify a ratio distribution R using the geometric partition.

    Parameters--
    R       : RatioDistribution to sparsify
    eps_s   : relative error parameter ε_s > 0
    delta_s : absolute error parameter δ_s > 0

    Returns--
    R_sparse : RatioDistribution with small support
               MTV(R, R_sparse) ≤ (1/2)(ε_s·TV(R) + δ_s)
    diagnostics : dict with support size, intervals used, etc.
    """
    intervals_below, intervals_above, m = build_geometric_partition(eps_s, delta_s)

    # Build a lookup: for each support point r, which interval does it fall in?

    all_intervals = []
    # Below 1: (lo, hi) — r in [lo, hi)
    for lo, hi, itype in intervals_below:
        all_intervals.append((lo, hi, itype, 'below'))
    # Above 1: (lo, hi) — r in (lo, hi]
    for lo, hi, itype in intervals_above:
        all_intervals.append((lo, hi, itype, 'above'))

    # Group support points by interval
    buckets = {i: [] for i in range(len(all_intervals))}
    singletons = []  # Points that don't fall in any interval (exactly r=1)

    for r, p in R.support():
        if abs(r - 1.0) < 1e-12:
            singletons.append((r, p))
            continue

        placed = False
        for i, (lo, hi, itype, side) in enumerate(all_intervals):
            if side == 'below' and lo <= r < hi:
                buckets[i].append((r, p))
                placed = True
                break
            elif side == 'above' and lo < r <= hi:
                buckets[i].append((r, p))
                placed = True
                break
            elif side == 'above' and hi == np.inf and r > lo:
                buckets[i].append((r, p))
                placed = True
                break

        if not placed:
            singletons.append((r, p))

    # Build sparsified support
    new_support = list(singletons)  # keep r=1 exactly

    intervals_used = 0
    for i, pts in buckets.items():
        if not pts:
            continue
        intervals_used += 1
        total_mass = sum(p for _, p in pts)
        if total_mass < 1e-15:
            continue
        # Representative r* = R†(bucket) / R(bucket)
        alt_mass = sum(r * p for r, p in pts)
        r_star = alt_mass / total_mass
        new_support.append((r_star, total_mass))

    # Renormalize to fix floating point drift
    total = sum(p for _, p in new_support)
    if total > 1e-15:
        new_support = [(r, p / total) for r, p in new_support]

    R_sparse = RatioDistribution(new_support)

    diagnostics = {
        "support_before": R.support_size(),
        "support_after": R_sparse.support_size(),
        "num_intervals": len(intervals_below) + len(intervals_above),
        "intervals_used": intervals_used,
        "m": m,
        "tv_before": R.tv_distance(),
        "tv_after": R_sparse.tv_distance(),
        "theoretical_bound": 0.5 * (eps_s * R.tv_distance() + delta_s),
    }

    return R_sparse, diagnostics


def compute_mtv_upper_bound(R, R_sparse, eps_s, delta_s):
    """
    Compute the theoretical MTV upper bound from Lemma 14
    """
    return 0.5 * (eps_s * R.tv_distance() + delta_s)


# # ── Self-test ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     from brute_force import brute_force_tv_vectorized, random_marginals

#     print("=== Step 5: Full Sparsify Algorithm ===\n")

#     # Test 1: Basic sparsification
#     print("Test 1: Sparsify a single-coordinate ratio")
#     P_m = np.array([0.7, 0.3])
#     Q_m = np.array([0.5, 0.5])
#     R = RatioDistribution.from_marginals(P_m, Q_m)
#     print(f"  Original: {R}")
#     R_sparse, diag = sparsify(R, eps_s=0.1, delta_s=0.01)
#     print(f"  Sparsified: {R_sparse}")
#     print(f"  Support: {diag['support_before']} → {diag['support_after']}")
#     print(f"  TV: {diag['tv_before']:.6f} → {diag['tv_after']:.6f}")
#     print(f"  Theoretical MTV bound: {diag['theoretical_bound']:.6f}")

#     # Test 2: Sparsify after n-step product (main use case)
#     print("\nTest 2: Sparsify n=10 product ratio, verify TV preserved")
#     rng = np.random.default_rng(0)
#     n = 10
#     mP = random_marginals(n, 2, rng)
#     mQ = random_marginals(n, 2, rng)

#     # Build exact ratio
#     R = RatioDistribution.from_marginals(mP[0], mQ[0])
#     for i in range(1, n):
#         Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
#         R = RatioDistribution.independent_product(R, Ri)

#     exact_tv = brute_force_tv_vectorized(mP, mQ)
#     print(f"  Exact TV: {exact_tv:.6f}")
#     print(f"  TV from full ratio: {R.tv_distance():.6f}")
#     print(f"  Support size (exact): {R.support_size()} = 2^{n}")

#     eps_s, delta_s = 0.1, 0.001
#     R_sparse, diag = sparsify(R, eps_s, delta_s)
#     print(f"\n  After sparsify(ε_s={eps_s}, δ_s={delta_s}):")
#     print(f"  Support: {diag['support_before']} → {diag['support_after']}")
#     print(f"  TV from sparsified: {R_sparse.tv_distance():.6f}")
#     print(f"  |TV error|: {abs(R_sparse.tv_distance() - exact_tv):.6f}")
#     print(f"  MTV bound:  {diag['theoretical_bound']:.6f}")
#     print(f"  Bound holds: {abs(R_sparse.tv_distance() - exact_tv) <= 2*diag['theoretical_bound'] + 1e-9}")

#     # Test 3: Support size vs ε_s
#     print("\nTest 3: Support size as a function of ε_s and δ_s")
#     print(f"  {'eps_s':>8} {'delta_s':>10} {'support':>8} {'theory':>8} {'m':>4}")
#     R_big = R  # the n=10 ratio with 2^10=1024 support
#     for eps_s in [0.5, 0.2, 0.1, 0.05, 0.01]:
#         for delta_s in [0.01, 0.001]:
#             R_sp, d = sparsify(R_big, eps_s, delta_s)
#             # Theoretical: O((1/ε_s) * log(1/δ_s))
#             theory = int(np.ceil(2 / eps_s * np.log(1 / delta_s)))
#             print(f"  {eps_s:>8.3f} {delta_s:>10.4f} {d['support_after']:>8} "
#                   f"{theory:>8} {d['m']:>4}")

#     # Test 4: Error bound holds across many random ratios
#     print("\nTest 4: MTV bound validation over 50 random n=8 ratios")
#     violations = 0
#     rng = np.random.default_rng(1)
#     eps_s, delta_s = 0.1, 0.005
#     for trial in range(50):
#         n = 8
#         mP = random_marginals(n, 2, rng)
#         mQ = random_marginals(n, 2, rng)
#         R_t = RatioDistribution.from_marginals(mP[0], mQ[0])
#         for i in range(1, n):
#             Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
#             R_t = RatioDistribution.independent_product(R_t, Ri)
#         R_sp, d = sparsify(R_t, eps_s, delta_s)
#         actual_change = abs(R_t.tv_distance() - R_sp.tv_distance())
#         bound = 2 * d['theoretical_bound']  # factor 2 from Lemma 11
#         if actual_change > bound + 1e-9:
#             violations += 1
#     print(f"  Violations: {violations}/50 (should be 0)")

#     # Test 5: Plot TV preservation vs ε_s
#     print("\nTest 5: TV error vs ε_s (saving plot)")
#     eps_vals = np.logspace(-2, -0.3, 30)
#     tv_errors = []
#     bounds = []
#     rng = np.random.default_rng(3)
#     n = 12
#     mP = random_marginals(n, 2, rng)
#     mQ = random_marginals(n, 2, rng)
#     R_ref = RatioDistribution.from_marginals(mP[0], mQ[0])
#     for i in range(1, n):
#         Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
#         R_ref = RatioDistribution.independent_product(R_ref, Ri)
#     true_tv = R_ref.tv_distance()

#     for eps_s in eps_vals:
#         delta_s = eps_s * 0.01
#         R_sp, d = sparsify(R_ref, eps_s, delta_s)
#         tv_errors.append(abs(R_sp.tv_distance() - true_tv))
#         bounds.append(d['theoretical_bound'])

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#     ax1.loglog(eps_vals, tv_errors, 'b-o', markersize=3, label='|TV error|')
#     ax1.loglog(eps_vals, bounds, 'r--', label='Bound (ε_s·TV/2 + δ_s/2)')
#     ax1.set_xlabel('ε_s')
#     ax1.set_ylabel('TV error after sparsification')
#     ax1.set_title('TV error vs ε_s (n=12)')
#     ax1.legend()
#     ax1.grid(True, which='both', alpha=0.3)

#     support_sizes = []
#     for eps_s in eps_vals:
#         delta_s = eps_s * 0.01
#         R_sp, d = sparsify(R_ref, eps_s, delta_s)
#         support_sizes.append(d['support_after'])

#     ax2.loglog(eps_vals, support_sizes, 'g-o', markersize=3)
#     ax2.axhline(y=2**n, color='gray', linestyle='--', label=f'Full support (2^{n}={2**n})')
#     ax2.set_xlabel('ε_s')
#     ax2.set_ylabel('Support size after sparsification')
#     ax2.set_title('Support compression vs ε_s')
#     ax2.legend()
#     ax2.grid(True, which='both', alpha=0.3)

#     plt.tight_layout()
#     plt.savefig('/home/claude/tv_distance/step5_sparsify.png', dpi=100)
#     plt.close()
#     print("  Plot saved to step5_sparsify.png")

#     print("\n✓ Step 5 complete.")
