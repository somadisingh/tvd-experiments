"""
Implements Algorithm 2- Deterministic FPTAS for TV distance

The algorithm:
  1. Compute d_LB = max_i TV(P_i, Q_i)  (lower bound on TV(P,Q)/n)
  2. Initialize R'_{1:1} = (P_1‖Q_1)
  3. For k = 1 to n-1:
       a. Sparsify: R̃_{1:k} = Sparsify(R'_{1:k}, ε/2n, ε·d_LB/2n)
       b. R_{k+1} = (P_{k+1}‖Q_{k+1})
       c. R'_{1:k+1} = R̃_{1:k} ·_indp R_{k+1}
  4. Return TV(R'_{1:n})

Theorem 1:
  (1-ε)·TV(P,Q) ≤ output ≤ TV(P,Q)
  Runtime: O(qn²/ε · log q · log(n / (ε·TV(P,Q))))
"""

import numpy as np
import time
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ratio_distribution import RatioDistribution
from sparsify_full import sparsify


def marginal_tv(p_m, q_m):
    """TV distance between two marginal distributions."""
    return 0.5 * float(np.sum(np.abs(np.array(p_m) - np.array(q_m))))


def compute_lower_bound(marginals_P, marginals_Q):
    """
    d_LB = max_i TV(P_i, Q_i)

    """
    return max(marginal_tv(marginals_P[i], marginals_Q[i])
               for i in range(len(marginals_P)))


def deterministic_tv(marginals_P, marginals_Q, epsilon,
                     return_diagnostics=False):
    """
    Deterministic FPTAS for TV(P, Q) with relative error ε.

    Parameters
    ----------
    marginals_P, marginals_Q : list of n arrays over [q]
    epsilon : relative error ε ∈ (0, 1)
    return_diagnostics : if True, return per-step info

    Returns
    -------
    float : estimate b̂ satisfying (1-ε)·TV(P,Q) ≤ b̂ ≤ TV(P,Q)
    """
    n = len(marginals_P)
    t_start = time.time()

    # Lower bound
    d_LB = compute_lower_bound(marginals_P, marginals_Q)

    # Edge case: P and Q are identical
    if d_LB < 1e-15:
        if return_diagnostics:
            return 0.0, {"n": n, "epsilon": epsilon, "d_LB": 0.0,
                         "elapsed": 0.0, "steps": []}
        return 0.0

    # Set sparsification parameters for each step
    eps_s = epsilon / (2 * n)
    delta_s = epsilon * d_LB / (2 * n)

    # Initialize with first coordinate's ratio
    R_current = RatioDistribution.from_marginals(marginals_P[0], marginals_Q[0])

    diagnostics = [] if return_diagnostics else None

    if return_diagnostics:
        diagnostics.append({
            "step": 0,
            "support_before_sparsify": R_current.support_size(),
            "support_after_sparsify": R_current.support_size(),
            "support_after_product": R_current.support_size(),
            "tv_estimate": R_current.tv_distance(),
            "sparsify_bound": 0.0,
        })

    # Iteratively convolve and sparsify
    for k in range(1, n):
    
        R_sparse, diag = sparsify(R_current, eps_s, delta_s)
        support_after_sparsify = R_sparse.support_size()

        R_next = RatioDistribution.from_marginals(
            marginals_P[k], marginals_Q[k])
        R_current = RatioDistribution.independent_product(R_sparse, R_next)

        if return_diagnostics:
            diagnostics.append({
                "step": k,
                "support_before_sparsify": diag["support_before"],
                "support_after_sparsify": support_after_sparsify,
                "support_after_product": R_current.support_size(),
                "tv_estimate": R_current.tv_distance(),
                "sparsify_bound": diag["theoretical_bound"],
            })

    result = R_current.tv_distance()
    elapsed = time.time() - t_start

    if return_diagnostics:
        return result, {
            "n": n,
            "epsilon": epsilon,
            "d_LB": d_LB,
            "eps_s": eps_s,
            "delta_s": delta_s,
            "elapsed": elapsed,
            "steps": diagnostics,
        }
    return result


# # ── Self-test ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     from brute_force import (brute_force_tv_vectorized, random_marginals,
#                               nearly_identical_marginals, adversarial_marginals)

#     print("=== Deterministic FPTAS ===\n")

#     # Test 1: Basic correctness
#     print("Test 1: Basic correctness, n=6, q=2")
#     rng = np.random.default_rng(42)
#     mP = random_marginals(6, 2, rng)
#     mQ = random_marginals(6, 2, rng)
#     exact = brute_force_tv_vectorized(mP, mQ)
#     est, diag = deterministic_tv(mP, mQ, epsilon=0.1, return_diagnostics=True)
#     rel_err = abs(est - exact) / exact
#     print(f"  Exact:     {exact:.6f}")
#     print(f"  Estimate:  {est:.6f}")
#     print(f"  Rel error: {rel_err:.4f}  (ε=0.1, should be ≤0.1)")
#     print(f"  Is lower bound: {est <= exact + 1e-9}")
#     print(f"  d_LB = {diag['d_LB']:.4f}")

#     # Test 2: Per-step diagnostics
#     print("\nTest 2: Support size growth per step")
#     print(f"  {'step':>5} {'before_spar':>12} {'after_spar':>10} "
#           f"{'after_prod':>10} {'tv_est':>8}")
#     for s in diag['steps']:
#         print(f"  {s['step']:>5} {s['support_before_sparsify']:>12} "
#               f"{s['support_after_sparsify']:>10} "
#               f"{s['support_after_product']:>10} "
#               f"{s['tv_estimate']:>8.4f}")

#     # Test 3: Guarantee holds for many ε values
#     print("\nTest 3: Relative error guarantee at various ε")
#     rng = np.random.default_rng(7)
#     mP = random_marginals(8, 2, rng)
#     mQ = random_marginals(8, 2, rng)
#     exact = brute_force_tv_vectorized(mP, mQ)
#     print(f"  Exact TV = {exact:.4f}")
#     print(f"  {'epsilon':>8} {'estimate':>10} {'rel_err':>8} {'ok':>5}")
#     for eps in [0.5, 0.3, 0.2, 0.1, 0.05]:
#         est = deterministic_tv(mP, mQ, epsilon=eps)
#         rel_err = abs(est - exact) / exact
#         ok = rel_err <= eps
#         print(f"  {eps:>8.2f} {est:>10.6f} {rel_err:>8.4f} {str(ok):>5}")

#     # Test 4: P = Q → estimate ≈ 0
#     print("\nTest 4: P = Q (should give 0)")
#     mP = random_marginals(6, 2, rng)
#     est = deterministic_tv(mP, mP, epsilon=0.1)
#     print(f"  Estimate: {est:.10f} (should be 0)")

#     # Test 5: Small TV distance
#     print("\nTest 5: Small TV distance regime")
#     mP, mQ = nearly_identical_marginals(8, 2, epsilon=0.005, rng=rng)
#     exact = brute_force_tv_vectorized(mP, mQ)
#     est = deterministic_tv(mP, mQ, epsilon=0.1)
#     rel_err = abs(est - exact) / exact if exact > 1e-10 else float('inf')
#     print(f"  Exact TV: {exact:.6f}")
#     print(f"  Estimate: {est:.6f}")
#     print(f"  Rel error: {rel_err:.4f}")

#     # Test 6: Adversarial distributions
#     print("\nTest 6: Adversarial (all marginals identical)")
#     for n in [4, 6, 8]:
#         mP, mQ = adversarial_marginals(n, 2)
#         exact = brute_force_tv_vectorized(mP, mQ)
#         est = deterministic_tv(mP, mQ, epsilon=0.1)
#         rel_err = abs(est - exact) / exact
#         print(f"  n={n}: exact={exact:.4f}, est={est:.4f}, "
#               f"rel_err={rel_err:.4f}, ok={rel_err<=0.1}")

#     # Test 7: Runtime vs n
#     print("\nTest 7: Runtime vs n")
#     import time
#     print(f"  {'n':>5} {'exact_time':>12} {'det_time':>12} {'rel_err':>8}")
#     rng = np.random.default_rng(99)
#     for n in [4, 6, 8, 10, 12, 15]:
#         mP = random_marginals(n, 2, rng)
#         mQ = random_marginals(n, 2, rng)
#         if n <= 20:
#             t0 = time.time()
#             exact = brute_force_tv_vectorized(mP, mQ)
#             t_exact = time.time() - t0
#         else:
#             exact = None
#             t_exact = None

#         t0 = time.time()
#         est = deterministic_tv(mP, mQ, epsilon=0.1)
#         t_det = time.time() - t0

#         if exact is not None:
#             rel_err = abs(est - exact) / exact
#             print(f"  {n:>5} {t_exact:>11.4f}s {t_det:>11.4f}s {rel_err:>8.4f}")
#         else:
#             print(f"  {n:>5} {'N/A':>12} {t_det:>11.4f}s {'N/A':>8}")

#     # Test 8: Plot support growth for different distribution types
#     print("\nTest 8: Support size evolution plot (saving)")
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     distribution_types = [
#         ("Random", lambda rng, n: (random_marginals(n, 2, rng),
#                                     random_marginals(n, 2, rng))),
#         ("Nearly Identical", lambda rng, n: nearly_identical_marginals(
#             n, 2, epsilon=0.01, rng=rng)),
#         ("Adversarial", lambda rng, n: adversarial_marginals(n, 2)),
#     ]
#     rng = np.random.default_rng(5)
#     n = 12
#     eps = 0.15

#     for ax, (name, dist_fn) in zip(axes, distribution_types):
#         mP, mQ = dist_fn(rng, n)
#         _, diag = deterministic_tv(mP, mQ, epsilon=eps, return_diagnostics=True)

#         steps = [s['step'] for s in diag['steps']]
#         before = [s['support_before_sparsify'] for s in diag['steps']]
#         after_s = [s['support_after_sparsify'] for s in diag['steps']]
#         after_p = [s['support_after_product'] for s in diag['steps']]

#         ax.plot(steps, before, 'b-o', markersize=4, label='Before sparsify')
#         ax.plot(steps, after_s, 'r-s', markersize=4, label='After sparsify')
#         ax.plot(steps, after_p, 'g-^', markersize=4, label='After product')
#         ax.set_xlabel('Step k')
#         ax.set_ylabel('Support size')
#         ax.set_title(f'{name} (n={n})')
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)

#     plt.suptitle(f'Support size per step (ε={eps})', y=1.02)
#     plt.tight_layout()
#     plt.savefig('/home/claude/tv_distance/step6_deterministic.png',
#                 dpi=100, bbox_inches='tight')
#     plt.close()
#     print("  Plot saved to step6_deterministic.png")

#     print("\n✓ Step 6 complete.")
