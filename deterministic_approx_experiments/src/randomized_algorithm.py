"""
===================================================
Implements the Monte Carlo algorithm - Randomized FPRAS for TV Distance

Core idea:
  - Let C be the coordinate-wise greedy coupling (each coordinate
    coupled optimally, independently).
  - Let π(ω) = Pr_C[X = ω | X ≠ Y]  (C conditioned on disagreement)
  - Define estimator f(ω) = Pr_O[X=ω, X≠Y] / Pr_C[X=ω, X≠Y]
  - Then E_π[f] = Pr_O[X≠Y] / Pr_C[X≠Y]
  - And TV(P,Q) = Pr_C[X≠Y] · E_π[f]

Key properties proved in the paper:
  - 1/n ≤ E_π[f] ≤ 1         
  - Var_π[f] ≤ E_π[f]      
  These ensure O(n/ε²) samples suffice.
"""

import numpy as np


# ── Core probability computations ───────────────────────────────────────────

def compute_greedy_coupling_prob(marginals_P, marginals_Q):
    """
    Compute Pr_C[X ≠ Y] for the coordinate-wise greedy coupling.

    """
    n = len(marginals_P)
    prob_agree = 1.0
    for i in range(n):
        tv_i = 0.5 * np.sum(np.abs(marginals_P[i] - marginals_Q[i]))
        prob_agree *= (1.0 - tv_i)
    return 1.0 - prob_agree


def sample_from_pi(marginals_P, marginals_Q, rng):
    """
    Sample ω ~ π = Pr_C[X = ω | X ≠ Y].

    We sample by computing the conditional marginals.

    """
    n = len(marginals_P)
    q = len(marginals_P[0])

    # Precompute suffix products
    suffix_agree = np.ones(n + 1)
    for i in range(n - 1, -1, -1):
        tv_i = 0.5 * np.sum(np.abs(marginals_P[i] - marginals_Q[i]))
        suffix_agree[i] = suffix_agree[i + 1] * (1.0 - tv_i)

    # Pr_C[X ≠ Y] = 1 - suffix_agree[0]
    prob_disagree = 1.0 - suffix_agree[0]

    omega = []
    prefix_min_over_p = 1.0
    prefix_p = 1.0

    for k in range(n):
        # Compute unnormalized probability of each value c for coordinate k
        unnorm = np.zeros(q)
        for c in range(q):
            p_c = marginals_P[k][c]
            if p_c < 1e-15:
                unnorm[c] = 0.0
                continue
            min_pq_c = min(marginals_P[k][c], marginals_Q[k][c])

            # Pr_C[X=Y | X_1=ω_1,...,X_k=c]
            pr_agree_given = (prefix_min_over_p * (min_pq_c / p_c)
                              * suffix_agree[k + 1])

            # Numerator of π_k(c | ω_1,...,ω_{k-1}):
            # (1 - pr_agree_given) * prod_{i<k} P_i(ω_i) * P_k(c)
            unnorm[c] = (1.0 - pr_agree_given) * prefix_p * p_c

        total = unnorm.sum()
        if total < 1e-15:
            c = rng.integers(0, q)
        else:
            probs = unnorm / total
            c = rng.choice(q, p=probs)

        omega.append(c)
        # Update running products
        p_c = marginals_P[k][c]
        min_pq_c = min(marginals_P[k][c], marginals_Q[k][c])
        prefix_min_over_p *= (min_pq_c / p_c) if p_c > 1e-15 else 1.0
        prefix_p *= p_c

    return tuple(omega)


def compute_estimator_f(omega, marginals_P, marginals_Q):
    """
    Compute f(ω) = max(0, P(ω) - Q(ω)) / Pr_C[X=ω, X≠Y]

    Returns 0 if P(ω) = 0 (ω outside support of P).
    """
    n = len(marginals_P)

    ratio_q_over_p = 1.0
    # Compute Pr_C[X=Y | X=ω] = prod_i min(P_i,Q_i)(ω_i) / P_i(ω_i)
    pr_agree_given_omega = 1.0

    for k in range(n):
        c = omega[k]
        p_c = marginals_P[k][c]
        q_c = marginals_Q[k][c]

        if p_c < 1e-15:
            return 0.0  # ω not in support of P, f(ω) = 0

        ratio_q_over_p *= q_c / p_c
        pr_agree_given_omega *= min(p_c, q_c) / p_c

    # f(ω) = max(0, 1 - Q(ω)/P(ω)) / (1 - Pr_C[X=Y | X=ω])
    numerator = max(0.0, 1.0 - ratio_q_over_p)
    denominator = 1.0 - pr_agree_given_omega

    if denominator < 1e-15:
        return 0.0

    return numerator / denominator


# ── Main algorithm ───────────────────────────────────────────────────────────

def randomized_tv_estimate(marginals_P, marginals_Q, epsilon, delta,
                            rng=None, return_diagnostics=False,
                            max_samples=2000):
    """
    FPRAS for TV(P, Q) with relative error ε and failure probability δ.

    max_samples : cap on total samples (for experiments; reduces accuracy
                  guarantee but keeps runtime tractable for large n/small ε)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(marginals_P)

    # Pr_C[X ≠ Y] — exactly computable
    prob_disagree = compute_greedy_coupling_prob(marginals_P, marginals_Q)

    if prob_disagree < 1e-15:
        return 0.0

    # Batch sizes from the paper
    m = int(np.ceil(10 * n / (epsilon ** 2)))
    s = int(np.ceil(10 * np.log(1.0 / delta)))

    # Cap total samples for tractability in experiments
    total_budget = m * s
    if total_budget > max_samples:
        m = max(1, max_samples // s)

    batch_means = []
    total_samples = 0

    for _ in range(s):
        batch_sum = 0.0
        for _ in range(m):
            omega = sample_from_pi(marginals_P, marginals_Q, rng)
            f_val = compute_estimator_f(omega, marginals_P, marginals_Q)
            batch_sum += f_val
            total_samples += 1
        batch_means.append(batch_sum / m)

    f_hat = float(np.median(batch_means))
    tv_estimate = prob_disagree * f_hat

    if return_diagnostics:
        return tv_estimate, {
            "m": m, "s": s, "total_samples": total_samples,
            "prob_disagree": prob_disagree,
            "f_hat": f_hat, "batch_means": batch_means,
        }
    return tv_estimate


# # ── Self-test ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, "/home/claude/tv_distance/src")
#     from brute_force import brute_force_tv_vectorized, random_marginals

#     print("=== Step 2: Randomized TV Distance Algorithm ===\n")
#     rng = np.random.default_rng(42)

#     # Test 1: n=1 (known answer)
#     print("Test 1: n=1, q=2, P=Bern(0.7), Q=Bern(0.3)")
#     P = [np.array([0.7, 0.3])]
#     Q = [np.array([0.3, 0.7])]
#     exact = brute_force_tv_vectorized(P, Q)
#     est, diag = randomized_tv_estimate(P, Q, epsilon=0.1, delta=0.05,
#                                         rng=rng, return_diagnostics=True)
#     print(f"  Exact: {exact:.6f}, Estimate: {est:.6f}, "
#           f"Rel error: {abs(est-exact)/exact:.4f}")
#     print(f"  Samples used: {diag['total_samples']}, "
#           f"Pr_C[X≠Y]: {diag['prob_disagree']:.4f}")

#     # Test 2: n=5, random
#     print("\nTest 2: n=5, q=2, random distributions")
#     mP = random_marginals(5, 2, rng)
#     mQ = random_marginals(5, 2, rng)
#     exact = brute_force_tv_vectorized(mP, mQ)
#     est = randomized_tv_estimate(mP, mQ, epsilon=0.1, delta=0.05, rng=rng)
#     rel_err = abs(est - exact) / exact if exact > 1e-10 else 0.0
#     print(f"  Exact: {exact:.6f}, Estimate: {est:.6f}, Rel error: {rel_err:.4f}")

#     # Test 3: P = Q (should give ~0)
#     print("\nTest 3: P = Q (expect ~0)")
#     mP = random_marginals(5, 2, rng)
#     est = randomized_tv_estimate(mP, mP, epsilon=0.1, delta=0.05, rng=rng)
#     print(f"  Estimate: {est:.8f}, expected ≈ 0")

#     # Test 4: Multiple trials to check ε-relative error guarantee
#     print("\nTest 4: 20 independent trials, n=8, ε=0.2, δ=0.1")
#     mP = random_marginals(8, 2, rng)
#     mQ = random_marginals(8, 2, rng)
#     exact = brute_force_tv_vectorized(mP, mQ)
#     epsilon = 0.2
#     trials = 20
#     successes = 0
#     for _ in range(trials):
#         est = randomized_tv_estimate(mP, mQ, epsilon=epsilon,
#                                       delta=0.1, rng=rng)
#         rel_err = abs(est - exact) / exact
#         if rel_err <= epsilon:
#             successes += 1
#     print(f"  Exact TV: {exact:.4f}")
#     print(f"  Success rate: {successes}/{trials} = {successes/trials:.2f} "
#           f"(expected ≥ {1-0.1:.2f})")

#     # Test 5: Small TV distance
#     print("\nTest 5: Nearly identical distributions (small TV)")
#     from brute_force import nearly_identical_marginals
#     mP, mQ = nearly_identical_marginals(10, 2, epsilon=0.005, rng=rng)
#     exact = brute_force_tv_vectorized(mP, mQ)
#     est = randomized_tv_estimate(mP, mQ, epsilon=0.2, delta=0.05, rng=rng)
#     rel_err = abs(est - exact) / exact if exact > 1e-10 else float('inf')
#     print(f"  Exact TV: {exact:.6f}, Estimate: {est:.6f}, "
#           f"Rel error: {rel_err:.4f}")
#     print("  (Note: small TV → 1/n lower bound on E_π[f] is tight → needs more samples)")

#     print("\n✓ Step 2 complete.")
