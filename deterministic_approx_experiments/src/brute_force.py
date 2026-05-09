"""
Brute Force TV Distance Calculator
==========================================
Ground truth for validating all other algorithms. Only feasible for small n.
"""

import numpy as np
from itertools import product as cartesian_product


def make_product_distribution(marginals):
    """
    Given marginals[i] = probability vector over [q] for coordinate i,
    return a function P(x) that computes the probability of outcome x.

    marginals: list of length n, each entry a 1D numpy array summing to 1.
    """
    for i, m in enumerate(marginals):
        assert abs(sum(m) - 1.0) < 1e-9, f"Marginal {i} does not sum to 1: {sum(m)}"
    
    def P(x):
        """x is a tuple of length n, each entry in {0, ..., q-1}."""
        prob = 1.0
        for i, xi in enumerate(x):
            prob *= marginals[i][xi]
        return prob
    
    return P


def brute_force_tv(marginals_P, marginals_Q):
    """
    Compute TV(P, Q) exactly by enumerating all outcomes.

    """
    n = len(marginals_P)
    assert len(marginals_Q) == n, "P and Q must have same dimension"
    
    q = len(marginals_P[0])
    assert all(len(m) == q for m in marginals_P + marginals_Q), \
        "All marginals must have the same domain size q"

    P = make_product_distribution(marginals_P)
    Q = make_product_distribution(marginals_Q)

    tv = 0.0
    # Enumerate all q^n outcomes
    for x in cartesian_product(range(q), repeat=n):
        px = P(x)
        qx = Q(x)
        tv += max(0.0, px - qx)

    return tv


def brute_force_tv_vectorized(marginals_P, marginals_Q):
    """
    Faster vectorized version using numpy.
    Same result as brute_force_tv but more efficient for moderate n.

    """
    n = len(marginals_P)
    q = len(marginals_P[0])

    # Build P and Q as n-dimensional tensors of shape (q, q, ..., q)
    # Start with a scalar 1 and take outer products
    P_table = np.array([1.0])
    Q_table = np.array([1.0])

    for i in range(n):
        P_table = np.outer(P_table, marginals_P[i]).flatten()
        Q_table = np.outer(Q_table, marginals_Q[i]).flatten()

    tv = np.sum(np.maximum(0.0, P_table - Q_table))
    return float(tv)


# ── Helper functions for generating test distributions ──────────────────────

def random_marginals(n, q, rng=None):
    """
    Generate n random marginals over [q], each drawn from a Dirichlet(1,...,1).
    """
    if rng is None:
        rng = np.random.default_rng()
    return [rng.dirichlet(np.ones(q)) for _ in range(n)]


def nearly_identical_marginals(n, q, epsilon=0.01, rng=None):
    """
    Generate P marginals, then Q = P + small perturbation.
    Useful for testing small TV distance regime.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    marginals_P = random_marginals(n, q, rng)
    marginals_Q = []
    for m in marginals_P:
        # Add small noise and renormalize
        noise = rng.uniform(-epsilon, epsilon, size=q)
        m_new = np.clip(m + noise, 1e-10, None)
        m_new = m_new / m_new.sum()
        marginals_Q.append(m_new)
    
    return marginals_P, marginals_Q


def adversarial_marginals(n, q):
    """
    All P_i are identical, all Q_i are identical.
    This is a worst case for sparsification since the ratio distribution
    concentrates on q^n points.

    """
    assert q == 2, "Adversarial marginals currently implemented for q=2 only"
    p_marginal = np.array([0.6, 0.4])
    q_marginal = np.array([0.4, 0.6])
    marginals_P = [p_marginal.copy() for _ in range(n)]
    marginals_Q = [q_marginal.copy() for _ in range(n)]
    return marginals_P, marginals_Q


def uniform_vs_p_marginals(n, marginals_P):
    """Q is uniform, used for testing determinstic algorithm for the restricted case."""
    q = len(marginals_P[0])
    marginals_Q = [np.ones(q) / q for _ in range(n)]
    return marginals_P, marginals_Q


# # ── Self-test ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("=== Step 1: Brute Force TV Distance ===\n")

#     # Test 1: n=1 (should equal |p1 - q1| by hand)
#     print("Test 1: n=1, q=2")
#     P = [np.array([0.7, 0.3])]
#     Q = [np.array([0.4, 0.6])]
#     tv = brute_force_tv_vectorized(P, Q)
#     expected = abs(0.7 - 0.4)  # For n=1 Bernoulli, TV = |p - q|
#     print(f"  TV = {tv:.6f}, expected = {expected:.6f}, match = {abs(tv - expected) < 1e-9}")

#     # Test 2: n=2, q=2 — verify against manual calculation
#     print("\nTest 2: n=2, q=2 (P = Bern(0.7)^2, Q = Bern(0.3)^2)")
#     P = [np.array([0.7, 0.3])] * 2
#     Q = [np.array([0.3, 0.7])] * 2
#     tv = brute_force_tv_vectorized(P, Q)
#     # Manual: outcomes (0,0),(0,1),(1,0),(1,1)
#     # P: 0.49, 0.21, 0.21, 0.09  Q: 0.09, 0.21, 0.21, 0.49
#     # TV = max(0, 0.49-0.09) + max(0, 0.21-0.21)*2 + max(0, 0.09-0.49)
#     #    = 0.40 + 0 + 0 = 0.40
#     print(f"  TV = {tv:.6f}, expected = 0.400000")

#     # Test 3: Identical distributions should give TV = 0
#     print("\nTest 3: P = Q (TV should be 0)")
#     P = [np.array([0.5, 0.3, 0.2])] * 3
#     Q = [np.array([0.5, 0.3, 0.2])] * 3
#     tv = brute_force_tv_vectorized(P, Q)
#     print(f"  TV = {tv:.10f}, expected = 0.0")

#     # Test 4: Completely disjoint supports (TV should be 1)
#     print("\nTest 4: Completely disjoint supports (TV should be 1)")
#     P = [np.array([1.0, 0.0])] * 5
#     Q = [np.array([0.0, 1.0])] * 5
#     tv = brute_force_tv_vectorized(P, Q)
#     print(f"  TV = {tv:.6f}, expected = 1.0")

#     # Test 5: Random distributions, compare slow vs fast
#     print("\nTest 5: Random n=8, q=2 — slow vs fast comparison")
#     rng = np.random.default_rng(42)
#     mP = random_marginals(8, 2, rng)
#     mQ = random_marginals(8, 2, rng)
#     tv_slow = brute_force_tv(mP, mQ)
#     tv_fast = brute_force_tv_vectorized(mP, mQ)
#     print(f"  Slow: {tv_slow:.8f}, Fast: {tv_fast:.8f}, Match: {abs(tv_slow - tv_fast) < 1e-10}")

#     # Test 6: Timing
#     import time
#     print("\nTest 6: Timing for various n (q=2)")
#     for n in [5, 10, 15, 18, 20]:
#         rng = np.random.default_rng(0)
#         mP = random_marginals(n, 2, rng)
#         mQ = random_marginals(n, 2, rng)
#         t0 = time.time()
#         tv = brute_force_tv_vectorized(mP, mQ)
#         elapsed = time.time() - t0
#         print(f"  n={n:2d}: TV={tv:.4f}, time={elapsed:.4f}s, outcomes={2**n}")

#     print("\n✓ All Step 1 tests passed.")
