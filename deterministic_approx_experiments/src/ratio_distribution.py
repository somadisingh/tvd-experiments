"""
Ratio Distribution Data Structure
==========================================
The core data structure for deterministic algorithm.

The likelihood ratio distribution (P‖Q) is the distribution of P(X)/Q(X)
when X ~ Q. We represent it as a sorted list of (ratio_value, probability)
pairs — a sparse table.

"""

import numpy as np
from collections import defaultdict


class RatioDistribution:
    """
    A discrete distribution over [0, ∞) representing a likelihood ratio.

    Stored as a sorted list of (r, p) pairs where r is the
    ratio value and p = Pr[ratio = r].

    Invariant: all p > 0, all r ≥ 0, sum(p) = 1.
    """

    def __init__(self, support):
        """
        Parameters- 
        support : list of (ratio, prob) pairs, or dict {ratio: prob}
                  Pairs with prob=0 are silently dropped.
        """
        if isinstance(support, dict):
            items = support.items()
        else:
            items = support

        # Aggregate duplicate ratio values
        table = defaultdict(float)
        for r, p in items:
            if p > 0:
                table[float(r)] += float(p)

        # Sort by ratio value
        self._table = sorted(table.items())  # list of (r, p), sorted by r

        # Validate
        if self._table:
            total = sum(p for _, p in self._table)
            assert abs(total - 1.0) < 1e-6, \
                f"Ratio Distribution probabilities sum to {total}, not 1"

    @classmethod
    def from_marginals(cls, p_marginal, q_marginal):
        """
        Compute (P_i ‖ Q_i) for a single coordinate.

        For each value c in [q]:
          - If Q_i(c) > 0: ratio r_c = P_i(c)/Q_i(c), probability = Q_i(c)
          - If Q_i(c) = 0: this point is not in the support of Q,
            so it doesn't contribute to (P‖Q) (which is defined under Q)

        """
        support = []
        for c in range(len(q_marginal)):
            q_c = float(q_marginal[c])
            p_c = float(p_marginal[c])
            if q_c > 1e-15:
                r = p_c / q_c
                support.append((r, q_c))
        return cls(support)

    @classmethod
    def degenerate(cls, ratio_val):
        """Point mass at ratio_val (used for testing)."""
        return cls([(ratio_val, 1.0)])

    # ── Core properties ──────────────────────────────────────────────────────

    def support(self):
        """Return list of (ratio, prob) pairs sorted by ratio."""
        return list(self._table)

    def support_size(self):
        return len(self._table)

    def ratios(self):
        return [r for r, _ in self._table]

    def probs(self):
        return [p for _, p in self._table]

    def expected_ratio(self):
        """E_R[R] — should be ≤ 1 for valid ratios."""
        return sum(r * p for r, p in self._table)

    def is_valid(self):
        return self.expected_ratio() <= 1.0 + 1e-9

    def tv_distance(self):
        """
        TV(R) = E_R[max(1 - R, 0)]

        """
        return sum(max(0.0, 1.0 - r) * p for r, p in self._table)

    def alternative(self):
        """
        Compute R† — the alternative ratio distribution.

        R†(r) = r · R(r) for r ∈ [0,∞)
        R†(∞) = 1 - E[R]   (the mass at infinity, representing P(Supp(Q)^c))

        """
        alt_support = [(r, r * p) for r, p in self._table if r * p > 1e-15]
        total = sum(p for _, p in alt_support)

        if abs(total) < 1e-15:
            # Degenerate case
            return RatioDistribution([(0.0, 1.0)])

        # Renormalize 
        return RatioDistribution([(r, p / total) for r, p in alt_support])

    # ── Operations ───────────────────────────────────────────────────────────

    @staticmethod
    def independent_product(R1, R2):
        """
        Compute R1 ·_indp R2 = distribution of R1*R2.
        Support size = |Supp(R1)| × |Supp(R2)|.

        """
        new_support = defaultdict(float)
        for r1, p1 in R1._table:
            for r2, p2 in R2._table:
                r = r1 * r2
                new_support[r] += p1 * p2
        return RatioDistribution(new_support)

    def __repr__(self):
        n = min(5, len(self._table))
        entries = ", ".join(f"({r:.4f}, {p:.4f})" for r, p in self._table[:n])
        if len(self._table) > 5:
            entries += f", ... [{len(self._table)} total]"
        return f"RatioDistribution([{entries}])"

    def summary(self):
        print(f"  Support size: {self.support_size()}")
        print(f"  E[R] = {self.expected_ratio():.6f}  (≤1 for valid ratio)")
        print(f"  TV(R) = {self.tv_distance():.6f}")
        print(f"  Valid: {self.is_valid()}")
        if self.support_size() <= 8:
            for r, p in self._table:
                print(f"    r={r:.4f}, p={p:.4f}")


# # ── Self-test ────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys
#     sys.path.insert(0, "/home/claude/tv_distance/src")
#     from brute_force import brute_force_tv_vectorized

#     print("=== Step 3: Ratio Distribution Data Structure ===\n")

#     # Test 1: Single coordinate
#     print("Test 1: Single coordinate, P=Bern(0.7), Q=Bern(0.5)")
#     P_m = np.array([0.7, 0.3])
#     Q_m = np.array([0.5, 0.5])
#     R = RatioDistribution.from_marginals(P_m, Q_m)
#     print("  Ratio distribution:")
#     R.summary()
#     # TV should match brute force
#     exact = brute_force_tv_vectorized([P_m], [Q_m])
#     print(f"  Brute force TV = {exact:.6f}")
#     print(f"  Match: {abs(R.tv_distance() - exact) < 1e-9}")

#     # Test 2: Independent product
#     print("\nTest 2: Independent product of two single-coord ratios")
#     P1 = np.array([0.7, 0.3])
#     Q1 = np.array([0.5, 0.5])
#     P2 = np.array([0.6, 0.4])
#     Q2 = np.array([0.4, 0.6])
#     R1 = RatioDistribution.from_marginals(P1, Q1)
#     R2 = RatioDistribution.from_marginals(P2, Q2)
#     R12 = RatioDistribution.independent_product(R1, R2)
#     exact = brute_force_tv_vectorized([P1, P2], [Q1, Q2])
#     print(f"  TV from product ratio: {R12.tv_distance():.6f}")
#     print(f"  Brute force TV:        {exact:.6f}")
#     print(f"  Match: {abs(R12.tv_distance() - exact) < 1e-9}")
#     print(f"  Support size after product: {R12.support_size()} (= 2×2 = 4)")

#     # Test 3: n-step product matches brute force
#     print("\nTest 3: n=6 product matches brute force")
#     rng = np.random.default_rng(7)
#     n = 6
#     mP = [rng.dirichlet([1, 1]) for _ in range(n)]
#     mQ = [rng.dirichlet([1, 1]) for _ in range(n)]

#     # Build ratio step by step
#     R = RatioDistribution.from_marginals(mP[0], mQ[0])
#     for i in range(1, n):
#         Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
#         R = RatioDistribution.independent_product(R, Ri)

#     exact = brute_force_tv_vectorized(mP, mQ)
#     print(f"  TV from ratio:    {R.tv_distance():.8f}")
#     print(f"  Brute force TV:   {exact:.8f}")
#     print(f"  Support size: {R.support_size()} (should be 2^{n}={2**n})")
#     print(f"  Match: {abs(R.tv_distance() - exact) < 1e-7}")

#     # Test 4: Validity check
#     print("\nTest 4: E[R] ≤ 1 for all cases")
#     for _ in range(10):
#         mP_i = rng.dirichlet([1, 1, 1])
#         mQ_i = rng.dirichlet([1, 1, 1])
#         Ri = RatioDistribution.from_marginals(mP_i, mQ_i)
#         assert Ri.is_valid(), f"Invalid ratio! E[R]={Ri.expected_ratio()}"
#     print("  All 10 random ratios are valid ✓")

#     # Test 5: Alternative ratio
#     print("\nTest 5: Alternative ratio R†")
#     P_m = np.array([0.8, 0.2])
#     Q_m = np.array([0.5, 0.5])
#     R = RatioDistribution.from_marginals(P_m, Q_m)
#     Rd = R.alternative()
#     print(f"  R  support: {R.support()}")
#     print(f"  R† support: {Rd.support()}")
#     # TV(R) should equal TV(R†, R) in the classical sense
#     print(f"  TV(R) = {R.tv_distance():.6f}")
#     print(f"  E[R†] = {Rd.expected_ratio():.6f}")  # should be ≥ 1

#     print("\n✓ Step 3 complete.")
