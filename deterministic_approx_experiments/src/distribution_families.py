"""
distribution_families.py
========================
Defines five distribution families used across all experiments.
Each family isolates a different structural property of the ratio
distribution R = (P||Q).

Families
--------
A - Identical:     P = Q exactly. TV = 0. Boundary/sanity case.
B - Adversarial:   All P_i identical, all Q_i identical.
C - Random:        Marginals drawn i.i.d. from Dirichlet(1,...,1).
D - Skewed:        P_i concentrated, Q_i uniform.
E - Nearly-Identical: Q_i = P_i + small noise.

"""

import numpy as np

# ── Shared helpers ────────────────────────────────────────────────────────────

def _normalize(v):
    v = np.clip(v, 1e-12, None)
    return v / v.sum()


def marginal_tv_single(p, q):
    return 0.5 * float(np.sum(np.abs(np.array(p) - np.array(q))))


def family_metadata(name, description, expected_ratio_spread,
                    expected_tv_regime, worst_case_for):
    return {
        "name": name,
        "description": description,
        "expected_ratio_spread": expected_ratio_spread,
        "expected_tv_regime": expected_tv_regime,
        "worst_case_for": worst_case_for,
    }


# ── Family A: Identical distributions ────────────────────────────────────────

def family_A_identical(n, q=2, rng=None):
    """
    P = Q: all marginals are the same.
    TV(P,Q) = 0 exactly.
    Ratio distribution is a point mass at r=1.
    """
    if rng is None:
        rng = np.random.default_rng()
    marginals = [_normalize(rng.dirichlet(np.ones(q))) for _ in range(n)]
    meta = family_metadata(
        name="Identical (A)",
        description="P = Q, TV = 0",
        expected_ratio_spread="None (all ratios = 1)",
        expected_tv_regime="Zero",
        worst_case_for="Lower bound quality (d_LB = 0)",
    )
    return list(marginals), list(marginals), meta


# ── Family B: Adversarial (identical marginals) ───────────────────────────────

def family_B_adversarial(n, q=2, p_val=0.7, rng=None):
    """
    All P_i are the same fixed distribution; all Q_i are the same fixed
    distribution. The ratio distribution after k steps has exactly k+1
    support points forming a geometric sequence r_0^k * (1/r_0)^(n-k).

    For q=2: P_i = [p_val, 1-p_val], Q_i = [1-p_val, p_val].
    """
    if q == 2:
        p_m = np.array([p_val, 1.0 - p_val])
        q_m = np.array([1.0 - p_val, p_val])
    else:
        if rng is None:
            rng = np.random.default_rng()
        p_m = _normalize(rng.dirichlet(np.ones(q) * 3.0))
        q_m = _normalize(rng.dirichlet(np.ones(q) * 0.5))

    marginals_P = [p_m.copy() for _ in range(n)]
    marginals_Q = [q_m.copy() for _ in range(n)]
    meta = family_metadata(
        name="Adversarial (B)",
        description="All P_i identical, all Q_i identical",
        expected_ratio_spread="Geometric sequence, width grows with n",
        expected_tv_regime="Moderate to large",
        worst_case_for="Support size (geometric ratio structure)",
    )
    meta["base_ratio"] = float(p_val / (1.0 - p_val))
    return marginals_P, marginals_Q, meta


# ── Family C: Random (diverse marginals) ─────────────────────────────────────

def family_C_random(n, q=2, rng=None):
    """
    All marginals drawn independently from Dirichlet(1,...,1) (flat prior).
    Ratio values are all distinct.

    This tests average-case performance.
    """
    if rng is None:
        rng = np.random.default_rng()
    marginals_P = [_normalize(rng.dirichlet(np.ones(q))) for _ in range(n)]
    marginals_Q = [_normalize(rng.dirichlet(np.ones(q))) for _ in range(n)]
    meta = family_metadata(
        name="Random (C)",
        description="Marginals drawn i.i.d. from Dirichlet(1,...,1)",
        expected_ratio_spread="Wide, diverse",
        expected_tv_regime="Large (typically 0.7-1.0 for large n)",
        worst_case_for="Nothing in particular; average-case baseline",
    )
    return marginals_P, marginals_Q, meta


# ── Family D: Skewed (heavy ratio tails) ─────────────────────────────────────

def family_D_skewed(n, q=2, concentration=10.0, rng=None):
    """
    P_i is concentrated on one value (near-deterministic);
    Q_i is close to uniform.

    This produces ratio distributions with heavy tails: some ratio values are
    very large (P(c)/Q(c) >> 1 where P concentrates) and some are near zero.

    concentration: higher = more peaked P_i (more extreme ratios)
    """
    if rng is None:
        rng = np.random.default_rng()

    marginals_P = []
    marginals_Q = []
    for _ in range(n):
        # P_i: concentrate on a random mode
        mode = rng.integers(0, q)
        alpha_P = np.ones(q) * 0.1
        alpha_P[mode] = concentration
        p_m = _normalize(rng.dirichlet(alpha_P))
        # Q_i: nearly uniform
        alpha_Q = np.ones(q) * 2.0
        q_m = _normalize(rng.dirichlet(alpha_Q))
        marginals_P.append(p_m)
        marginals_Q.append(q_m)

    meta = family_metadata(
        name="Skewed (D)",
        description="P_i concentrated, Q_i near-uniform (heavy ratio tails)",
        expected_ratio_spread="Very wide, extreme values above 1",
        expected_tv_regime="Large",
        worst_case_for="Sparsification above r=1 (J_t intervals)",
    )
    meta["concentration"] = concentration
    return marginals_P, marginals_Q, meta


# ── Family E: Nearly-Identical (ratios cluster near 1) ───────────────────────

def family_E_nearly_identical(n, q=2, perturbation=0.05, rng=None):
    """
    Q_i = P_i + small random perturbation (renormalized).
    All ratio values P_i(c)/Q_i(c) are close to 1.

    perturbation: controls how far Q_i deviates from P_i
    """
    if rng is None:
        rng = np.random.default_rng()

    marginals_P = [_normalize(rng.dirichlet(np.ones(q))) for _ in range(n)]
    marginals_Q = []
    for p_m in marginals_P:
        noise = rng.uniform(-perturbation, perturbation, size=q)
        q_m = _normalize(p_m + noise)
        marginals_Q.append(q_m)

    meta = family_metadata(
        name="Nearly Identical (E)",
        description=f"Q_i = P_i + noise(pert={perturbation}), small TV",
        expected_ratio_spread="Narrow, all ratios near 1",
        expected_tv_regime="Small (scales with perturbation)",
        worst_case_for="log(1/TV) runtime factor; fine intervals near r=1",
    )
    meta["perturbation"] = perturbation
    return marginals_P, marginals_Q, meta


# ── Registry: all families in one place ──────────────────────────────────────

ALL_FAMILIES = {
    "A_identical":       family_A_identical,
    "B_adversarial":     family_B_adversarial,
    "C_random":          family_C_random,
    "D_skewed":          family_D_skewed,
    "E_nearly_identical": family_E_nearly_identical,
}

FAMILY_COLORS = {
    "A_identical":        "#888888",  # grey
    "B_adversarial":      "#d62728",  # red
    "C_random":           "#1f77b4",  # blue
    "D_skewed":           "#ff7f0e",  # orange
    "E_nearly_identical": "#2ca02c",  # green
}

FAMILY_MARKERS = {
    "A_identical":        "x",
    "B_adversarial":      "s",
    "C_random":           "o",
    "D_skewed":           "^",
    "E_nearly_identical": "D",
}


def get_family(name, n, q=2, rng=None, **kwargs):
    """Convenience wrapper. Returns (mP, mQ, meta)."""
    fn = ALL_FAMILIES[name]
    return fn(n, q=q, rng=rng, **kwargs)


def ratio_spread(ratio_dist):
    """
    log10(max_r / min_r) for a ratio distribution.
    Measures how spread out the support is on the log scale.
    """
    ratios = [r for r, _ in ratio_dist.support() if r > 1e-15]
    if len(ratios) < 2:
        return 0.0
    return float(np.log10(max(ratios) / min(ratios)))


def mass_near_one(ratio_dist, window=0.1):
    """
    Fraction of probability mass with r in [1-window, 1+window].
    High value → ratios cluster near 1 → small TV.
    """
    return sum(p for r, p in ratio_dist.support()
               if abs(r - 1.0) <= window)


def ratio_entropy(ratio_dist):
    """Shannon entropy of the ratio distribution."""
    probs = [p for _, p in ratio_dist.support() if p > 1e-15]
    probs = np.array(probs)
    return float(-np.sum(probs * np.log(probs + 1e-300)))


# ── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from brute_force import brute_force_tv_vectorized
    from ratio_distribution import RatioDistribution

    rng = np.random.default_rng(42)
    print("=== Distribution Families Quick Test ===\n")

    for fname in ALL_FAMILIES:
        try:
            mP, mQ, meta = get_family(fname, n=6, q=2, rng=rng)
            tv = brute_force_tv_vectorized(mP, mQ)
            # Build product ratio
            R = RatioDistribution.from_marginals(mP[0], mQ[0])
            for i in range(1, 6):
                Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
                R = RatioDistribution.independent_product(R, Ri)
            spread = ratio_spread(R)
            mnear = mass_near_one(R)
            print(f"  {meta['name']:25s}  TV={tv:.4f}  "
                  f"supp={R.support_size():4d}  "
                  f"spread={spread:.2f}  mass@1={mnear:.3f}")
        except Exception as e:
            print(f"  {fname}: ERROR - {e}")

    print("\n✓ distribution_families.py OK")
