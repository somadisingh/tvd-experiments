"""
sparsification_bound.py
========================================
Empirical Validation of the Sparsification Error Bound

Since direct MTV computation requires solving an optimisation, we use two
computable proxies that are bounded by MTV:

  (1) W1(R, R~): Wasserstein-1 distance between ratio PMFs.
      This measures how much probability mass must be "moved" to go from R to R~.

  (2) CDF-L1(R, R~): L1 distance between CDFs, equals the W1 distance.
      This is easier to compute and equals W1 for 1D distributions.

Outputs
-------
- w1_distance.png       : W1 distance (shape change) vs eps_s per family
- tightness_ratio.png   : W1 / theoretical_bound (tightness) per family
- compression_vs_n.png  : compression ratio at forcing regime (large n)
- cumulative_error.png  : cumulative TV error through full algorithm 
- sparsification_results.txt
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brute_force import brute_force_tv_vectorized
from ratio_distribution import RatioDistribution
from sparsify_full import sparsify
from deterministic_algorithm import compute_lower_bound
from distribution_families import (get_family, ALL_FAMILIES,
                                    FAMILY_COLORS, FAMILY_MARKERS)


# ── Proxy metrics for MTV ─────────────────────────────────────────────────────

def wasserstein1(R1, R2):
    """
    Wasserstein-1 distance between two 1D ratio distributions.
    For 1D distributions this equals the L1 distance between CDFs:
      W1(R1, R2) = integral |F1(r) - F2(r)| dr
    Computed exactly from the discrete supports.
    """
    pts = sorted(set([r for r, _ in R1.support()] +
                     [r for r, _ in R2.support()]))
    if not pts:
        return 0.0

    d1 = dict(R1.support())
    d2 = dict(R2.support())

    # Build CDFs at each unique point
    cdf1 = cdf2 = 0.0
    w1 = 0.0
    prev_r = 0.0
    for r in pts:
        w1 += abs(cdf1 - cdf2) * (r - prev_r)
        cdf1 += d1.get(r, 0.0)
        cdf2 += d2.get(r, 0.0)
        prev_r = r
    return float(w1)


def build_product_ratio(mP, mQ):
    R = RatioDistribution.from_marginals(mP[0], mQ[0])
    for i in range(1, len(mP)):
        Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
        R = RatioDistribution.independent_product(R, Ri)
    return R


def measure_sparsification(mP, mQ, eps_s, delta_s):

    R = build_product_ratio(mP, mQ)
    R_sp, diag = sparsify(R, eps_s, delta_s)

    w1    = wasserstein1(R, R_sp)
    bound = diag["theoretical_bound"]
    tight = w1 / bound if bound > 1e-15 else 0.0
    comp  = R.support_size() / max(R_sp.support_size(), 1)

    return {
        "tv":            R.tv_distance(),
        "w1_distance":   w1,
        "bound":         bound,
        "tightness_w1":  tight,
        "compression":   comp,
        "support_before": diag["support_before"],
        "support_after":  diag["support_after"],
    }


def measure_cumulative_error(mP, mQ, epsilon):
    """
    Run the full algorithm and track cumulative |TV change| vs theoretical bound.
    """
    n = len(mP)
    exact = brute_force_tv_vectorized(mP, mQ)
    d_LB  = compute_lower_bound(mP, mQ)
    if d_LB < 1e-15:
        return None

    eps_s   = epsilon / (2 * n)
    delta_s = epsilon * d_LB / (2 * n)

    R_current = build_product_ratio(mP[:1], mQ[:1])
    R_true    = build_product_ratio(mP[:1], mQ[:1])

    steps, tv_errors, w1_errors, theory_bounds, compressions = [0], [0.], [0.], [0.], [1.]

    for k in range(1, n):
        R_sp, diag = sparsify(R_current, eps_s, delta_s)
        comp = R_current.support_size() / max(R_sp.support_size(), 1)

        R_next  = RatioDistribution.from_marginals(mP[k], mQ[k])
        R_current = RatioDistribution.independent_product(R_sp, R_next)

        Ri_true = RatioDistribution.from_marginals(mP[k], mQ[k])
        R_true  = RatioDistribution.independent_product(R_true, Ri_true)

        tv_err   = abs(R_current.tv_distance() - R_true.tv_distance())
        w1_err   = wasserstein1(R_current, R_true)
        theory_b = (k / (2 * n)) * epsilon * exact

        steps.append(k);         tv_errors.append(tv_err)
        w1_errors.append(w1_err); theory_bounds.append(theory_b)
        compressions.append(comp)

    return {"steps": steps, "tv_errors": tv_errors, "w1_errors": w1_errors,
            "theory_bounds": theory_bounds, "compressions": compressions,
            "exact_tv": exact}


def run_sparsification_bound(n_instances=30, seed=0, out_dir="."):
    rng = np.random.default_rng(seed)
    families = [f for f in ALL_FAMILIES if f != "A_identical"]
    lines = ["=" * 70,
             "Sparsification Shape-Change Measurement",
             "=" * 70, "",
             ""]

    # ── W1 tightness per family at forcing regime ─────────────────────────
    lines.append("    W1 tightness ratio W1(R,R~)/bound per family")
    lines.append("    support > interval count so merging is actually forced.")
    lines.append("    eps_s=0.3, delta_s=0.015  (37 intervals; support>37 for n>=10)")
    lines.append("")

    eps_s_force, delta_s_force = 0.3, 0.015

    tightness_by_family = {}
    # Use larger n so support >> interval count
    n_for_1a = {"B_adversarial": 14, "C_random": 12,
                 "D_skewed": 12, "E_nearly_identical": 10}

    for fname in families:
        n_use = n_for_1a.get(fname, 12)
        tlist = []
        for _ in range(n_instances):
            mP, mQ, _ = get_family(fname, n=n_use, q=2, rng=rng)
            r = measure_sparsification(mP, mQ, eps_s_force, delta_s_force)
            if r["bound"] > 1e-15 and r["compression"] > 1.001:
                # Only include trials where merging actually occurred
                tlist.append(r["tightness_w1"])
        tightness_by_family[fname] = tlist
        if tlist:
            lines.append(f"  {fname:25s} n={n_use:2d}  "
                         f"mean_tightness={np.mean(tlist):.4f}  "
                         f"max={np.max(tlist):.4f}  "
                         f"frac>0.5={np.mean([t>0.5 for t in tlist]):.2f}  "
                         f"n_merged={len(tlist)}/{n_instances}")
        else:
            lines.append(f"  {fname:25s} n={n_use:2d}  no merging occurred")

    fig, axes = plt.subplots(1, len(families),
                              figsize=(4 * len(families), 4))
    for ax, fname in zip(axes, families):
        vals = tightness_by_family.get(fname, [])
        if vals:
            ax.hist(vals, bins=20, range=(0, 1.05),
                    color=FAMILY_COLORS[fname], edgecolor='black', alpha=0.8)
            ax.axvline(x=np.mean(vals), color='black', linestyle='--',
                       linewidth=1.5, label=f'mean={np.mean(vals):.3f}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No merging\noccurred',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('W1 tightness\n(W1 / bound)')
        ax.set_ylabel('Count')
        ax.set_title(fname.replace("_", " ").title() +
                     f'\n(n={n_for_1a.get(fname,12)})')
        ax.set_xlim(0, 1.05)
    plt.suptitle('W1 shape-change tightness (forcing regime)\n'
                 '(1 = W1 equals theoretical bound; 0 = very loose)\n'
                 'TV is algebraically preserved — W1 measures decision-problem distortion',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tightness_ratio.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # ── W1 vs bound across eps_s values at forcing n ─────────────────────
    lines.append("\nW1 distance vs bound across eps_s (forcing n per family)")
    eps_s_vals = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    fig, axes = plt.subplots(1, len(families),
                              figsize=(4 * len(families), 4))
    for ax, fname in zip(axes, families):
        n_use = n_for_1a.get(fname, 12)
        w1_means, bound_means, comp_means = [], [], []
        for eps_s in eps_s_vals:
            ws, bs, cs = [], [], []
            for _ in range(12):
                mP, mQ, _ = get_family(fname, n=n_use, q=2, rng=rng)
                r = measure_sparsification(mP, mQ, eps_s, eps_s * 0.05)
                ws.append(r["w1_distance"])
                bs.append(r["bound"])
                cs.append(r["compression"])
            w1_means.append(np.mean(ws))
            bound_means.append(np.mean(bs))
            comp_means.append(np.mean(cs))

        ax.loglog(eps_s_vals, w1_means, 'b-o', markersize=5,
                  label='W1 (actual shape change)', linewidth=2)
        ax.loglog(eps_s_vals, bound_means, 'r--s', markersize=5,
                  label='MTV bound', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(eps_s_vals, comp_means, 'g:^', markersize=5,
                 label='Compression ratio', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Compression ratio', color='green', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='green')

        ax.set_xlabel('ε_s')
        ax.set_ylabel('W1 distance / bound')
        ax.set_title(fname.replace("_", " ").title() + f'\n(n={n_use})')
        lines_h, labels_h = ax.get_legend_handles_labels()
        l2, la2 = ax2.get_legend_handles_labels()
        ax.legend(lines_h + l2, labels_h + la2, fontsize=7)
        ax.grid(True, which='both', alpha=0.3)

        tightness_vals = [w/b for w, b in zip(w1_means, bound_means) if b > 0]
        lines.append(f"  {fname:25s} n={n_use:2d}: mean tightness = "
                     f"{np.mean(tightness_vals):.4f}  "
                     f"compression range = "
                     f"{min(comp_means):.1f}x – {max(comp_means):.1f}x")

    plt.suptitle('W1 shape change (blue solid) vs MTV bound (red dashed)\n'
                 'Green dotted = compression ratio (right axis)\n'
                 'TV change is always zero — W1 measures decision-problem distortion',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bound_tightness.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # ── Cumulative W1 and TV through full algorithm ───────────────────────
    lines.append("\nCumulative W1 and TV error through full algorithm (n=12)")
    families_c = ["B_adversarial", "C_random", "E_nearly_identical"]
    fig, axes = plt.subplots(2, len(families_c),
                              figsize=(5 * len(families_c), 7))

    for col, fname in enumerate(families_c):
        mP, mQ, _ = get_family(fname, n=12, q=2, rng=rng)
        cum = measure_cumulative_error(mP, mQ, epsilon=0.2)
        if cum is None:
            continue

        # Top: TV error (should be ~0) and theory bound
        ax_top = axes[0, col]
        ax_top.semilogy(cum["steps"],
                        [max(v, 1e-17) for v in cum["tv_errors"]],
                        'b-o', markersize=4, label='TV change (≡ 0)', linewidth=2)
        ax_top.semilogy(cum["steps"], [max(v,1e-17) for v in cum["theory_bounds"]],
                        'r--', linewidth=2, label='MTV bound')
        ax_top.set_xlabel('Step k')
        ax_top.set_ylabel('Error (log scale)')
        ax_top.set_title(fname.replace("_"," ").title() +
                         f'\nTV(P,Q)={cum["exact_tv"]:.3f}')
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        # Bottom: W1 distance and compression
        ax_bot = axes[1, col]
        ax_bot.plot(cum["steps"], cum["w1_errors"], 'g-D', markersize=5,
                    label='W1 shape change', linewidth=2)
        ax_bot.plot(cum["steps"], cum["theory_bounds"], 'r--', linewidth=2,
                    label='MTV bound')
        ax_bot.set_xlabel('Step k')
        ax_bot.set_ylabel('W1 distance / bound')
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)
        ax2 = ax_bot.twinx()
        ax2.plot(cum["steps"], cum["compressions"], 'm:s', markersize=4,
                 label='Compression', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Compression ratio', color='m', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='m')

        lines.append(f"  {fname:25s}: "
                     f"final TV_error={cum['tv_errors'][-1]:.2e}  "
                     f"final W1={cum['w1_errors'][-1]:.6f}  "
                     f"final bound={cum['theory_bounds'][-1]:.6f}  "
                     f"W1_tightness={cum['w1_errors'][-1]/max(cum['theory_bounds'][-1],1e-15):.4f}")

    plt.suptitle('Cumulative errors through algorithm (n=12, ε=0.2)\n'
                 'Top: TV change ≡ 0 (algebraic invariant) vs MTV bound\n'
                 'Bottom: W1 shape distortion vs MTV bound',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_error.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    with open(os.path.join(out_dir, "sparsification_results.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  Figures saved to {out_dir}")
    return tightness_by_family


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    run_sparsification_bound(n_instances=25, out_dir=out)
