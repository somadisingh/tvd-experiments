"""
support_growth.py
======================
Support Size Growth Through the Algorithm
How does support size evolve at each step? Does sparsification keep it
under control? Which distribution family hits the worst-case bound?

Outputs
-------
- support_trajectory.png : support size per step, all families
- peak_support.png       : max support vs n, per family
- support_vs_eps.png     : final support vs eps_s, per family
- support_growth_results.txt
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brute_force import brute_force_tv_vectorized
from deterministic_algorithm import deterministic_tv
from src.distribution_families import (get_family, ALL_FAMILIES,
                                    FAMILY_COLORS, FAMILY_MARKERS,
                                    ratio_spread, mass_near_one, ratio_entropy)
from ratio_distribution import RatioDistribution


def run_support_growth(seed=2, out_dir="."):
    rng = np.random.default_rng(seed)
    lines = ["=" * 70, "Support Size Growth", "=" * 70, ""]

    # ── Full trajectory for each family ──────────────────────────────────
    lines.append("Support trajectory (n=12, eps=0.15)")
    families_a = ["B_adversarial", "C_random", "D_skewed", "E_nearly_identical"]
    n, eps = 12, 0.15

    traj_data = {}
    for fname in families_a:
        mP, mQ, _ = get_family(fname, n=n, q=2, rng=rng)
        exact = brute_force_tv_vectorized(mP, mQ)
        if exact < 1e-10:
            continue
        _, diag = deterministic_tv(mP, mQ, epsilon=eps, return_diagnostics=True)
        steps   = [s['step'] for s in diag['steps']]
        before  = [s['support_before_sparsify'] for s in diag['steps']]
        after_s = [s['support_after_sparsify']  for s in diag['steps']]
        after_p = [s['support_after_product']   for s in diag['steps']]
        traj_data[fname] = dict(steps=steps, before=before,
                                 after_s=after_s, after_p=after_p,
                                 exact=exact)
        lines.append(f"  {fname:25s}  TV={exact:.4f}  "
                     f"peak_before={max(before)}  "
                     f"peak_after_s={max(after_s)}  "
                     f"final={after_p[-1] if after_p else 'N/A'}")

    fig, axes = plt.subplots(1, len(traj_data), figsize=(5 * len(traj_data), 4))
    if len(traj_data) == 1:
        axes = [axes]
    for ax, (fname, d) in zip(axes, traj_data.items()):
        ax.plot(d['steps'], d['before'],  color=FAMILY_COLORS[fname],
                marker='o', markersize=4, linestyle='-',  label='Before sparsify', linewidth=2)
        ax.plot(d['steps'], d['after_s'], color=FAMILY_COLORS[fname],
                marker='s', markersize=4, linestyle='--', label='After sparsify',  linewidth=1.5)
        ax.plot(d['steps'], d['after_p'], color=FAMILY_COLORS[fname],
                marker='^', markersize=4, linestyle=':',  label='After product',   linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Step k'); ax.set_ylabel('Support size')
        ax.set_title(f"{fname.replace('_',' ').title()}\n(TV={d['exact']:.3f})")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle(f'Support size trajectory (n={n}, ε={eps})', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "support_trajectory.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # ── Peak support vs n ─────────────────────────────────────────────────
    lines.append("\nPeak support vs n")
    n_vals = [6, 8, 10, 12, 14]
    fig, ax = plt.subplots(figsize=(8, 4))

    for fname in families_a:
        peaks = []
        for n_val in n_vals:
            ps = []
            for _ in range(3):
                mP, mQ, _ = get_family(fname, n=n_val, q=2, rng=rng)
                exact = brute_force_tv_vectorized(mP, mQ)
                if exact < 1e-10:
                    ps.append(1)
                    continue
                _, diag = deterministic_tv(mP, mQ, epsilon=0.15,
                                            return_diagnostics=True)
                befores = [s['support_before_sparsify'] for s in diag['steps']]
                ps.append(max(befores) if befores else 1)
            peaks.append(np.mean(ps))
        ax.semilogy(n_vals, peaks, color=FAMILY_COLORS[fname],
                    marker=FAMILY_MARKERS[fname], label=fname.replace("_"," "),
                    linewidth=2, markersize=6)
        lines.append(f"  {fname:25s}  peaks(n={n_vals}) = "
                     f"{[int(p) for p in peaks]}")

    # Reference: 2^n worst case
    ax.semilogy(n_vals, [2**n for n in n_vals], 'k--', alpha=0.4,
                label='2^n (worst case)')
    ax.set_xlabel('Dimension n'); ax.set_ylabel('Peak support size (log scale)')
    ax.set_title('Peak support size vs n (ε=0.15)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "peak_support.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # ── Distributional properties as predictors ───────────────────────────
    lines.append("\nRatio spread vs compression ratio (scatter)")
    spreads, compressions_all, family_labels = [], [], []

    for fname in families_a:
        for _ in range(25):
            mP, mQ, _ = get_family(fname, n=8, q=2, rng=rng)
            exact = brute_force_tv_vectorized(mP, mQ)
            if exact < 1e-10:
                continue
            # Compute full product ratio
            R = RatioDistribution.from_marginals(mP[0], mQ[0])
            for i in range(1, 8):
                Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
                R = RatioDistribution.independent_product(R, Ri)
            sp = ratio_spread(R)
            mn = mass_near_one(R)
            # Sparsify and compute compression
            from sparsify_full import sparsify as _sparsify
            R_sp, _ = _sparsify(R, 0.1, 0.005)
            comp = R.support_size() / max(R_sp.support_size(), 1)
            spreads.append(sp)
            compressions_all.append(comp)
            family_labels.append(fname)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for fname in families_a:
        idxs = [i for i, f in enumerate(family_labels) if f == fname]
        xs = [spreads[i]       for i in idxs]
        ys = [compressions_all[i] for i in idxs]
        axes[0].scatter(xs, ys, color=FAMILY_COLORS[fname], alpha=0.6,
                         marker=FAMILY_MARKERS[fname],
                         label=fname.replace("_"," "), s=40)
    axes[0].set_xlabel('Ratio spread log10(max_r / min_r)')
    axes[0].set_ylabel('Compression ratio')
    axes[0].set_title('Ratio spread vs compression\n'
                       'Wide spread → more values per interval → better compression')
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Mass near 1 vs compression
    mnears = []
    for i, fname in enumerate(family_labels):
        mP, mQ, _ = get_family(fname, n=8, q=2, rng=rng)
        R = RatioDistribution.from_marginals(mP[0], mQ[0])
        for j in range(1, 8):
            Ri = RatioDistribution.from_marginals(mP[j], mQ[j])
            R = RatioDistribution.independent_product(R, Ri)
        mnears.append(mass_near_one(R))

    for fname in families_a:
        idxs = [i for i, f in enumerate(family_labels) if f == fname]
        xs = [mnears[i]          for i in idxs]
        ys = [compressions_all[i] for i in idxs]
        axes[1].scatter(xs, ys, color=FAMILY_COLORS[fname], alpha=0.6,
                         marker=FAMILY_MARKERS[fname],
                         label=fname.replace("_"," "), s=40)
    axes[1].set_xlabel('Mass near r=1 (within ±0.1)')
    axes[1].set_ylabel('Compression ratio')
    axes[1].set_title('Mass-near-1 vs compression\n'
                       'High mass near 1 → fine intervals needed → less compression')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "support_vs_eps.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    with open(os.path.join(out_dir, "support_growth_results.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  Figures saved to {out_dir}")


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    run_exp3(out_dir=out)
