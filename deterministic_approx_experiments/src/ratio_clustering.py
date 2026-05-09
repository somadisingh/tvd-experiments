"""
ratio_clustering.py
==========================

From support size analysis:
  Family E values near r=1 cluster extremely tightly (0.9950, 0.9970, 0.9990...)
  Each marginal ratio crowd into the same intervals and cannot be merged
  because the interval widths shrink to zero proportionally as r->1.

What we measure:
  1. Distribution of ratio values within [0.9, 1.1] for B vs E at various n
  2. Interval occupancy: how many support points fall into each fine interval
  3. Merge potential: for each interval, how many points it contains
  4. The relationship between clustering width and compression failure

Outputs
-------
  ratio_clustering.png  : histogram of ratio values near 1 for B vs E
  interval_occupancy.png: how support points distribute across intervals
  merge_potential.png   : merge counts per interval, B vs E
  ratio_clustering_results.txt
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ratio_distribution import RatioDistribution
from sparsify_full import build_geometric_partition
from distribution_families import get_family, FAMILY_COLORS


def build_product_ratio(mP, mQ):
    R = RatioDistribution.from_marginals(mP[0], mQ[0])
    for i in range(1, len(mP)):
        Ri = RatioDistribution.from_marginals(mP[i], mQ[i])
        R = RatioDistribution.independent_product(R, Ri)
    return R


def ratio_values_near_one(R, window=0.1):
    """Return all ratio values r with |r - 1| <= window."""
    return [(r, p) for r, p in R.support() if abs(r - 1.0) <= window]


def interval_occupancy(R, eps_s, delta_s):
    """
    For each geometric interval in the partition, count how many support
    points fall inside it and what their total probability is.
    Returns list of (interval_lo, interval_hi, count, total_prob).
    """
    intervals_below, intervals_above, _ = build_geometric_partition(eps_s, delta_s)
    all_ivs = [(lo, hi, 'below') for lo, hi, _ in intervals_below] + \
              [(lo, hi, 'above') for lo, hi, _ in intervals_above]

    occupancy = []
    for lo, hi, side in all_ivs:
        if side == 'below':
            pts = [(r, p) for r, p in R.support() if lo <= r < hi]
        else:
            pts = [(r, p) for r, p in R.support()
                   if lo < r <= (hi if hi < np.inf else 1e18)]
        if pts:
            occupancy.append((lo, hi, len(pts), sum(p for _, p in pts)))
    return occupancy


def run_ratio_clustering(seed=10, out_dir="."):
    rng = np.random.default_rng(seed)
    lines = ["=" * 70,
             "Ratio Clustering Near r=1",
             "=" * 70, ""]

    eps_s, delta_s = 0.1, 0.005

    # ── Histogram of ratio values near 1, B vs E at n=8,12 ──────────────
    lines.append("Distribution of ratio values within [0.9, 1.1]")
    lines.append(f"    eps_s={eps_s}, delta_s={delta_s}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for col, n in enumerate([8, 12]):
        for row, fname in enumerate(["B_adversarial", "E_nearly_identical"]):
            ax = axes[row, col]
            near_vals_all = []
            for trial in range(5):
                mP, mQ, _ = get_family(fname, n=n, q=2, rng=rng)
                R = build_product_ratio(mP, mQ)
                near = ratio_values_near_one(R, window=0.1)
                near_vals_all.extend([r for r, _ in near])

            if near_vals_all:
                color = FAMILY_COLORS[fname]
                ax.hist(near_vals_all, bins=60, range=(0.9, 1.1),
                        color=color, edgecolor='black', alpha=0.8, linewidth=0.3)
                ax.set_xlabel('Ratio value r')
                ax.set_ylabel('Count (5 trials)')
                ax.set_title(f'{fname.replace("_"," ")} | n={n}\n'
                             f'Values in [0.9, 1.1]: {len(near_vals_all)}')
                ax.axvline(x=1.0, color='black', linestyle='--',
                           alpha=0.5, linewidth=1)

                # Annotate geometric intervals in this window
                intervals_below, _, _ = build_geometric_partition(eps_s, delta_s)
                near_ivs = [(lo, hi) for lo, hi, _ in intervals_below
                            if lo >= 0.88 and hi <= 1.02]
                for lo, hi in near_ivs[:8]:
                    ax.axvline(x=lo, color='gray', linestyle=':', alpha=0.4,
                               linewidth=0.8)
                ax.grid(True, alpha=0.2)
                lines.append(f"  n={n:2d}, {fname:25s}: "
                             f"{len(near_vals_all):5d} values near 1 "
                             f"(across 5 trials)")
            else:
                ax.set_title(f'{fname} | n={n}\nNo values near 1')

    plt.suptitle('Ratio values within [0.9, 1.1]\n'
                 'Grey dotted lines = geometric interval boundaries\n',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratio_clustering.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # ── Interval occupancy — how many points per interval ─────────────────
    lines.append("\nInterval occupancy (points per geometric interval)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = 10

    for ax, fname in zip(axes, ["B_adversarial", "E_nearly_identical"]):
        all_counts = []
        for trial in range(8):
            mP, mQ, _ = get_family(fname, n=n, q=2, rng=rng)
            R = build_product_ratio(mP, mQ)
            occ = interval_occupancy(R, eps_s, delta_s)
            all_counts.extend([c for _, _, c, _ in occ])

        if all_counts:
            max_count = max(all_counts)
            bins = list(range(1, min(max_count + 2, 25)))
            ax.hist(all_counts, bins=bins, align='left',
                    color=FAMILY_COLORS[fname], edgecolor='black',
                    alpha=0.8, linewidth=0.3)
            ax.set_xlabel('Points per interval')
            ax.set_ylabel('Number of intervals (8 trials)')
            ax.set_title(f'{fname.replace("_"," ")} | n={n}\n'
                         f'Mean pts/interval = {np.mean(all_counts):.2f}  '
                         f'Max = {max_count}')
            ax.grid(True, alpha=0.3, axis='y')

            singles = sum(1 for c in all_counts if c == 1)
            multi   = sum(1 for c in all_counts if c > 1)
            pct_single = 100 * singles / len(all_counts) if all_counts else 0
            lines.append(f"  n={n}, {fname:25s}: "
                         f"{pct_single:.1f}% of intervals have exactly 1 point "
                         f"(cannot merge), max={max_count}")

    plt.suptitle('Points per interval after partitioning\n',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "interval_occupancy.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    with open(os.path.join(out_dir, "ratio_clustering_results.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  Figures saved to {out_dir}")


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    run_ratio_clustering(out_dir=out)
