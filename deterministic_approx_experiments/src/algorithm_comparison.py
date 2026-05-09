"""
algorithm_comparison.py
=============================
Deterministic vs Randomized Algorithm Comparison

When does each algorithm win? Theory predicts:
  - Deterministic: runtime ~ 1/eps (better eps-scaling)
  - Randomized: runtime ~ 1/eps^2
  - Deterministic has extra log(1/TV) that blows up for small TV.

Outputs
-------
- runtime_vs_n.png     : runtime scaling with dimension
- runtime_vs_tv.png    : runtime vs TV distance (the log(1/TV) test)
- runtime_vs_eps.png   : runtime vs eps (1/e vs 1/e^2 slopes)
- accuracy.png         : relative error comparison
- results.txt          : numerical summary
"""

import numpy as np
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from brute_force import brute_force_tv_vectorized
from randomized_algorithm import randomized_tv_estimate
from deterministic_algorithm import deterministic_tv
from src.distribution_families import (get_family, FAMILY_COLORS, FAMILY_MARKERS)


def compare_once(mP, mQ, epsilon, delta=0.1, rng=None, max_rand_samples=400):
    exact = brute_force_tv_vectorized(mP, mQ)
    if rng is None:
        rng = np.random.default_rng()

    t0 = time.perf_counter()
    det = deterministic_tv(mP, mQ, epsilon=epsilon)
    t_det = time.perf_counter() - t0

    t0 = time.perf_counter()
    rand = randomized_tv_estimate(mP, mQ, epsilon=epsilon, delta=delta,
                                   rng=rng, max_samples=max_rand_samples)
    t_rand = time.perf_counter() - t0

    det_err  = abs(det  - exact) / exact if exact > 1e-10 else 0.0
    rand_err = abs(rand - exact) / exact if exact > 1e-10 else 0.0
    return dict(exact=exact, det=det, rand=rand,
                t_det=t_det, t_rand=t_rand,
                det_err=det_err, rand_err=rand_err)


def run_algo_compare(seed=1, out_dir="."):
    rng = np.random.default_rng(seed)
    lines = ["=" * 70, "Deterministic vs Randomized Comparison", "=" * 70, ""]

    # ── Runtime vs n ───────────────────────────────────────────────────────
    lines.append("Runtime vs n (eps=0.2)")
    lines.append(f"  {'family':25s}  {'n':>4}  {'t_det(s)':>10}  {'t_rand(s)':>10}  "
                 f"{'det_err':>8}  {'rand_err':>9}")

    n_values = list(range(3, 10))
    families_a = ["B_adversarial", "C_random", "E_nearly_identical"]
    rt_n = {f: {"ns": [], "td": [], "tr": [], "ed": [], "er": []}
            for f in families_a}

    for fname in families_a:
        for n in n_values:
            tds, trs, eds, ers = [], [], [], []
            for _ in range(3):
                mP, mQ, _ = get_family(fname, n=n, q=2, rng=rng)
                r = compare_once(mP, mQ, epsilon=0.2, rng=rng)
                tds.append(r["t_det"]); trs.append(r["t_rand"])
                eds.append(r["det_err"]); ers.append(r["rand_err"])
            rt_n[fname]["ns"].append(n)
            rt_n[fname]["td"].append(np.mean(tds))
            rt_n[fname]["tr"].append(np.mean(trs))
            rt_n[fname]["ed"].append(np.mean(eds))
            rt_n[fname]["er"].append(np.mean(ers))
            lines.append(f"  {fname:25s}  {n:>4}  {np.mean(tds):>10.5f}  "
                         f"{np.mean(trs):>10.5f}  {np.mean(eds):>8.5f}  "
                         f"{np.mean(ers):>9.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for fname in families_a:
        c = FAMILY_COLORS[fname]; m = FAMILY_MARKERS[fname]
        label = fname.replace("_", " ")
        axes[0].semilogy(rt_n[fname]["ns"], rt_n[fname]["td"],
                          color=c, marker=m, label=f'Det {label}', linewidth=2)
        axes[0].semilogy(rt_n[fname]["ns"], rt_n[fname]["tr"],
                          color=c, marker=m, linestyle='--',
                          label=f'Rand {label}', linewidth=1.5, alpha=0.7)
    axes[0].set_xlabel('Dimension n'); axes[0].set_ylabel('Runtime (s)')
    axes[0].set_title('Runtime vs n (ε=0.2)\nSolid=Deterministic, Dashed=Randomized')
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(True, alpha=0.3)

    for fname in families_a:
        c = FAMILY_COLORS[fname]; m = FAMILY_MARKERS[fname]
        axes[1].plot(rt_n[fname]["ns"], rt_n[fname]["ed"],
                      color=c, marker=m, label=f'Det {fname[:8]}', linewidth=2)
        axes[1].plot(rt_n[fname]["ns"], rt_n[fname]["er"],
                      color=c, marker=m, linestyle='--',
                      label=f'Rand {fname[:8]}', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=0.2, color='gray', linestyle=':', label='ε=0.2')
    axes[1].set_xlabel('Dimension n'); axes[1].set_ylabel('Relative error')
    axes[1].set_title('Accuracy vs n'); axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_vs_n.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # ── Runtime vs TV distance ────────────────────────────────────────────
    lines.append("\nRuntime vs TV distance (n=6, eps=0.2)")
    lines.append("  Testing the log(1/TV) factor in deterministic algorithm")
    lines.append(f"  {'perturbation':>13}  {'TV_mean':>8}  {'t_det':>8}  {'t_rand':>8}")

    pert_levels = [0.4, 0.2, 0.08, 0.03, 0.01, 0.003]
    tv_means, td_means, tr_means = [], [], []

    for pert in pert_levels:
        tds, trs, tvs = [], [], []
        for _ in range(4):
            mP, mQ, _ = get_family("E_nearly_identical", n=6, q=2,
                                    rng=rng, perturbation=pert)
            r = compare_once(mP, mQ, epsilon=0.2, rng=rng)
            tds.append(r["t_det"]); trs.append(r["t_rand"]); tvs.append(r["exact"])
        tv_means.append(np.mean(tvs))
        td_means.append(np.mean(tds))
        tr_means.append(np.mean(trs))
        lines.append(f"  {pert:>13.3f}  {np.mean(tvs):>8.5f}  "
                     f"{np.mean(tds):>8.5f}  {np.mean(trs):>8.5f}")

    for _ in range(4):
        mP, mQ, _ = get_family("C_random", n=6, q=2, rng=rng)
        r = compare_once(mP, mQ, epsilon=0.2, rng=rng)
        tv_means.append(r["exact"]); td_means.append(r["t_det"])
        tr_means.append(r["t_rand"])

    order = np.argsort(tv_means)[::-1]
    tv_sorted  = [tv_means[i] for i in order]
    td_sorted  = [td_means[i] for i in order]
    tr_sorted  = [tr_means[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(tv_sorted, td_sorted, 'b-o', label='Deterministic', linewidth=2, markersize=6)
    ax.semilogy(tv_sorted, tr_sorted, 'r-s', label='Randomized',    linewidth=2, markersize=6)
    ax.set_xlabel('TV distance'); ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime vs TV distance (n=6, ε=0.2)\n'
                 'Deterministic has log(1/TV) factor → slower for small TV')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_vs_tv.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # ── Runtime vs eps ────────────────────────────────────────────────────
    lines.append("\nRuntime vs eps (n=6, C_random)")
    lines.append(f"  {'eps':>6}  {'t_det':>8}  {'t_rand':>8}  "
                 f"{'det_err':>8}  {'rand_err':>9}")

    eps_values = [0.4, 0.3, 0.2, 0.15, 0.1, 0.07]
    mP_fixed, mQ_fixed, _ = get_family("C_random", n=6, q=2, rng=rng)
    exact_fixed = brute_force_tv_vectorized(mP_fixed, mQ_fixed)
    td_eps, tr_eps = [], []

    for eps in eps_values:
        tds, trs = [], []
        for _ in range(4):
            r = compare_once(mP_fixed, mQ_fixed, epsilon=eps, rng=rng)
            tds.append(r["t_det"]); trs.append(r["t_rand"])
        td_eps.append(np.mean(tds)); tr_eps.append(np.mean(trs))
        lines.append(f"  {eps:>6.2f}  {np.mean(tds):>8.5f}  {np.mean(trs):>8.5f}  "
                     f"{abs(deterministic_tv(mP_fixed,mQ_fixed,eps)-exact_fixed)/exact_fixed:>8.5f}")

    # Fit slopes on log-log
    log_eps = np.log(eps_values)
    slope_det  = np.polyfit(log_eps, np.log(td_eps),  1)[0]
    slope_rand = np.polyfit(log_eps, np.log(tr_eps), 1)[0]
    lines.append(f"\n  Log-log slope det={slope_det:.2f} (theory: -1.0)  "
                 f"rand={slope_rand:.2f} (theory: -2.0)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(eps_values, td_eps, 'b-o', label=f'Deterministic (slope≈{slope_det:.2f})',
              linewidth=2, markersize=6)
    ax.loglog(eps_values, tr_eps, 'r-s', label=f'Randomized (slope≈{slope_rand:.2f})',
              linewidth=2, markersize=6)
    # Reference lines
    ref = np.array(eps_values)
    scale_d = td_eps[0] / (1/eps_values[0])
    scale_r = tr_eps[0] / (1/eps_values[0]**2)
    ax.loglog(eps_values, scale_d / ref,       'b--', alpha=0.4, label='∝ 1/ε (theory)')
    ax.loglog(eps_values, scale_r / ref**2,    'r--', alpha=0.4, label='∝ 1/ε² (theory)')
    ax.set_xlabel('ε'); ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime vs ε (n=6, C_random)\nExpected slopes: det=-1, rand=-2')
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_vs_eps.png"),
                dpi=120, bbox_inches='tight')
    plt.close()

    # Write report
    with open(os.path.join(out_dir, "algo_compare_results.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n  Figures saved to {out_dir}")


if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    run_algo_compare(out_dir=out)
