"""
run_paper_experiments.py
========================
Runs all four experiments needed to reproduce all figures
and results in the final paper.

Usage:
  python run_paper_experiments.py
"""

import sys, os, time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR  = os.path.join(_THIS_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

OUT = _THIS_DIR


def run(name, fn, **kwargs):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    t0 = time.time()
    try:
        fn(out_dir=OUT, **kwargs)
        print(f"  ✓ done in {time.time()-t0:.1f}s")
    except Exception as e:
        import traceback
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    from sparsification_bound  import run_sparsification_bound
    from support_growth        import run_support_growth
    from ratio_clustering     import run_ratio_clustering
    from algorithm_comparison  import run_algo_compare

    run("Sparsification bound (Finding 1)", run_sparsification_bound, n_instances=25, seed=0)
    run("Support size growth (Finding 2)", run_support_growth, seed=2)
    run("Ratio clustering near r=1 (Finding 2)", run_ratio_clustering, seed=10)
    run("Algorithm comparison (Finding 3)", run_algo_compare, seed=1)

    print("\n\nAll done. Figures saved to:", OUT)
