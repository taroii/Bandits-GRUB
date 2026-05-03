"""Quick screen of ChEMBL targets for BAI suitability.

Fetches top-K most-active compounds for each candidate target, prints
gap structure and smoothness diagnostics, picks the targets where:
  - smallest non-zero gap is reasonably large (BAI is tractable)
  - H_classical / H_graph > 1.5 (graph regularization has analytical credit)
  - no ties at the best arm
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils import chembl_loader, hardness  # noqa: E402

CANDIDATES = [
    'CHEMBL204',   # thrombin (ALREADY USED, baseline)
    'CHEMBL230',   # carbonic anhydrase II
    'CHEMBL220',   # acetylcholinesterase
    'CHEMBL325',   # histamine H3
    'CHEMBL287',   # sigma-1 receptor
    'CHEMBL2147',  # PIM1 kinase
    'CHEMBL206',   # estrogen receptor alpha
    'CHEMBL3155',  # carbonic anhydrase XII
    'CHEMBL2820',  # cathepsin K
]

K = 100
KNN = 10


def screen_one(target):
    try:
        payload = chembl_loader.build_instance(
            target=target, subsample_k=K, knn_k=KNN,
            normalize=False, max_pages=4, select='top',
            verbose=False)
    except Exception as e:
        print(f"  {target}: ERROR -- {e}")
        return None
    mu = payload['pIC50']
    A = payload['A']
    D = payload['D']
    a_star = int(np.argmax(mu))
    Delta = mu[a_star] - mu
    nz = Delta[Delta > 0]
    n_ties = int((Delta == 0).sum() - 1)  # subtract a* itself
    smallest_gap = float(nz.min()) if nz.size else float('nan')
    sorted_mu = np.sort(mu)[::-1]
    gap_to_2nd = float(sorted_mu[0] - sorted_mu[1])
    L = D - A
    eps_L = float(np.sqrt(max(mu @ L @ mu, 0.0)))
    H_cls = float(hardness.classical_hardness(mu))
    H_gr = float(hardness.graph_hardness(mu, A, D, rho=1.0))
    n_records = payload['n_raw_records']
    return dict(target=target, K=int(mu.shape[0]), n_records=int(n_records),
                pic50_min=float(mu.min()), pic50_max=float(mu.max()),
                smallest_gap=smallest_gap, gap_to_2nd=gap_to_2nd,
                n_ties_with_best=n_ties, eps_L=eps_L,
                H_cls=H_cls, H_gr=H_gr, ratio=H_cls / max(H_gr, 1e-9))


def main():
    print(f"Screening {len(CANDIDATES)} ChEMBL targets at K={K}, knn={KNN}")
    print(f"{'target':<12} {'K':>4} {'pIC50 range':<16} {'gap_2nd':>8}"
          f" {'min_nz_gap':>10} {'ties':>5} {'eps_L':>7} {'H_cls':>8}"
          f" {'H_gr':>8} {'ratio':>6}")
    rows = []
    for t in CANDIDATES:
        r = screen_one(t)
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['target']:<10} {r['K']:>4}"
              f" [{r['pic50_min']:.2f},{r['pic50_max']:.2f}]"
              f" {r['gap_to_2nd']:>8.3f} {r['smallest_gap']:>10.4f}"
              f" {r['n_ties_with_best']:>5d} {r['eps_L']:>7.2f}"
              f" {r['H_cls']:>8.2f} {r['H_gr']:>8.2f}"
              f" {r['ratio']:>5.2f}x")

    print("\nRanking by H_cls / H_graph * (gap_to_2nd > 0.3 filter)")
    ok = [r for r in rows if r['gap_to_2nd'] > 0.3 and r['n_ties_with_best'] == 0
          and np.isfinite(r['H_cls'])]
    ok.sort(key=lambda r: -r['ratio'])
    for r in ok[:5]:
        print(f"  {r['target']}: ratio={r['ratio']:.2f}x, "
              f"gap_to_2nd={r['gap_to_2nd']:.3f}, "
              f"min_nz_gap={r['smallest_gap']:.3f}, "
              f"H_cls={r['H_cls']:.1f}, H_gr={r['H_gr']:.1f}")


if __name__ == "__main__":
    main()
