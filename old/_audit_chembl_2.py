import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

z = np.load('experiments/outputs/chembl_2_results.npz', allow_pickle=False)
targets = [str(t) for t in z['targets']]
rhos = list(z['rhos'].tolist())
seeds = list(z['seeds'].tolist())
algos = ['TS-Explore', 'GRUB', 'Basic TS', 'KL-LUCB']

print(f"=== run config ===")
print(f"  targets = {targets}")
print(f"  rhos    = {rhos}")
print(f"  seeds   = {len(seeds)}")
print(f"  top_k   = {int(z['top_k'])}")
print(f"  knn_k   = {int(z['knn_k'])}")
print()
print(f"=== per-target diagnostics ===")
for ti, t in enumerate(targets):
    print(f"  {t}: eps_L = {float(z['eps_L'][ti]):.2f}, "
          f"H_cls = {float(z['H_classical'][ti]):.2f}, "
          f"H_graph = {float(z['H_graph'][ti]):.2f}, "
          f"smallest_gap = {float(z['smallest_gap'][ti]):.3f}")

print()
for ti, t in enumerate(targets):
    print(f"\n##### target = {t} #####")
    print(f"{'rho':>7s}  " +
          '  '.join(f'{a:>11s}' for a in algos) +
          '  ' + 'B/TSx' + '  ' + 'KLLU/TSx' + '  ' + 'TSE-corr')
    for ri, rho in enumerate(rhos):
        meds = {a: float(np.nanmedian(z[f'{t}__{a}__stop'][ri])) for a in algos}
        cors = {a: float(z[f'{t}__{a}__correct'][ri].mean()) for a in algos}
        b_ratio = meds['Basic TS'] / max(meds['TS-Explore'], 1.0)
        kl_ratio = meds['TS-Explore'] / max(meds['KL-LUCB'], 1.0)
        row = '  '.join(f'{meds[a]:>11.0f}' for a in algos)
        print(f"  {rho:>5.0f}  {row}  {b_ratio:>5.2f}x  {kl_ratio:>7.2f}x  {cors['TS-Explore']:>5.0%}")
    print()
    # also show all algo correctness across rhos
    print(f"  correctness across rhos:")
    for a in algos:
        corrs = [float(z[f'{t}__{a}__correct'][ri].mean()) for ri in range(len(rhos))]
        print(f"    {a:11s}: {[f'{c:.0%}' for c in corrs]}")

print()
print(f"=== best per-target Basic / TS-Explore ratio (across rhos, requiring 100% correct) ===")
for ti, t in enumerate(targets):
    best_rho_i = -1
    best_ratio = 0
    for ri, rho in enumerate(rhos):
        if z[f'{t}__TS-Explore__correct'][ri].mean() == 1.0:
            ratio = float(np.nanmedian(z[f'{t}__Basic TS__stop'][ri]) /
                          max(np.nanmedian(z[f'{t}__TS-Explore__stop'][ri]), 1.0))
            if ratio > best_ratio:
                best_ratio = ratio
                best_rho_i = ri
    if best_rho_i >= 0:
        print(f"  {t}: best Basic/TSE = {best_ratio:.2f}x at rho = {rhos[best_rho_i]}")
    else:
        print(f"  {t}: NO RHO with 100% TS-Explore correctness")
