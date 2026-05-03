import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

z = np.load('experiments/outputs/chembl_1_results.npz', allow_pickle=False)
print('=== run config ===')
print(f"  K          = {int(z['K'])}")
print(f"  rho        = {float(z['rho'])}")
print(f"  seeds      = {z['seeds'].size}")
print(f"  target     = {str(z['target'])}")
print(f"  best arm   = {int(z['best_arm'])}")
print(f"  smallest gap (raw pIC50)   = {float(z['smallest_gap']):.4f}")
print(f"  epsilon_L (combinatorial)  = {float(z['eps_L']):.3f}")
print(f"  H_classical                = {float(z['H_classical']):.2f}")
print(f"  H_graph (rho=1)            = {float(z['H_graph']):.2f}")
print()
print('=== per-algo stopping times ===')
print(f"{'algo':<13} {'med':>8} {'mean':>8} {'std':>8} {'p25':>8} {'p75':>8} {'min':>8} {'max':>8}  correct")
for a in ['TS-Explore', 'Basic TS', 'GRUB']:
    ts = z[f'{a}_stop'].astype(float)
    cor = z[f'{a}_correct'].astype(bool)
    print(f"  {a:<11s} {np.median(ts):>8.0f} {ts.mean():>8.0f} {ts.std():>8.0f} "
          f"{np.percentile(ts,25):>8.0f} {np.percentile(ts,75):>8.0f} "
          f"{ts.min():>8.0f} {ts.max():>8.0f}  "
          f"{cor.sum()}/{len(cor)}")

print()
med_ts = np.median(z['TS-Explore_stop'])
med_basic = np.median(z['Basic TS_stop'])
med_grub = np.median(z['GRUB_stop'])
print(f"  Basic TS / TS-Explore = {med_basic / med_ts:.2f}x")
print(f"  GRUB     / TS-Explore = {med_grub / med_ts:.2f}x")

# pull-count diagnostics
print()
print('=== pull counts (median across seeds, top arms by pulls) ===')
mu = z['mu']
a_star = int(z['best_arm'])
order = np.argsort(-mu)[:6]
for a in ['TS-Explore', 'Basic TS', 'GRUB']:
    pulls = z[f'{a}_pulls'].astype(float)
    med_p = np.median(pulls, axis=0)
    print(f"  {a}:")
    for i in order:
        gap = mu[a_star] - mu[i]
        print(f"    rank-{int(np.argsort(-mu).tolist().index(i))} arm{i:>4d}  "
              f"mu={mu[i]:.3f}  gap={gap:.3f}  pulls={med_p[i]:8.0f}")

# distribution diagnostics
print()
print('=== reward distribution ===')
print(f"  K = {len(mu)}")
print(f"  pIC50 range: [{mu.min():.3f}, {mu.max():.3f}], median = {np.median(mu):.3f}")
sorted_mu = np.sort(mu)[::-1]
print(f"  top 5 pIC50:  {sorted_mu[:5].tolist()}")
print(f"  bottom 5:     {sorted_mu[-5:].tolist()}")
gaps = sorted_mu[0] - sorted_mu[1:]
print(f"  Delta to runner-up = {gaps[0]:.4f}  (1/Delta^2 = {1/gaps[0]**2:.2f})")
print(f"  Delta to 3rd       = {gaps[1]:.4f}")
print(f"  Delta to 5th       = {gaps[3]:.4f}")
print(f"  Delta to 10th      = {gaps[8]:.4f}")
