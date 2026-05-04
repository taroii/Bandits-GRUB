"""Logic-only test of chembl_2's --rerun-algos invalidation.

Loads the saved checkpoint, simulates the resume + invalidation
branches without saving, and prints what got invalidated vs preserved.
Avoids any disk I/O so the Windows file-lock issue doesn't bite.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALGOS = ['TS-Explore', 'GRUB', 'Basic TS', 'KL-LUCB']
RHO_FREE = ['Basic TS', 'KL-LUCB']

zr = np.load('experiments/outputs/chembl_2_results.npz', allow_pickle=False)
targets = [str(t) for t in zr['targets']]
rhos = list(zr['rhos'].tolist())
seeds = list(zr['seeds'].tolist())
n_targets = len(targets)
n_rhos = len(rhos)
n_seeds = len(seeds)

# Initialize state
stop_times = {(t, a): np.full((n_rhos, n_seeds), np.nan)
              for t in targets for a in ALGOS}
correct = {(t, a): np.zeros((n_rhos, n_seeds), dtype=bool)
           for t in targets for a in ALGOS}
done = {(t, a): np.zeros(n_rhos, dtype=bool) for t in targets for a in ALGOS}
rho_free_done = {(t, a): False for t in targets for a in RHO_FREE}

# Resume
for t in targets:
    for a in ALGOS:
        key = f'{t}__{a}'
        stop_times[(t, a)] = zr[f'{key}__stop']
        correct[(t, a)] = zr[f'{key}__correct'].astype(bool)
        done[(t, a)] = zr[f'{key}__done'].astype(bool)
        if a in RHO_FREE:
            rho_free_done[(t, a)] = bool(done[(t, a)].all())

print('=== state after resume (no invalidation) ===')
for a in ALGOS:
    print(f'  {a:11s}: ' + '  '.join(
        f'{t}={int(done[(t, a)].sum())}/{n_rhos}' for t in targets))

# Apply invalidation: simulating --rerun-algos KL-LUCB
rerun = ['KL-LUCB']
for t in targets:
    for a in rerun:
        done[(t, a)] = np.zeros(n_rhos, dtype=bool)
        stop_times[(t, a)] = np.full((n_rhos, n_seeds), np.nan)
        correct[(t, a)] = np.zeros((n_rhos, n_seeds), dtype=bool)
        if a in RHO_FREE:
            rho_free_done[(t, a)] = False

print(f'\n=== state after --rerun-algos {rerun} ===')
for a in ALGOS:
    n = sum(int(done[(t, a)].sum()) for t in targets)
    total = len(targets) * n_rhos
    if a in rerun:
        marker = '  <-- will be re-run'
    else:
        marker = ''
    print(f'  {a:11s}: ' + '  '.join(
        f'{t}={int(done[(t, a)].sum())}/{n_rhos}'
        for t in targets) + marker)

# Verify rho_free_done is reset for KL-LUCB
print(f'\nrho_free_done after invalidation:')
for t in targets:
    for a in RHO_FREE:
        flag = rho_free_done[(t, a)]
        marker = '  <-- will run' if not flag else ''
        print(f'  {t} / {a}: {flag}{marker}')
