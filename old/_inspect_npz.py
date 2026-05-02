import numpy as np
import os
files = [
    'experiments/outputs/chembl_204_smoke.npz',
    'experiments/outputs/chembl_204_v2.npz',
    'experiments/outputs/chembl_1_results.npz',
    'experiments/outputs/movielens_1_results.npz',
]
for f in files:
    print(f'--- {f} ---')
    if not os.path.exists(f):
        print('  (missing)')
        continue
    try:
        z = np.load(f, allow_pickle=False)
        print('  files:', sorted(z.files))
        if 'pIC50' in z.files:
            pic50 = z['pIC50']
            print(f'  K = {len(pic50)}, pIC50 range = [{pic50.min():.3f}, {pic50.max():.3f}]')
            if 'target' in z.files:
                print(f'  target = {z["target"]}')
        if 'rhos' in z.files:
            print(f'  rhos = {z["rhos"]}')
        if 'seeds' in z.files:
            print(f'  n_seeds = {len(z["seeds"])}')
        if 'done' in z.files:
            print(f'  done = {z["done"]}')
    except Exception as e:
        print(f'  error: {e}')
