"""Verify both movielens and chembl modules import without errors."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[imports] importing experiments.utils.movielens ... ", end='', flush=True)
from experiments.utils import movielens
print("OK")

print("[imports] importing experiments.utils.chembl_loader ... ",
      end='', flush=True)
try:
    from experiments.utils import chembl_loader
    print("OK")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print("[imports] importing experiments.utils.instances "
      "+ checking movielens_top_k ... ", end='', flush=True)
from experiments.utils import instances
assert hasattr(instances, 'movielens_top_k'), "missing movielens_top_k wrapper"
print("OK")

print("[imports] importing graph_algo ... ", end='', flush=True)
import graph_algo
assert hasattr(graph_algo, 'ThompsonSampling')
assert hasattr(graph_algo, 'BasicThompsonSampling')
assert hasattr(graph_algo, 'MaxVarianceArmAlgo')
print("OK")

print("[imports] all imports succeed")
