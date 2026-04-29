"""Run every experiment in sequence and report acceptance outcomes.

Usage:
    python experiments/run_all.py [--quick] [--n-jobs N] [--seeds S]

Uses all experiments' individual CLI flags so ``--quick`` propagates.
Experiments 1 (delta down to 1e-6) and 3 (eps down to 1e-3) are the most
expensive; for a fast sanity run prefer ``--quick``.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time


EXPERIMENTS = [
    'exp1_delta_scaling.py',
    'exp2_density_sweep.py',
    'exp3_smoothness_asymptotics.py',
    'exp4_kernel_comparison.py',
    'exp5_competitive_set.py',
    'exp6_fig1_with_ci.py',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--n-jobs', type=int, default=max(mp.cpu_count() - 1, 1))
    parser.add_argument('--seeds', type=int, default=None,
                        help='override seeds in every experiment')
    parser.add_argument('--skip', nargs='*', default=(),
                        help='experiment filenames to skip')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, 'outputs'), exist_ok=True)
    timings = {}
    exit_codes = {}

    for exp in EXPERIMENTS:
        if exp in args.skip:
            print(f"[skip] {exp}")
            continue
        cmd = [sys.executable, '-u', os.path.join(here, exp),
               '--n-jobs', str(args.n_jobs)]
        if args.quick:
            cmd.append('--quick')
        if args.seeds is not None:
            cmd += ['--seeds', str(args.seeds)]
        print(f"\n{'='*70}\n=== {exp}\n{'='*70}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        timings[exp] = time.time() - t0
        exit_codes[exp] = rc
        print(f"[{exp}] finished in {timings[exp]:.1f}s (exit {rc})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for exp in EXPERIMENTS:
        t = timings.get(exp, None)
        rc = exit_codes.get(exp, None)
        tag = 'OK' if rc == 0 else ('SKIP' if rc is None else 'FAIL')
        ts = f"{t:6.1f}s" if t is not None else '   -'
        print(f"  {exp:32s}  {ts}  [{tag}]")

    return 0 if all(rc == 0 for rc in exit_codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
