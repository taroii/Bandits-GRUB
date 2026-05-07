"""Seeded multi-run harness for graph-bandit algorithms.

A *run* fixes the numpy random state, instantiates an algorithm via
``algo_factory()``, and drives ``play_round(1)`` until convergence.  The
runner records:

* ``stopping_time``  - number of real arm pulls when the run terminated
* ``selected_arm``   - the arm the algorithm returned
* ``correct``        - whether that matches ``argmax(means)``
* ``elimination_curve`` - list of (t, num_remaining) tuples (useful for
  Experiment 6; for Thompson Sampling the curve only contains the initial
  and final points, since TS does not expose intermediate eliminations).
* ``pull_counts``    - direct pulls per arm at termination
* ``converged_flag`` - whether the algorithm hit its stopping rule (False
  means we exited on ``max_steps``).

``algo_factory`` is invoked once per seed in ``run_many``. The factory
should NOT seed numpy itself --- the runner handles seeding before it
is called. Execution is strictly sequential; multiprocessing was
removed because parallel pools were causing the shared compute server
to crash under load.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _run_round(algo) -> None:
    algo.play_round(1)


def _is_converged(algo) -> bool:
    if getattr(algo, 'converged', False):
        return True
    if len(getattr(algo, 'remaining_nodes', [])) <= 1:
        return True
    return False


def _current_pulls(algo) -> np.ndarray:
    if hasattr(algo, 'counter') and algo.counter is not None and algo.counter.size:
        return np.diag(algo.counter).copy()
    if hasattr(algo, 'pull_counts') and algo.pull_counts is not None:
        return np.asarray(algo.pull_counts).copy()
    if hasattr(algo, 'counts') and algo.counts is not None:
        return np.asarray(algo.counts).copy()
    raise AttributeError("algo has no pull-count attribute")


def _current_t(algo) -> int:
    if hasattr(algo, 't'):
        return int(algo.t)
    if hasattr(algo, 'counter'):
        return int(np.trace(algo.counter))
    return 0


def _true_means(algo) -> np.ndarray:
    return np.asarray(algo.means, dtype=float).flatten()


def _elimination_sample_rate(algo) -> bool:
    """True if algo exposes intermediate eliminations (UCB-style)."""
    return not isinstance(getattr(algo, 't', 0), type(None)) and \
        hasattr(algo, 'remaining_nodes') and \
        not getattr(algo, 'converged', False).__class__ is bool.__class__ and False \
        or hasattr(algo, 'remaining_nodes')


def run_algorithm(algo_factory: Callable,
                  seed: int,
                  max_steps: int = 1_000_000,
                  record_elimination: bool = True) -> Dict:
    np.random.seed(int(seed))
    algo = algo_factory()
    K = len(_true_means(algo))
    a_star_true = int(np.argmax(_true_means(algo)))

    curve: List[tuple] = []
    remaining_prev = len(getattr(algo, 'remaining_nodes', range(K)))
    t0 = _current_t(algo)
    curve.append((t0, remaining_prev))

    steps = 0
    while not _is_converged(algo) and steps < max_steps:
        _run_round(algo)
        steps += 1
        if record_elimination:
            remaining_now = len(algo.remaining_nodes)
            if remaining_now != remaining_prev:
                curve.append((_current_t(algo), remaining_now))
                remaining_prev = remaining_now
    converged = _is_converged(algo)
    if converged:
        remaining_now = len(algo.remaining_nodes)
        curve.append((_current_t(algo), remaining_now))

    selected = int(algo.remaining_nodes[0]) if algo.remaining_nodes else -1
    return {
        'stopping_time': _current_t(algo),
        'selected_arm': selected,
        'correct': selected == a_star_true,
        'elimination_curve': curve,
        'pull_counts': _current_pulls(algo),
        'converged_flag': converged,
    }


def run_many(algo_factory: Callable,
             seeds: List[int],
             max_steps: int = 1_000_000,
             record_elimination: bool = True,
             progress: bool = False,
             **_legacy_kwargs) -> List[Dict]:
    """Sequentially run ``algo_factory`` for each seed.

    Earlier versions accepted an ``n_jobs`` argument that dispatched
    to ``multiprocessing.Pool``; that path was removed because parallel
    pools were causing the shared compute server to crash under load.
    Any ``n_jobs`` keyword passed by legacy callers is silently
    ignored via ``**_legacy_kwargs``.
    """
    out = []
    for i, s in enumerate(seeds):
        out.append(run_algorithm(algo_factory, s,
                                 max_steps=max_steps,
                                 record_elimination=record_elimination))
        if progress:
            print(f"  seed {s} done ({i+1}/{len(seeds)})", flush=True)
    return out


def summarize(results: List[Dict]) -> Dict:
    """Aggregate seed-level results into summary statistics."""
    t_arr = np.array([r['stopping_time'] for r in results], dtype=float)
    return {
        'median': float(np.median(t_arr)),
        'p25': float(np.percentile(t_arr, 25)),
        'p75': float(np.percentile(t_arr, 75)),
        'mean': float(np.mean(t_arr)),
        'correct_rate': float(np.mean([r['correct'] for r in results])),
        'n': len(results),
        'raw': t_arr,
    }
