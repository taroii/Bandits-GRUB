"""Shared plotting helpers."""
from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ALGO_STYLE = {
    'ThompsonSampling':      {'color': '#d62728', 'marker': 's', 'ls': '-'},
    'BasicThompsonSampling': {'color': '#2ca02c', 'marker': '^', 'ls': '--'},
    'GraphFeedbackTS':       {'color': '#9467bd', 'marker': 'D', 'ls': '-'},
    'MaxDiffVarAlgo':        {'color': '#1f77b4', 'marker': 'o', 'ls': '-'},
    'CyclicAlgo':            {'color': '#ff7f0e', 'marker': 'x', 'ls': '-.'},
    'OneStepMinSumAlgo':     {'color': '#e377c2', 'marker': 'v', 'ls': '-'},
    'MaxVarianceArmAlgo':    {'color': '#8c564b', 'marker': 'p', 'ls': ':'},
    'NoGraphAlgo':           {'color': '#7f7f7f', 'marker': '*', 'ls': '--'},
}


def style_for(name: str) -> dict:
    return ALGO_STYLE.get(name, {'color': None, 'marker': 'o', 'ls': '-'})


def plot_with_ci(ax, x, runs, label, color=None, marker='o', ls='-'):
    """Plot median with 25-75 percentile shading.

    ``runs`` should be a 2D array-like of shape (len(x), n_seeds)
    containing stopping times.
    """
    runs = np.asarray(runs, dtype=float)
    med = np.median(runs, axis=1)
    lo = np.percentile(runs, 25, axis=1)
    hi = np.percentile(runs, 75, axis=1)
    ax.plot(x, med, color=color, marker=marker, linestyle=ls, label=label, linewidth=1.8)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


def nice_log_axis(ax, which='x'):
    if which in ('x', 'both'):
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    if which in ('y', 'both'):
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())


def hline(ax, y, label=None, color='k', ls='--', alpha=0.7):
    ax.axhline(y, color=color, linestyle=ls, alpha=alpha, label=label)


def vline(ax, x, label=None, color='k', ls=':', alpha=0.7):
    ax.axvline(x, color=color, linestyle=ls, alpha=alpha, label=label)
