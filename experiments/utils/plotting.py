"""Shared plotting helpers and global paper style.

The convention is:
  * No on-figure titles -- captions do that work.
  * One color per algorithm, fixed across every figure in the paper
    (Okabe-Ito, colorblind-safe).
  * Solid lines, distinct markers, IQR shading at low alpha.
  * Major gridlines only.
  * Every figure is saved as both PDF (vector, used by LaTeX) and PNG
    (raster, used for quick inline previews) -- see ``save_figure``.
"""
from __future__ import annotations

import os

import matplotlib as mpl
import numpy as np


# Okabe-Ito palette (colorblind-safe).
_VERMILLION     = '#D55E00'
_BLUISH_GREEN   = '#009E73'
_SKY_BLUE       = '#56B4E9'
_BLUE           = '#0072B2'
_ORANGE         = '#E69F00'
_REDDISH_PURPLE = '#CC79A7'
_GRAY           = '#555555'


# Per-algorithm style. Every figure looks up the same dict so series read
# the same across figures.
ALGO_STYLE = {
    'TS-Explore':    {'color': _VERMILLION,     'marker': 's'},
    'TS-Explore-GF': {'color': _ORANGE,         'marker': 's'},
    'Basic TS':      {'color': _GRAY,           'marker': 'o'},
    'GRUB':          {'color': _SKY_BLUE,       'marker': '^'},
    'UCB-N':         {'color': _BLUE,           'marker': 'D'},
    'UCB+cover':     {'color': _BLUE,           'marker': 's'},
    'KL-LUCB':       {'color': _REDDISH_PURPLE, 'marker': 'v'},
    # mis_1 variants of TS-Explore.
    'TS_tuned':      {'color': _VERMILLION,     'marker': 's'},
    'TS_rho1':       {'color': _BLUISH_GREEN,   'marker': 'D'},
    'Basic':         {'color': _GRAY,           'marker': 'o'},
    # kernel_1 variants.
    'TS-L_G':        {'color': _VERMILLION,     'marker': 's'},
    'TS-K_G':        {'color': _BLUISH_GREEN,   'marker': 'P'},
    # graph-feedback ablation: color = stopping rule, marker = pull rule.
    'TS+cover':      {'color': _ORANGE,         'marker': 's'},
    'TS+width':      {'color': _ORANGE,         'marker': 'D'},
}


def apply_paper_style():
    """Set rcParams to a consistent paper-figure style."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman', 'Times'],
        'mathtext.fontset': 'cm',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.6,
        'lines.markersize': 5.5,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def style_for(name):
    return ALGO_STYLE.get(name, {'color': None, 'marker': 'o'})


def plot_with_iqr(ax, x, runs, label, color=None, marker='o',
                  linewidth=None, markersize=None, zorder=None,
                  linestyle='-'):
    """Plot median with 25-75 IQR shading."""
    runs = np.asarray(runs, dtype=float)
    med = np.nanmedian(runs, axis=1)
    lo = np.nanpercentile(runs, 25, axis=1)
    hi = np.nanpercentile(runs, 75, axis=1)
    plot_kwargs = dict(color=color, marker=marker, linestyle=linestyle,
                       label=label)
    if linewidth is not None:
        plot_kwargs['linewidth'] = linewidth
    if markersize is not None:
        plot_kwargs['markersize'] = markersize
    if zorder is not None:
        plot_kwargs['zorder'] = zorder
    ax.plot(x, med, **plot_kwargs)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0,
                    zorder=(zorder - 0.1) if zorder is not None else None)


def grid_only_major(ax):
    ax.grid(which='major', alpha=0.3)
    ax.grid(which='minor', visible=False)


def save_figure(fig, path):
    """Save ``fig`` as both .pdf (vector, for LaTeX) and .png (raster preview).

    ``path`` may have any extension; the base is taken via
    ``os.path.splitext`` and both ``base.pdf`` and ``base.png`` are written.
    Returns the list of paths written.
    """
    base, _ = os.path.splitext(path)
    written = []
    for ext in ('.pdf', '.png'):
        out = base + ext
        fig.savefig(out)
        written.append(out)
    return written


def legend_above(ax, ncol=None, **kwargs):
    """Place a horizontal legend just above the axes (no overlap with data)."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if ncol is None:
        ncol = len(handles)
    defaults = dict(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        borderpad=0.0,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    defaults.update(kwargs)
    return ax.legend(handles, labels, **defaults)


def legend_above_figure(fig, axes, ncol=None, y=1.0, **kwargs):
    """Single deduplicated horizontal legend above all panels."""
    handles, labels, seen = [], [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    if not handles:
        return None
    if ncol is None:
        ncol = len(handles)
    defaults = dict(
        loc='lower center',
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        borderpad=0.0,
        borderaxespad=0.0,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    defaults.update(kwargs)
    return fig.legend(handles, labels, **defaults)


# ---------------------------------------------------------------------
# Backwards-compatible shims (older scripts may still import these).
# ---------------------------------------------------------------------

def plot_with_ci(ax, x, runs, label, color=None, marker='o', ls='-'):
    """Deprecated alias kept for older scripts; ignores ``ls``."""
    plot_with_iqr(ax, x, runs, label=label, color=color, marker=marker)


def nice_log_axis(ax, which='x'):
    if which in ('x', 'both'):
        ax.set_xscale('log')
    if which in ('y', 'both'):
        ax.set_yscale('log')


def hline(ax, y, label=None, color='k', ls='--', alpha=0.7):
    ax.axhline(y, color=color, linestyle=ls, alpha=alpha, label=label)


def vline(ax, x, label=None, color='k', ls=':', alpha=0.7):
    ax.axvline(x, color=color, linestyle=ls, alpha=alpha, label=label)
