"""MovieLens-100K loader and item-item similarity graph builder.

Builds a real-world bandit instance for graph-smooth pure exploration:

    arms        : K most-rated movies
    mu_i        : empirical mean rating of movie i (over its raters)
    Adjacency A : top-k mutual-neighbor sparsification of item-item
                  adjusted-cosine similarity (positive correlations only)
    Degree D    : diag(row sums of A)

The reward model is Gaussian noise around mu_i (matched to the synthetic
experiments), so a "pull" of arm i samples r_t ~ N(mu_i, sigma^2).  The
empirical mu_i is treated as the ground truth.

Recipe is the standard MovieLens preprocessing used in graph-bandit
work (Zong et al. 2016, Wu et al. 2019): adjusted cosine on user
ratings, top-k neighbor sparsification, no edge weights (unit
adjacency matches the synthetic experiments).
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = _REPO / "data" / "ml-100k"


def download_and_extract(cache_dir: Path | str | None = None) -> Path:
    """Download ml-100k.zip into cache_dir.parent, extract to cache_dir.

    No-op if cache_dir/u.data already exists.  Returns the cache_dir
    path.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    if (cache_dir / "u.data").exists():
        return cache_dir
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir.parent / "ml-100k.zip"
    if not zip_path.exists():
        print(f"[movielens] downloading {ML_100K_URL}\n"
              f"  -> {zip_path}", flush=True)
        urllib.request.urlretrieve(ML_100K_URL, str(zip_path))
    print(f"[movielens] extracting -> {cache_dir.parent}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(str(cache_dir.parent))
    if not (cache_dir / "u.data").exists():
        raise RuntimeError(f"extraction did not produce {cache_dir / 'u.data'}")
    return cache_dir


def load_ratings(cache_dir=None):
    """Return (R, M) where R[u, i] = rating, M[u, i] = whether rated.

    Both have shape (n_users, n_items).  Movie/user IDs are remapped to
    0-indexed contiguous integers (the source data uses 1-indexed IDs).
    """
    cache = download_and_extract(cache_dir)
    raw = np.loadtxt(cache / "u.data", delimiter="\t", dtype=int)
    user_ids = raw[:, 0]
    item_ids = raw[:, 1]
    ratings = raw[:, 2].astype(float)
    n_users = int(user_ids.max())
    n_items = int(item_ids.max())
    R = np.zeros((n_users, n_items), dtype=float)
    M = np.zeros((n_users, n_items), dtype=bool)
    R[user_ids - 1, item_ids - 1] = ratings
    M[user_ids - 1, item_ids - 1] = True
    return R, M


def load_movie_titles(cache_dir=None):
    """Return list of movie titles indexed by 0-based item ID."""
    cache = download_and_extract(cache_dir)
    titles = []
    with open(cache / "u.item", encoding="latin-1") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("|")
            titles.append(fields[1])
    return titles


def adjusted_cosine_similarity(R_sub, M_sub, min_common=5):
    """Adjusted cosine: mean-center each item over its raters globally,
    then take cosine over user pairs.  Returns (sim, common_counts).
    """
    sums = (R_sub * M_sub).sum(axis=0)
    counts = M_sub.sum(axis=0).astype(float)
    means = sums / np.maximum(counts, 1.0)
    R_c = (R_sub - means[None, :]) * M_sub
    num = R_c.T @ R_c
    norms = np.sqrt((R_c ** 2).sum(axis=0))
    denom = np.outer(norms, norms)
    denom_safe = np.where(denom > 0, denom, 1.0)
    sim = num / denom_safe
    sim = np.where(denom > 0, sim, 0.0)
    common_counts = M_sub.astype(np.int32).T @ M_sub.astype(np.int32)
    sim = np.where(common_counts >= min_common, sim, 0.0)
    np.fill_diagonal(sim, 0.0)
    return sim, common_counts


def build_instance(K=100, top_k_neighbors=5, min_common=5, min_ratings=None,
                   cache_dir=None, return_meta=False):
    """Build a (mu, A, D) instance from MovieLens 100K.

    Parameters
    ----------
    K : int
        Number of arms (movies).  We take the K most-rated movies.
    top_k_neighbors : int
        Each item connects to its top-k similar items (mutual union).
    min_common : int
        Skip a similarity entry if the two items share fewer than this
        many users in their rating sets.
    min_ratings : int or None
        Optionally drop items with fewer than this many ratings before
        ranking by popularity.  None = no filter beyond the top-K cut.
    cache_dir : str or None
        Where to cache ml-100k (default: <repo>/data/ml-100k/).
    return_meta : bool
        If True, additionally return a dict with keys:
        ``selected_item_ids`` (0-indexed into ml-100k items),
        ``titles`` (movie titles aligned to mu's order),
        ``rating_counts`` (number of raters per arm),
        ``similarity`` (the dense similarity matrix on the K subset).

    Returns
    -------
    mu, A, D  (and optionally meta) -- numpy arrays in the format used
    by the rest of the experiments.
    """
    R, M = load_ratings(cache_dir)
    n_raters_per_item = M.sum(axis=0)
    eligible = (n_raters_per_item >= (min_ratings or 1))
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) < K:
        raise ValueError(
            f"only {len(eligible_idx)} items pass min_ratings={min_ratings}; "
            f"need K={K}"
        )
    order = eligible_idx[np.argsort(n_raters_per_item[eligible_idx])[::-1]]
    selected = order[:K]

    R_sub = R[:, selected]
    M_sub = M[:, selected]

    sums = (R_sub * M_sub).sum(axis=0)
    counts = M_sub.sum(axis=0).astype(float)
    mu = sums / np.maximum(counts, 1.0)

    sim, _ = adjusted_cosine_similarity(R_sub, M_sub, min_common=min_common)
    sim_pos = np.maximum(sim, 0.0)
    np.fill_diagonal(sim_pos, 0.0)

    A = np.zeros((K, K), dtype=float)
    order_per_row = np.argsort(sim_pos, axis=1)
    for i in range(K):
        for j in order_per_row[i, -top_k_neighbors:]:
            if sim_pos[i, j] > 0:
                A[i, j] = 1.0
    A = np.maximum(A, A.T)
    D = np.diag(A.sum(axis=1))

    if return_meta:
        try:
            titles = load_movie_titles(cache_dir)
            titles_sel = [titles[i] for i in selected]
        except Exception:
            titles_sel = [f"movie_{i}" for i in selected]
        # Per-arm observed rating arrays (1-D float arrays of variable length)
        # for empirical-rating reward sampling.
        ratings_per_arm = []
        for ai in range(K):
            col = R_sub[:, ai]
            mask = M_sub[:, ai]
            ratings_per_arm.append(np.ascontiguousarray(col[mask], dtype=float))
        # Per-arm rating standard deviation, useful for reporting.
        rating_stds = np.array(
            [r.std(ddof=0) if len(r) > 1 else 0.0 for r in ratings_per_arm]
        )
        meta = dict(
            selected_item_ids=selected,
            titles=titles_sel,
            rating_counts=counts,
            similarity=sim_pos,
            ratings_per_arm=ratings_per_arm,
            rating_stds=rating_stds,
        )
        return mu, A, D, meta
    return mu, A, D


def make_empirical_reward_fn(ratings_per_arm, rng=None):
    """Return a callable ``f(arm) -> reward`` that samples uniformly with
    replacement from the observed user ratings of ``arm``.

    Replaces the synthetic Gaussian reward model in
    ``support_func.gaussian_reward`` for the MovieLens experiments, so
    the reward stream uses the dataset's actual rating distribution
    (per-movie heteroscedastic noise, integer 1--5 categorical values)
    rather than ``N(mu_i, sigma=1)``.

    ``rng`` is an optional ``numpy.random.Generator``; if None, the
    function uses ``np.random.choice`` (legacy global RNG seeded by the
    runner via ``np.random.seed``).
    """
    arm_arrays = [np.asarray(r, dtype=float) for r in ratings_per_arm]
    if any(len(r) == 0 for r in arm_arrays):
        raise ValueError("at least one arm has zero observed ratings; cannot "
                         "sample empirically")
    if rng is None:
        def _draw(arm):
            r = arm_arrays[int(arm)]
            return float(r[np.random.randint(len(r))])
    else:
        def _draw(arm):
            r = arm_arrays[int(arm)]
            return float(rng.choice(r))
    return _draw
