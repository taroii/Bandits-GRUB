"""ChEMBL bioactivity loader -> graph-bandit instance.

Fetches all measured IC50 bioactivities for a single ChEMBL target,
deduplicates per molecule, computes ECFP4 (Morgan, radius 2, 2048-bit)
fingerprints via RDKit, builds a k-nearest-neighbor Tanimoto-similarity
graph over the molecules, and caches the resulting instance to a single
``.npz`` so the downstream bandit experiment is self-contained and
reproducible.

Default target is CHEMBL204 (thrombin / coagulation factor II), a
canonical QSAR target with thousands of measured ligands and broad
chemical diversity.

Cached arrays in the output ``.npz``:
    pIC50           (K,)                normalized to [0, 1]
    pIC50_raw       (K,)                raw pIC50 = 9 - log10(IC50_nM)
    A               (K, K)              k-NN adjacency (symmetric, 0/1)
    D               (K, K)              degree matrix (diagonal)
    chembl_ids      (K,) <U20           ChEMBL molecule identifiers
    smiles          (K,) <U400          canonical SMILES
    target          ()    <U20          target ChEMBL ID
    knn_k           ()    int           k used for k-NN graph
    fp_bits         ()    int           fingerprint size in bits
    fp_radius       ()    int           Morgan radius
    n_raw_records   ()    int           # raw IC50 records before dedup

Usage:
    python -m experiments.utils.chembl_loader \
        --target CHEMBL204 --top-k 200 --knn-k 5 \
        --out experiments/outputs/chembl_204_data.npz

This module is imported by ``experiments/chembl_1.py`` and the pre-flight
script in ``old/chembl_preflight.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def _fetch_activities(target_chembl_id, page_size=1000, max_pages=50):
    """Stream all IC50/'='/nM activities for a target via ChEMBL REST.

    Returns a list of dicts with keys: molecule_chembl_id, standard_value
    (nM), canonical_smiles.  Pagination is via the ``next`` field of the
    API's ``page_meta`` block.
    """
    import requests
    url = (f"{_CHEMBL_API}/activity.json"
           f"?target_chembl_id={target_chembl_id}"
           f"&standard_type=IC50"
           f"&standard_relation=%3D"          # URL-encoded "="
           f"&standard_units=nM"
           f"&limit={page_size}&offset=0")
    out = []
    for page in range(max_pages):
        t0 = time.time()
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        js = r.json()
        out.extend(js.get("activities", []))
        nxt = js.get("page_meta", {}).get("next")
        elapsed = time.time() - t0
        print(f"  [api] page {page+1}: +{len(js.get('activities', []))} "
              f"records ({elapsed:.1f}s, total={len(out)})", flush=True)
        if not nxt:
            break
        url = "https://www.ebi.ac.uk" + nxt
    return out


def _dedup_to_pic50(records):
    """Group records by ``molecule_chembl_id``, take median pIC50.

    pIC50 = 9 - log10(IC50_nM)  (so pIC50 = 9 means 1 nM, 6 means 1 uM, etc.)
    Discards records with missing/zero/negative IC50 or missing SMILES.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    smiles_of = {}
    for rec in records:
        mid = rec.get("molecule_chembl_id")
        sval = rec.get("standard_value")
        smi = rec.get("canonical_smiles")
        if not (mid and smi and sval):
            continue
        try:
            ic50_nM = float(sval)
        except (TypeError, ValueError):
            continue
        if ic50_nM <= 0:
            continue
        pic50 = 9.0 - np.log10(ic50_nM)
        buckets[mid].append(pic50)
        smiles_of[mid] = smi
    chembl_ids, pic50s, smis = [], [], []
    for mid, vals in buckets.items():
        chembl_ids.append(mid)
        pic50s.append(float(np.median(vals)))
        smis.append(smiles_of[mid])
    order = np.argsort(chembl_ids)  # deterministic ordering
    chembl_ids = [chembl_ids[i] for i in order]
    pic50s = [pic50s[i] for i in order]
    smis = [smis[i] for i in order]
    return chembl_ids, np.asarray(pic50s, dtype=float), smis


def _morgan_fps(smiles_list, radius=2, n_bits=2048):
    """ECFP4 (radius 2) Morgan fingerprints as RDKit ExplicitBitVect."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    valid = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(fp)
        valid.append(i)
    return fps, valid


def _tanimoto_matrix(fps):
    """Pairwise Tanimoto similarity matrix using BulkTanimotoSimilarity."""
    from rdkit import DataStructs
    n = len(fps)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        S[i, :] = sims
    return S


def _knn_graph_from_similarity(S, k):
    """Symmetric k-NN graph over similarity matrix S (diagonal ignored)."""
    n = S.shape[0]
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        sims = S[i].copy()
        sims[i] = -np.inf  # exclude self
        # top-k neighbors of i
        idx = np.argpartition(sims, -k)[-k:]
        A[i, idx] = 1.0
    A = np.maximum(A, A.T)  # symmetrize: i->j or j->i implies an edge
    np.fill_diagonal(A, 0.0)
    D = np.diag(A.sum(axis=1))
    return A, D


def build_instance(target='CHEMBL204', top_k=None, knn_k=5,
                   fp_radius=2, fp_bits=2048, normalize=True,
                   page_size=1000, max_pages=50, verbose=True):
    """End-to-end: fetch activities -> fingerprint -> k-NN Tanimoto graph.

    If ``top_k`` is given, restrict to the top-k most-active molecules
    (by raw pIC50) so the bandit instance has a manageable K.

    Returns a dict suitable for ``np.savez``.
    """
    if verbose:
        print(f"[chembl] fetching activities for {target}...", flush=True)
    records = _fetch_activities(target, page_size=page_size,
                                max_pages=max_pages)
    if verbose:
        print(f"  -> {len(records)} raw records", flush=True)
    if not records:
        raise RuntimeError(f"no IC50 records returned for {target}")

    chembl_ids, pic50_raw, smiles = _dedup_to_pic50(records)
    if verbose:
        print(f"[chembl] deduplicated to {len(chembl_ids)} molecules; "
              f"pIC50 range [{pic50_raw.min():.2f}, {pic50_raw.max():.2f}], "
              f"median {np.median(pic50_raw):.2f}", flush=True)

    if top_k is not None and top_k < len(chembl_ids):
        order = np.argsort(-pic50_raw)[:top_k]  # most active first
        chembl_ids = [chembl_ids[i] for i in order]
        pic50_raw = pic50_raw[order]
        smiles = [smiles[i] for i in order]
        if verbose:
            print(f"[chembl] kept top-{top_k} by pIC50; new range "
                  f"[{pic50_raw.min():.2f}, {pic50_raw.max():.2f}]",
                  flush=True)

    if verbose:
        print(f"[chembl] computing Morgan fingerprints "
              f"(radius={fp_radius}, bits={fp_bits})...", flush=True)
    fps, valid = _morgan_fps(smiles, radius=fp_radius, n_bits=fp_bits)
    if len(valid) < len(chembl_ids):
        if verbose:
            n_drop = len(chembl_ids) - len(valid)
            print(f"  dropped {n_drop} unparseable SMILES", flush=True)
        chembl_ids = [chembl_ids[i] for i in valid]
        pic50_raw = pic50_raw[valid]
        smiles = [smiles[i] for i in valid]

    if verbose:
        print(f"[chembl] {len(fps)} fingerprints; computing Tanimoto "
              f"matrix...", flush=True)
    S = _tanimoto_matrix(fps)
    if verbose:
        print(f"[chembl] building {knn_k}-NN graph...", flush=True)
    A, D = _knn_graph_from_similarity(S, knn_k)

    if normalize:
        # Normalize pIC50 to [0, 1] so the bandit setup is on the same
        # scale as the synthetic experiments.
        lo = float(pic50_raw.min())
        hi = float(pic50_raw.max())
        pic50 = (pic50_raw - lo) / max(hi - lo, 1e-9)
    else:
        pic50 = pic50_raw.copy()

    return dict(
        pIC50=pic50,
        pIC50_raw=pic50_raw,
        A=A,
        D=D,
        chembl_ids=np.asarray(chembl_ids, dtype='<U20'),
        smiles=np.asarray(smiles, dtype='<U400'),
        target=np.asarray(target, dtype='<U20'),
        knn_k=int(knn_k),
        fp_bits=int(fp_bits),
        fp_radius=int(fp_radius),
        n_raw_records=int(len(records)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='CHEMBL204',
                        help="ChEMBL target ID (default: thrombin)")
    parser.add_argument('--top-k', type=int, default=200,
                        help="restrict to top-k molecules by pIC50")
    parser.add_argument('--knn-k', type=int, default=5,
                        help="k-NN graph degree")
    parser.add_argument('--fp-radius', type=int, default=2)
    parser.add_argument('--fp-bits', type=int, default=2048)
    parser.add_argument('--out', type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.dirname(os.path.abspath(__file__))),
                            'outputs', 'chembl_204_data.npz'))
    parser.add_argument('--no-normalize', action='store_true')
    parser.add_argument('--max-pages', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = build_instance(target=args.target, top_k=args.top_k,
                             knn_k=args.knn_k,
                             fp_radius=args.fp_radius, fp_bits=args.fp_bits,
                             normalize=not args.no_normalize,
                             max_pages=args.max_pages)
    tmp = args.out + '.tmp.npz'
    np.savez(tmp, **payload)
    os.replace(tmp, args.out)
    K = payload['pIC50'].shape[0]
    n_edges = int(payload['A'].sum() / 2)
    deg = payload['A'].sum(axis=1)
    print(f"\nSaved {args.out}")
    print(f"  K = {K}, edges = {n_edges}, deg mean/max = "
          f"{deg.mean():.1f}/{int(deg.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
