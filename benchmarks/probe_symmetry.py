"""Probe whether a cached filter operator supports the V-cycle preconditioner.

The V-cycle requires K = D*S to be symmetric (S in the PSD convention). This
reads a save_to_file() NPZ cache directly (no jax) and reports the relative
asymmetry of K, classified with the same two-tier gate as
implicit_filter.utils._vcycle: <= 1e-10 clean float64 roundoff, <= 1e-6
storage-precision roundoff (warn tier, e.g. float32-saved caches), above
that structural (the V-cycle setup refuses it).

Usage:
    python benchmarks/probe_symmetry.py CACHE.npz --family {tri|latlon}
"""
import argparse

import numpy as np
import scipy.sparse as sp


def probe(path, family):
    z = np.load(path)
    ss = np.asarray(z["_ss"], dtype=np.float64)
    ii = np.asarray(z["_ii"]).astype(np.int64)
    jj = np.asarray(z["_jj"]).astype(np.int64)
    area = np.asarray(z["_area"], dtype=np.float64)
    n = area.size
    if family == "latlon":
        ss = -ss                       # assembly is negative-semidefinite
        area = area ** 2               # tensor-product symmetrizing weight
    S = sp.csr_matrix((ss, (ii, jj)), shape=(n, n))
    K = sp.diags(area) @ S
    diff = (K - K.T).tocoo()
    denom = np.abs(K.data).max()
    rel = np.abs(diff.data).max() / denom if diff.nnz else 0.0
    verdict = _verdict(rel)
    print(f"{path} [{family}]: n={n} nnz={S.nnz} rel_asymmetry={rel:.3e} -> {verdict}")

    if family == "tri" and "_ss_e" in z.files:
        ss_e = np.asarray(z["_ss_e"], dtype=np.float64)
        ii_e = np.asarray(z["_ii_e"]).astype(np.int64)
        jj_e = np.asarray(z["_jj_e"]).astype(np.int64)
        ea = np.asarray(z["_elem_area"], dtype=np.float64)
        Se = sp.csr_matrix((ss_e, (ii_e, jj_e)), shape=(ea.size, ea.size))
        Ke = sp.diags(ea) @ Se
        de = (Ke - Ke.T).tocoo()
        rele = np.abs(de.data).max() / np.abs(Ke.data).max() if de.nnz else 0.0
        print(f"{path} [tri elements]: n={ea.size} nnz={Se.nnz} "
              f"rel_asymmetry={rele:.3e} -> {_verdict(rele)}")


def _verdict(rel):
    if rel <= 1e-10:
        return "OK (float64 roundoff)"
    if rel <= 1e-6:
        return "OK with warning (storage-precision roundoff)"
    return "STRUCTURALLY ASYMMETRIC (V-cycle setup refuses this operator)"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cache")
    p.add_argument("--family", choices=["tri", "latlon"], required=True)
    a = p.parse_args()
    probe(a.cache, a.family)
