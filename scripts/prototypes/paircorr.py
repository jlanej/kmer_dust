"""Same-label pair correlation G(r) over a positioned FracMinHash stream.

The crystallography import: treat retained hashes as a labelled point process on
the chromosome and take its second-order statistic.  For a window W,

    G(r) = #{ (i,j) : h_i == h_j , p_j - p_i == r }

Peaks of G sit at the recurring distances of the structure -- i.e. at the repeat
unit length -- with NO monomer library, NO period prior and NO alignment.  This
is Patterson's 1934 trick: the autocorrelation of a structure is measurable even
when the absolute phase (here, the array's offset, which indels destroy) is not.

Requires the MULTISET of hashes.  kmer-dust's shards collapse duplicates within
a bin (sketch.py:311-317), which deletes 74% of the instances in an active alpha
satellite HOR array, so this must run off positioned sketches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pair_correlation(pos: np.ndarray, hsh: np.ndarray, rmax: int = 40_000,
                     rmin: int = 30) -> np.ndarray:
    """Histogram of same-hash lags, 1 bp resolution, index r == lag."""
    G = np.zeros(rmax + 1, dtype=np.int64)
    order = np.argsort(hsh, kind="stable")
    h, p = hsh[order], pos[order]
    edges = np.flatnonzero(np.r_[True, h[1:] != h[:-1], True])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2:
            continue
        q = np.sort(p[a:b])
        for i in range(q.size - 1):
            d = q[i + 1:] - q[i]
            d = d[(d >= rmin) & (d <= rmax)]
            if d.size:
                np.add.at(G, d, 1)
    return G


def smooth(G: np.ndarray, w: int = 5) -> np.ndarray:
    k = np.ones(w) / w
    return np.convolve(G, k, mode="same")


def top_peaks(G: np.ndarray, n: int = 6, sep: int = 150, rmin: int = 30):
    """Local maxima of G, greedily separated so one peak is not reported twice."""
    S = smooth(G)
    S[:rmin] = 0
    out = []
    work = S.copy()
    for _ in range(n):
        r = int(work.argmax())
        if work[r] <= 0:
            break
        out.append((r, float(S[r]), int(G[max(0, r - 2):r + 3].sum())))
        work[max(0, r - sep):r + sep + 1] = 0
    return out


def fundamental(peaks, tol: float = 0.06):
    """Smallest lag of which the strong peaks are near-integer multiples."""
    if not peaks:
        return None
    cands = sorted(p[0] for p in peaks)
    for c in cands:
        if c < 60:
            continue
        ok = sum(
            1 for r, _, _ in peaks
            if abs(r / c - round(r / c)) < tol and round(r / c) >= 1
        )
        if ok >= max(2, len(peaks) // 2):
            return c
    return cands[0]


def windows_by_feature(rows_path: str, ann_path: str, assembly: str):
    rows = pd.read_parquet(rows_path)
    ann = pd.read_parquet(ann_path)[["bin_uid", "dominant_feature"]]
    d = rows[rows.assembly == assembly].merge(ann, on="bin_uid")
    d = d[d.dominant_feature.notna() & (d.dominant_feature != "")]
    return d[["start", "end", "dominant_feature"]].sort_values("start")
