"""Generate the JSON payload for the paired-landmark explorer."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pairtok as pt
import pandas as pd

R = Path(os.environ.get("KD_RUN", "results/chr21"))
POS = Path(os.environ.get("KD_POS", "pos"))
RNG = np.random.default_rng(7)

HAPS = [
    "HG00097_hap1_hprc_r2_v1.0.1",
    "HG02723_pat_hprc_r2_v1.0.1",
    "NA19468_hap1_hprc_r2_v1.0.1",
]
MAX_PTS = 22_000


def match(tokA, posA, tokB, posB, max_mult=40, cap=200):
    uA, cA = np.unique(tokA, return_counts=True)
    uB, cB = np.unique(tokB, return_counts=True)
    common = np.intersect1d(uA[cA <= max_mult], uB[cB <= max_mult])
    idxA, idxB = defaultdict(list), defaultdict(list)
    s = np.isin(tokA, common)
    for t, p in zip(tokA[s], posA[s]):
        idxA[t].append(p)
    s = np.isin(tokB, common)
    for t, p in zip(tokB[s], posB[s]):
        idxB[t].append(p)
    ap, off = [], []
    for t in common:
        A, B = idxA[t], idxB[t]
        if len(A) * len(B) > cap:
            continue
        for a in A:
            for b in B:
                ap.append(a)
                off.append(b - a)
    return np.asarray(ap, np.int64), np.asarray(off, np.int64)


def collinear_flags(ap, off, win=100_000, tol=10_000):
    flag = np.zeros(ap.size, bool)
    if ap.size == 0:
        return flag, 0.0
    for s in range(0, int(ap.max()) + 1, win):
        m = (ap >= s) & (ap < s + win)
        if m.sum() < 10:
            continue
        o = off[m]
        h, e = np.histogram(o, bins=np.arange(o.min(), o.max() + 2 * tol, tol))
        peak = e[h.argmax()]
        idx = np.flatnonzero(m)
        flag[idx[np.abs(o - peak) <= tol]] = True
    return flag, float(flag.mean())


def subsample(*arrays, n=MAX_PTS):
    k = arrays[0].size
    if k <= n:
        return arrays
    i = np.sort(RNG.choice(k, n, replace=False))
    return tuple(a[i] for a in arrays)


def main() -> None:
    rows = pd.read_parquet(R / "matrix/rows.parquet")
    ann = pd.read_parquet(R / "annotate/annotations.parquet")[["bin_uid", "dominant_feature"]]
    ref = rows[rows.assembly == "chm13v2.0"].merge(ann, on="bin_uid")
    ref = ref.sort_values("start")
    track = [
        {"s": int(r.start), "e": int(r.end), "f": str(r.dominant_feature or "")}
        for r in ref.itertuples()
    ]

    pa, ha = pt.load_stream(POS / "chm13v2.0.pos.parquet")
    feat = dict(zip(ref.start // 10000, ref.dominant_feature.fillna("")))

    # --- specificity by feature -------------------------------------------
    spec = {}
    for label, f, g in [("single", 0, "rank"), ("paired", 3, "rank")]:
        tok, anc = pt.pair_tokens(pa, ha, fanout=f, gap_mode=g)
        u, inv, cnt = np.unique(tok, return_inverse=True, return_counts=True)
        mult = cnt[inv]
        fv = np.array([feat.get(p // 10000, "") for p in anc])
        for name in np.unique(fv):
            if not name:
                continue
            s = mult[fv == name]
            if s.size < 200:
                continue
            spec.setdefault(name, {})[label] = {
                "n": int(s.size),
                "mean": round(float(s.mean()), 2),
                "median": round(float(np.median(s)), 1),
            }
    spec = {k: v for k, v in spec.items() if len(v) == 2}

    # --- correspondence per haplotype -------------------------------------
    out_haps = {}
    for hap in HAPS:
        p = POS / f"{hap}.pos.parquet"
        if not p.exists():
            continue
        pb, hb = pt.load_stream(p)
        reps = {}
        for label, f, g in [("single", 0, "rank"), ("paired", 3, "rank")]:
            tA, aA = pt.pair_tokens(pa, ha, fanout=f, gap_mode=g)
            tB, aB = pt.pair_tokens(pb, hb, fanout=f, gap_mode=g)
            ap, off = match(tA, aA, tB, aB)
            flag, prec = collinear_flags(ap, off)
            sap, soff, sfl = subsample(ap, off, flag)
            reps[label] = {
                "x": (sap // 1000).astype(int).tolist(),          # kb
                "y": np.clip(soff // 1000, -3000, 3000).astype(int).tolist(),
                "ok": sfl.astype(np.uint8).tolist(),
                "n_votes": int(ap.size),
                "precision": round(100 * prec, 1),
            }
            print(f"  {hap:34s} {label:7s} votes={ap.size:>8,} collinear={100*prec:.1f}%", flush=True)
        out_haps[hap] = reps

    payload = {
        "ref_len_kb": int(pa.max() // 1000),
        "track": track,
        "spec": spec,
        "haps": out_haps,
        "n_hashes_ref": int(ha.size),
    }
    Path("viz_data.json").write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote viz_data.json", Path("viz_data.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
