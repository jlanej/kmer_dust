"""Paired-landmark tokens over a FracMinHash stream (the Shazam trick).

A single retained hash is order-blind: inside a satellite array the same hash
recurs thousands of times, so a match tells you almost nothing about WHERE.
Shazam's fix for audio is to hash PAIRS of landmarks together with the spacing
between them, so each token carries local context and becomes far more specific.

Here the stream is the ordered list of retained canonical-31-mer hashes along a
chromosome.  For anchor i we emit tokens pairing it with the next F hashes:

    token = splitmix64( h_i * C1 ^ h_j * C2 ^ quantized_gap )

Two choices of "gap" are compared, because this is the crux of the design:

  rank  -- gap measured in SKETCH space (j - i, the number of retained hashes
           between the two landmarks).  Invariant to any indel that does not
           create or destroy a retained hash, which is most of them.
  base  -- gap measured in BASE space (pos_j - pos_i), log-quantized.  Carries
           more information but an indel between the landmarks shifts it.

Also emits the degenerate F=0 case (single hashes) as the baseline to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

C1 = np.uint64(0x9E3779B97F4A7C15)
C2 = np.uint64(0xC2B2AE3D27D4EB4F)


def _mix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def log_bucket(gap: np.ndarray, base: float = 1.25) -> np.ndarray:
    """Log-quantize a base-space gap so small indels stay in the same bucket."""
    g = np.maximum(np.asarray(gap, dtype=np.float64), 1.0)
    return np.floor(np.log(g) / np.log(base)).astype(np.uint64)


def pair_tokens(
    pos: np.ndarray,
    hsh: np.ndarray,
    *,
    fanout: int = 3,
    gap_mode: str = "rank",
    max_base_gap: int = 20_000,
):
    """Return (token, anchor_pos) arrays.

    fanout=0 returns the single-hash baseline (token == hash).
    """
    pos = np.asarray(pos, dtype=np.int64)
    hsh = np.asarray(hsh, dtype=np.uint64)
    if fanout <= 0:
        return hsh.copy(), pos.copy()

    toks, anchors = [], []
    n = pos.shape[0]
    for d in range(1, fanout + 1):
        if n <= d:
            break
        hi, hj = hsh[:-d], hsh[d:]
        pi, pj = pos[:-d], pos[d:]
        gapb = pj - pi
        keep = gapb <= max_base_gap
        if gap_mode == "rank":
            q = np.full(hi.shape[0], d, dtype=np.uint64)
        elif gap_mode == "base":
            q = log_bucket(gapb)
        elif gap_mode == "none":
            q = np.zeros(hi.shape[0], dtype=np.uint64)
        else:
            raise ValueError(gap_mode)
        with np.errstate(over="ignore"):
            t = _mix(hi * C1) ^ _mix(hj * C2) ^ _mix(q + np.uint64(0x9E37))
        toks.append(t[keep])
        anchors.append(pi[keep])
    if not toks:
        return np.empty(0, np.uint64), np.empty(0, np.int64)
    t = np.concatenate(toks)
    a = np.concatenate(anchors)
    o = np.argsort(a, kind="stable")
    return t[o], a[o]


def load_stream(path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_parquet(path)
    return d["pos"].to_numpy(np.int64), d["hash"].to_numpy(np.uint64)


def specificity(tok: np.ndarray) -> dict:
    """How discriminative is this token vocabulary within one assembly?"""
    _, counts = np.unique(tok, return_counts=True)
    return {
        "n_tokens": int(tok.size),
        "n_distinct": int(counts.size),
        "frac_unique": float((counts == 1).mean()),
        "median_mult": float(np.median(counts)),
        "p99_mult": float(np.percentile(counts, 99)),
        "max_mult": int(counts.max()),
    }
