"""Positioned FracMinHash sketches: (position, hash) in contig order.

The trick: kmer_dust.hashing.sketch_contig attributes each k-mer to
``start_position // bin_size``.  Call it with ``bin_size=1`` and the returned
"bin index" IS the exact 0-based position of the k-mer's first base.  Nothing
new to implement; the numba kernel does the work.

Output per assembly: <assembly>.pos.parquet with columns
    pos   int64   0-based position of the k-mer's first base on the contig
    hash  uint64  splitmix64 of the canonical 31-mer, below the scaled threshold
sorted by pos (i.e. in contig order, which the stored kmer-dust sketch destroys
by sorting on (bin_idx, hash)).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kmer_dust import hashing
from kmer_dust.fasta import FastaSource, load_chrom_alias

K = 31
SCALED = 200


def positioned_sketch(seq: bytes, k: int = K, scaled: int = SCALED):
    codes = hashing.encode_bases(seq)
    pos, hsh = hashing.sketch_contig(
        codes, k=k, bin_size=1, max_hash=hashing.max_hash_for_scaled(scaled)
    )
    return pos.astype(np.int64), hsh


def contig_for(src: FastaSource, alias_path: str, want: str = "chr21") -> str:
    """Resolve `want` to this assembly's own contig name via its chromAlias.

    HPRC contigs are named SAMPLE#HAP#ACCESSION and contain no 'chr21' anywhere,
    so name matching silently returns the wrong sequence -- the alias table is
    the only correct route.
    """
    names = set(src.contigs)
    if alias_path:
        alias = load_chrom_alias(alias_path, cache_dir=Path(os.environ.get("KD_CACHE", "data/cache")))
        hits = [c for c, norm in alias.items() if norm == want and c in names]
        if hits:
            return hits[0]
    if want in names:
        return want
    raise KeyError(f"{want} not resolvable among {len(names)} contigs")


def main(manifest_path: str, outdir: str, limit: int = 0, chrom: str = "chr21") -> None:
    man = pd.read_csv(manifest_path, sep="\t")
    if limit:
        man = man.head(limit)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    for _, row in man.iterrows():
        asm = row["assembly"]
        dest = out / f"{asm}.pos.parquet"
        if dest.exists():
            print(f"  {asm:42s} cached", flush=True)
            continue
        t0 = time.time()
        src = FastaSource(
            str(row["fasta"]),
            fai=str(row.get("fai") or ""),
            gzi=str(row.get("gzi") or ""),
        )
        try:
            name = contig_for(src, str(row.get("chrom_alias") or ""), chrom)
            seq = src.fetch(name)
            pos, hsh = positioned_sketch(seq)
            pd.DataFrame({"pos": pos, "hash": hsh}).to_parquet(dest, index=False)
            print(
                f"  {asm:42s} {name:28s} {len(seq)/1e6:6.1f} Mb  "
                f"{len(pos):7d} hashes  {time.time()-t0:5.1f}s",
                flush=True,
            )
        finally:
            src.close()


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    )
