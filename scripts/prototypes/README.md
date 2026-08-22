# Prototypes

Throwaway scripts that produced the measurements in
[`docs/REPRESENTATION.md`](../../docs/REPRESENTATION.md). They are **not** part of
the package, are not imported by `kmer_dust`, and are not covered by the test
suite. They are here so the numbers in that document can be reproduced rather
than taken on trust.

Run them from the repository root with the project venv active.

| script | what it does |
| --- | --- |
| `possketch.py` | Re-sketches each assembly retaining **hash positions**. The stored shards sort on `(bin_idx, hash)` and so destroy contig order; every order-aware measurement needs this instead. Uses `hashing.sketch_contig(..., bin_size=1)`, which makes the returned "bin index" the exact base position — no new sketching code. |
| `pairtok.py` | Paired-landmark tokens over a positional stream (the audio-fingerprinting construction), plus the single-hash baseline. |
| `paircorr.py` | Same-label pair correlation `G(r)` — the crystallographic Patterson construction — for repeat-period recovery. |
| `mkviz.py` | Builds the JSON payload behind `docs/paired_landmark_explorer.html`. |

```bash
python scripts/prototypes/possketch.py results/chr21/manifest.tsv /tmp/pos
```

Then, with `/tmp/pos` on `sys.path`'s data side:

```bash
KD_RUN=results/chr21 KD_POS=/tmp/pos python scripts/prototypes/mkviz.py
```

`possketch.py` streams each chromosome from S3 by HTTP range request; chr21 for
24 assemblies takes roughly four minutes and about 70 MB on disk.

Environment overrides: `KD_RUN` (run directory, default `results/chr21`),
`KD_POS` (positional sketch directory, default `pos`), `KD_CACHE` (download
cache, default `data/cache`).
