# What a real run actually shows

A single exploration run, reproduced by:

```bash
kmer-dust run --config workflow/config/chr21.yaml -s manifest.max_samples=20
```

**Inputs.** HPRC release-2 haplotypes whose chr21 is a single gapless
sequence (`manifest.require_t2t_chrom: true`) — that filter drops 424 of the
464 release-2 haplotypes, leaving 40 candidates, from which the seeded,
superpopulation-balanced sampler took **23 haplotypes across 20 samples**, plus
**T2T-CHM13v2.0**. Nothing was downloaded whole: `pysam` range-requests each
assembly's chr21 straight out of the S3 bgzip file.

**Parameters.** k = 31, 10 kb bins, `scaled = 200` (≈50 hashes/bin), IDF + L2,
64 components, UMAP (n_neighbors 30, cosine), HDBSCAN (min_cluster_size 50).

## Cost

| stage | result | wall time |
| --- | --- | --- |
| manifest | 424/464 haplotypes dropped by the T2T-completeness filter | 9.6 s |
| **sketch** | 24 chromosomes streamed from S3 → 105,013 bins, 4.9 M hashes | **46 s** |
| select | 247,978 distinct → 224,683 (min_bins) → 222,676 (prevalence) → **200,349** | 1.4 s |
| matrix | 105,007 × 200,349, 4.39 M nnz, 0.021 % dense, 34 MiB | 2.6 s |
| decompose | randomized SVD, k = 64, σ₀ = 36.4 | **6.5 s** |
| embed | UMAP, 105,007 rows | **1084 s** |
| cluster | HDBSCAN → 926 clusters, 13.4 % noise | 23 s |
| annotate | 24 per-haplotype track sets + the T2T tracks (~6 GB) | ~65 min |
| enrich + backprop + report | 23,175 enrichment tests, 25 BEDs, 9.5 MB HTML | 7 s |

Two of those are worth internalising. **The SVD is not the bottleneck — the
embedding is**, by a factor of 160, because `embed.deterministic: true` pins
UMAP to one thread. And **`annotate` moves more bytes than `sketch` does**: the
per-haplotype RepeatMasker BEDs average 167 MB each.

## Result 1 — the map recovers synteny with no aligner

A cluster is typically *the same locus in every haplotype*:

| | observed | size-matched random baseline |
| --- | --- | --- |
| median positional spread of a cluster (IQR of chr21 coordinates) | **3.20 Mb** | 21.6 Mb |
| concentration factor | **6.7×** | 1× |
| clusters containing ≥80 % of the 24 haplotypes | **96.0 %** | — |
| median haplotypes per cluster | **24 of 24** | — |

The tightest clusters span 12 kb. Nothing in the pipeline ever saw a coordinate,
a chain file or an alignment.

## Result 2 — satellite identity survives back-propagation

`backprop/cluster_transfer.parquet` asks the question the whole design exists to
answer: name a cluster from **CHM13's** annotation, then check what the *HPRC*
bins in that same cluster are called by each assembly's own, independently
generated cenSat/RepeatMasker/segdup tracks.

Over 288 testable clusters (≥5 reference bins, ≥20 assembly bins):

| reference top feature | clusters | median per-bin agreement | top-feature match |
| --- | --- | --- | --- |
| `hsat1a` | 7 | **1.00** | **100 %** |
| `bsat` | 8 | **0.97** | **100 %** |
| `hsat3` | 8 | **0.96** | **100 %** |
| `asat_mon` | 4 | 0.75 | **100 %** |
| `rdna` | 13 | 0.58 | 62 % |
| `segdup` | 8 | 0.42 | 62 % |
| `sine` | 17 | 0.37 | 94 % |
| `line` | 93 | 0.31 | 84 % |
| `ltr` | 22 | 0.27 | 86 % |
| `ct` (centromeric transition) | 98 | 0.12 | 42 % |

The split is exactly what the biology predicts. **Satellite arrays are
vocabulary-defined**, so a cluster built purely from shared k-mers *is* the
satellite family, and the label transfers essentially perfectly. **Euchromatin
is a mosaic** of LINE/SINE/LTR fragments: the cluster's *modal* label still
agrees 84–94 % of the time, but no single class dominates any individual bin, so
per-bin agreement is necessarily low. `ct` is the honest floor — "somewhere in
the centromeric transition" is not a vocabulary, and it does not transfer.

Strongest individual enrichments (log₂, hypergeometric):

| cluster | feature | bins | fraction of cluster | log₂ enrichment | −log₁₀ p |
| --- | --- | --- | --- | --- | --- |
| C4 | `subterminal` | 133 | 0.81 | 9.09 | 382 |
| C865 | `gsat` | 51 | 0.59 | 8.63 | 124 |
| C319 | `hsat1b` | 52 | **1.00** | 8.58 | 137 |
| C620 | `rrna` | 159 | 0.86 | 8.33 | 403 |
| C535 | `asat_hor` | 76 | 0.85 | 7.33 | 160 |

## Choosing the clustering granularity

`cluster.min_cluster_size` trades cluster count against noise, and there is no
universally right answer -- the fine setting is a map of *loci*, the coarse one
a map of *feature classes*. Re-running only the `cluster` stage takes 25 s
because the embedding is already on disk, so sweep it:

| `min_cluster_size` | `min_samples` | clusters | noise |
| --- | --- | --- | --- |
| 50 | 10 | 926 | 13.4 % |
| 300 | 10 | 92 | 35.8 % |
| 400 | 50 | 63 | 44.4 % |
| 500 | 10 | 54 | 40.9 % |
| 800 | 50 | 28 | 53.8 % |

The results above use 50/10. Note that `cluster_selection_epsilon > 0` -- the
obvious knob for merging neighbouring clusters -- trips an upstream bug in
scikit-learn's Cython condensed-tree traversal (`traverse_upwards` raises a bare
`TypeError: only 0-dimensional arrays can be converted to Python scalars`) that
fires only at scale. `cluster.py` catches it and re-raises with the diagnosis;
use `min_cluster_size`, or `cluster.method: dbscan`, instead.

## A methodological trap worth knowing about

The first version of this run reported a median transfer agreement of **0.000**,
and the clustering was not at fault. CHM13 has a gene annotation and a telomere
track; the HPRC per-assembly track set does not. So every euchromatic
*reference* bin was labelled `gene` while the identical locus in an *assembly*
was labelled `line` — a disagreement manufactured entirely by asymmetric track
availability.

`cluster_transfer_report` now recomputes both sides' dominant feature over the
**intersection** of what each track set could possibly produce, and writes that
vocabulary to `backprop/transfer_features.json`. Median agreement went from
0.000 to 0.32, and the satellite numbers above appeared. If you add or remove a
track, read that JSON before reading the score.

## Reproducing

Everything above comes from `results/chr21/`. The run is deterministic given the
config: same seed, same manifest, same clusters. `config.resolved.yaml` in the
output directory is the exact input.
