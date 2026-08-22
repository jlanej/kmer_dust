# What a real run actually shows

A single exploration run, reproduced by:

```bash
kmer-dust run --config workflow/config/chr21.yaml -s manifest.max_samples=20
```

**Inputs.** HPRC release-2 haplotypes whose chr21 is a single gapless
sequence. The filters run in sequence: `require_annotations: [censat]` drops 2
of the 464 release-2 haplotypes, then `require_t2t_chrom` drops 424 of the
remaining 462, leaving **38** candidates across 35 samples — from which the
seeded, superpopulation-balanced sampler took **23 haplotypes across 20
samples**, plus **T2T-CHM13v2.0**. Nothing was downloaded whole: `pysam` range-requests each
assembly's chr21 straight out of the S3 bgzip file.

**Parameters.** k = 31, 10 kb bins, `scaled = 200` (≈50 hashes/bin), IDF + L2,
64 components, UMAP (n_neighbors 30, cosine), HDBSCAN (min_cluster_size 50).

## Cost

| stage | result | wall time |
| --- | --- | --- |
| manifest | 426 of 464 haplotypes dropped (2 no cenSat, then 424 not T2T-complete) | 9.6 s |
| **sketch** | 24 chromosomes streamed from S3 → 105,013 bins, 4.9 M hashes | **46 s** |
| select | 247,978 distinct → 224,683 (min_bins) → 222,676 (prevalence) → **200,349** | 1.4 s |
| matrix | 105,007 × 200,349, 4.39 M nnz, 0.021 % dense, 34 MiB | 2.6 s |
| decompose | randomized SVD, k = 64, σ₀ = 36.4 | **7.2 s** |
| embed | UMAP, 105,007 rows | **1088 s** |
| cluster | HDBSCAN → 926 clusters, 13.4 % noise | 23 s |
| annotate | 23 per-haplotype track sets + the T2T tracks (9.9 GB) | ~66 min |
| enrich + backprop + report | 23,175 enrichment tests, 25 BEDs, 9.5 MB HTML | 7 s |

Two of those are worth internalising. **The SVD is not the bottleneck — the
embedding is**, by a factor of 151, because `embed.deterministic: true` pins
UMAP to one thread. And **`annotate` moves more bytes than `sketch` does**: the
per-haplotype RepeatMasker BEDs average 414 MB each (measured; 9.9 GB for this
run, and ~192 GB if it were run across all 464 haplotypes).

## Result 1 — the map recovers synteny with no aligner

A cluster is typically *the same locus in every haplotype*:

| | observed | size-matched random baseline |
| --- | --- | --- |
| median positional spread of a cluster (IQR of chr21 coordinates) | **3.20 Mb** | 21.42 Mb |
| concentration factor | **6.7×** | 1× |
| clusters containing ≥80 % of the 24 assemblies | **96.0 %** | — |
| median assemblies per cluster (23 haplotypes + CHM13) | **24 of 24** | — |

The tightest single cluster has a positional IQR of 12.5 kb (its full span is
50 kb, which is also the smallest span in the run). Nothing in the pipeline ever saw a coordinate,
a chain file or an alignment.

> Measured on localised bins only. This run predates the placement fix, so every
> bin it contains sits on a whole `chrN` contig and `start` is a genuine
> chromosome coordinate. On a run that includes `chrN_*_random` contigs the same
> statistic must filter on `placed`, because those coordinates are contig-local —
> see the acrocentric section below.

## Result 2 — satellite identity survives back-propagation

`backprop/cluster_transfer.parquet` asks the question the whole design exists to
answer: name a cluster from **CHM13's** annotation, then check what the *HPRC*
bins in that same cluster are called by each assembly's own, independently
generated cenSat/RepeatMasker/segdup tracks.

Over 289 testable clusters (≥5 reference bins, ≥20 assembly bins, and a
reference top feature inside the shared vocabulary):

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
| `ct` (centromeric transition) | 99 | 0.10 | 41 % |

The split is exactly what the biology predicts. **Satellite arrays are
vocabulary-defined**, so a cluster built purely from shared k-mers *is* the
satellite family, and the label transfers essentially perfectly. **Euchromatin
is a mosaic** of LINE/SINE/LTR fragments: the cluster's *modal* label still
agrees 84–94 % of the time, but no single class dominates any individual bin, so
per-bin agreement is necessarily low. `ct` is the honest floor — "somewhere in
the centromeric transition" is not a vocabulary, and it does not transfer.

A selection of the strongest enrichments (log₂, hypergeometric). Two others
rank higher and are omitted only because they are less interpretable —
C826 `low_complexity` at 8.84 and C177 `retroposon` at 8.62:

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


---

# The acrocentric run: two opposite predictions, both held

A larger run designed so it could fail. 33 assemblies (32 HPRC release-2
haplotypes + T2T-CHM13v2.0) across chr13/14/15/21/22 — **1,303,159 bins**,
61.6 M hashes, 1.77 M selected k-mers.

The acrocentrics force the method to commit to two predictions that point in
**opposite** directions, so a method that trivially clusters by chromosome fails
one and a method that ignores chromosome fails the other.

## P1 — alpha-satellite HOR clusters should be chromosome-*pure*

Each chromosome carries its own higher-order repeat variant. Over 193 asat
clusters with ≥20 bins (29,151 bins):

| | |
| --- | --- |
| median chromosome purity | **1.000** |
| clusters ≥0.9 pure | **81 %** |
| median distinct chromosomes per cluster | **1** |

And the exception is the one the literature predicts. chr13/chr21 share a
suprachromosomal family and are near-indistinguishable; so do chr14/chr22. Of
the 36 impure clusters:

| chromosome pair | clusters | |
| --- | --- | --- |
| chr14 + chr22 | 27 | **predicted** |
| chr13 + chr21 | 4 | **predicted** |
| chr14 + chr21 | 3 | |
| chr13 + chr14 | 1 | |
| chr13 + chr15 | 1 | |

**86 % of all the mixing is the two predicted pairs.**

## P2 — rDNA clusters should be chromosome-*mixed*

The rDNA unit is the same on all five acrocentric short arms. Over 43 rDNA
clusters with ≥20 bins (6,744 bins):

| | |
| --- | --- |
| median chromosome purity | **0.268** (0.200 would be perfectly even across five) |
| clusters ≥0.9 pure | **0 %** |
| median distinct chromosomes per cluster | **5 of 5** |

Same pipeline, same run, opposite behaviour — decided entirely by whether the
underlying sequence family is chromosome-specific.

## The unlocalised material is not an island

**1,061,514 of the 1,303,159 bins (81.5 %)** come from `chrN_*_random` contigs
with no chromosome coordinates; of the 799,448 bins that were clustered at all,
649,671 (81.3 %) are unlocalised. **Only 14 of the 3,021 clusters (0.46 %) are
made entirely of unlocalised bins; 99.54 % mix localised and unlocalised** — so
this material integrates with the placed sequence rather than forming its own
compartment of assembly artefacts.

## Assigning a chromosome to an unplaced contig — and knowing when not to

Let the cluster's chromosome be decided by its *localised* members only, then
predict the chromosome of each unlocalised bin. Its own `chrN_*_random` alias is
the held-out answer (648,363 bins):

| | bins | accuracy |
| --- | --- | --- |
| all | 648,363 | 34.6 % |
| cluster chromosome purity ≥0.8 | 44,597 | **93.4 %** |
| cluster chromosome purity ≥0.95 | 36,335 | **96.9 %** |
| satellite bins | 67,392 | 62.8 % |

The honest reading: chromosome assignment works **where chromosome-specific
vocabulary exists** — alpha-satellite HOR — and fails where sequence is shared
between chromosomes, which is most of the genome. The purity of the cluster's
localised vote separates the two cases cleanly, so this is a "knows when it
knows" result rather than a general contig-placement method. It covers 7 % of
unlocalised bins at 93 %+ accuracy.

## Cost, and what it should cost now

| stage | this run | after the fixes below |
| --- | --- | --- |
| sketch | 702 s | 702 s |
| select | 41 s | 41 s |
| matrix | 54 s | 54 s |
| decompose | 106 s | 106 s |
| embed | 240 s | 240 s |
| **cluster** | **4,819 s** | **~5 s** (`fast_hdbscan`) |
| **annotate** | **2,484 s** | **~60 s** (reference-only is the default) |
| **total** | **2 h 24 m** | **~20 min** |

Two things dominated this run and neither had to. scikit-learn's HDBSCAN has no
Boruvka MST, so it is O(n²) even on a 2-D embedding — `fast_hdbscan` does the
same clustering ~1,025× faster -- closely agreeing rather than identical
(ARI 0.83). And per-assembly annotation was never an
input to the method, only a way to check it; annotating the reference alone is
now the default.
