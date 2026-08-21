# Configuration reference

One YAML file describes a run completely. Unknown keys are a hard error, so a
typo can never silently change nothing. The resolved config is copied to
`<outdir>/config.resolved.yaml` at the start of every run.

```yaml
run_name: chr21              # goes into the report title and BED track names
outdir: results/chr21        # every stage writes under here
datadir: data                # catalogs and downloads are cached in data/cache
threads: 8                   # default parallelism
seed: 7                      # propagates to every stage that has its own seed
```

---

## `manifest:` — which assemblies take part

| key | default | meaning |
| --- | --- | --- |
| `source` | `hprc_release2` | `hprc_release2` resolves the live HPRC index; `file` reads a TSV you wrote; `local_dir` globs FASTAs in a directory |
| `path` | `""` | the TSV or directory when `source` is not `hprc_release2` |
| `chroms` | `[chr21]` | which chromosomes to bin. `[]` means every contig |
| `max_samples` | `0` | cap on distinct samples (`0` = no cap). Sampling is seeded and balances superpopulations |
| `max_assemblies` | `0` | cap on haplotype assemblies |
| `require_t2t_chrom` | `false` | keep only haplotypes where each requested chromosome is a single gapless sequence. Costs one small HTTP GET per assembly, cached |
| `require_annotations` | `[censat]` | drop assemblies missing these per-assembly tracks |
| `include_reference` | `true` | add T2T-CHM13v2.0 as its own assembly — this is what makes cluster naming possible |
| `populations` | `[]` | restrict to these 1000G-style population codes |
| `samples` / `exclude_samples` | `[]` | explicit allow/deny lists |

> **Why prevalence is computed over samples, not haplotypes.** The two
> haplotypes of one individual share most of their k-mer vocabulary, so counting
> haplotypes would let a single person's private variation look like a
> population-level signal.

## `sketch:` — tiling and FracMinHash

| key | default | meaning |
| --- | --- | --- |
| `k` | `31` | k-mer length. Must be odd (so a canonical k-mer is unambiguous) and ≤ 31 (so it fits in 64 bits) |
| `bin_size` | `10000` | bin width in bp. A k-mer belongs to the bin containing its **first** base |
| `scaled` | `200` | keep ~1/`scaled` of k-mer space. Expected sketch size per bin ≈ `bin_size / scaled` |
| `min_bin_acgt_frac` | `0.5` | drop bins that are mostly `N` |
| `min_bin_sketch` | `5` | drop bins with too few retained hashes to be comparable |
| `include_unplaced` | `false` | keep contigs with no chromosome assignment (`chrom` becomes `""`) |
| `drop_partial_terminal_bin` | `true` | drop the ragged last bin of each contig so every row covers the same span |
| `threads` | `4` | assemblies sketched concurrently |

`scaled` is the main cost/resolution dial. Rough numbers per haplotype genome
(3.1 Gb): `scaled=1000` → ~3.1 M hashes, `scaled=200` → ~15 M, `scaled=20` → ~155 M.

## `select:` — choosing the k-mer columns

| key | default | meaning |
| --- | --- | --- |
| `min_sample_prevalence` | `0.10` | drop k-mers seen in fewer than this fraction of samples (assembly noise, private variation) |
| `max_sample_prevalence` | `1.0` | drop k-mers above this sample prevalence. Defaults to keeping everything — see the note below |
| `min_bins` | `2` | drop k-mers seen in a single bin genome-wide |
| `max_features` | `200000` | cap on matrix columns; `0` keeps everything that passed |
| `n_buckets` | `16` | power-of-two radix partitions for out-of-core counting. Raise it when a bucket no longer fits in RAM |

> **Why the upper prevalence bound defaults to 1.0.** A k-mer shared by every
> *sample* is not a k-mer shared by every *bin*. An HSat2 31-mer occurs in all
> 232 samples and in ~0.1 % of bins — it is one of the most informative columns
> in the matrix, and cutting it would delete exactly the repeat-family signal the
> clustering is supposed to find. Bin-level ubiquity is handled by IDF weighting
> in `matrix:`. Lower this only if you deliberately want to suppress the shared
> core vocabulary and look at polymorphic content alone.

Sub-sampling to `max_features` is a *second* FracMinHash
(`splitmix64(hash ^ seed)` below a threshold), so it is order-independent: the
same k-mers are chosen no matter what order the shards were processed in.

## `matrix:`

| key | default | meaning |
| --- | --- | --- |
| `weighting` | `idf` | `none`, `idf` (`log(n_rows/df)`), or `log` |
| `row_norm` | `l2` | `none`, `l1`, or `l2`. L2 + IDF makes the SVD a latent semantic analysis |
| `drop_empty_rows` | `true` | bins that retained no selected k-mer are removed and `row_idx` renumbered |

## `decompose:`

| key | default | meaning |
| --- | --- | --- |
| `n_components` | `64` | truncated SVD rank. Clamped to `min(shape) - 1` |
| `n_oversamples` | `20` | randomized-SVD oversampling |
| `n_iter` | `7` | power iterations. Raise for a slowly-decaying spectrum |
| `drop_first` | `0` | discard leading components. Component 1 often just encodes sketch depth; set to `1` if the UMAP looks like a coverage gradient |
| `keep_components` | `true` | write `components.npy` so you can ask which k-mers drive an axis |

## `embed:`

| key | default | meaning |
| --- | --- | --- |
| `n_neighbors` | `30` | UMAP locality. Small = fine structure, large = global geography |
| `min_dist` | `0.05` | how tightly points may pack |
| `metric` | `cosine` | cosine is right for L2-normalised vocabulary vectors |
| `n_components` | `2` | `2` or `3` |
| `max_fit_rows` | `400000` | fit on a seeded subsample, transform the rest |
| `deterministic` | `true` | keep UMAP's `random_state`, which pins it to **one thread** |
| `n_jobs` | `0` | cores when `deterministic: false` (`0` = use `threads`) |

> **The embedding is the slow stage.** Setting UMAP's `random_state` forces it
> and pynndescent onto a single thread — the parallel paths race on the
> negative-sample RNG. At 10⁵–10⁶ bins that dominates the whole run. Setting
> `deterministic: false` hands UMAP `n_jobs` cores; the *layout* then changes
> between runs while the *structure* does not. Right for exploration, wrong for
> a figure you intend to publish. Whichever was used is recorded in
> `embed/umap_params.json`.

## `cluster:`

| key | default | meaning |
| --- | --- | --- |
| `method` | `hdbscan` | `hdbscan` or `dbscan` |
| `space` | `embedding` | cluster the UMAP coordinates or the PCs directly |
| `min_cluster_size` | `50` | smallest thing you are willing to call a cluster |
| `min_samples` | `10` | conservativeness; higher = more noise |
| `cluster_selection_epsilon` | `0.0` | merge clusters closer than this |
| `eps` | `0.5` | DBSCAN only |

If the run comes back >90 % noise, lower `min_samples` before anything else.

## `annotate:`

| key | default | meaning |
| --- | --- | --- |
| `reference_tracks` | `[censat, repeatmasker, segdup, telomere, gene]` | T2T-CHM13 tracks to overlay |
| `assembly_tracks` | `[censat, repeatmasker, segdup]` | per-assembly tracks from the HPRC annotation index |
| `annotate_assemblies` | `true` | set `false` for a reference-only quick look |

> **`repeatmasker` is the expensive per-assembly track.** Measured on a real
> run: the HPRC per-haplotype RepeatMasker BEDs average **167 MB** each
> (cenSat ~250 kB, segdups ~5 MB), so annotating 24 haplotypes moves ~6 GB and
> all 464 would move ~70 GB. The stage prefetches them concurrently (`threads`,
> capped at 8) and caches the *parsed* intervals under `datadir/cache/tracks/`,
> so you pay once — but on a cluster with no compute-node egress, warm that
> cache from the login node first, or set `assembly_tracks: [censat, segdup]`
> and keep RepeatMasker for the reference only. The reference tracks are a
> fixed ~350 MB regardless of run size.
| `min_frac_for_dominant` | `0.25` | a bin's `dominant_feature` needs at least this covered fraction |

## `enrich:`

| key | default | meaning |
| --- | --- | --- |
| `min_cluster_size` | `10` | clusters smaller than this are not tested |
| `min_frac` | `0.25` | a bin "carries" a feature above this covered fraction |
| `top_features` | `3` | how many features to list per cluster name |

## `report:`

| key | default | meaning |
| --- | --- | --- |
| `max_points` | `300000` | scatter is subsampled above this, and the report says so |
| `title` / `subtitle` | `""` | override the generated header |
| `embed_plotlyjs` | `true` | inline plotly.js so the HTML works with no network |
| `point_size` | `3.0` | marker size |
