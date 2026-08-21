# kmer-dust

**Cluster the human genome without ever aligning it.**

`kmer-dust` tiles every high-quality [HPRC release-2](https://humanpangenome.org/)
haplotype assembly and [T2T-CHM13v2.0](https://github.com/marbl/CHM13) into fixed
10 kb bins, represents each bin by a *scaled MinHash* sketch of its canonical
31-mers, stacks those sketches into one enormous sparse **bin × k-mer** matrix,
and then does the obvious thing to it: randomized SVD → UMAP → HDBSCAN.

The result is a map of the genome in which position is decided purely by k-mer
vocabulary. No aligner, no reference coordinate, no orthology. Bins land next to
each other because they *use the same words*, and it turns out that alpha-satellite
HOR arrays, HSat2/3 blocks, rDNA, segmental duplications, subtelomeres and plain
unique euchromatin each speak a recognisable dialect.

Because T2T-CHM13 is one of the assemblies in the matrix, every cluster can be
handed a name from the reference's own annotation tracks (cenSat, RepeatMasker,
segmental duplications) — and then those names can be **back-propagated onto all
464 HPRC haplotypes**, including into regions where no reference annotation
exists at all.

[![ci](https://github.com/jlanej/kmer_dust/actions/workflows/ci.yml/badge.svg)](https://github.com/jlanej/kmer_dust/actions/workflows/ci.yml)
[![docker](https://github.com/jlanej/kmer_dust/actions/workflows/docker.yml/badge.svg)](https://github.com/jlanej/kmer_dust/actions/workflows/docker.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of contents

- [What it actually computes](#what-it-actually-computes)
- [Does it work?](#does-it-work)
- [Quick start](#quick-start)
- [Real test data](#real-test-data)
- [Configuration](#configuration)
- [Running on an HPC cluster](#running-on-an-hpc-cluster)
- [Outputs](#outputs)
- [Design notes](#design-notes)
- [Development](#development)

---

## What it actually computes

```
              assemblies (464 HPRC haplotypes + CHM13v2.0)
                                │
   ┌────────────────────────────▼────────────────────────────┐
   │ 1. sketch    tile into 10 kb bins; for every canonical   │
   │              31-mer keep it iff splitmix64(kmer) <       │
   │              2^64 / scaled     ← FracMinHash             │
   └────────────────────────────┬────────────────────────────┘
                                │  bins.parquet + sketch.parquet, per assembly
   ┌────────────────────────────▼────────────────────────────┐
   │ 2. select    count, per hash, how many *samples* carry   │
   │              it; keep the middle of the prevalence       │
   │              spectrum, then sub-sample deterministically │
   └────────────────────────────┬────────────────────────────┘
                                │  kmers.parquet  (the feature space)
   ┌────────────────────────────▼────────────────────────────┐
   │ 3. matrix    sparse binary bin × k-mer, IDF-weighted,    │
   │              L2 row-normalised                           │
   └────────────────────────────┬────────────────────────────┘
                                │  matrix.npz + rows.parquet
   ┌────────────────────────────▼────────────────────────────┐
   │ 4. decompose randomized SVD  →  5. embed UMAP            │
   │              →  6. cluster HDBSCAN                       │
   └────────────────────────────┬────────────────────────────┘
                                │  pcs.npy, umap.npy, clusters.parquet
   ┌────────────────────────────▼────────────────────────────┐
   │ 7. annotate  cenSat / RepeatMasker / segdup / telomere / │
   │              gene coverage per bin (reference AND every  │
   │              HPRC assembly, which have their own tracks) │
   │ 8. enrich    cluster × feature log2 enrichment + names   │
   │ 9. backprop  per-assembly BED9 + label-transfer report   │
   │ 10. report   one self-contained interactive HTML         │
   └─────────────────────────────────────────────────────────┘
```

The k-mer selection step is the interesting one. A k-mer present in **one**
sample is mostly assembly noise, so a floor on *sample* prevalence (default: 10 %
of samples) throws it away. The ceiling defaults to keeping everything, because a
k-mer shared by every sample is not the same thing as a k-mer shared by every
*bin* — an HSat2 31-mer occurs in all 232 samples and in 0.1 % of bins, which
makes it one of the most informative columns in the whole matrix. Bin-level
ubiquity is handled downstream by IDF weighting, not by throwing columns away.

---

## Quick start

```bash
pip install -e '.[dev]'
```

Grab a few megabases of **real** sequence from real assemblies, then run the
whole thing:

```bash
kmer-dust fetch --dest data/testdata --samples 6 --chrom chr21 --span-mb 8
```

```bash
kmer-dust run --config workflow/config/smoke.yaml
```

Open `results/smoke/report/kmer_dust_report.html`.

To re-tune only the tail of the pipeline without re-sketching anything:

```bash
kmer-dust run --config workflow/config/smoke.yaml --force-from embed -s embed.n_neighbors=15
```

Any config value can be overridden from the command line with `-s section.key=value`.

---

## Real test data

`kmer-dust fetch` does not ship a synthetic toy. It uses `pysam` to issue HTTP
range requests against the **actual** HPRC release-2 bgzip assemblies (which
publish `.fa.gz.fai` and `.fa.gz.gzi` next to them) and against the uncompressed
T2T-CHM13v2.0 FASTA, pulls out the requested slice of the requested chromosome,
and re-bgzips it locally with its own index. It slices the matching cenSat,
RepeatMasker and segmental-duplication BED records for the same interval and
rewrites their coordinates into the slice's frame.

The default download is well under 150 MB and is what CI runs against, so the
smoke test exercises the same code paths as a 464-haplotype run.

---

## Configuration

One YAML file describes a run completely; it is copied into the output directory
as `config.resolved.yaml`. Three presets ship in `workflow/config/`:

| preset | scope | rough cost |
| --- | --- | --- |
| `smoke.yaml` | the fetched test slices | seconds, laptop |
| `chr21.yaml` | chr21 of every release-2 haplotype | ~1 h, one node |
| `full.yaml` | all autosomes, all haplotypes | HPC, Slurm array |

See `docs/CONFIG.md` for every knob.

---

## Running on an HPC cluster

The image is built by GitHub Actions on every commit and published to GHCR.

```bash
bash hpc/build_sif.sh ghcr.io/OWNER/kmer-dust:main kmer-dust.sif
```

```bash
sbatch hpc/submit.sbatch workflow/config/full.yaml
```

That submits a Snakemake controller job which fans the `sketch` stage out over
one Slurm task per assembly (`--use-apptainer`, `snakemake-executor-plugin-slurm`),
then walks the rest of the DAG. `hpc/run_stage.sbatch` runs a single stage if you
would rather drive it yourself. `hpc/README.md` has the site-specific details.

---

## Outputs

Everything lands under `outdir/` (see the module docstring of
[`schemas.py`](src/kmer_dust/schemas.py) for the exact layout and column
contracts):

| file | what it is |
| --- | --- |
| `report/kmer_dust_report.html` | the interactive map, linked genome ribbon, enrichment heatmap |
| `cluster/clusters.parquet` | cluster label + membership probability per bin |
| `enrich/cluster_names.parquet` | each cluster's inferred identity and purity |
| `backprop/<assembly>.clusters.bed` | load straight into IGV or the UCSC browser |
| `backprop/cluster_transfer.parquet` | does a reference-named cluster keep its meaning in the assemblies? |
| `matrix/matrix.npz`, `decompose/pcs.npy`, `embed/umap.npy` | the intermediate spaces, for your own analysis |

---

## Does it work?

Yes, and [`docs/RESULTS.md`](docs/RESULTS.md) has the numbers from a real run —
23 HPRC release-2 haplotypes with a gapless chr21, plus CHM13, 105,007 bins.

**The map recovers synteny without an aligner.** A cluster is typically the same
locus in every haplotype: median positional spread 3.20 Mb against a
size-matched random baseline of 21.6 Mb (**6.7× concentration**), with 96 % of
clusters containing at least 80 % of the haplotypes and a median of **24 of 24**.
The tightest span 12 kb. Nothing in the pipeline ever saw a coordinate.

**Satellite identity survives back-propagation.** Name a cluster from CHM13's
annotation, then ask what each HPRC assembly's own independently-generated
tracks call the assembly bins in that same cluster:

| reference top feature | clusters | median per-bin agreement | top-feature match |
| --- | --- | --- | --- |
| `hsat1a` | 7 | **1.00** | **100 %** |
| `bsat` | 8 | **0.97** | **100 %** |
| `hsat3` | 8 | **0.96** | **100 %** |
| `asat_mon` | 4 | 0.75 | **100 %** |
| `line` | 93 | 0.31 | 84 % |
| `ltr` | 22 | 0.27 | 86 % |

That split is the biology, not a bug: satellite arrays *are* defined by their
vocabulary, so a k-mer cluster is the satellite family. Euchromatin is a mosaic
of LINE/SINE/LTR fragments, so the cluster's modal label still agrees 84–94 % of
the time while no single class owns any individual bin.

Cost, on one laptop: sketching 24 chromosomes straight out of S3 took **46 s**;
the randomized SVD took **6.5 s**; the UMAP took **18 minutes**, because
reproducibility pins it to a single thread.

---

## Design notes

**Why FracMinHash rather than a fixed k-mer list?** The threshold
`splitmix64(kmer) ≤ 2⁶⁴/scaled` is a *consistent* random sample of k-mer space:
every assembly independently keeps the same subset without any coordination, so
sketches from bins that were never compared to each other are still directly
intersectable. `scaled` is the single knob that trades resolution for cost —
`scaled=200` keeps ~50 hashes per 10 kb bin, `scaled=20` keeps ~500.

**Why no mean-centering before the SVD?** Centering a sparse presence matrix
destroys the sparsity that makes the problem tractable. An IDF-weighted,
L2-normalised presence matrix with a truncated SVD is exactly latent semantic
analysis, and "which k-mer vocabulary does this bin use" is precisely the
question LSA answers.

**Why bins, not windows?** Non-overlapping bins keep every k-mer in exactly one
row, so a column's document frequency is interpretable and the matrix has no
built-in autocorrelation between neighbouring rows. Any structure you see
between adjacent bins is biology, not a sliding window.

**What is the honest test?** `backprop/cluster_transfer.parquet`. A cluster is
named from the reference's annotation; the report then measures how often
*assembly* bins in that same cluster carry the same annotation in their own,
independently generated per-assembly tracks. That number is the whole claim.

---

## Development

```bash
python -m pytest -q
```

```bash
ruff check .
```

Network-touching tests are marked; skip them with `-m 'not network'`.

`docs/API.md` is the internal contract every module is written against — read it
before changing a signature.

## Licence

MIT. HPRC and T2T data are used under their own terms; see
[humanpangenome.org](https://humanpangenome.org/) and the
[T2T consortium](https://github.com/marbl/CHM13).
