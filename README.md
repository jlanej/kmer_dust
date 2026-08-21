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
named from annotation and then written back onto **every haplotype in the run**
as BED9 — including onto clusters that contain no reference bin at all, which a
reference-database lookup cannot produce.

The run reported below covers chr21 of 23 HPRC release-2 haplotypes (20 samples)
plus CHM13: 24 assemblies, 105,007 bins. The pipeline is written to scale to all
464 release-2 haplotypes and all autosomes (`workflow/config/full.yaml`), but
**that run has not been executed** and none of the numbers here describe it.

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
- [Prior work, and how this differs](#prior-work-and-how-this-differs)
- [References](#references)
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

**Cluster membership is strongly non-random in position**, even though no
coordinate enters the feature space. The median cluster's chr21 coordinates have
an interquartile range of **3.20 Mb**, against **21.4 Mb** when cluster labels are
permuted at random and **15.8 Mb** when they are permuted only among bins sharing
the same dominant repeat annotation — a **4.9× concentration over the stricter
null**, 6.7× over the uniform one. 96 % of clusters contain at least 80 % of the
haplotypes, with a median of **24 of 24**.

Call this positional concentration, not synteny. The median cluster covers only
~40 kb of any one haplotype, so a 3.20 Mb interquartile range is roughly eighty
times wider than a single shared locus, and only 4.3 % of clusters fall inside a
1 Mb IQR. A cross-chromosome run — not yet done — is the test that separates
locus recovery from repeat-class recovery.

**Satellite identity survives back-propagation.** Take a cluster's inferred name,
then ask what each HPRC assembly's own independently-generated tracks call the
assembly bins in that cluster:

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

**Where do cluster names come from?** From *pooled* annotation. `enrich.py` runs
its hypergeometric tests over every clustered bin, reference and assembly alike,
against a background of all 105,007 bins — CHM13 contributes 4,509 of them
(4.3 %), so a name is overwhelmingly determined by the HPRC per-assembly tracks,
not by the reference. This is deliberate: it is what lets a cluster with no CHM13
bin still receive a name. It also means the transfer report below is not a clean
reference-to-assembly test, and should not be read as one.

**What is the honest test?** `backprop/cluster_transfer.parquet`. It compares the
dominant feature of a cluster's *reference* bins against the dominant feature of
its *assembly* bins, over the intersection of the two track sets' reachable
vocabularies. For the 18 clusters with no reference bin, `asm_agreement` is `NaN`
— there is nothing to transfer from — but the cluster is still named and still
written to every BED.

---

## Prior work, and how this differs

Nothing in this pipeline is a new algorithm. Every stage is a standard technique
borrowed from a field that already uses it, and two independent literatures have
published the whole ten-stage skeleton before. This section says exactly what was
borrowed from where, what the closest existing work is, and what — narrowly — is
left that appears to be new. Everything cited here was checked against a primary
source; where a claim could not be confirmed, it is not made.

### What this is assembled from

| stage | the idea, and where it comes from |
| --- | --- |
| sketch | **FracMinHash / scaled MinHash** — keep a hash iff it falls below `2⁶⁴/scaled`, a *consistent* sample of k-mer space that needs no coordination between assemblies. Formalised for genomics by Irber *et al.* and implemented in `sourmash`; the underlying mod-*p* shingle selection is Broder *et al.* 1997, which also introduced the sparse binary document × feature structure and the practice of suppressing ubiquitous features. MinHash reached genomics through `Mash`. |
| matrix | **IDF weighting** (Spärck Jones 1972) on a binary presence matrix. |
| decompose | **Latent semantic analysis** (Deerwester *et al.* 1990) — truncated SVD of an IDF-weighted term × document matrix — computed with the **randomized SVD** of Halko, Martinsson & Tropp. "Which vocabulary does this bin use" is the question LSA answers. |
| embed, cluster | **UMAP** (McInnes *et al.*) and **HDBSCAN** (Campello *et al.*). |
| the whole numeric chain | **scATAC-seq LSI**: binarise fixed genomic bins, TF-IDF, SVD, embed, cluster. Established by Cusanovich *et al.* (2015, 2018) and productionised in `Signac` and `ArchR`. kmer-dust is that recipe with the matrix transposed — bin as document, k-mer as word. |
| annotate → enrich → BED | **Segmentation and genome annotation (SAGA)**: bin the genome, learn classes unsupervised, name them by enrichment against independent annotation, emit a track. `ChromHMM` (2012) onward; reviewed by Libbrecht, Chan & Hoffman. The same pattern names Hi-C subcompartments in Rao *et al.* |
| binning by composition | **Metagenomic binning** — chop sequence, featurise by k-mer composition, reduce, cluster (`CONCOCT`) — and, before it, oligonucleotide-frequency maps of fixed genomic fragments (Abe *et al.* 2003). |

### The closest existing work

| work | what it already does | what it does not do |
| --- | --- | --- |
| **GDA** (Aunin *et al.* 2022) | The closest tool in this genre. Non-overlapping windows, window × feature matrix, UMAP, HDBSCAN on the 2-D embedding, clusters named by statistical enrichment, `clusters.bed`, browser app, human demo recovering centromeres and pericentromeric segdups. Runs on a bare FASTA, and `gda_concatenate_tsv_tables.py` clusters windows from several genomes in one shared table. | Its features are ~30 curated summary statistics per window, not a k-mer feature space — its only k-mer features collapse all k-mer identity into two scalars at k=3 and k=4. No sketching, no IDF, no SVD. Features are computed independently per assembly, and it is built for a handful of genomes across species, not hundreds of haplotypes of one. |
| **Carnation T2T** (Lan *et al.* 2024) | The closest published *operating point*: GDA at **10 kb non-overlapping windows** — kmer-dust's exact bin size — UMAP + HDBSCAN, run jointly across both haplotypes of a haplotype-resolved T2T assembly, with cluster composition compared per haplotype. | 27 hand-engineered features, no factorisation at all, two haplotypes of one plant genome, no cross-assembly name transfer. |
| **KaryoScope** (Ranallo-Benavidez *et al.* 2026) | The closest *competitor*, on the same corpus: alignment-free 31-mer annotation of HPRC release-2 haplotypes against CHM13-derived feature sets, at base-pair resolution, in minutes per haplotype, validated against per-haplotype tracks. | Supervised projection from a pre-built database. It cannot emit a class that has no reference counterpart — unmatched sequence is a failure mode, not an output. No latent space, no clusters, no discovery. |
| **ModDotPlot**, **StainedGlass** | Fixed-window sketching of human satellite and centromeric arrays, including cross-haplotype comparison. | The matrix is window × *window* similarity, so cost is O(n²) and there is no shared feature space to factorise. |
| **polyCRACKER** (Gordon *et al.* 2019) | Within-species unsupervised partitioning from a sparse fragment × k-mer matrix with cosine geometry and iterative label propagation. | 250 kb fragments, k=26, raw counts and no weighting, inverted k-mer selection polarity, one assembly at a time. |
| **Smash++** (Hosseini *et al.* 2020) | Alignment-free positional correspondence between genomes, with coordinate output and native inverted-repeat handling, at whole-chromosome scale. | Strictly pairwise homologous-region pairs, not a many-haplotype map of sequence classes. |

### What is actually different here

Four things, stated as narrowly as the evidence supports.

**1. The shared column space is k-mer identities.** Every assembly in a run is
sketched against the same FracMinHash threshold, so the matrix columns *are*
individual k-mers and any two bins — from any two haplotypes — are directly
comparable rows. Other tools share a feature space across genomes (GDA
concatenates per-genome window tables; PanKmer builds one canonical k-mer index),
but they share either summary statistics or whole-genome rows. Bins from hundreds
of conspecific assemblies as rows of one k-mer matrix is the part we could not
find anywhere.

**2. Nothing but the assembly is needed to place a bin on the map.** No
functional-genomics assay, no repeat library, no gene models. That separates it
from assay-driven SAGA methods, which annotate one sample at a time from that
sample's own ChIP-seq or DNase panel, and from supervised k-mer annotation, which
needs a labelled reference database. It does *not* separate it from GDA, which
also runs on a bare FASTA. Naming and scoring the clusters is a separate matter
and does consume per-assembly annotation tracks.

**3. Classes are discovered before they are named.** A cluster can therefore exist
with no counterpart in the reference. In the chr21 run, **18 of 926 clusters
contain no CHM13 bin at all** (median 88 bins each, carried by a median of 13.5 of
the 23 haplotypes). Their inferred names are almost all satellite: 11 of the 18
are `asat_hor_active` or `asat_hor`, the rest `hsat1a`/`hsat1b`/`hsat3`/`bsat`
and one `simple_repeat`. They are named from pooled annotation and written to
the BEDs like any other
cluster; only their `asm_agreement` is `NaN`, because there is no reference label
to transfer. A reference-database lookup is structurally incapable of producing
these. Sun *et al.* report that 56.4 % of the 195 alpha-satellite HOR arrays they
identify are absent from T2T-CHM13, so this is not a corner case.

**4. Satellite sequence is signal, not something to mask.** It is also where the
method works best, since a satellite array is defined by its vocabulary. This is
not an empty field — `SRF`, `TRASH`, `HiCAT` and `StringDecomposer` all address
this sequence directly. What differs is that satellite and non-satellite bins are
described in one feature space and clustered together, rather than handed to a
separate repeat-specific pipeline.

### What is explicitly *not* claimed

* **Not a new numeric method.** TF-IDF → SVD → UMAP → HDBSCAN on fixed genomic
  bins is scATAC LSI and GDA. Randomized SVD → nonlinear embedding → density
  clustering on alignment-free fragment features was published by Kouchaki *et
  al.* in 2019.
* **Not the first alignment-free correspondence between genomes.** Smash++ does
  that, with coordinates.
* **Not the first alignment-free annotation of HPRC release 2 from CHM13.**
  KaryoScope does that, faster and at finer resolution.
* **Not "no aligner anywhere".** No alignment, reference coordinate or chain file
  enters the *feature space* — bins are compared only through shared k-mer hashes,
  and no bin is ever placed in another assembly's coordinate frame. But
  `annotate.py` intersects bins with annotation tracks by coordinate so clusters
  can be named and scored, the manifest uses HPRC's reference-derived
  chromosome-completeness table to pick haplotypes, and the RepeatMasker and
  segdup tracks are themselves alignment-derived products made upstream.
* **Not demonstrated at pangenome scale.** One chromosome, 24 assemblies. See the
  note under [What it actually computes](#what-it-actually-computes).

### Known limitations

Read these before trusting a number.

* **HDBSCAN runs on the 2-D UMAP embedding** (`cluster.space: embedding`), not on
  the 64-D SVD scores. UMAP does not preserve density or global distance, and the
  lineage this borrows from (`Signac`, `ArchR`) clusters in latent space and
  reserves UMAP for display. See Chari & Pachter and the Lause *et al.* reply.
* **926 clusters is a parameter setting, not a discovery.** `docs/RESULTS.md`
  sweeps `min_cluster_size` from 50 to 800 and gets 926 → 28 clusters at 13 % →
  54 % noise.
* **The enrichment p-values are a ranking statistic.** Bins are spatially
  autocorrelated and 24 near-copy haplotypes are not independent draws;
  `enrich.py` says so in its own docstring. Do not read `−log₁₀ p` as evidence.
* **The positional null is weak.** The 6.7× figure is against a uniform
  permutation; against an annotation-matched permutation it is 4.9×. Neither
  preserves spatial autocorrelation — see Kanduri *et al.* for why that matters,
  Gotelli for the measured Type I inflation of equiprobable nulls, and Strona *et
  al.* for a cheap fixed-margin alternative.
* **The transfer score is unadjusted accuracy on an imbalanced label set**, with
  no chance correction and no rejection option. `hsat1a = 1.00` is uninterpretable
  without the base rate. Compare Chicco & Jurman and the label-transfer reporting
  in Abdelaal *et al.*
* **The satellite rows of that table are the least informative, not the most.**
  RepeatMasker and cenSat assign labels by matching sequence against a repeat
  library; kmer-dust clusters by shared k-mers. On a tandem array both are reading
  the same vocabulary, so `hsat1a` at 1.00 is close to a restatement. The
  `line` / `ltr` / `ct` rows carry the information, because those labels are *not*
  recoverable from local vocabulary.
* **No alignment baseline has been run.** The comparison that would settle the
  method's value — against `liftOver`/`Liftoff` chains, or against MashMap3
  minmers, which return a coordinate from the same sketch primitive in one step —
  does not exist yet.
* **Bin phase is unquantified off the gapless subset.** `chr21.yaml` sets
  `require_t2t_chrom: true` (dropping 424 of 464 haplotypes) precisely because
  N-runs shift every downstream bin boundary; `full.yaml` sets it `false` and
  leans on `min_bin_acgt_frac`, which drops low-ACGT bins but does not restore
  phase.
* **`full.yaml` has not been validated.** At `scaled: 2000` and 10 kb bins the
  expected sketch is ~5 hashes per bin, which is also `min_bin_sketch`. Every
  reported number comes from the 10× denser `scaled: 200`.

---

## References

Verified against Crossref, Europe PMC or the publisher record. HPRC and T2T data
are used under their own terms.

**Data**

* Nurk S, Koren S, Rhie A, Rautiainen M, *et al.* The complete sequence of a human genome. *Science* 376(6588):44–53, 2022. [10.1126/science.abj6987](https://doi.org/10.1126/science.abj6987)
* Altemose N, Logsdon GA, Bzikadze AV, Sidhwani P, *et al.* Complete genomic and epigenetic maps of human centromeres. *Science* 376(6588):eabl4178, 2022. [10.1126/science.abl4178](https://doi.org/10.1126/science.abl4178) — provenance of the cenSat track.
* Hoyt SJ, Storer JM, Hartley GA, Grady PGS, *et al.* From telomere to telomere: the transcriptional and epigenetic state of human repeat elements. *Science* 376(6588):eabk3112, 2022. [10.1126/science.abk3112](https://doi.org/10.1126/science.abk3112) — provenance of the CHM13 RepeatMasker track.
* Liao WW, Asri M, Ebler J, Doerr D, *et al.* A draft human pangenome reference. *Nature* 617(7960):312–324, 2023. [10.1038/s41586-023-05896-x](https://doi.org/10.1038/s41586-023-05896-x)
* Lucas JK, Hebbar P, Liao WW, Macias-Velasco JF, *et al.* HPRC2: a human pangenome reference with near-complete coverage of common genetic variation. bioRxiv 2026.07.21.739710, 2026. [10.64898/2026.07.21.739710](https://doi.org/10.64898/2026.07.21.739710)
* Numanagić I, Gökkaya AS, Zhang L, Berger B, Alkan C, Hach F. Fast characterization of segmental duplications in genome assemblies. *Bioinformatics* 34(17):i706–i714, 2018. [10.1093/bioinformatics/bty586](https://doi.org/10.1093/bioinformatics/bty586) — SEDEF produced the per-haplotype segdup BEDs.
* Logsdon GA, Rozanski AN, Ryabov F, *et al.* The variation and evolution of complete human centromeres. *Nature* 629(8010):136–145, 2024. [10.1038/s41586-024-07278-3](https://doi.org/10.1038/s41586-024-07278-3)
* Guarracino A, Buonaiuto S, de Lima LG, Potapova T, *et al.* Recombination between heterologous human acrocentric chromosomes. *Nature* 617(7960):335–343, 2023. [10.1038/s41586-023-05976-y](https://doi.org/10.1038/s41586-023-05976-y)
* Gao S, Oshima KK, Chuang SC, Loftus M, *et al.* A global view of human centromere variation and evolution. *Nature*, online 29 July 2026. [10.1038/s41586-026-10841-9](https://doi.org/10.1038/s41586-026-10841-9)
* Sun Y, Wan S, Nie L, *et al.* Multidimensional variation and population stratification across 8000 complete human centromeres. bioRxiv 2026.07.22.740206, 2026. [10.64898/2026.07.22.740206](https://doi.org/10.64898/2026.07.22.740206)

**Method components**

* Broder AZ, Glassman SC, Manasse MS, Zweig G. Syntactic clustering of the Web. *Computer Networks and ISDN Systems* 29(8–13):1157–1166, 1997. [10.1016/S0169-7552(97)00031-7](https://doi.org/10.1016/S0169-7552(97)00031-7)
* Ondov BD, Treangen TJ, Melsted P, *et al.* Mash: fast genome and metagenome distance estimation using MinHash. *Genome Biology* 17(1):132, 2016. [10.1186/s13059-016-0997-x](https://doi.org/10.1186/s13059-016-0997-x)
* Irber L, Brooks PT, Reiter T, Pierce-Ward NT, Hera MR, Koslicki D, Brown CT. Lightweight compositional analysis of metagenomes with FracMinHash and minimum metagenome covers. bioRxiv 2022.01.11.475838, 2022. [10.1101/2022.01.11.475838](https://doi.org/10.1101/2022.01.11.475838)
* Irber L, Pierce-Ward NT, Abuelanin M, *et al.* sourmash v4: a multitool to quickly search, compare, and analyze genomic and metagenomic data sets. *JOSS* 9(98):6830, 2024. [10.21105/joss.06830](https://doi.org/10.21105/joss.06830)
* Spärck Jones K. A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation* 28(1):11–21, 1972. [10.1108/eb026526](https://doi.org/10.1108/eb026526)
* Deerwester S, Dumais ST, Furnas GW, Landauer TK, Harshman R. Indexing by latent semantic analysis. *JASIS* 41(6):391–407, 1990. [10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9)
* Halko N, Martinsson PG, Tropp JA. Finding structure with randomness. *SIAM Review* 53(2):217–288, 2011. [10.1137/090771806](https://doi.org/10.1137/090771806)
* McInnes L, Healy J, Melville J. UMAP: uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426, 2018. [10.48550/arXiv.1802.03426](https://doi.org/10.48550/arXiv.1802.03426)
* Campello RJGB, Moulavi D, Sander J. Density-based clustering based on hierarchical density estimates. *PAKDD 2013*, LNCS 7819:160–172. [10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14)
* McInnes L, Healy J, Astels S. hdbscan: hierarchical density based clustering. *JOSS* 2(11):205, 2017. [10.21105/joss.00205](https://doi.org/10.21105/joss.00205)

**Closest prior work**

* Aunin E, Berriman M, Reid AJ. Characterising genome architectures using genome decomposition analysis. *BMC Genomics* 23(1):398, 2022. [10.1186/s12864-022-08616-3](https://doi.org/10.1186/s12864-022-08616-3)
* Lan L, Leng L, Liu W, *et al.* The haplotype-resolved telomere-to-telomere carnation (*Dianthus caryophyllus*) genome reveals the correlation between genome architecture and gene expression. *Horticulture Research* 11(1):uhad244, 2024. [10.1093/hr/uhad244](https://doi.org/10.1093/hr/uhad244)
* Ranallo-Benavidez TR, Chen Y-A, Potapova T, *et al.* KaryoScope: rapid, alignment-free sequence annotation for the pangenome era. bioRxiv 2026.05.15.725544, 2026. [10.64898/2026.05.15.725544](https://doi.org/10.64898/2026.05.15.725544)
* Sweeten AP, Schatz MC, Phillippy AM. ModDotPlot — rapid and interactive visualization of tandem repeats. *Bioinformatics* 40(8):btae493, 2024. [10.1093/bioinformatics/btae493](https://doi.org/10.1093/bioinformatics/btae493)
* Vollger MR, Kerpedjiev P, Phillippy AM, Eichler EE. StainedGlass: interactive visualization of massive tandem repeat structures with identity heatmaps. *Bioinformatics* 38(7):2049–2051, 2022. [10.1093/bioinformatics/btac018](https://doi.org/10.1093/bioinformatics/btac018)
* Gordon SP, Levy JJ, Vogel JP. PolyCRACKER, a robust method for the unsupervised partitioning of polyploid subgenomes by signatures of repetitive DNA evolution. *BMC Genomics* 20(1):580, 2019. [10.1186/s12864-019-5828-5](https://doi.org/10.1186/s12864-019-5828-5)
* Hosseini M, Pratas D, Morgenstern B, Pinho AJ. Smash++: an alignment-free and memory-efficient tool to find genomic rearrangements. *GigaScience* 9(5):giaa048, 2020. [10.1093/gigascience/giaa048](https://doi.org/10.1093/gigascience/giaa048)
* Kouchaki S, Tapinos A, Robertson DL. A signal processing method for alignment-free metagenomic binning: multi-resolution genomic binary patterns. *Scientific Reports* 9:2159, 2019. [10.1038/s41598-018-38197-9](https://doi.org/10.1038/s41598-018-38197-9)
* Kille B, Garrison E, Treangen TJ, Phillippy AM. Minmers are a generalization of minimizers that enable unbiased local Jaccard estimation. *Bioinformatics* 39(9):btad512, 2023. [10.1093/bioinformatics/btad512](https://doi.org/10.1093/bioinformatics/btad512)

**Segmentation, binning and the LSI lineage**

* Cusanovich DA, Daza R, Adey A, *et al.* Multiplex single-cell profiling of chromatin accessibility by combinatorial cellular indexing. *Science* 348(6237):910–914, 2015. [10.1126/science.aab1601](https://doi.org/10.1126/science.aab1601)
* Cusanovich DA, Hill AJ, Aghamirzaie D, *et al.* A single-cell atlas of in vivo mammalian chromatin accessibility. *Cell* 174(5):1309–1324.e18, 2018. [10.1016/j.cell.2018.06.052](https://doi.org/10.1016/j.cell.2018.06.052)
* Stuart T, Srivastava A, Madad S, Lareau CA, Satija R. Single-cell chromatin state analysis with Signac. *Nature Methods* 18(11):1333–1341, 2021. [10.1038/s41592-021-01282-5](https://doi.org/10.1038/s41592-021-01282-5)
* Granja JM, Corces MR, Pierce SE, *et al.* ArchR is a scalable software package for integrative single-cell chromatin accessibility analysis. *Nature Genetics* 53(3):403–411, 2021. [10.1038/s41588-021-00790-6](https://doi.org/10.1038/s41588-021-00790-6)
* Ernst J, Kellis M. ChromHMM: automating chromatin-state discovery and characterization. *Nature Methods* 9(3):215–216, 2012. [10.1038/nmeth.1906](https://doi.org/10.1038/nmeth.1906)
* Libbrecht MW, Chan RCW, Hoffman MM. Segmentation and genome annotation algorithms for identifying chromatin state and other genomic patterns. *PLoS Computational Biology* 17(10):e1009423, 2021. [10.1371/journal.pcbi.1009423](https://doi.org/10.1371/journal.pcbi.1009423)
* Rao SSP, Huntley MH, Durand NC, *et al.* A 3D map of the human genome at kilobase resolution reveals principles of chromatin looping. *Cell* 159(7):1665–1680, 2014. [10.1016/j.cell.2014.11.021](https://doi.org/10.1016/j.cell.2014.11.021)
* Alneberg J, Bjarnason BS, de Bruijn I, *et al.* Binning metagenomic contigs by coverage and composition. *Nature Methods* 11(11):1144–1146, 2014. [10.1038/nmeth.3103](https://doi.org/10.1038/nmeth.3103)
* Abe T, Kanaya S, Kinouchi M, Ichiba Y, Kozuki T, Ikemura T. Informatics for unveiling hidden genome signatures. *Genome Research* 13(4):693–702, 2003. [10.1101/gr.634603](https://doi.org/10.1101/gr.634603)
* Chen KM, Wong AK, Troyanskaya OG, Zhou J. A sequence-based global map of regulatory activity for deciphering human genetics. *Nature Genetics* 54(7):940–949, 2022. [10.1038/s41588-022-01102-2](https://doi.org/10.1038/s41588-022-01102-2)

**Evaluation and critique**

* Gotelli NJ. Null model analysis of species co-occurrence patterns. *Ecology* 81(9):2606–2621, 2000. [10.1890/0012-9658(2000)081[2606:NMAOSC]2.0.CO;2](https://doi.org/10.1890/0012-9658(2000)081%5B2606:NMAOSC%5D2.0.CO;2)
* Strona G, Nappo D, Boccacci F, Fattorini S, San-Miguel-Ayanz J. A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals. *Nature Communications* 5:4114, 2014. [10.1038/ncomms4114](https://doi.org/10.1038/ncomms4114)
* Kanduri C, Bock C, Gundersen S, Hovig E, Sandve GK. Colocalization analyses of genomic elements: approaches, recommendations and challenges. *Bioinformatics* 35(9):1615–1624, 2019. [10.1093/bioinformatics/bty835](https://doi.org/10.1093/bioinformatics/bty835)
* Chari T, Pachter L. The specious art of single-cell genomics. *PLoS Computational Biology* 19(8):e1011288, 2023. [10.1371/journal.pcbi.1011288](https://doi.org/10.1371/journal.pcbi.1011288)
* Lause J, Berens P, Kobak D. The art of seeing the elephant in the room: 2D embeddings of single-cell data do make sense. *PLoS Computational Biology* 20(10):e1012403, 2024. [10.1371/journal.pcbi.1012403](https://doi.org/10.1371/journal.pcbi.1012403)
* Chicco D, Jurman G. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics* 21(1):6, 2020. [10.1186/s12864-019-6413-7](https://doi.org/10.1186/s12864-019-6413-7)
* Abdelaal T, Michielsen L, Cats D, *et al.* A comparison of automatic cell identification methods for single-cell RNA sequencing data. *Genome Biology* 20(1):194, 2019. [10.1186/s13059-019-1795-z](https://doi.org/10.1186/s13059-019-1795-z)
* Foroozandeh Shahraki M, Farahbod M, Libbrecht MW. Robust chromatin state annotation. *Genome Research* 34(3):469–483, 2024. [10.1101/gr.278343.123](https://doi.org/10.1101/gr.278343.123)
* Shumate A, Salzberg SL. Liftoff: accurate mapping of gene annotations. *Bioinformatics* 37(12):1639–1643, 2021. [10.1093/bioinformatics/btaa1016](https://doi.org/10.1093/bioinformatics/btaa1016)

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
