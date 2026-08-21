# Internal API contract

Every module below is written against `kmer_dust.schemas` (on-disk tables) and
`kmer_dust.config.Config` (run parameters).  `cli.py` and `pipeline.py` are the
only callers; they use **exactly** these signatures.  Do not change a signature
without changing this file.

Conventions used everywhere:

* `cfg: Config` -- the resolved run config.  Output paths come from
  `cfg.stage_dir("<stage>")`, never from ad-hoc arguments.
* `cache_dir: Path` -- `Path(cfg.datadir) / "cache"`, for downloaded catalogs.
* Every stage is **restartable**: it writes to `*.tmp` and renames, and skips
  work whose output already exists unless `force=True`.
* Every stage returns the in-memory object it wrote, so `pipeline.py` can chain
  without re-reading -- but it must also be readable from disk by the matching
  `load_*` helper so stages can run as separate Slurm jobs.
* Randomness always flows from `cfg.seed` / the stage's own `seed`.

---

## `kmer_dust/fasta.py`

```python
class FastaSource:
    def __init__(self, fasta: str, fai: str = "", gzi: str = "", cache_dir: Path | None = None): ...
    @property
    def contigs(self) -> list[str]: ...
    def contig_lengths(self) -> dict[str, int]: ...
    def fetch(self, contig: str, start: int = 0, end: int | None = None) -> bytes: ...
    def iter_contigs(self, contigs: Sequence[str] | None = None,
                     block: int = 8_000_000) -> Iterator[tuple[str, int, bytes]]: ...
    def close(self) -> None: ...

def load_chrom_alias(path_or_url: str, cache_dir: Path | None = None) -> dict[str, str]: ...
def normalize_chrom(name: str) -> str: ...
def open_text(path_or_url: str, cache_dir: Path | None = None) -> IO[str]: ...
def download(url: str, dest: Path, *, force: bool = False) -> Path: ...
```

* `FastaSource` must transparently handle: plain local FASTA, local bgzip FASTA,
  `https://` plain FASTA with a remote `.fai`, and `https://` bgzip FASTA with
  remote `.fai` + `.gzi`.  Use `pysam.FastaFile` for the indexed cases and a
  streaming parser otherwise.
* `iter_contigs` yields `(contig_name, contig_length, sequence_block)` in
  `block`-sized pieces **with no overlap**; the caller stitches k-mers across
  block boundaries by keeping the last `k-1` bases (so blocks must be exact and
  contiguous).  For unindexed streaming input `contigs=None` means "all", and
  `contig_length` is **`-1`** -- a streaming parser cannot know the length until
  it has passed the contig.  Consumers that need a real length must call
  `contig_lengths()` first, which costs a full pass.
* `normalize_chrom` maps `chr1`/`1`/`CM085953.1`-style names to `chr1` when
  possible, and returns `""` for anything unplaced.

## `kmer_dust/sketch.py`

```python
def sketch_assembly(row: Mapping[str, Any], cfg: Config, outdir: Path,
                    *, force: bool = False, cache_dir: Path | None = None) -> Path: ...
def sketch_manifest(manifest: pd.DataFrame, cfg: Config,
                    *, threads: int | None = None, force: bool = False,
                    cache_dir: Path | None = None) -> pd.DataFrame: ...
def load_sketch_shard(outdir: Path, assembly: str) -> tuple[pd.DataFrame, pd.DataFrame]: ...
def sketch_shard_paths(outdir: Path, assembly: str) -> dict[str, Path]: ...
```

Writes `<outdir>/<assembly>.bins.parquet` (`schemas.BIN_COLUMNS`),
`<assembly>.sketch.parquet` (`schemas.SKETCH_COLUMNS`, sorted by
`(bin_idx, hash)`) and a `<assembly>.done` marker written **last**.
`sketch_manifest` returns one row per assembly with columns
`assembly, n_bins, n_hashes, n_contigs, seconds, status`, plus `cached` (bool)
and `error` (string).  `status` is exactly `"ok"` or `"failed"`; a shard skipped
because its `.done` marker was already valid reports `status="ok", cached=True`.

`sketch_assembly` returns the path of the `.done` marker (written last, so its
existence means the shard is complete).  The marker is JSON carrying the sketch
parameters: changing `k`, `bin_size`, `scaled`, the drop rules or `chroms`
invalidates it and forces a re-sketch, because mixing `k=21` and `k=31` shards
would produce a plausible and completely wrong matrix.

Bin rules: non-overlapping `cfg.sketch.bin_size` windows from position 0 of each
contig; a trailing partial bin is dropped when
`cfg.sketch.drop_partial_terminal_bin`; bins failing
`n_acgt / (end-start) >= cfg.sketch.min_bin_acgt_frac` or
`n_sketch >= cfg.sketch.min_bin_sketch` are dropped (and their hashes with
them).  A k-mer belongs to the bin containing its first base.  Contigs are
selected by `cfg.manifest.chroms` via the row's `chrom_alias`; when
`cfg.sketch.include_unplaced` is true, unaliased contigs are kept with
`chrom == ""`.

## `kmer_dust/catalog/hprc.py`, `catalog/t2t.py`, `catalog/manifest.py`

```python
# hprc.py
HPRC_DATA_TABLES_BASE: str
def s3_to_https(uri: str) -> str: ...
def fetch_table(url: str, cache_dir: Path, *, force: bool = False, sep: str = ",") -> pd.DataFrame: ...
def release2_index(cache_dir: Path, *, force: bool = False) -> pd.DataFrame: ...
def annotation_index(kind: str, cache_dir: Path, *, force: bool = False) -> pd.DataFrame: ...
def sample_metadata(cache_dir: Path, *, force: bool = False) -> pd.DataFrame: ...
def t2t_sequence_table(cache_dir: Path, *, force: bool = False) -> pd.DataFrame: ...
POPULATION_TO_SUPERPOPULATION: dict[str, str]

# t2t.py
T2T_ASSEMBLY: str          # "chm13v2.0"
T2T_FASTA: str; T2T_FAI: str
T2T_TRACKS: dict[str, str] # censat|repeatmasker|segdup|telomere|gene -> URL
def reference_manifest_row() -> dict[str, str]: ...

# manifest.py
def build_manifest(cfg: Config, cache_dir: Path, *, force: bool = False) -> pd.DataFrame: ...
def write_manifest(df: pd.DataFrame, path: Path) -> Path: ...
def read_manifest(path: Path) -> pd.DataFrame: ...
```

`build_manifest` returns a frame with exactly `schemas.MANIFEST_COLUMNS`,
reference row first when `cfg.manifest.include_reference`, deterministic order,
honouring every filter in `ManifestConfig`.

## `kmer_dust/select.py`

```python
def select_kmers(sketch_dir: Path, manifest: pd.DataFrame, cfg: Config, outdir: Path,
                 *, force: bool = False) -> pd.DataFrame: ...
def load_kmers(outdir: Path) -> pd.DataFrame: ...
```

Two-pass, out-of-core: pass 1 partitions every shard's hashes into
`cfg.select.n_buckets` files keyed by the **high** bits of the hash; pass 2
loads one bucket at a time and counts, per hash, the number of distinct samples,
distinct assemblies and bins.  Prevalence is over **samples**, not haplotypes.
Both prevalence bounds are **inclusive**: a k-mer survives when
`lo <= n_samples <= hi`, with `lo = ceil(min_prev * n_samples_total)` and
`hi = floor(max_prev * n_samples_total)` computed with an epsilon so that
`0.7 * 10` does not silently drop a legitimate 7-of-10 k-mer.

Sub-sampling to `cfg.select.max_features` uses a second FracMinHash pass
(`splitmix64(hash ^ seed)` below a threshold) so it is order-independent and
reproducible; the threshold is chosen from the observed count.  Writes
`kmers.parquet` (`schemas.KMER_COLUMNS`, sorted by hash) and
`prevalence.parquet`.

## `kmer_dust/matrix.py`

```python
def build_matrix(sketch_dir: Path, kmers: pd.DataFrame, manifest: pd.DataFrame,
                 cfg: Config, outdir: Path, *, force: bool = False
                 ) -> tuple[sparse.csr_matrix, pd.DataFrame]: ...
def load_matrix(outdir: Path) -> tuple[sparse.csr_matrix, pd.DataFrame]: ...
```

Rows are every surviving bin across every shard, in manifest order then
`bin_idx` order; `rows.parquet` is `schemas.BIN_COLUMNS` + `row_idx` (int64).
Values start as 1.0, then `cfg.matrix.weighting` (`idf` = `log(n_rows / df)`,
`log` = `log1p(1)`) and `cfg.matrix.row_norm` are applied.  Empty rows are
dropped when `cfg.matrix.drop_empty_rows` (and `row_idx` renumbered).

## `kmer_dust/decompose.py`, `embed.py`, `cluster.py`

```python
def decompose(matrix: sparse.csr_matrix, cfg: Config, outdir: Path,
              *, force: bool = False) -> np.ndarray: ...
def load_pcs(outdir: Path) -> np.ndarray: ...

def embed(pcs: np.ndarray, cfg: Config, outdir: Path, *, force: bool = False) -> np.ndarray: ...
def load_embedding(outdir: Path) -> np.ndarray: ...

def cluster(coords: np.ndarray, rows: pd.DataFrame, cfg: Config, outdir: Path,
            *, force: bool = False) -> pd.DataFrame: ...
def load_clusters(outdir: Path) -> pd.DataFrame: ...
```

`decompose` uses `sklearn.utils.extmath.randomized_svd`, writes `pcs.npy`
(`U * S`, float32), `components.npy` and `svd.json`
(`singular_values`, `explained_variance_ratio`, `n_components`, `shape`).
`embed` fits UMAP on at most `cfg.embed.max_fit_rows` rows (seeded subsample)
and transforms the rest.  `cluster` returns `schemas.CLUSTER_COLUMNS`.

## `kmer_dust/annotate.py`, `enrich.py`, `backprop.py`

```python
# annotate.py
def normalize_censat_name(raw: str) -> str: ...      # -> schemas.CENSAT_CLASSES member or ""
def normalize_repeat_class(cls: str, family: str = "") -> str: ...  # -> REPEAT_CLASSES member or ""
def read_bed(path_or_url: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """columns: chrom(str) start(int64) end(int64) name(str) score(float) strand(str) extra(list)"""
def bin_feature_fractions(bins: pd.DataFrame, intervals: pd.DataFrame,
                          features: Sequence[str]) -> np.ndarray:
    """(n_bins, n_features) float32 covered fraction; intervals need a 'feature' column.

    JOIN KEY: the raw FASTA **contig** name, never the normalised `chrom`.
    `bins` is keyed on its `contig` column (falling back to `chrom`), intervals
    on their `chrom` column (falling back to `contig`).  This is the one place a
    wrong join silently annotates the wrong sequence, and the contig name is the
    only key that is correct for both track flavours: a T2T reference BED says
    `chr21` and the CHM13 FASTA contig *is* `chr21`, while an HPRC per-assembly
    BED says `HG00408#1#CM085953.1` and so does that assembly's FASTA.  Mapping
    either side through `normalize_chrom` first would break the second case.
    """
def annotate_bins(rows: pd.DataFrame, manifest: pd.DataFrame, cfg: Config, outdir: Path,
                  *, cache_dir: Path | None = None, force: bool = False) -> pd.DataFrame: ...
def load_annotations(outdir: Path) -> pd.DataFrame: ...

# enrich.py
def enrich_clusters(rows: pd.DataFrame, clusters: pd.DataFrame, annotations: pd.DataFrame,
                    cfg: Config, outdir: Path, *, force: bool = False
                    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...   # (enrichment, cluster_names)
def load_enrichment(outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]: ...

# backprop.py
def write_cluster_beds(rows: pd.DataFrame, clusters: pd.DataFrame, names: pd.DataFrame,
                       cfg: Config, outdir: Path, *, force: bool = False) -> list[Path]: ...
def cluster_transfer_report(rows, clusters, annotations, names, cfg, outdir) -> pd.DataFrame: ...
```

`annotate_bins` annotates reference bins from `t2t.T2T_TRACKS` and HPRC bins
from the per-assembly BEDs named in the manifest, producing
`schemas.ANNOTATION_ID_COLUMNS` + one `frac_<feature>` column per
`schemas.FEATURE_VOCAB` entry, one row per row of `rows` and in the same order.

`cluster_transfer_report` measures how well a cluster label learned on the
reference predicts the annotation of *assembly* bins in the same cluster --
the honest test of the whole idea.  Columns:
`cluster, name, n_ref_bins, n_asm_bins, ref_top_feature, asm_top_feature,
 asm_agreement, asm_annotated_frac`.

Both sides' dominant feature is recomputed over the **intersection** of the
feature vocabularies their track sets can reach
(`schemas.features_for_tracks(cfg.annotate.reference_tracks)` ∩ the same for
`assembly_tracks`), and the vocabulary used is written to
`backprop/transfer_features.json`.  Without that restriction the reference's
gene and telomere tracks -- which the HPRC per-assembly track set does not have
-- label every euchromatic reference bin `gene` while the same locus in an
assembly is `line`, and the score collapses to zero for reasons that have
nothing to do with the clustering.  This was observed on a real run: median
agreement went from 0.000 to 0.32 once the comparison was made symmetric.

## `kmer_dust/viz/report.py`

```python
def build_report(cfg: Config, outdir: Path, *, force: bool = False) -> Path: ...
def collect_report_frame(cfg: Config, outdir: Path) -> pd.DataFrame: ...
```

Reads every stage output under `outdir` and writes
`report/kmer_dust_report.html` plus `report/summary.json`.

---

## `kmer_dust/pipeline.py`

```python
STAGES: tuple[str, ...]     # dependency order, drives the CLI subcommands

class RunContext:
    def __init__(self, cfg: Config): ...
    # lazily loads and memoises every stage output: .manifest .rows .matrix
    # .kmers .pcs .embedding .clusters .annotations .enrichment

def run_stage(cfg: Config, stage: str, *, force: bool = False,
              ctx: RunContext | None = None, **kwargs) -> StageResult: ...
def run_all(cfg: Config, *, stages: tuple[str, ...] = STAGES, force: bool = False,
            force_from: str | None = None) -> list[StageResult]: ...
```

`stage_sketch` additionally accepts `assemblies: Sequence[str] | None`, surfaced
as `kmer-dust sketch --assembly <id>` (repeatable) and `--assembly-file`.  That
is the Slurm-array entry point: one task per haplotype, each touching only its
own shard and writing its own `sketch_summary.<tag>.tsv`.  Without it, N
parallel `kmer-dust sketch` jobs would each walk the whole manifest and race.

`run_all` writes `config.resolved.yaml` before the first stage and rewrites
`run_summary.json` after each one, so a run that dies half way still says how
far it got.
