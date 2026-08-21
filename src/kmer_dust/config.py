"""Run configuration: dataclasses, YAML (de)serialisation and validation.

A kmer-dust run is fully described by one YAML file.  Every stage takes the same
:class:`Config` object, so a run is reproducible from ``config.resolved.yaml``
written into the output directory.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------


@dataclass
class ManifestConfig:
    """Which assemblies take part."""

    source: str = "hprc_release2"  # hprc_release2 | file | local_dir
    path: str = ""  # manifest TSV (source=file) or FASTA dir (source=local_dir)
    chroms: list[str] = field(default_factory=lambda: ["chr21"])
    max_samples: int = 0  # 0 == no cap
    max_assemblies: int = 0  # 0 == no cap
    require_t2t_chrom: bool = False  # keep only haplotypes where every requested
    # chromosome is a single gapless sequence
    require_annotations: list[str] = field(default_factory=lambda: ["censat"])
    include_reference: bool = True  # add T2T-CHM13v2.0 as its own "assembly"
    populations: list[str] = field(default_factory=list)  # keep only these (empty == all)
    samples: list[str] = field(default_factory=list)  # explicit allow-list
    exclude_samples: list[str] = field(default_factory=list)
    seed: int = 7


@dataclass
class SketchConfig:
    """FracMinHash sketching of binned sequence."""

    k: int = 31
    bin_size: int = 10_000
    scaled: int = 200  # keep ~1/scaled of k-mer space
    min_bin_acgt_frac: float = 0.5  # drop bins that are mostly N
    min_bin_sketch: int = 5  # drop bins with too few retained hashes
    include_unplaced: bool = False  # bins on contigs with no chrom assignment
    drop_partial_terminal_bin: bool = True
    threads: int = 4


@dataclass
class SelectConfig:
    """Choosing the k-mer columns of the matrix.

    The *lower* prevalence bound is the load-bearing one: it removes private
    variation and assembly artefacts, which are the k-mers that would otherwise
    make every haplotype its own cluster.  The upper bound defaults to 1.0
    because a k-mer shared by every sample is not the same thing as a k-mer
    shared by every *bin* -- an HSat2 31-mer occurs in all 232 samples and in
    0.1 % of bins, which makes it one of the most informative columns in the
    matrix.  Bin-level ubiquity is handled by IDF weighting in `matrix`, not
    here.  Lower it only if you specifically want to suppress the core shared
    vocabulary.
    """

    min_sample_prevalence: float = 0.10
    max_sample_prevalence: float = 1.0
    min_bins: int = 2  # ignore k-mers seen in fewer bins than this
    max_features: int = 200_000  # 0 == keep everything that passes prevalence
    n_buckets: int = 16  # radix partitions for out-of-core counting
    seed: int = 7


@dataclass
class MatrixConfig:
    weighting: str = "idf"  # none | idf | log
    row_norm: str = "l2"  # none | l1 | l2
    drop_empty_rows: bool = True


@dataclass
class DecomposeConfig:
    n_components: int = 64
    n_oversamples: int = 20
    n_iter: int = 7
    keep_components: bool = True  # write components.npy for loading inspection
    drop_first: int = 0  # discard leading components (they often encode depth)
    seed: int = 7


@dataclass
class EmbedConfig:
    """UMAP settings.

    ``deterministic`` is the one knob with a real trade-off.  Setting UMAP's
    ``random_state`` forces it (and pynndescent under it) onto a single thread,
    because the parallel code paths race on the negative-sample RNG.  That is
    the right default -- a run that cannot be reproduced cannot be debugged --
    but it makes the embedding the slowest stage in the pipeline by a wide
    margin at 10^5-10^6 bins.  Set it to false, with ``n_jobs``, to trade exact
    reproducibility for the cores you actually have; the layout will differ run
    to run while the structure will not.
    """

    n_neighbors: int = 30
    min_dist: float = 0.05
    metric: str = "cosine"
    n_components: int = 2
    max_fit_rows: int = 400_000  # fit on a subsample, transform the rest
    deterministic: bool = True
    n_jobs: int = 0  # 0 == use cfg.threads; ignored when deterministic
    seed: int = 7


@dataclass
class ClusterConfig:
    method: str = "hdbscan"  # hdbscan | dbscan
    space: str = "embedding"  # embedding | pcs
    min_cluster_size: int = 50
    min_samples: int = 10
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"  # eom | leaf
    eps: float = 0.5  # dbscan only
    seed: int = 7


@dataclass
class AnnotateConfig:
    reference_tracks: list[str] = field(
        default_factory=lambda: ["censat", "repeatmasker", "segdup", "telomere", "gene"]
    )
    assembly_tracks: list[str] = field(default_factory=lambda: ["censat", "repeatmasker", "segdup"])
    min_frac_for_dominant: float = 0.25
    annotate_assemblies: bool = True


@dataclass
class EnrichConfig:
    min_cluster_size: int = 10
    min_frac: float = 0.25  # a bin "carries" a feature above this covered fraction
    top_features: int = 3


@dataclass
class ReportConfig:
    max_points: int = 300_000
    title: str = ""
    subtitle: str = ""
    point_size: float = 3.0
    embed_plotlyjs: bool = True  # inline plotly.js so the HTML works offline
    seed: int = 7


@dataclass
class Config:
    run_name: str = "kmer-dust"
    outdir: str = "results/kmer-dust"
    datadir: str = "data"
    threads: int = 4
    seed: int = 7
    manifest: ManifestConfig = field(default_factory=ManifestConfig)
    sketch: SketchConfig = field(default_factory=SketchConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    decompose: DecomposeConfig = field(default_factory=DecomposeConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    annotate: AnnotateConfig = field(default_factory=AnnotateConfig)
    enrich: EnrichConfig = field(default_factory=EnrichConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    # ---------------- construction ----------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        data = dict(data or {})
        sections = {
            "manifest": ManifestConfig,
            "sketch": SketchConfig,
            "select": SelectConfig,
            "matrix": MatrixConfig,
            "decompose": DecomposeConfig,
            "embed": EmbedConfig,
            "cluster": ClusterConfig,
            "annotate": AnnotateConfig,
            "enrich": EnrichConfig,
            "report": ReportConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sections.items():
            section = data.pop(key, None) or {}
            unknown = set(section) - {f.name for f in dataclasses.fields(klass)}
            if unknown:
                raise ValueError(f"unknown key(s) in config section '{key}': {sorted(unknown)}")
            kwargs[key] = klass(**section)
        top_fields = {f.name for f in dataclasses.fields(cls)} - set(sections)
        unknown = set(data) - top_fields
        if unknown:
            raise ValueError(f"unknown top-level config key(s): {sorted(unknown)}")
        kwargs.update({k: v for k, v in data.items() if k in top_fields})
        cfg = cls(**kwargs)
        cfg.propagate_seed()
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Config:
        with open(path) as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def dump(self, path: str | os.PathLike[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    # ---------------- derived paths ----------------

    @property
    def out(self) -> Path:
        return Path(self.outdir)

    def path(self, *parts: str) -> Path:
        return self.out.joinpath(*parts)

    def stage_dir(self, stage: str) -> Path:
        p = self.path(stage)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------- validation ----------------

    def propagate_seed(self) -> None:
        """A single top-level ``seed`` drives every stage unless overridden."""
        for section in (
            self.manifest,
            self.select,
            self.decompose,
            self.embed,
            self.cluster,
            self.report,
        ):
            if getattr(section, "seed", None) == 7 and self.seed != 7:
                section.seed = self.seed

    def validate(self) -> None:
        errs: list[str] = []
        if not 1 <= self.sketch.k <= 31:
            errs.append("sketch.k must be in 1..31 (a k-mer must fit in 64 bits)")
        if self.sketch.k % 2 == 0:
            errs.append("sketch.k must be odd so that canonical k-mers are unambiguous")
        if self.sketch.bin_size < self.sketch.k:
            errs.append("sketch.bin_size must be >= sketch.k")
        if self.sketch.scaled < 1:
            errs.append("sketch.scaled must be >= 1")
        if not 0.0 <= self.sketch.min_bin_acgt_frac <= 1.0:
            errs.append("sketch.min_bin_acgt_frac must be in [0, 1]")
        lo, hi = self.select.min_sample_prevalence, self.select.max_sample_prevalence
        if not 0.0 <= lo <= hi <= 1.0:
            errs.append("require 0 <= select.min_sample_prevalence <= max_sample_prevalence <= 1")
        if self.select.n_buckets < 1 or self.select.n_buckets & (self.select.n_buckets - 1):
            errs.append("select.n_buckets must be a power of two")
        if self.matrix.weighting not in {"none", "idf", "log"}:
            errs.append("matrix.weighting must be one of none|idf|log")
        if self.matrix.row_norm not in {"none", "l1", "l2"}:
            errs.append("matrix.row_norm must be one of none|l1|l2")
        if self.decompose.n_components < 2:
            errs.append("decompose.n_components must be >= 2")
        if self.decompose.drop_first >= self.decompose.n_components:
            errs.append("decompose.drop_first must be < decompose.n_components")
        if self.embed.n_components not in (2, 3):
            errs.append("embed.n_components must be 2 or 3 (the report plots it)")
        if self.cluster.method not in {"hdbscan", "dbscan"}:
            errs.append("cluster.method must be hdbscan or dbscan")
        if self.cluster.space not in {"embedding", "pcs"}:
            errs.append("cluster.space must be embedding or pcs")
        if self.manifest.source not in {"hprc_release2", "file", "local_dir"}:
            errs.append("manifest.source must be hprc_release2|file|local_dir")
        if self.manifest.source in {"file", "local_dir"} and not self.manifest.path:
            errs.append(f"manifest.path is required when manifest.source={self.manifest.source}")
        if errs:
            raise ValueError("invalid configuration:\n  - " + "\n  - ".join(errs))

    # ---------------- misc ----------------

    @property
    def max_hash(self) -> int:
        """Inclusive FracMinHash threshold implied by ``sketch.scaled``."""
        return ((1 << 64) - 1) // self.sketch.scaled


def default_config() -> Config:
    return Config()
