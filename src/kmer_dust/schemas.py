"""Canonical on-disk data contracts for every kmer-dust stage.

Every stage of the pipeline reads and writes tables described here.  Keeping the
column names, dtypes and file names in one place means a stage can be re-run,
replaced or parallelised without any other stage needing to know.

Layout of a run directory (``--outdir``)::

    <outdir>/
      manifest.tsv                 assembly manifest actually used (MANIFEST_COLUMNS)
      config.resolved.yaml         the fully-resolved config for provenance
      sketch/
        <assembly>.bins.parquet    BIN_COLUMNS, with shard-local bin_idx 0..n-1
        <assembly>.sketch.parquet  SKETCH_COLUMNS, referencing that bin_idx
        <assembly>.done            marker written last, used for restartability
      kmers/
        buckets/bucket_<nn>.parquet   intermediate, BUCKET_COLUMNS
        kmers.parquet              KMER_COLUMNS (sorted by hash), the feature set
        prevalence.parquet         PREVALENCE_HIST_COLUMNS, diagnostics
      matrix/
        matrix.npz                 scipy.sparse CSR, float32, shape (n_bins, n_kmers)
        rows.parquet               BIN_COLUMNS + row_idx, global bin table
      decompose/
        pcs.npy                    float32 (n_bins, n_components)
        components.npy             float32 (n_components, n_kmers)  [optional]
        svd.json                   singular values + explained variance
      embed/
        umap.npy                   float32 (n_bins, n_embed_dims)
      cluster/
        clusters.parquet           CLUSTER_COLUMNS
      annotate/
        annotations.parquet        ANNOTATION_ID_COLUMNS + frac_<feature> columns
      enrich/
        enrichment.parquet         ENRICHMENT_COLUMNS
        cluster_names.parquet      CLUSTER_NAME_COLUMNS
      backprop/
        <assembly>.clusters.bed    BED9, one line per bin
        clusters.all.bed.gz        every assembly concatenated
      report/
        kmer_dust_report.html      the interactive report
        summary.json               headline numbers
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

# --------------------------------------------------------------------------
# manifest -- one row per haplotype assembly (plus one row for the reference)
# --------------------------------------------------------------------------

MANIFEST_COLUMNS: Final[dict[str, str]] = {
    "assembly": "string",  # unique id, e.g. HG00408_pat_hprc_r2_v1.0.1 or chm13v2.0
    "sample": "string",  # HG00408, CHM13
    "haplotype": "string",  # pat | mat | hap1 | hap2 | ref
    "source": "string",  # hprc | t2t | local
    "fasta": "string",  # path or https:// URL to (b)gzipped or plain FASTA
    "fai": "string",  # optional companion .fai URL/path ('' if none)
    "gzi": "string",  # optional companion .gzi URL/path ('' if none)
    "chrom_alias": "string",  # optional contig -> chr name mapping ('' if none)
    "censat_bed": "string",  # optional per-assembly cenSat BED ('' if none)
    "repeatmasker_bed": "string",  # optional per-assembly RepeatMasker BED ('' if none)
    "segdup_bed": "string",  # optional per-assembly segmental-duplication BED ('' if none)
    "population": "string",  # 1000G-style population abbreviation, '' if unknown
    "superpopulation": "string",  # AFR/AMR/EAS/EUR/SAS or '' if unknown
    "sex": "string",  # male | female | ''
}

#: Manifest columns that must be non-empty for a row to be usable.
MANIFEST_REQUIRED: Final[tuple[str, ...]] = ("assembly", "sample", "haplotype", "source", "fasta")

# --------------------------------------------------------------------------
# sketch stage
# --------------------------------------------------------------------------

BIN_COLUMNS: Final[dict[str, str]] = {
    "bin_idx": "int32",  # shard-local index, 0..n-1, matches sketch.bin_idx
    "bin_uid": "string",  # "<assembly>|<contig>|<start>" -- globally unique
    "assembly": "string",
    "sample": "string",
    "haplotype": "string",
    "source": "string",
    "contig": "string",  # name as it appears in the FASTA
    "chrom": "string",  # normalised chr name via chrom_alias, else '' (unplaced)
    # True when the contig is a whole, localised chromosome, so `start`/`end`
    # are genuine chromosome coordinates.  False for a `chrN_*_random` contig:
    # the chromosome is known but the coordinates are contig-local, so anything
    # reasoning about genomic *position* must filter on this.  See
    # fasta.parse_ucsc_placement.
    "placed": "bool",
    "start": "int64",  # 0-based, inclusive
    "end": "int64",  # 0-based, exclusive
    "n_acgt": "int64",  # unambiguous bases in the bin
    "n_kmers": "int64",  # valid canonical k-mers whose first base falls in the bin
    "n_sketch": "int32",  # k-mers retained by the FracMinHash filter
    "gc": "float32",  # G+C / n_acgt, NaN if n_acgt == 0
    "nfrac": "float32",  # 1 - n_acgt / (end - start)
}

SKETCH_COLUMNS: Final[dict[str, str]] = {
    "bin_idx": "int32",
    "hash": "uint64",  # splitmix64 of the canonical 2-bit k-mer code
}

# --------------------------------------------------------------------------
# k-mer selection stage
# --------------------------------------------------------------------------

BUCKET_COLUMNS: Final[dict[str, str]] = {
    "hash": "uint64",
    "sample_bits": "binary",  # not used by the default path; reserved
}

KMER_COLUMNS: Final[dict[str, str]] = {
    "hash": "uint64",  # sorted ascending -- np.searchsorted-able
    "col_idx": "int32",  # 0..n_features-1, equals the row order
    "n_samples": "int32",  # distinct samples containing the k-mer
    "n_assemblies": "int32",  # distinct haplotype assemblies containing it
    "n_bins": "int64",  # bins containing it (document frequency)
}

PREVALENCE_HIST_COLUMNS: Final[dict[str, str]] = {
    "n_samples": "int32",
    "n_kmers": "int64",
    "selected": "bool",
}

# --------------------------------------------------------------------------
# clustering stage
# --------------------------------------------------------------------------

CLUSTER_COLUMNS: Final[dict[str, str]] = {
    "row_idx": "int64",
    "bin_uid": "string",
    "cluster": "int32",  # -1 == noise
    "probability": "float32",
    "outlier_score": "float32",
}

# --------------------------------------------------------------------------
# annotation stage
# --------------------------------------------------------------------------

ANNOTATION_ID_COLUMNS: Final[dict[str, str]] = {
    "bin_uid": "string",
    "dominant_feature": "string",  # argmax over frac_* columns, 'unannotated' if all 0
    "dominant_frac": "float32",
    "annotated": "bool",  # False when the assembly had no annotation tracks at all
}

#: Normalised satellite / repeat vocabulary.  Raw track names from the T2T cenSat
#: BED and the per-assembly HPRC cenSat BEDs are mapped onto these classes so
#: that reference and assembly bins are directly comparable.
CENSAT_CLASSES: Final[tuple[str, ...]] = (
    "asat_hor_active",  # active alpha-satellite higher-order repeat array
    "asat_hor",  # (inactive/divergent) alpha-satellite HOR
    "asat_mon",  # monomeric alpha satellite
    "hsat1a",
    "hsat1b",
    "hsat2",
    "hsat3",
    "bsat",  # beta satellite
    "gsat",  # gamma satellite
    "censat_other",  # other peri/centromeric satellite
    "rdna",
    "ct",  # centromeric transition region
    "subterminal",  # subterminal / subtelomeric satellite
    "mon",  # monomeric, unclassified
)

#: RepeatMasker class vocabulary (column 7 of the T2T RepeatMasker BED).
REPEAT_CLASSES: Final[tuple[str, ...]] = (
    "line",
    "sine",
    "ltr",
    "dna",
    "satellite",
    "simple_repeat",
    "low_complexity",
    "rrna",
    "trna",
    "snrna",
    "retroposon",
    "rc",
    "repeat_unknown",
)

#: Extra single-track features.
EXTRA_FEATURES: Final[tuple[str, ...]] = ("segdup", "telomere", "gene")

#: Full ordered feature vocabulary used for the ``frac_<feature>`` columns.
FEATURE_VOCAB: Final[tuple[str, ...]] = CENSAT_CLASSES + REPEAT_CLASSES + EXTRA_FEATURES


#: Which features each annotation track kind can possibly produce.
#:
#: This matters for the label-transfer report.  The reference has a gene
#: annotation and the HPRC per-assembly track set does not, so a euchromatic
#: reference bin comes out ``gene`` while the same locus in an assembly comes
#: out ``line`` -- a disagreement caused entirely by asymmetric track
#: availability, not by the clustering.  Comparing the two sides only over the
#: intersection of what each *could* have been labelled makes the number honest.
TRACK_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "censat": CENSAT_CLASSES,
    "repeatmasker": REPEAT_CLASSES,
    "segdup": ("segdup",),
    "telomere": ("telomere",),
    "gene": ("gene",),
}


def features_for_tracks(kinds: Sequence[str]) -> tuple[str, ...]:
    """Feature vocabulary reachable from a set of track kinds, in vocab order."""
    reachable: set[str] = set()
    for kind in kinds or ():
        reachable.update(TRACK_FEATURES.get(str(kind), ()))
    return tuple(f for f in FEATURE_VOCAB if f in reachable)


def feature_column(feature: str) -> str:
    """Column name holding the covered fraction of ``feature`` in a bin."""
    return f"frac_{feature}"


FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(feature_column(f) for f in FEATURE_VOCAB)

# --------------------------------------------------------------------------
# enrichment stage
# --------------------------------------------------------------------------

ENRICHMENT_COLUMNS: Final[dict[str, str]] = {
    "cluster": "int32",
    "feature": "string",
    "n_bins_cluster": "int64",  # bins in the cluster carrying the feature
    "n_bins_total": "int64",  # bins genome-wide carrying the feature
    "cluster_size": "int64",
    "background_size": "int64",
    "frac_cluster": "float64",
    "frac_background": "float64",
    "log2_enrichment": "float64",
    "neg_log10_p": "float64",  # hypergeometric survival function
}

CLUSTER_NAME_COLUMNS: Final[dict[str, str]] = {
    "cluster": "int32",
    "name": "string",  # human-readable, e.g. "C7 asat_hor_active"
    "top_features": "string",  # ';'-joined "feature:log2fc" pairs
    "size": "int64",
    "n_assemblies": "int32",
    "n_chroms": "int32",
    "purity": "float64",  # fraction of annotated bins carrying the top feature
}

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

#: Hashes are always unsigned 64-bit.
HASH_DTYPE: Final = np.uint64
#: Largest representable hash, used to turn a "scaled" factor into a threshold.
HASH_MAX: Final[int] = (1 << 64) - 1


def bin_uid(assembly: str, contig: str, start: int) -> str:
    """Globally unique identifier for a bin.  ``|`` is illegal in FASTA names."""
    return f"{assembly}|{contig}|{start}"


def parse_bin_uid(uid: str) -> tuple[str, str, int]:
    assembly, contig, start = uid.rsplit("|", 2)
    return assembly, contig, int(start)


def enforce(df, columns: dict[str, str], *, subset: bool = False):
    """Return ``df`` with exactly ``columns`` in order and with the right dtypes.

    ``subset=True`` allows extra columns to survive (appended after the
    contract columns), which is what the annotation and row tables need.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = df.copy()
    for col, dtype in columns.items():
        if dtype == "binary":
            continue
        if dtype == "string":
            out[col] = out[col].fillna("").astype("string")
        else:
            out[col] = out[col].astype(dtype)
    ordered = list(columns)
    if subset:
        ordered += [c for c in out.columns if c not in columns]
    else:
        out = out[ordered]
        return out
    return out[ordered]


def empty_frame(columns: dict[str, str]):
    """An empty DataFrame with the contract's dtypes -- handy for edge cases."""
    import pandas as pd

    data = {}
    for col, dtype in columns.items():
        if dtype == "binary":
            data[col] = pd.Series([], dtype="object")
        elif dtype == "string":
            data[col] = pd.Series([], dtype="string")
        else:
            data[col] = pd.Series([], dtype=dtype)
    return pd.DataFrame(data)
