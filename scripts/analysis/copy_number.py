"""Satellite array copy number per haplotype, counted in bins, with no aligner.

A 10 kb bin *is* 10 kb of assembled sequence, so the number of bins a haplotype
contributes to a satellite family is already a copy-number estimate -- no
mapping, no coordinates, no reference to project onto.  That matters precisely
because satellite arrays are the material alignment-based copy number handles
worst: a read from an alpha-satellite HOR maps equally well to a hundred places,
so depth-of-coverage callers either collapse the array or refuse to call it.
Counting bins sidesteps the problem entirely, because a bin never has to be
placed anywhere to be counted.

The whole exercise lives or dies on one confounder, so it is measured rather
than asserted: **a haplotype can look satellite-rich simply by having assembled
more sequence.**  Three independent controls are computed here.

1.  *Non-satellite assembled megabases* is the completeness proxy.  It excludes
    every satellite class, so it cannot be inflated by the quantity under test.
    Each family's copy number is regressed on it and the residual CV reported;
    if the residual CV collapses, the "variation" was completeness.
2.  *Ratios between families* cancel any purely multiplicative completeness
    factor.  If haplotype H simply assembled 10 % more of everything, its
    hsat1a/hsat3 ratio is unchanged.
3.  *Within-sample concordance.*  The two haplotype assemblies of one donor are
    partitioned from one read set, so per-sample technical effects act on both.
    A permutation test over re-pairings of the cohort measures it.  Note the
    interpretation is the opposite of the intuitive one: the two haplotypes of a
    person are one paternal and one maternal gamete, i.e. two draws from the
    same population and *not* relatives, so excess within-sample similarity is
    evidence of a shared technical signature, not of heritability.  Three nulls
    are therefore reported -- unrestricted, within superpopulation, and within
    assembly-phasing stratum -- and a finding is only interesting if it survives
    all three.

Two structural facts about the input constrain the code.  ``placed`` is False
for ~80 % of bins, whose ``start`` is contig-local, so nothing here ever touches
a coordinate: only counts.  And the run this must scale to has ~19 M bins, so
every parquet file is streamed in aligned batches and reduced into a small
(assembly, chrom, feature) table rather than being joined in memory.

**Where the per-bin label comes from is now a property of the run, not of this
script.**  ``annotate.annotate_assemblies`` defaults to False, so a modern run
annotates the reference only and ``annotations.parquet`` says ``unannotated``
for every assembly bin; the assembly labels live in ``backprop/inferred.parquet``
instead, transferred from the reference through cluster membership.  Reading the
wrong one yields a table of exact zeros with no error anywhere, which is the
worst possible failure mode, so the source is *probed* rather than assumed and
is printed with its coverage before any number is shown.

That switch changes what the numbers mean, and the report says so rather than
papering over it.  A reference-transferred label exists only where a cluster
contains reference bins at all; on a 463-haplotype run CHM13 is 0.22 % of the
matrix, 23 % of bins sit in clusters it never entered, and only 32 % of bins get
a label.  Every megabase figure is therefore a *floor* whose depth depends on
cohort size -- absolute values are not comparable across runs, while comparisons
between haplotypes *within* one run are, because every haplotype is labelled
through the same cluster-to-name map.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kmer_dust import schemas  # noqa: E402

__all__ = ["aggregate_counts", "copy_number_report", "main"]

#: Satellite families this report is about.  Everything in the vocabulary is
#: written to the parquet; this is the subset the printed tables focus on,
#: ordered by descending expected array size.  ``ct`` (centromeric transition)
#: and ``censat_other`` are deliberately excluded -- they are residual
#: categories, not arrays with a copy number.
SATELLITE_FEATURES: tuple[str, ...] = (
    "asat_hor_active",
    "asat_hor",
    "asat_mon",
    "hsat1a",
    "hsat1b",
    "hsat2",
    "hsat3",
    "bsat",
    "gsat",
    "rdna",
)

#: Every class that is satellite-derived.  Subtracted from the assembly total to
#: build the completeness proxy, so that the proxy cannot contain the signal.
_SATELLITE_LIKE: frozenset[str] = frozenset(schemas.CENSAT_CLASSES) | {"satellite"}

#: Label written when a bin's argmax fell below ``annotate.min_frac_for_dominant``.
_UNANNOTATED = "unannotated"

#: ``manifest.source`` of the finished reference assembly.  Its bins are the only
#: ones allowed to name a cluster in the annotation-free cross-check, and it is
#: reported apart from the cohort everywhere else -- it is one haplotype, and a
#: complete one, so pooling it would flatter every distribution.
_REFERENCE_SOURCE = "t2t"

#: A cluster names a satellite family only if the reference contributed at least
#: this many bins and this share of them agree.  Below that the "name" is noise.
_MIN_REF_BINS = 5
_MIN_REF_PURITY = 0.5

#: How well a cluster-transferred label reproduces the family's per-haplotype
#: megabases, measured on ``results/acro3`` where 32 haplotypes carried their own
#: independently generated cenSat/RepeatMasker tracks *and* a reference-only
#: labelling could be computed for the same bins.  That run is the only place the
#: two can be compared, so these numbers are carried here as a constant and used
#: to mark families whose reference-only copy number should not be trusted.
#:
#: ``asat_hor_active`` is the outlier and the reason this exists: each chromosome
#: carries its own higher-order repeat variant, so a cluster built from CHM13's
#: HOR arrays does not name the corresponding array in another haplotype.
TRANSFER_FIDELITY: dict[str, float] = {
    "bsat": 0.991,
    "rdna": 0.960,
    "hsat1a": 0.908,
    "asat_hor": 0.885,
    "asat_mon": 0.874,
    "hsat3": 0.819,
    "hsat1b": 0.711,
    "hsat2": 0.588,
    "asat_hor_active": 0.338,
}

#: Below this correlation a family's reference-only copy number is reported with
#: a warning rather than as a result.
_MIN_TRANSFER_FIDELITY = 0.7

#: Rows pulled through per streaming batch.  Bounds peak memory at roughly
#: ``batch * n_feature_columns * 4`` bytes for the annotation side.
_BATCH_ROWS = 500_000

#: Bit layout packing (assembly, chrom, feature) into one int64 so that a batch
#: can be reduced with a single ``np.unique``.  Allows 2**20 assemblies and
#: 2**20 contigs-worth of chromosome labels, which no plausible run approaches.
_SHIFT_ASSEMBLY = 40
_SHIFT_CHROM = 20


class _Codebook:
    """Growing str -> int map, so streaming never holds the strings twice."""

    def __init__(self) -> None:
        self._to_code: dict[str, int] = {}
        self.values: list[str] = []

    def encode(self, labels: Sequence[str]) -> np.ndarray:
        """Codes for ``labels``, registering any label not yet seen."""
        to_code = self._to_code
        out = np.empty(len(labels), dtype=np.int64)
        for i, label in enumerate(labels):
            code = to_code.get(label)
            if code is None:
                code = len(self.values)
                to_code[label] = code
                self.values.append(label)
            out[i] = code
        return out

    def encode_dictionary(self, array: pa.Array) -> np.ndarray:
        """Codes for an Arrow string array, decoding its distinct values only."""
        dictionary = array.dictionary_encode()
        if isinstance(dictionary, pa.ChunkedArray):
            dictionary = dictionary.combine_chunks()
        local = self.encode([str(v) for v in dictionary.dictionary.to_pylist()])
        indices = dictionary.indices.to_numpy(zero_copy_only=False)
        return local[indices.astype(np.int64)]


def _aligned_batches(
    sources: Sequence[tuple[Path, list[str]]],
    batch_rows: int = _BATCH_ROWS,
) -> Iterator[tuple[pa.Table, ...]]:
    """Stream several row-aligned parquet files in matching chunks.

    ``iter_batches`` never spans a row group, so two files written with
    different row-group sizes yield differently shaped batches even when their
    rows correspond one-to-one.  Every side is therefore re-chunked to a common
    size here rather than zipped naively.
    """

    def rechunk(path: Path, columns: list[str]) -> Iterator[pa.Table]:
        held: list[pa.RecordBatch] = []
        n_held = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_rows, columns=columns
        ):
            held.append(batch)
            n_held += batch.num_rows
            while n_held >= batch_rows:
                table = pa.Table.from_batches(held)
                yield table.slice(0, batch_rows)
                rest = table.slice(batch_rows)
                held = rest.to_batches()
                n_held = rest.num_rows
        if n_held:
            yield pa.Table.from_batches(held)

    yield from zip(*(rechunk(path, columns) for path, columns in sources))


class FeatureSource(NamedTuple):
    """Which file supplies the per-bin feature label, and how far it reaches."""

    kind: str  # "annotations" | "inferred"
    path: Path
    column: str
    cohort_bins: int
    cohort_labelled: int
    reference_bins: int

    @property
    def coverage(self) -> float:
        return self.cohort_labelled / self.cohort_bins if self.cohort_bins else 0.0


def probe_feature_source(outdir: Path) -> FeatureSource:
    """Decide whether assembly bins are labelled in ``annotate`` or ``backprop``.

    ``annotate.annotate_assemblies: false`` is the default, and under it every
    assembly bin in ``annotations.parquet`` is ``unannotated`` while the real
    labels sit in ``backprop/inferred.parquet``.  Guessing wrong produces a table
    of zeros and no exception, so this counts rather than assumes: two single
    columns are streamed and the choice is made on how many *non-reference* bins
    actually carry a label.
    """
    rows_path = outdir / "matrix" / "rows.parquet"
    ann_path = outdir / "annotate" / "annotations.parquet"
    inferred_path = outdir / "backprop" / "inferred.parquet"
    for path in (rows_path, ann_path):
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")

    cohort = labelled = reference = 0
    for rows_batch, ann_batch in _aligned_batches(
        [(rows_path, ["source"]), (ann_path, ["annotated", "dominant_feature"])]
    ):
        is_ref = np.asarray(
            pc.equal(rows_batch.column("source"), _REFERENCE_SOURCE).to_numpy(
                zero_copy_only=False
            ),
            dtype=bool,
        )
        has_label = np.asarray(
            pc.and_(
                ann_batch.column("annotated"),
                pc.not_equal(ann_batch.column("dominant_feature"), _UNANNOTATED),
            ).to_numpy(zero_copy_only=False),
            dtype=bool,
        )
        reference += int(is_ref.sum())
        cohort += int((~is_ref).sum())
        labelled += int((has_label & ~is_ref).sum())

    if labelled > 0:
        return FeatureSource(
            "annotations", ann_path, "dominant_feature", cohort, labelled, reference
        )
    if not inferred_path.exists():
        raise FileNotFoundError(
            f"no assembly bin is labelled in {ann_path} (the run used "
            f"annotate.annotate_assemblies: false) and {inferred_path} does not "
            "exist, so there is nothing to count. Run the backprop stage."
        )
    inf_labelled = 0
    for rows_batch, inf_batch in _aligned_batches(
        [(rows_path, ["source"]), (inferred_path, ["inferred_feature"])]
    ):
        is_ref = np.asarray(
            pc.equal(rows_batch.column("source"), _REFERENCE_SOURCE).to_numpy(
                zero_copy_only=False
            ),
            dtype=bool,
        )
        has_label = np.asarray(
            pc.not_equal(inf_batch.column("inferred_feature"), "").to_numpy(
                zero_copy_only=False
            ),
            dtype=bool,
        )
        inf_labelled += int((has_label & ~is_ref).sum())
    return FeatureSource(
        "inferred", inferred_path, "inferred_feature", cohort, inf_labelled, reference
    )


def aggregate_counts(
    outdir: Path, feature_source: FeatureSource
) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame, pd.DataFrame]:
    """Reduce the run to per-(assembly, chrom, feature) counts and covered bases.

    Returns the long count table, a per-assembly total table, the bin size in
    base pairs inferred from the bin table, the (cluster, feature) counts
    restricted to *reference* bins -- which is what lets a cluster be named
    without consulting any assembly's own annotation -- and, when labels came
    from ``backprop``, a per-assembly summary of how well supported they were.

    ``covered_mb`` is only meaningful when the labels came from ``annotate``:
    the ``frac_*`` columns are per-bin track coverage, and under a reference-only
    run they are zero for every assembly bin.  They are then not read at all,
    which is also most of the speed-up on a 18 M-bin run.
    """
    rows_path = outdir / "matrix" / "rows.parquet"
    ann_path = outdir / "annotate" / "annotations.parquet"
    for path in (rows_path, ann_path):
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")
    cluster_path = outdir / "cluster" / "clusters.parquet"
    from_annotations = feature_source.kind == "annotations"
    # Reference cluster names are only wanted to cross-check assembly tracks
    # against; when the labels already came through the clusters, deriving them
    # again would be circular, so the column is not even read.
    use_clusters = cluster_path.exists() and from_annotations

    ann_schema = pq.ParquetFile(ann_path).schema_arrow
    features = [
        name.removeprefix("frac_") for name in ann_schema.names if name.startswith("frac_")
    ]
    frac_columns = [f"frac_{f}" for f in features] if from_annotations else []

    assemblies = _Codebook()
    chroms = _Codebook()
    # Seeded so that a feature's code equals its index in ``features``, which is
    # what lets the covered-fraction matrix and the count table share an axis.
    dominant = _Codebook()
    dominant.encode([*features, _UNANNOTATED])

    counts: defaultdict[int, int] = defaultdict(int)
    covered: defaultdict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(len(features), dtype=np.float64)
    )
    bins_per_assembly: defaultdict[int, int] = defaultdict(int)
    acgt_per_assembly: defaultdict[int, float] = defaultdict(float)
    ref_cluster_feature: defaultdict[tuple[int, int], int] = defaultdict(int)
    #: per assembly: [labelled bins, bins in a novel cluster, sum of label purity]
    provenance: defaultdict[int, np.ndarray] = defaultdict(lambda: np.zeros(3))
    widths: list[int] = []
    n_rows = 0

    label_columns = [feature_source.column]
    if not from_annotations:
        label_columns += ["novel", "purity"]
    sources: list[tuple[Path, list[str]]] = [
        (rows_path, ["bin_uid", "assembly", "chrom", "source", "start", "end", "n_acgt"]),
        (feature_source.path, ["bin_uid", *label_columns, *frac_columns]),
    ]
    labels = ("the feature source", "cluster/clusters")
    if use_clusters:
        sources.append((cluster_path, ["bin_uid", "cluster"]))

    for batches in _aligned_batches(sources):
        rows_batch, ann_batch = batches[0], batches[1]
        for other, label in zip(batches[1:], labels):
            if rows_batch.num_rows != other.num_rows:
                raise RuntimeError(
                    f"matrix/rows.parquet and {label} have different lengths "
                    f"({rows_batch.num_rows} vs {other.num_rows} in one batch). "
                    "Every stage promises one row per bin in bin order; re-run it."
                )
            if not pc.all(
                pc.equal(rows_batch.column("bin_uid"), other.column("bin_uid"))
            ).as_py():
                raise RuntimeError(
                    f"bin_uid order differs between matrix/rows.parquet and {label}"
                    ". Every aggregate here assumes the positional alignment "
                    "that the pipeline guarantees; re-run that stage against this "
                    "rows.parquet rather than trusting a join."
                )

        n = rows_batch.num_rows
        n_rows += n
        a_codes = assemblies.encode_dictionary(rows_batch.column("assembly"))
        c_codes = chroms.encode_dictionary(rows_batch.column("chrom"))
        # backprop writes '' where a cluster had no reference support; that is
        # the same statement as annotate's 'unannotated', so map it onto one code.
        raw = ann_batch.column(feature_source.column)
        if not from_annotations:
            raw = pc.if_else(pc.equal(raw, ""), _UNANNOTATED, raw)
        f_codes = dominant.encode_dictionary(raw)

        packed = (a_codes << _SHIFT_ASSEMBLY) | (c_codes << _SHIFT_CHROM) | f_codes
        keys, n_key = np.unique(packed, return_counts=True)
        for key, count in zip(keys.tolist(), n_key.tolist()):
            counts[key] += count

        if use_clusters:
            is_ref = np.asarray(
                pc.equal(rows_batch.column("source"), _REFERENCE_SOURCE).to_numpy(
                    zero_copy_only=False
                ),
                dtype=bool,
            )
            if is_ref.any():
                cluster = batches[2].column("cluster").to_numpy(zero_copy_only=False)
                pairs, n_pair = np.unique(
                    np.stack([cluster[is_ref].astype(np.int64), f_codes[is_ref]], axis=1),
                    axis=0,
                    return_counts=True,
                )
                for (cl, fc), count in zip(pairs.tolist(), n_pair.tolist()):
                    ref_cluster_feature[(cl, fc)] += count

        # Per-assembly reductions.  Bins are written assembly by assembly, so
        # runs are few; reduceat is correct for any ordering and avoids one mask
        # per assembly per batch.
        stack = [rows_batch.column("n_acgt").to_numpy(zero_copy_only=False).astype(np.float64)]
        if from_annotations:
            stack += [
                ann_batch.column(col).to_numpy(zero_copy_only=False).astype(np.float64)
                for col in frac_columns
            ]
        else:
            labelled = (f_codes != dominant.encode([_UNANNOTATED])[0]).astype(np.float64)
            stack += [
                labelled,
                ann_batch.column("novel").to_numpy(zero_copy_only=False).astype(np.float64),
                labelled * ann_batch.column("purity").to_numpy(zero_copy_only=False),
            ]
        block = np.column_stack(stack)
        starts = np.concatenate(([0], np.flatnonzero(a_codes[1:] != a_codes[:-1]) + 1))
        sums = np.add.reduceat(block, starts, axis=0)
        run_len = np.diff(np.append(starts, n))
        for i, code in enumerate(a_codes[starts].tolist()):
            acgt_per_assembly[code] += float(sums[i, 0])
            bins_per_assembly[code] += int(run_len[i])
            if from_annotations:
                covered[code] += sums[i, 1:]
            else:
                provenance[code] += sums[i, 1:]

        if len(widths) < 4:
            start = rows_batch.column("start").to_numpy(zero_copy_only=False)
            end = rows_batch.column("end").to_numpy(zero_copy_only=False)
            widths.append(int(np.median(end - start)))

    bin_size = int(np.median(widths)) if widths else 10_000

    key_array = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
    long = pd.DataFrame(
        {
            "assembly": pd.Categorical.from_codes(
                key_array >> _SHIFT_ASSEMBLY, categories=assemblies.values
            ),
            "chrom": pd.Categorical.from_codes(
                (key_array >> _SHIFT_CHROM) & ((1 << _SHIFT_CHROM) - 1),
                categories=chroms.values,
            ),
            "feature": pd.Categorical.from_codes(
                key_array & ((1 << _SHIFT_CHROM) - 1), categories=dominant.values
            ),
            "n_bins": np.fromiter(counts.values(), dtype=np.int64, count=len(counts)),
        }
    )

    covered_long = pd.DataFrame(
        [
            {"assembly": assemblies.values[code], "feature": f, "covered_bp": vec[j] * bin_size}
            for code, vec in covered.items()
            for j, f in enumerate(features)
            if vec[j] > 0
        ],
        columns=["assembly", "feature", "covered_bp"],
    )

    totals = pd.DataFrame(
        {
            "assembly": [assemblies.values[c] for c in bins_per_assembly],
            "assembly_bins": list(bins_per_assembly.values()),
            "assembly_acgt": [acgt_per_assembly[c] for c in bins_per_assembly],
        }
    )
    totals["assembly_mb"] = totals["assembly_bins"] * bin_size / 1e6
    if n_rows == 0:
        raise RuntimeError(f"no bins found under {outdir}")

    ref_clusters = pd.DataFrame(
        [
            {"cluster": cl, "feature": dominant.values[fc], "n_ref_bins": count}
            for (cl, fc), count in ref_cluster_feature.items()
        ],
        columns=["cluster", "feature", "n_ref_bins"],
    )
    label_quality = pd.DataFrame(
        [
            {
                "assembly": assemblies.values[code],
                "labelled_bins": int(vec[0]),
                "novel_bins": int(vec[1]),
                "mean_purity": float(vec[2] / vec[0]) if vec[0] else np.nan,
            }
            for code, vec in provenance.items()
        ],
        columns=["assembly", "labelled_bins", "novel_bins", "mean_purity"],
    )
    return (
        long,
        totals.merge(covered_long, on="assembly", how="left"),
        bin_size,
        ref_clusters,
        label_quality,
    )


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _cv(x: np.ndarray) -> float:
    """Coefficient of variation; NaN when the mean is zero."""
    mean = float(np.mean(x))
    return float(np.std(x, ddof=1) / mean) if mean > 0 and x.size > 1 else float("nan")


def _fold(x: np.ndarray) -> float:
    """Max/min, NaN when some haplotype carries none of the family."""
    lo = float(np.min(x))
    return float(np.max(x) / lo) if lo > 0 else float("nan")


def _robust_fold(x: np.ndarray) -> float:
    """p90/p10 -- max/min is two single haplotypes and grows with cohort size."""
    lo = float(np.percentile(x, 10))
    return float(np.percentile(x, 90) / lo) if lo > 0 else float("nan")


def _paired_concordance(z: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Double-entry correlation of exchangeable pairs, vectorised over columns.

    Entering each pair in both orders makes the statistic independent of which
    haplotype was arbitrarily called ``hap1``, which a plain Pearson r is not.
    Because a re-pairing permutation leaves the pooled mean and variance
    untouched, the statistic reduces to the mean cross-product of the centred
    values and the whole null can be computed by fancy indexing.
    """
    centred = z - z.mean(axis=0, keepdims=True)
    var = (centred**2).mean(axis=0)
    prod = (centred[left] * centred[right]).mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(var > 0, prod / var, np.nan)


def _random_pairings(
    strata: np.ndarray, n_perm: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``n_perm`` random perfect matchings that never pair across strata.

    A stratum of odd size drops one member per permutation, which is the honest
    way to keep the null comparable rather than smuggling in a cross-stratum
    pair.
    """
    order = np.argsort(strata, kind="stable")
    bounds = np.flatnonzero(np.diff(strata[order])) + 1
    blocks = np.split(order, bounds)
    lefts, rights = [], []
    for _ in range(n_perm):
        left_p, right_p = [], []
        for block in blocks:
            shuffled = rng.permutation(block)
            usable = len(shuffled) - (len(shuffled) % 2)
            left_p.append(shuffled[:usable:2])
            right_p.append(shuffled[1:usable:2])
        lefts.append(np.concatenate(left_p))
        rights.append(np.concatenate(right_p))
    return np.array(lefts), np.array(rights)


def _concordance_test(
    values: pd.DataFrame,
    sample: np.ndarray,
    strata: dict[str, np.ndarray],
    n_perm: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Within-sample concordance of each column against several nulls."""
    frame = pd.DataFrame({"sample": sample}).reset_index(drop=True)
    pairs = [g.index.to_numpy() for _, g in frame.groupby("sample") if len(g) == 2]
    if len(pairs) < 4:
        return pd.DataFrame()
    left = np.array([p[0] for p in pairs])
    right = np.array([p[1] for p in pairs])

    z = values.to_numpy(dtype=float)
    sd = z.std(axis=0, ddof=1)
    keep = sd > 0
    z = (z[:, keep] - z[:, keep].mean(axis=0)) / sd[keep]
    columns = list(values.columns[keep])
    observed = _paired_concordance(z, left, right)

    out: dict[str, Any] = {"feature": columns, "n_pairs": len(pairs), "concordance": observed}
    for name, stratum in strata.items():
        perm_left, perm_right = _random_pairings(stratum, n_perm, rng)
        null = np.array(
            [_paired_concordance(z, perm_left[t], perm_right[t]) for t in range(n_perm)]
        )
        out[f"p_similar[{name}]"] = (null >= observed).mean(axis=0)
        out[f"p_different[{name}]"] = (null <= observed).mean(axis=0)
    return pd.DataFrame(out)


def reference_free_copy_number(
    outdir: Path, ref_clusters: pd.DataFrame, families: Sequence[str], bin_size: int
) -> pd.DataFrame:
    """Per-assembly megabases per family using only the *reference's* annotation.

    This is the version of the measurement with nothing else in it: the clusters
    were built from shared k-mers with no coordinates, each cluster is named
    from CHM13's bins alone, and a query haplotype's copy number is then just
    how many of its bins landed in the clusters carrying that name.  No track,
    no BED and no aligner ever touched the query assembly.  Agreement with the
    per-assembly-annotation numbers is therefore a real check, not a tautology.
    """
    if ref_clusters.empty:
        return pd.DataFrame()
    grid = ref_clusters.pivot_table(
        index="cluster", columns="feature", values="n_ref_bins", aggfunc="sum", fill_value=0
    )
    total = grid.sum(axis=1)
    modal = grid.idxmax(axis=1)
    purity = grid.max(axis=1) / total
    keep = (total >= _MIN_REF_BINS) & (purity >= _MIN_REF_PURITY) & modal.isin(families)
    keep &= grid.index >= 0  # -1 is HDBSCAN noise, which names nothing
    named = modal[keep]
    if named.empty:
        return pd.DataFrame()

    family_code = {f: i for i, f in enumerate(families)}
    lookup = np.full(int(named.index.max()) + 1, -1, dtype=np.int64)
    lookup[named.index.to_numpy()] = [family_code[f] for f in named]

    assemblies = _Codebook()
    tally: defaultdict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(len(families), dtype=np.int64)
    )
    for rows_batch, cluster_batch in _aligned_batches(
        [
            (outdir / "matrix" / "rows.parquet", ["assembly"]),
            (outdir / "cluster" / "clusters.parquet", ["cluster"]),
        ]
    ):
        a_codes = assemblies.encode_dictionary(rows_batch.column("assembly"))
        cluster = cluster_batch.column("cluster").to_numpy(zero_copy_only=False).astype(np.int64)
        in_range = (cluster >= 0) & (cluster < lookup.size)
        fam = np.where(in_range, lookup[np.where(in_range, cluster, 0)], -1)
        hit = fam >= 0
        if not hit.any():
            continue
        packed = a_codes[hit] * len(families) + fam[hit]
        keys, n_key = np.unique(packed, return_counts=True)
        for key, count in zip(keys.tolist(), n_key.tolist()):
            tally[key // len(families)][key % len(families)] += count

    return pd.DataFrame(
        {
            "assembly": [assemblies.values[c] for c in tally],
            **{
                f: [tally[c][i] * bin_size / 1e6 for c in tally]
                for i, f in enumerate(families)
            },
        }
    ).set_index("assembly")


def _ols_group_effects(
    y: np.ndarray, group: np.ndarray, covariate: np.ndarray
) -> pd.DataFrame:
    """Least-squares group means adjusted for one categorical covariate.

    Deviation coding, so each level's coefficient is its offset from the grand
    mean rather than from an arbitrary reference level, and every level gets an
    interval.  Written out rather than pulled from statsmodels because that is
    not a dependency of this repo, and the model is small enough that the normal
    equations are the whole implementation.
    """
    g_levels = list(pd.unique(group))
    c_levels = list(pd.unique(covariate))
    n = len(y)
    design = [np.ones(n)]
    names: list[str] = ["(mean)"]
    for level in g_levels[:-1]:
        design.append((group == level).astype(float) - (group == g_levels[-1]).astype(float))
        names.append(str(level))
    for level in c_levels[:-1]:
        design.append(
            (covariate == level).astype(float) - (covariate == c_levels[-1]).astype(float)
        )
        names.append(f"[{level}]")
    X = np.column_stack(design)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - np.linalg.matrix_rank(X)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * float(resid @ resid) / dof)
    crit = float(stats.t.ppf(0.975, dof))

    # The dropped level's deviation is minus the sum of the others; recover it
    # with the same linear algebra so it gets an honest interval too.
    out = []
    for i, name in enumerate(names):
        out.append((name, beta[i], se[i]))
    for levels, offset, brackets in ((g_levels, 1, False), (c_levels, 1 + len(g_levels) - 1, True)):
        k = len(levels) - 1
        contrast = np.zeros(X.shape[1])
        contrast[offset : offset + k] = -1.0
        value = float(contrast @ beta)
        err = float(np.sqrt(contrast @ xtx_inv @ contrast * float(resid @ resid) / dof))
        label = f"[{levels[-1]}]" if brackets else str(levels[-1])
        out.append((label, value, err))

    frame = pd.DataFrame(out, columns=["term", "effect", "se"])
    frame["ci_lo"] = frame["effect"] - crit * frame["se"]
    frame["ci_hi"] = frame["effect"] + crit * frame["se"]
    frame["t"] = frame["effect"] / frame["se"]
    frame["p"] = 2 * stats.t.sf(np.abs(frame["t"]), dof)
    # Omnibus F for the group term, i.e. is ancestry doing anything at all once
    # the covariate has had its say.
    keep = [i for i, nm in enumerate(names) if not nm.startswith("[") and nm != "(mean)"]
    reduced = np.delete(X, keep, axis=1)
    r_beta, *_ = np.linalg.lstsq(reduced, y, rcond=None)
    r_resid = y - reduced @ r_beta
    f_stat = ((float(r_resid @ r_resid) - float(resid @ resid)) / len(keep)) / (
        float(resid @ resid) / dof
    )
    frame.attrs["F"] = f_stat
    frame.attrs["p_F"] = float(stats.f.sf(f_stat, len(keep), dof))
    frame.attrs["dof"] = dof
    return frame


def _phasing_stratum(haplotype: pd.Series) -> pd.Series:
    """Group haplotypes by the naming scheme of their assembly's phasing.

    ``mat``/``pat`` come from trio binning and ``hap1``/``hap2`` from a
    read-based phasing, and the two pipelines behave differently on satellite
    arrays.  Both haplotypes of one donor always share a stratum, so this is a
    candidate explanation for any within-sample concordance and has to be
    permuted against.
    """
    label = haplotype.astype(str)
    return label.map(lambda h: {"mat": "mat/pat", "pat": "mat/pat"}.get(h, "hap1/hap2"))


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _ancestry_adjusted(cohort: pd.DataFrame, families: Sequence[str]) -> None:
    """Re-test every ancestry association with assembly phasing as a covariate.

    Both haplotypes of a donor come from one read set and one assembler run, so
    462 haplotypes are 231 independent observations, not 462.  Everything here is
    therefore done on the donor mean.  Phasing method is a per-donor property
    too, which is why it can go in as a covariate at all -- and why, if it were
    perfectly confounded with ancestry, no amount of modelling would separate
    them.  The crossing is printed so that can be judged.

    Every family is modelled, not only the ones that looked interesting
    unadjusted: selecting which to adjust on the strength of the unadjusted test
    is the forking path that makes an adjustment meaningless.
    """
    if "sample" not in cohort.columns:
        return
    donor = cohort.reset_index().assign(method=_phasing_stratum(cohort["haplotype"]).to_numpy())
    donor = (
        donor.groupby(["sample", "superpopulation", "method"], observed=True)[list(families)]
        .mean()
        .reset_index()
    )
    modelled = [f for f in families if donor[f].std(ddof=1) > 0]
    if not modelled:
        return
    print(f"\n  donors: {len(donor)} (one row per sample, the mean of its two haplotypes)")
    print("  crossing of ancestry with phasing at donor level:")
    print(pd.crosstab(donor["method"], donor["superpopulation"]).to_string())

    fits = {
        f: _ols_group_effects(
            donor[f].to_numpy(dtype=float),
            donor["superpopulation"].to_numpy(),
            donor["method"].to_numpy(),
        )
        for f in modelled
    }
    omnibus = pd.DataFrame(
        [
            {
                "family": f,
                "grand_mean_Mb": float(donor[f].mean()),
                "ancestry_F": fits[f].attrs["F"],
                "p": fits[f].attrs["p_F"],
                "p_bonferroni": min(1.0, fits[f].attrs["p_F"] * len(modelled)),
                "phasing_p": float(
                    fits[f].loc[fits[f]["term"].str.startswith("["), "p"].iloc[0]
                ),
            }
            for f in modelled
        ]
    )
    print("\n  ancestry effect AFTER adjusting for phasing, every family:")
    _show(omnibus, "%.4g")
    survivors = omnibus.loc[omnibus["p_bonferroni"] < 0.05, "family"].tolist()
    if not survivors:
        print("  nothing survives Bonferroni once phasing is in the model.")
        return
    for family in survivors:
        y = donor[family].to_numpy(dtype=float)
        fit = fits[family]
        print(f"\n  {family}: deviation from the grand mean, Mb, adjusted for phasing")
        print(f"  ancestry omnibus F = {fit.attrs['F']:.2f}, p = {fit.attrs['p_F']:.2e}, "
              f"dof = {fit.attrs['dof']}")
        _show(fit[fit["term"] != "(mean)"][["term", "effect", "ci_lo", "ci_hi", "p"]], "%.4f")
        print(f"  grand mean {y.mean():.3f} Mb.  A [bracketed] term is the phasing")
        print("  covariate, not ancestry.")
        # A model can only adjust; a stratified read shows whether the direction
        # is the same inside each phasing method, which no single p-value does.
        rows_s = []
        for method, block in donor.groupby("method", observed=True):
            groups = [
                (str(k), g[family].to_numpy(dtype=float))
                for k, g in block.groupby("superpopulation", observed=True)
                if len(g) > 1
            ]
            if len(groups) < 2:
                continue
            H, p = stats.kruskal(*[v for _, v in groups])
            rows_s.append(
                {"stratum": method, "n": len(block), "kruskal_H": H, "p": p,
                 **{k: float(np.median(v)) for k, v in groups}}
            )
        if rows_s:
            print("  the same test run separately inside each phasing stratum:")
            _show(pd.DataFrame(rows_s), "%.4f")


def _fidelity_note(feature: str, from_annotations: bool) -> dict[str, Any]:
    """Transfer fidelity column, populated only when it is the operative caveat."""
    if from_annotations or feature not in TRANSFER_FIDELITY:
        return {}
    r = TRANSFER_FIDELITY[feature]
    return {"fidelity": r, "trust": "" if r >= _MIN_TRANSFER_FIDELITY else "  <-- NO"}


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _show(frame: pd.DataFrame, floatfmt: str = "%.3f") -> None:
    if frame.empty:
        print("  (nothing to report)")
        return
    print(frame.to_string(index=False, float_format=lambda v: floatfmt % v))


def copy_number_report(outdir: Path, n_perm: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Print the report and return the table written to ``analysis/``."""
    rng = np.random.default_rng(seed)
    manifest = pd.read_csv(outdir / "manifest.tsv", sep="\t", dtype=str)
    feature_source = probe_feature_source(outdir)
    long, totals, bin_size, ref_clusters, label_quality = aggregate_counts(
        outdir, feature_source
    )
    from_annotations = feature_source.kind == "annotations"

    long["mb"] = long["n_bins"] * bin_size / 1e6
    per_chrom = long.copy()
    per_assembly = (
        long.groupby(["assembly", "feature"], observed=True)[["n_bins", "mb"]]
        .sum()
        .reset_index()
    )

    covered = totals[["assembly", "feature", "covered_bp"]].dropna()
    covered["covered_mb"] = covered["covered_bp"] / 1e6
    assembly_totals = totals[
        ["assembly", "assembly_bins", "assembly_mb", "assembly_acgt"]
    ].drop_duplicates()
    assembly_totals["assembly_acgt_mb"] = assembly_totals.pop("assembly_acgt") / 1e6

    per_assembly = per_assembly.merge(
        covered[["assembly", "feature", "covered_mb"]], on=["assembly", "feature"], how="outer"
    ).fillna({"n_bins": 0, "mb": 0.0, "covered_mb": 0.0})
    per_assembly["n_bins"] = per_assembly["n_bins"].astype("int64")

    meta_cols = [
        c
        for c in ("sample", "haplotype", "source", "population", "superpopulation", "sex")
        if c in manifest.columns
    ]
    meta = manifest[["assembly", *meta_cols]]

    out = pd.concat(
        [
            per_assembly.assign(scope="assembly", chrom=""),
            per_chrom.assign(scope="chrom", covered_mb=np.nan),
        ],
        ignore_index=True,
    )
    out = out.merge(meta, on="assembly", how="left").merge(
        assembly_totals, on="assembly", how="left"
    )
    out["frac_of_assembly"] = out["mb"] / out["assembly_mb"]
    out = out[
        [
            "scope",
            "assembly",
            *meta_cols,
            "chrom",
            "feature",
            "n_bins",
            "mb",
            "covered_mb",
            "assembly_bins",
            "assembly_mb",
            "assembly_acgt_mb",
            "frac_of_assembly",
        ]
    ].sort_values(["scope", "assembly", "chrom", "feature"], ignore_index=True)

    # ---------------------------------------------------------------- setup
    wide = (
        per_assembly.pivot_table(
            index="assembly", columns="feature", values="mb", aggfunc="sum", fill_value=0.0
        )
        .join(
            assembly_totals.set_index("assembly")[
                ["assembly_mb", "assembly_bins", "assembly_acgt_mb"]
            ]
        )
        .join(meta.set_index("assembly"))
    )
    satellite_like = [c for c in wide.columns if c in _SATELLITE_LIKE]
    wide["satellite_mb"] = wide[satellite_like].sum(axis=1)
    wide["nonsatellite_mb"] = wide["assembly_mb"] - wide["satellite_mb"]

    source = (
        wide["source"].astype(str) if "source" in wide.columns else pd.Series("", index=wide.index)
    )
    reference = wide[source == _REFERENCE_SOURCE]
    cohort = wide[source != _REFERENCE_SOURCE]
    families = [f for f in SATELLITE_FEATURES if f in wide.columns]

    print("=" * 78)
    print(f"satellite copy number without an aligner -- {outdir}")
    print("=" * 78)
    print(f"  bin size                {bin_size:,} bp")
    print(f"  assemblies              {len(wide)}  ({len(cohort)} cohort, {len(reference)} reference)")
    print(f"  samples in cohort       {cohort['sample'].nunique() if 'sample' in cohort else 0}")
    print(f"  bins                    {int(wide['assembly_bins'].sum()):,}")
    print(f"  chromosome labels       {per_chrom['chrom'].nunique()}")
    print(f"  satellite families      {len(families)} of {len(SATELLITE_FEATURES)} present")
    # bins x bin size is the copy-number unit, so state up front how much of a
    # nominal bin is really ACGT and how far that varies between assemblies.
    acgt = wide["assembly_acgt_mb"] / wide["assembly_mb"]
    print(
        f"  ACGT / nominal bin      median {acgt.median():.4f}, "
        f"min {acgt.min():.4f} over assemblies"
    )

    # ------------------------------------------------------- 0. label source
    _section("0. Where the per-bin labels came from")
    rel = feature_source.path.relative_to(outdir)
    print(f"  source                  {rel}:{feature_source.column}")
    if from_annotations:
        print("  each assembly carries its own cenSat/RepeatMasker/segdup tracks")
        print("  (annotate.annotate_assemblies was true), so labels are independent")
        print("  of the clustering and the megabase totals are directly comparable")
        print("  to any other run.")
    else:
        print("  no assembly bin is annotated in annotate/annotations.parquet, so labels")
        print("  are the reference's, transferred through cluster membership by backprop.")
    print(
        f"  cohort bins             {feature_source.cohort_bins:,}"
        f"  ({feature_source.reference_bins:,} reference)"
    )
    print(
        f"  cohort bins labelled    {feature_source.cohort_labelled:,}"
        f"  ({feature_source.coverage * 100:.1f} %)"
    )
    if not from_annotations and not label_quality.empty:
        lq = label_quality.merge(
            assembly_totals[["assembly", "assembly_bins"]], on="assembly"
        )
        lq = lq[lq["assembly"].isin(cohort.index)]
        novel = lq["novel_bins"].sum() / lq["assembly_bins"].sum()
        print(f"  cohort bins in a novel cluster (no reference bin at all)   {novel * 100:.1f} %")
        print(
            f"  mean purity of a transferred label   {lq['mean_purity'].mean():.3f}"
            f"  (range {lq['mean_purity'].min():.3f}-{lq['mean_purity'].max():.3f})"
        )
        print("\n  EVERY megabase below is therefore a floor: a bin is only counted when")
        print("  its cluster contains reference bins.  Absolute values are not comparable")
        print("  across runs of different size -- the reference is a smaller share of a")
        print("  bigger matrix -- but comparisons between haplotypes inside this run are,")
        print("  because every haplotype is labelled through the same cluster-name map.")

    # ------------------------------------------------- 1. how much is carried
    label = "dominant_feature" if from_annotations else "inferred_feature"
    _section(f"1. Megabases carried per haplotype ({label} x {bin_size // 1000} kb)")
    ref_name = reference.index[0] if len(reference) else None
    summary = pd.DataFrame(
        [
            {
                "family": f,
                "min": cohort[f].min(),
                "median": cohort[f].median(),
                "max": cohort[f].max(),
                "mean": cohort[f].mean(),
                "sd": cohort[f].std(ddof=1),
                "CV": _cv(cohort[f].to_numpy()),
                "fold": _fold(cohort[f].to_numpy()),
                # max/min is one haplotype over one haplotype, and at n=462 the
                # extremes are cheap; p90/p10 is the same statement made robustly.
                "fold_p90_p10": _robust_fold(cohort[f].to_numpy()),
                "reference": reference[f].iloc[0] if ref_name else np.nan,
                **_fidelity_note(f, from_annotations),
            }
            for f in families + ["satellite_mb", "nonsatellite_mb", "assembly_mb"]
        ]
    )
    _show(summary)
    if ref_name:
        print(f"\n  reference column is {ref_name}; it is one haplotype and is excluded")
        print("  from every cohort statistic above and below.")
    if not from_annotations:
        bad = [
            f
            for f in families
            if TRANSFER_FIDELITY.get(f, 1.0) < _MIN_TRANSFER_FIDELITY and cohort[f].max() > 0
        ]
        if bad:
            print(f"\n  *** DO NOT TRUST THE {', '.join(b.upper() for b in bad)} ROW"
                  f"{'S' if len(bad) > 1 else ''} ***")
            for f in bad:
                print(
                    f"  {f}: a reference-named cluster reproduces the per-haplotype total at "
                    f"only r = {TRANSFER_FIDELITY[f]:.2f}"
                )
            print("  (measured on results/acro3, the one run with both labellings). For")
            print("  asat_hor_active the cause is structural, not statistical: each")
            print("  chromosome carries its own higher-order repeat variant, so a cluster")
            print("  built from CHM13's HOR arrays does not name another haplotype's.")
            print("  The 'fidelity' column carries that r for every family.")
    silent = [
        f
        for f in families
        if cohort[f].max() == 0
        and not covered.empty
        and float(covered.loc[covered["feature"] == f, "covered_mb"].sum()) > 0
    ]
    if silent:
        print(
            f"\n  never dominant in any bin, though present as covered sequence: {', '.join(silent)}"
        )
        print(f"  -- their arrays are shorter than the {bin_size:,} bp bin, so the argmax")
        print("  always goes to whatever they sit inside.  See covered_mb in the parquet.")

    # ---------------------------------------------------- 2. the confounder
    _section("2. Is it just 'this haplotype assembled more'?")
    nonsat = cohort["nonsatellite_mb"].to_numpy(dtype=float)
    print("  completeness proxy = assembled Mb whose dominant feature is not satellite-derived")
    print(
        f"    range {nonsat.min():.1f} - {nonsat.max():.1f} Mb, "
        f"CV {_cv(nonsat):.4f}, fold {_fold(nonsat):.3f}"
    )
    conf = []
    for f in families:
        x = cohort[f].to_numpy(dtype=float)
        if np.std(x) == 0:
            continue
        r, p = stats.pearsonr(x, nonsat)
        resid = x - np.polyval(np.polyfit(nonsat, x, 1), nonsat)
        conf.append(
            {
                "family": f,
                "CV": _cv(x),
                "r_completeness": r,
                "p": p,
                "r2": r * r,
                "residual_CV": float(np.std(resid, ddof=1) / np.mean(x)),
                "CV_explained": 1 - float(np.std(resid, ddof=1) / np.std(x, ddof=1)),
            }
        )
    _show(pd.DataFrame(conf), "%.4f")
    print("\n  ratios between families cancel any multiplicative completeness factor:")
    ratios = []
    anchor = max(families, key=lambda f: cohort[f].median()) if families else None
    for f in families:
        if anchor is None or f == anchor or cohort[f].min() <= 0 or cohort[anchor].min() <= 0:
            continue
        r = (cohort[f] / cohort[anchor]).to_numpy(dtype=float)
        ratios.append(
            {"ratio": f"{f}/{anchor}", "min": r.min(), "median": np.median(r), "max": r.max(),
             "fold": _fold(r), "CV": _cv(r)}
        )
    _show(pd.DataFrame(ratios))

    # ------------------------------------------- 3. within-sample concordance
    _section("3. Are the two haplotypes of one donor more alike than chance?")
    if "sample" in cohort.columns:
        strata: dict[str, np.ndarray] = {"any": np.zeros(len(cohort), dtype=np.int64)}
        if "superpopulation" in cohort.columns and cohort["superpopulation"].nunique() > 1:
            strata["superpop"] = pd.factorize(cohort["superpopulation"].astype(str))[0]
        if "haplotype" in cohort.columns:
            phasing = _phasing_stratum(cohort["haplotype"])
            if phasing.nunique() > 1:
                strata["phasing"] = pd.factorize(phasing)[0]
        # The two haplotype assemblies of one donor are carved out of one read
        # set, so their *sizes* already trade off; a family has to beat that.
        # Residualising each family on total assembled Mb asks whether anything
        # family-specific is left once the shared partition is taken out.
        panel = cohort[[*families, "assembly_mb", "nonsatellite_mb"]].copy()
        total = cohort["assembly_mb"].to_numpy(dtype=float)
        for f in families:
            x = cohort[f].to_numpy(dtype=float)
            if np.std(x) > 0:
                panel[f"{f} ~asm"] = x - np.polyval(np.polyfit(total, x, 1), total)
        conc = _concordance_test(
            panel, cohort["sample"].astype(str).to_numpy(), strata, n_perm, rng
        )
        if conc.empty:
            print("  fewer than four donors contribute two haplotypes each; skipped")
        else:
            _show(conc, "%.4f")
            print("\n  concordance is the double-entry (order-free) correlation of the two")
            print("  haplotypes of a donor.  p_similar is the fraction of random re-pairings")
            print("  at least as concordant; p_different the converse.  assembly_mb and")
            print("  nonsatellite_mb are the technical baseline: whatever they do is what a")
            print("  family has to beat to be more than an assembly-partitioning effect.")
            print("  A '~asm' row is the family residualised on total assembled Mb: what")
            print("  survives there is family-specific, what does not was the size trade-off.")
            print("  Note the two haplotypes of a donor are one paternal and one maternal")
            print("  gamete, i.e. not relatives, so excess similarity is a shared technical")
            print("  signature and not evidence of heritability.")
    else:
        print("  manifest has no sample column; skipped")

    # ---------------------------------------- 4. annotation-free cross-check
    _section("4. The same numbers with no query annotation at all")
    if not from_annotations:
        # The labels already came through the clusters, so re-deriving them here
        # would be a tautology.  Report what can be checked instead: how well
        # supported each label is, and what the same comparison gave on the one
        # run where both labellings existed.
        print("  labels already come from the clusters, so this cross-check would be")
        print("  circular on this run.  Reporting label support instead, and carrying")
        print("  the fidelity measured on results/acro3 where both labellings existed.")
        fid = pd.DataFrame(
            [
                {
                    "family": f,
                    "median_Mb": float(cohort[f].median()),
                    "labelled_by": "reference cluster",
                    "fidelity_r_acro3": TRANSFER_FIDELITY.get(f, np.nan),
                    "trust": "yes"
                    if TRANSFER_FIDELITY.get(f, 1.0) >= _MIN_TRANSFER_FIDELITY
                    else "NO",
                }
                for f in families
                if cohort[f].max() > 0
            ]
        )
        _show(fid)
        if not label_quality.empty:
            lq = (
                label_quality.merge(assembly_totals[["assembly", "assembly_bins"]], on="assembly")
                .set_index("assembly")
                .reindex(cohort.index)
                .join(meta.set_index("assembly"))
            )
            lq["labelled_frac"] = lq["labelled_bins"] / lq["assembly_bins"]
            lq["novel_frac"] = lq["novel_bins"] / lq["assembly_bins"]
            print(
                f"\n  per-haplotype labelled bins: median {lq['labelled_bins'].median():,.0f}"
                f", range {lq['labelled_bins'].min():,.0f}-{lq['labelled_bins'].max():,.0f}"
                f", CV of the labelled fraction {_cv(lq['labelled_frac'].to_numpy()):.4f}"
            )
            print(
                f"  per-haplotype novel-cluster bins: median {lq['novel_bins'].median():,.0f}"
                f", range {lq['novel_bins'].min():,.0f}-{lq['novel_bins'].max():,.0f}"
            )
            print("  -- a haplotype that got labelled less would look satellite-poor for")
            print("  every family at once, so this CV bounds the artefactual component.")
            if "superpopulation" in lq.columns and lq["superpopulation"].nunique() > 1:
                # The reference is one genome from one ancestry, so the obvious way
                # for an "ancestry effect" to be manufactured is for one group's
                # sequence to be less nameable from it.  That is checkable.
                bias = lq.groupby("superpopulation", observed=True)[
                    ["labelled_frac", "novel_frac"]
                ].median()
                bias["n"] = lq.groupby("superpopulation", observed=True).size()
                print("\n  label reach by ancestry -- if the reference names one group's")
                print("  sequence less well, that group looks satellite-poor for free:")
                _show(bias.reset_index(), "%.4f")
        free = pd.DataFrame()
    else:
        free = reference_free_copy_number(outdir, ref_clusters, families, bin_size)
    if from_annotations and free.empty:
        print("  no clustering in this run, or no cluster the reference could name; skipped")
    elif from_annotations:
        free = free.reindex(cohort.index).fillna(0.0)
        check = []
        for f in families:
            a = cohort[f].to_numpy(dtype=float)
            b = free[f].to_numpy(dtype=float)
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            check.append(
                {
                    "family": f,
                    "own_track_Mb": float(np.median(a)),
                    "ref_named_Mb": float(np.median(b)),
                    "recovered": float(np.median(b) / np.median(a)) if np.median(a) else np.nan,
                    "pearson_r": stats.pearsonr(a, b)[0],
                    "spearman": stats.spearmanr(a, b).statistic,
                }
            )
        _show(pd.DataFrame(check))
        print("\n  left column: each haplotype's own cenSat/RepeatMasker BEDs.  right column:")
        print("  bins in clusters whose name came from the reference's bins only.  The second")
        print("  never looked at the query assembly's annotation, so the correlation is a")
        print("  genuine check.  'recovered' < 1 because a cluster must be reference-nameable.")

    # ------------------------------------------------------ 5. per chromosome
    _section("5. Megabases per chromosome (cohort median [reference])")
    chrom_tab = per_chrom[per_chrom["feature"].isin(families)].merge(meta, on="assembly", how="left")
    is_ref = (
        chrom_tab["source"].astype(str) == _REFERENCE_SOURCE
        if "source" in chrom_tab.columns
        else pd.Series(False, index=chrom_tab.index)
    )
    # A haplotype with no bins of a family on a chromosome has no row at all, and
    # dropping it from the median would turn "carries none" into "not measured".
    dense = (
        chrom_tab[~is_ref]
        .pivot_table(
            index=["assembly", "chrom"],
            columns="feature",
            values="mb",
            aggfunc="sum",
            fill_value=0.0,
            observed=True,
        )
        .reindex(
            pd.MultiIndex.from_product(
                [cohort.index, sorted(per_chrom["chrom"].unique())], names=["assembly", "chrom"]
            ),
            fill_value=0.0,
        )
    )
    grid = dense.groupby(level="chrom", observed=True).median()
    ref_grid = chrom_tab[is_ref].pivot_table(
        index="chrom", columns="feature", values="mb", aggfunc="sum", fill_value=0.0, observed=True
    )
    if not grid.empty:
        show = grid.reindex(sorted(grid.index)).round(2)
        if not ref_grid.empty:
            ref_grid = ref_grid.reindex(index=show.index, columns=show.columns, fill_value=0.0)
            show = show.astype(str) + ref_grid.round(2).astype(str).radd("[").add("]")
        print(show.to_string())

    # ---------------------------------------------------- 5. cohort structure
    if "superpopulation" in cohort.columns and cohort["superpopulation"].nunique() > 1:
        _section("6. Does copy number track ancestry label?")
        groups = cohort.groupby("superpopulation", observed=True)
        rowsk = []
        for f in families:
            arrays = [g[f].to_numpy(dtype=float) for _, g in groups if len(g) > 1]
            if len(arrays) < 2 or np.std(np.concatenate(arrays)) == 0:
                continue
            H, p = stats.kruskal(*arrays)
            rowsk.append(
                {"family": f, "kruskal_H": H, "p": p, "p_bonferroni": min(1.0, p * len(families)),
                 **{str(k): float(np.median(g[f])) for k, g in groups}}
            )
        _show(pd.DataFrame(rowsk), "%.4f")
        if "haplotype" in cohort.columns:
            cross = pd.crosstab(_phasing_stratum(cohort["haplotype"]), cohort["superpopulation"])
            print("\n  phasing stratum x superpopulation:")
            print(cross.to_string())
            _ancestry_adjusted(cohort, families)

    _section("Caveats that this report cannot remove")
    print("  * A bin is counted where its dominant feature won the argmax.  An array")
    print("    shorter than one bin can never be dominant, so families whose arrays are")
    print("    smaller than the bin size are undercounted by construction; the")
    print("    covered_mb column in the parquet is the argmax-free alternative.")
    print("  * Unplaced bins keep a chromosome label from the assembly's own alias")
    print("    file, not from an alignment.  Per-chromosome numbers inherit whatever")
    print("    that assignment got wrong; per-assembly totals do not.")
    print("  * A collapsed or dropped array is indistinguishable from a short one.")
    print("    Every number here is a floor on true copy number.")

    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    outdir = Path(args[0]) if args else Path("results/acro3")
    table = copy_number_report(outdir)
    dest = outdir / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "copy_number.parquet"
    table.to_parquet(path, index=False)
    print(f"\nwrote {path}  ({len(table):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
