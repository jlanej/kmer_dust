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
both parquet files are streamed in aligned batches and reduced into a small
(assembly, chrom, feature) table rather than being joined in memory.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

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


def aggregate_counts(outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    """Reduce the run to per-(assembly, chrom, feature) counts and covered bases.

    Returns the long count table, a per-assembly total table, the bin size in
    base pairs inferred from the bin table, and -- when the run got as far as
    clustering -- the (cluster, feature) counts restricted to *reference* bins,
    which is what lets a cluster be named without consulting any assembly's own
    annotation.
    """
    rows_path = outdir / "matrix" / "rows.parquet"
    ann_path = outdir / "annotate" / "annotations.parquet"
    for path in (rows_path, ann_path):
        if not path.exists():
            raise FileNotFoundError(f"missing input: {path}")
    cluster_path = outdir / "cluster" / "clusters.parquet"
    use_clusters = cluster_path.exists()

    ann_schema = pq.ParquetFile(ann_path).schema_arrow
    features = [
        name.removeprefix("frac_") for name in ann_schema.names if name.startswith("frac_")
    ]
    frac_columns = [f"frac_{f}" for f in features]

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
    widths: list[int] = []
    n_rows = 0

    sources: list[tuple[Path, list[str]]] = [
        (rows_path, ["bin_uid", "assembly", "chrom", "source", "start", "end", "n_acgt"]),
        (ann_path, ["bin_uid", "dominant_feature", *frac_columns]),
    ]
    if use_clusters:
        sources.append((cluster_path, ["bin_uid", "cluster"]))

    for batches in _aligned_batches(sources):
        rows_batch, ann_batch = batches[0], batches[1]
        if rows_batch.num_rows != ann_batch.num_rows:
            raise RuntimeError(
                "rows.parquet and annotations.parquet have different lengths "
                f"({rows_batch.num_rows} vs {ann_batch.num_rows} in one batch). "
                "annotate promises one row per bin in bin order; re-run it."
            )
        for other, label in zip(batches[1:], ("annotate/annotations", "cluster/clusters")):
            if not pc.all(
                pc.equal(rows_batch.column("bin_uid"), other.column("bin_uid"))
            ).as_py():
                raise RuntimeError(
                    f"bin_uid order differs between matrix/rows.parquet and {label}"
                    ".parquet. Every aggregate here assumes the positional alignment "
                    "that the pipeline guarantees; re-run that stage against this "
                    "rows.parquet rather than trusting a join."
                )

        n = rows_batch.num_rows
        n_rows += n
        a_codes = assemblies.encode_dictionary(rows_batch.column("assembly"))
        c_codes = chroms.encode_dictionary(rows_batch.column("chrom"))
        f_codes = dominant.encode_dictionary(ann_batch.column("dominant_feature"))

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

        fracs = np.empty((n, len(frac_columns)), dtype=np.float64)
        for j, col in enumerate(frac_columns):
            fracs[:, j] = ann_batch.column(col).to_numpy(zero_copy_only=False)
        acgt = rows_batch.column("n_acgt").to_numpy(zero_copy_only=False).astype(np.float64)
        # Bins are written assembly by assembly, so runs are few; reduceat is
        # correct for any ordering and avoids a mask per assembly per batch.
        starts = np.concatenate(([0], np.flatnonzero(a_codes[1:] != a_codes[:-1]) + 1))
        sums = np.add.reduceat(fracs, starts, axis=0)
        acgt_sums = np.add.reduceat(acgt, starts)
        run_len = np.diff(np.append(starts, n))
        for i, code in enumerate(a_codes[starts].tolist()):
            covered[code] += sums[i]
            bins_per_assembly[code] += int(run_len[i])
            acgt_per_assembly[code] += float(acgt_sums[i])

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
        ]
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
        ]
    )
    return long, totals.merge(covered_long, on="assembly", how="left"), bin_size, ref_clusters


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
    long, totals, bin_size, ref_clusters = aggregate_counts(outdir)

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

    # ------------------------------------------------- 1. how much is carried
    _section("1. Megabases carried per haplotype (dominant_feature x 10 kb)")
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
                "reference": reference[f].iloc[0] if ref_name else np.nan,
            }
            for f in families + ["satellite_mb", "nonsatellite_mb", "assembly_mb"]
        ]
    )
    _show(summary)
    if ref_name:
        print(f"\n  reference column is {ref_name}; it is one haplotype and is excluded")
        print("  from every cohort statistic above and below.")
    silent = [
        f
        for f in families
        if cohort[f].max() == 0
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
    free = reference_free_copy_number(outdir, ref_clusters, families, bin_size)
    if free.empty:
        print("  no clustering in this run, or no cluster the reference could name; skipped")
    else:
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
            print("\n  phasing stratum x superpopulation -- read the table above against this:")
            print(cross.to_string())

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
