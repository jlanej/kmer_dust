"""What the reference cannot describe: novel clusters, and cross-chromosome sharing.

Two questions, one script, because they are the same question asked twice --
*what does a cluster know that a coordinate system does not?*

--------------------------------------------------------------------------
Q1.  Is "novel" real?
--------------------------------------------------------------------------
``backprop.infer_annotations`` marks a cluster ``novel`` when no reference bin
fell in it: sequence the assemblies share with each other and not with CHM13.
That framing has an obvious failure mode and it is the reason this script
exists.  The reference is *one assembly out of N*.  If a cluster has 70 bins and
the reference contributes 3.2 % of bins, the chance that none of them is a
reference bin is ~10 % **with no biology involved at all**.  Novelty measured
against a single genome is mostly a statement about sample size.

So the report computes three nulls, in increasing order of how much they know:

1. **Hypergeometric.**  ``P(0 reference bins | cluster size)`` under a random
   draw.  Cheap, analytic, and already enough to disqualify small clusters.
2. **Label permutation.**  Shuffle the reference/assembly label across
   clustered bins, preserving every cluster size.  Confirms (1) empirically.
3. **Leave-one-assembly-out.**  The one that matters.  For every assembly in
   the run, count the clusters that contain no bin *from that assembly*.  The
   reference is then just one point in an empirical distribution of N such
   counts.  If CHM13 sits inside that distribution, "novel with respect to the
   reference" is not a property of the reference -- it is the ordinary rate at
   which any single genome misses a cluster, and the correct interpretation of
   the novel set is presence/absence polymorphism, not reference gaps.

Nulls (1) and (2) are *conservative* here and it is worth knowing why: a cluster
is a locus, and a locus contributes roughly one bin per assembly, so the
reference is far less likely to be missing than a random draw of the same size
would suggest.  (3) has no such assumption because it uses the real clusters.

A sharper version of (3) is also reported: the number of clusters for which a
given assembly is the *sole* absentee.  A cluster carried by every haplotype but
one is the strongest possible claim of assembly-specific absence, and comparing
the reference's count against every haplotype's count is a like-for-like test.

--------------------------------------------------------------------------
Q2.  What is shared across the acrocentric short arms, beyond rDNA?
--------------------------------------------------------------------------
Guarracino et al. (Nature, 2023) showed the acrocentric short arms recombine
with each other.  kmer-dust should see that as clusters drawing bins from
several chromosomes *within one haplotype* -- not merely across the cohort,
which a locus that simply sits on different chromosomes in different people
would also produce.  The distinction needs two nulls, and they answer different
questions:

* **free null** -- permute chromosome labels *within each assembly* over all
  clustered bins.  Preserves each assembly's chromosome composition and every
  cluster's size, and destroys all chromosome structure.  This is literally
  "given its size and the overall chromosome composition, how multi-chromosomal
  would this cluster be by chance".  Nearly every cluster falls *below* it --
  that is the signal that clusters are loci.
* **fixed null** -- permute chromosome labels *within each cluster*.  The
  cluster's own chromosome composition is held fixed and only its distribution
  over haplotypes is randomised.  Observed at the null means each haplotype
  samples the cluster's whole chromosome repertoire (genuine within-haplotype
  sharing); observed far below it means each haplotype carries the cluster on
  one chromosome and it is *which* chromosome that varies between people.

Two more traps are worth naming because both were live on the acrocentric run.
The chromosome-pair table's null is scoped to the *selected clusters' own bins*:
a genome-wide null under-predicts every chr21 and chr22 pair, because the short
arm is a far larger share of those chromosomes, and every short-arm cluster then
looks enriched for them whether or not anything is shared.  And a chromosome
that takes part in sharing at all inflates every pair it appears in, so the pair
table also reports each chromosome's marginal ratio and a ``pair_only`` column
with both marginals divided out.  That last column is the one a
pseudo-homologous-region claim actually has to move.

Both permutations are bin-level, which would be wrong if a cluster's bins came
in long contiguous runs -- the bins would not be independent draws.  The report
prints the observed bins-per-contiguous-segment so the assumption is checkable;
on the acrocentric run it is 1.08, i.e. cluster membership is not contiguous and
bin-level permutation is the right granularity.

Everything is keyed on ``chrom``, which is valid for unplaced bins (it comes
from the ``chrN_*_random`` alias); ``start``/``end`` are only ever used where
``placed`` is true, and positional statements are restricted to the reference,
which is the only assembly guaranteed to be complete.

Nothing here hard-codes the cohort size, the chromosome list, or the feature
vocabulary: all three are read off the run.  Per-assembly annotation is used
when it exists (``annotate.annotate_assemblies: true``) and silently skipped
when it does not, because it is a cross-check and never an input.

Usage::

    PYTHONPATH=src python scripts/analysis/novel.py [OUTDIR] [N_PERMUTATIONS]

Writes ``<OUTDIR>/analysis/novel.parquet``: one row per cluster, every statistic
below, plus a ``frac_<chrom>`` column per chromosome in the run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import gammaln

# --------------------------------------------------------------------------
# thresholds -- all arbitrary, so all named and in one place
# --------------------------------------------------------------------------

#: A cluster is "chromosome-specific" when a haplotype carrying it puts it on
#: essentially one chromosome.  Alpha-satellite HOR sits at 1.0.
PURE_CHROM_PER_ASM = 1.25

#: ...and "multi-chromosomal" when a haplotype carrying it puts it on two or
#: more.  Halfway between one chromosome and two is the only defensible line.
SHARED_CHROM_PER_ASM = 1.5

#: How far below the fixed null a cluster must fall before we say the mixing
#: happens *between* haplotypes rather than inside them.  Three standard
#: deviations of a permutation statistic.
WITHIN_Z = -3.0

#: Reference bins must be this proximal (fraction below the reference's own
#: alpha-satellite array) for a cluster to count as short-arm/pericentromeric.
PROXIMAL_FRAC = 0.8

#: ...over at least this many placed reference bins, or the fraction is noise.
MIN_REF_PLACED = 5

#: Feature classes excluded from "what else is shared": the known answer and
#: the vocabulary-free ones.
EXCLUDE_FEATURES = ("rdna", "rrna", "simple_repeat", "low_complexity", "satellite")

NOISE_CLUSTER = -1


# --------------------------------------------------------------------------
# loading -- 19 M bins, so integer codes and never a Python string per row
# --------------------------------------------------------------------------


@dataclass
class Run:
    """Everything the report needs, as parallel arrays over clustered bins."""

    cluster: np.ndarray  # compact cluster index, 0..K-1
    cluster_id: np.ndarray  # original cluster labels, len K
    asm: np.ndarray  # compact assembly index
    asm_name: np.ndarray
    smp: np.ndarray  # compact sample index
    smp_name: np.ndarray
    chrom: np.ndarray  # compact chromosome index
    chrom_name: np.ndarray
    contig: np.ndarray
    start: np.ndarray
    end: np.ndarray
    placed: np.ndarray
    gc: np.ndarray
    n_sketch: np.ndarray
    is_ref: np.ndarray
    feature: np.ndarray  # compact feature index, -1 = unannotated/absent
    feature_name: np.ndarray
    n_noise: int
    n_bins_total: int
    ref_name: str
    has_asm_annotation: bool
    pop_of_sample: dict[str, str]

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_id)

    @property
    def n_assemblies(self) -> int:
        return len(self.asm_name)

    @property
    def n_chroms(self) -> int:
        return len(self.chrom_name)


def _column(table: pa.Table, name: str) -> pa.ChunkedArray:
    if name not in table.column_names:
        raise KeyError(f"{name!r} missing; found {table.column_names}")
    return table.column(name)


def _factorize(table: pa.Table, name: str, fill: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Dictionary-encode a string column into (int32 codes, categories).

    Arrow does this without ever materialising 19 M Python strings, which is the
    difference between a 400 MB array and a 3 GB one.
    """
    arr = _column(table, name).combine_chunks()
    if arr.null_count:
        arr = arr.fill_null(fill)
    if not pa.types.is_dictionary(arr.type):
        arr = arr.dictionary_encode()
    codes = arr.indices.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    cats = np.asarray(arr.dictionary.to_pylist(), dtype=object)
    return codes, cats


def _numpy(table: pa.Table, name: str, dtype: str) -> np.ndarray:
    return _column(table, name).combine_chunks().to_numpy(zero_copy_only=False).astype(dtype)


def _aligned(reference: Path, other: Path, key: str = "bin_uid") -> np.ndarray | None:
    """Positional index of ``other``'s rows in ``reference`` order, or None if identical.

    Every table in an output directory is written from one row order, so the
    fast path is the real path; the merge exists so a hand-assembled directory
    still works instead of silently mis-joining.  Both key columns are read and
    dropped inside this function -- at 19 M bins ``bin_uid`` is the single most
    expensive column in the run and it is needed for nothing else.
    """
    left = pq.read_table(reference, columns=[key])
    right = pq.read_table(other, columns=[key])
    if left.num_rows == right.num_rows:
        same = pa.compute.all(pa.compute.equal(_column(left, key), _column(right, key)))
        if same.is_valid and same.as_py():
            return None
    lookup = pd.Index(_column(right, key).to_pandas())
    idx = lookup.get_indexer(pd.Index(_column(left, key).to_pandas()))
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} bin_uid in {reference} have no match in {other}")
    return idx


def _load_populations(outdir: Path) -> dict[str, str]:
    path = outdir / "manifest.tsv"
    if not path.exists():
        return {}
    manifest = pd.read_csv(path, sep="\t", dtype=str)
    for col in ("superpopulation", "population"):
        if col in manifest.columns and manifest[col].notna().any():
            keep = manifest[["sample", col]].dropna()
            return dict(zip(keep["sample"].astype(str), keep[col].astype(str)))
    return {}


#: Columns worth reading as Arrow dictionaries: low cardinality, so the codes
#: come straight off the parquet pages and the strings are never materialised.
_DICT_COLUMNS = ("assembly", "sample", "haplotype", "source", "contig", "chrom")


def load_run(outdir: Path) -> Run:
    """Join rows, clusters and annotations, and drop noise."""
    rows_path = outdir / "matrix" / "rows.parquet"
    cluster_path = outdir / "cluster" / "clusters.parquet"
    ann_path = outdir / "annotate" / "annotations.parquet"

    idx = _aligned(rows_path, cluster_path)
    labels = _numpy(pq.read_table(cluster_path, columns=["cluster"]), "cluster", "int64")
    if idx is not None:
        labels = labels[idx]
    keep = labels != NOISE_CLUSTER
    n_noise = int((~keep).sum())
    n_bins_total = len(labels)

    feature_codes = np.full(n_bins_total, -1, dtype=np.int32)
    feature_names: np.ndarray = np.empty(0, dtype=object)
    if ann_path.exists():
        aidx = _aligned(rows_path, ann_path)
        ann = pq.read_table(ann_path, columns=["dominant_feature"],
                            read_dictionary=["dominant_feature"])
        codes, feature_names = _factorize(ann, "dominant_feature", fill="unannotated")
        del ann
        if aidx is not None:
            codes = codes[aidx]
        unlabelled = np.flatnonzero(feature_names == "unannotated")
        feature_codes = codes.astype(np.int32)
        if len(unlabelled):
            feature_codes[feature_codes == unlabelled[0]] = -1

    available = set(pq.read_schema(rows_path).names)
    wanted = [*_DICT_COLUMNS, "placed", "start", "end", "gc", "n_sketch"]
    missing = [c for c in ("assembly", "chrom", "start", "end") if c not in available]
    if missing:
        raise ValueError(f"{rows_path} lacks required column(s) {missing}")
    rows = pq.read_table(
        rows_path,
        columns=[c for c in wanted if c in available],
        read_dictionary=[c for c in _DICT_COLUMNS if c in available],
    )

    asm, asm_name = _factorize(rows, "assembly")
    is_ref = np.zeros(rows.num_rows, dtype=bool)
    for name, marker in (("source", "t2t"), ("haplotype", "ref")):
        if name in available:
            codes, vocab = _factorize(rows, name)
            is_ref |= np.isin(codes, [c for c, v in enumerate(vocab) if v == marker])

    smp, smp_name = (_factorize(rows, "sample") if "sample" in available else (asm, asm_name))
    chrom, chrom_name = _factorize(rows, "chrom")
    contig, _ = _factorize(rows, "contig") if "contig" in available else (chrom, chrom_name)
    start = _numpy(rows, "start", "int32")
    end = _numpy(rows, "end", "int32")
    # No ``placed`` column means an output directory older than that flag.  The
    # reference is a T2T assembly whose contigs are chromosomes, so its
    # coordinates are safe; nothing else is, and the caveat is that unplaced
    # start/end are contig-local.
    placed = _numpy(rows, "placed", "bool") if "placed" in available else is_ref.copy()
    gc = (_numpy(rows, "gc", "float32") if "gc" in available
          else np.full(rows.num_rows, np.nan, dtype="float32"))
    n_sketch = (_numpy(rows, "n_sketch", "float32") if "n_sketch" in available
                else np.full(rows.num_rows, np.nan, dtype="float32"))
    del rows

    cluster_id, cluster = np.unique(labels[keep], return_inverse=True)
    ref_names = sorted({str(asm_name[a]) for a in np.unique(asm[is_ref])})
    asm_bins = keep & ~is_ref

    return Run(
        cluster=cluster.astype(np.int32),
        cluster_id=cluster_id,
        asm=asm[keep], asm_name=asm_name,
        smp=smp[keep], smp_name=smp_name,
        chrom=chrom[keep], chrom_name=chrom_name,
        contig=contig[keep],
        start=start[keep], end=end[keep], placed=placed[keep],
        gc=gc[keep], n_sketch=n_sketch[keep],
        is_ref=is_ref[keep],
        feature=feature_codes[keep],
        feature_name=feature_names,
        n_noise=n_noise,
        n_bins_total=n_bins_total,
        ref_name=", ".join(ref_names) if ref_names else "",
        has_asm_annotation=bool(np.any(feature_codes[asm_bins] >= 0)),
        pop_of_sample=_load_populations(outdir),
    )


# --------------------------------------------------------------------------
# small vectorised helpers
# --------------------------------------------------------------------------


def _blocks(key: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted order and group boundaries, for shuffling inside groups."""
    order = np.argsort(key, kind="stable")
    bounds = np.searchsorted(key[order], np.arange(n_groups + 1))
    return order, bounds


def _shuffle_within(
    values: np.ndarray, order: np.ndarray, bounds: np.ndarray, rng: np.random.Generator,
    out: np.ndarray,
) -> np.ndarray:
    for i in range(len(bounds) - 1):
        block = order[bounds[i] : bounds[i + 1]]
        out[block] = rng.permutation(values[block])
    return out


def _mode_per_group(
    group: np.ndarray, value: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    """Most frequent ``value`` per group, and its share.  Ties break on value.

    Distinct (group, value) pairs are found by one sort of a packed key rather
    than ``np.unique(..., axis=1)``, which copies the whole 2 x N block.
    """
    span = int(value.max()) + 1 if len(value) else 1
    packed, counts = np.unique(group.astype(np.int64) * span + value.astype(np.int64),
                               return_counts=True)
    g, v = packed // span, packed % span
    order = np.lexsort((v, -counts, g))
    g, v, c = g[order], v[order], counts[order]
    first = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    best = np.zeros(n_groups, dtype=np.int64)
    share = np.zeros(n_groups, dtype=np.float64)
    total = np.bincount(group, minlength=n_groups)
    best[g[first]] = v[first]
    share[g[first]] = c[first] / np.maximum(total[g[first]], 1)
    return best, share


def _median_per_group(group: np.ndarray, value: np.ndarray, n_groups: int) -> np.ndarray:
    """Median of ``value`` per group, by one lexsort instead of a pandas groupby."""
    order = np.lexsort((value, group))
    g = group[order]
    v = value[order]
    starts = np.searchsorted(g, np.arange(n_groups), side="left")
    ends = np.searchsorted(g, np.arange(n_groups), side="right")
    size = ends - starts
    out = np.full(n_groups, np.nan)
    ok = size > 0
    lo = starts[ok] + (size[ok] - 1) // 2
    hi = starts[ok] + size[ok] // 2
    out[ok] = 0.5 * (v[lo].astype(np.float64) + v[hi].astype(np.float64))
    return out


def _popcount(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    out = np.zeros_like(values)
    work = values.copy()
    while np.any(work):
        out += (work & 1).astype(out.dtype)
        work >>= 1
    return out


def _hypergeom_zero(size: np.ndarray, n_total: int, n_ref: int) -> np.ndarray:
    """P(a random draw of ``size`` bins contains no reference bin)."""
    n_other = n_total - n_ref
    size = np.asarray(size, dtype=np.float64)
    ok = size <= n_other
    out = np.zeros_like(size)
    log_p = (
        gammaln(n_other + 1) - gammaln(np.maximum(n_other - size, 0) + 1)
        + gammaln(n_total - size + 1) - gammaln(n_total + 1)
    )
    out[ok] = np.exp(log_p[ok])
    return out


def _fmt_bp(bp: float) -> str:
    return f"{bp / 1e6:,.1f} Mb"


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _sub(title: str) -> None:
    print()
    print(f"-- {title}")


# --------------------------------------------------------------------------
# Q1 -- is "novel" anything more than the reference being one genome?
# --------------------------------------------------------------------------


def cluster_presence(run: Run) -> np.ndarray:
    """Boolean ``n_clusters x n_assemblies`` presence matrix."""
    seen = np.zeros(run.n_clusters * run.n_assemblies, dtype=bool)
    seen[run.cluster.astype(np.int64) * run.n_assemblies + run.asm] = True
    return seen.reshape(run.n_clusters, run.n_assemblies)


def report_novel(
    run: Run, seen: np.ndarray, size: np.ndarray, n_perm: int, findings: dict[str, object]
) -> pd.DataFrame:
    """Q1: inventory the novel clusters, then try three ways to explain them away."""
    n_ref_bins = np.bincount(run.cluster, weights=run.is_ref, minlength=run.n_clusters).astype(np.int64)
    novel = n_ref_bins == 0
    n_clustered = len(run.cluster)
    n_ref_clustered = int(run.is_ref.sum())
    bp = np.bincount(run.cluster, weights=(run.end - run.start), minlength=run.n_clusters)
    n_asm = seen.sum(axis=1)

    _rule("Q1  NOVEL MATERIAL -- clusters with no reference bin")
    print(f"reference            : {run.ref_name or '(none in this run)'}")
    print(f"clustered bins       : {n_clustered:,}  ({run.n_noise:,} noise, "
          f"{run.n_noise / max(run.n_bins_total, 1):.1%} of all bins)")
    print(f"reference share      : {n_ref_clustered:,} of {n_clustered:,} clustered bins "
          f"= {n_ref_clustered / max(n_clustered, 1):.2%}")
    print(f"clusters             : {run.n_clusters:,}")
    asm_bins = int((~run.is_ref).sum())
    novel_bins = int(size[novel].sum())
    print(f"novel clusters       : {int(novel.sum()):,}  ({novel.mean():.1%} of clusters)")
    print(f"novel bins           : {novel_bins:,} = {_fmt_bp(bp[novel].sum())} of assembly sequence "
          f"({novel_bins / max(asm_bins, 1):.2%} of clustered assembly bins)")
    if run.n_assemblies > 1:
        print(f"                       {_fmt_bp(bp[novel].sum() / (run.n_assemblies - 1))} per haplotype")

    findings["novel_clusters"] = int(novel.sum())
    findings["novel_bp"] = float(bp[novel].sum())

    if n_ref_clustered == 0:
        print("\nno reference bins in this run: every cluster is trivially novel, nothing to test.")
        return pd.DataFrame({"cluster": run.cluster_id, "novel": novel, "n_ref_bins": n_ref_bins})

    # --- null 1: analytic ---------------------------------------------------
    p_chance = _hypergeom_zero(size, n_clustered, n_ref_clustered)
    findings["expected_by_chance"] = float(p_chance.sum())
    _sub("null 1 -- hypergeometric: a size-matched random draw missing the reference")
    print(f"expected novel clusters by chance : {p_chance.sum():.1f}")
    print(f"observed novel clusters           : {int(novel.sum())}")
    edges = np.array([50, 60, 80, 120, 200, 400, 1000, np.inf])
    edges = np.r_[size.min(), edges[edges > size.min()]]
    print(f"{'cluster size':>16}  {'clusters':>9}  {'observed':>9}  {'expected':>9}  {'excess':>8}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (size >= lo) & (size < hi)
        if not m.any():
            continue
        label = f"{int(lo)}-{int(hi) - 1}" if np.isfinite(hi) else f">={int(lo)}"
        print(f"{label:>16}  {int(m.sum()):>9,}  {int(novel[m].sum()):>9,}  "
              f"{p_chance[m].sum():>9.1f}  {novel[m].sum() - p_chance[m].sum():>8.1f}")

    # --- null 2: label permutation -----------------------------------------
    rng = np.random.default_rng(7)
    draws = np.empty(n_perm, dtype=np.int64)
    for i in range(n_perm):
        shuffled = rng.permutation(run.is_ref)
        draws[i] = int((np.bincount(run.cluster, weights=shuffled, minlength=run.n_clusters) == 0).sum())
    _sub(f"null 2 -- permuting the reference label over clustered bins ({n_perm} draws)")
    print(f"null novel-cluster count : mean {draws.mean():.1f}  sd {draws.std():.1f}  "
          f"range {draws.min()}-{draws.max()}")
    print(f"observed                 : {int(novel.sum())}")
    print("both nulls are conservative: a cluster is a locus and contributes about one bin per")
    print("assembly, so the reference is less droppable than a random draw of the same size.")

    # --- null 3: leave one assembly out ------------------------------------
    ref_col = _reference_column(run)
    absent = ~seen
    per_assembly = absent.sum(axis=0)
    loao = pd.DataFrame({
        "assembly": [str(a) for a in run.asm_name],
        "is_reference": np.arange(run.n_assemblies) == (ref_col if ref_col is not None else -1),
        "clustered_bins": np.bincount(run.asm, minlength=run.n_assemblies),
        "clusters_absent": per_assembly,
        "bins_in_absent": [int(size[absent[:, j]].sum()) for j in range(run.n_assemblies)],
    })
    _sub("null 3 -- leave-one-assembly-out: is the reference special at all?")
    if ref_col is None:
        print("could not identify a single reference assembly; skipping.")
    else:
        others = np.delete(per_assembly, ref_col)
        rank = int((per_assembly <= per_assembly[ref_col]).sum())
        findings["loao_rank"] = rank
        findings["loao_ref"] = int(per_assembly[ref_col])
        findings["loao_median"] = float(np.median(others))
        findings["loao_max"] = int(others.max())
        findings["loao_p"] = (run.n_assemblies - rank + 1) / run.n_assemblies
        print(f"clusters absent from the reference        : {per_assembly[ref_col]:,}")
        print(f"clusters absent from another assembly     : median {np.median(others):,.0f}, "
              f"range {others.min():,}-{others.max():,}")
        print(f"the reference ranks {rank} of {run.n_assemblies} "
              f"(empirical one-sided p = {(run.n_assemblies - rank + 1) / run.n_assemblies:.3f})")
        # recurrence-stratified: does the reference miss the *well supported* clusters?
        print()
        print(f"{'recurrence floor':>18}  {'reference':>10}  {'others: median':>15}"
              f"  {'max':>7}  {'rank':>6}")
        n_other_asm = seen.sum(axis=1)[:, None] - seen
        for frac in (0.25, 0.5, 0.75, 0.95):
            floor = int(np.ceil(frac * (run.n_assemblies - 1)))
            counts = ((absent) & (n_other_asm >= floor)).sum(axis=0)
            rest = np.delete(counts, ref_col)
            r = int((counts <= counts[ref_col]).sum())
            print(f"{f'>={floor} others':>18}  {counts[ref_col]:>10,}  {np.median(rest):>15,.0f}  "
                  f"{rest.max():>7,}  {r:>3}/{run.n_assemblies}")

        # the sharpest form: sole absentee
        sole = seen.sum(axis=1) == run.n_assemblies - 1
        sole_of = np.full(run.n_clusters, -1, dtype=np.int64)
        if sole.any():
            sole_of[sole] = np.argmax(absent[sole], axis=1)
        tally = np.bincount(sole_of[sole], minlength=run.n_assemblies) if sole.any() else np.zeros(
            run.n_assemblies, dtype=np.int64)
        rest = np.delete(tally, ref_col)
        _sub("null 3b -- clusters present in every assembly but one")
        print(f"total such clusters                 : {int(sole.sum()):,}")
        print(f"the reference is the sole absentee   : {tally[ref_col]:,}")
        print(f"a haplotype is the sole absentee     : median {np.median(rest):,.0f}, "
              f"range {rest.min():,}-{rest.max():,}")
        print(f"the reference ranks {int((tally <= tally[ref_col]).sum())} of {run.n_assemblies}")
        findings["sole_ref"] = int(tally[ref_col])
        findings["sole_median"] = float(np.median(rest))
        findings["sole_rank"] = int((tally <= tally[ref_col]).sum())

    loao_sorted = loao.sort_values("clusters_absent").reset_index(drop=True)
    keep_rows = loao_sorted
    if len(loao_sorted) > 24:
        edge = pd.concat([loao_sorted.head(10), loao_sorted.tail(10)])
        keep_rows = pd.concat([edge, loao_sorted[loao_sorted["is_reference"]]]).drop_duplicates()
        keep_rows = keep_rows.sort_values("clusters_absent")
    shown = ("extremes plus the reference" if len(keep_rows) < len(loao_sorted) else "full table")
    _sub(f"leave-one-assembly-out ({shown}, reference marked *)")
    print(f"   {'assembly':<44} {'clustered':>10}  {'absent':>6}  {'bins':>9}")
    for _, row in keep_rows.iterrows():
        star = "*" if bool(row["is_reference"]) else " "
        print(f" {star} {str(row['assembly'])[:44]:<44} {row['clustered_bins']:>10,}  "
              f"{row['clusters_absent']:>6,}  {row['bins_in_absent']:>9,}")

    return pd.DataFrame({
        "cluster": run.cluster_id,
        "novel": novel,
        "n_ref_bins": n_ref_bins,
        "p_novel_chance": p_chance,
        "n_missing_assemblies": run.n_assemblies - n_asm,
    })


def _reference_column(run: Run) -> int | None:
    """Assembly index of the reference, or None when it is absent or plural."""
    codes = np.unique(run.asm[run.is_ref])
    return int(codes[0]) if len(codes) == 1 else None


def describe_novel(
    run: Run, stats: pd.DataFrame, seen: np.ndarray, findings: dict[str, object]
) -> None:
    """What the novel material *is*, using only run-internal evidence."""
    novel = stats["novel"].to_numpy()
    per_bin_novel = novel[run.cluster]
    asm_bins = ~run.is_ref

    _sub("what the novel bins look like (assembly bins only)")
    rowsdef = [
        ("median GC", run.gc),
        ("median sketch size (distinct 31-mers/bin)", run.n_sketch),
        ("fraction on chromosome-placed contigs", run.placed.astype(np.float32)),
    ]
    print(f"{'':<44}{'novel':>12}{'reference-supported':>22}")
    for label, values in rowsdef:
        a = values[asm_bins & per_bin_novel]
        b = values[asm_bins & ~per_bin_novel]
        fn = np.mean if label.startswith("fraction") else np.median
        left, right = (float(fn(a)) if len(a) else np.nan, float(fn(b)) if len(b) else np.nan)
        print(f"{label:<44}{left:>12.3f}{right:>22.3f}")
        if label.startswith("median sketch"):
            findings["sketch_novel"], findings["sketch_supported"] = left, right
    print()
    print("a low sketch size is the tell: a bin whose 31-mers repeat inside it collapses to a")
    print("few distinct hashes, so this is satellite, not novel euchromatin.")

    _sub("chromosome attribution of novel bins")
    tab = pd.DataFrame({
        "novel": np.bincount(run.chrom[asm_bins & per_bin_novel], minlength=run.n_chroms),
        "supported": np.bincount(run.chrom[asm_bins & ~per_bin_novel], minlength=run.n_chroms),
    }, index=[str(c) for c in run.chrom_name])
    tab["novel %"] = 100 * tab["novel"] / max(tab["novel"].sum(), 1)
    tab["supported %"] = 100 * tab["supported"] / max(tab["supported"].sum(), 1)
    print(tab.to_string(float_format=lambda x: f"{x:.1f}"))

    if run.has_asm_annotation:
        _sub("independently annotated identity of the novel clusters (per-assembly tracks)")
        top = _cluster_feature(run, reference_only=False)
        frame = pd.DataFrame({"feature": top, "novel": novel})
        cross = pd.crosstab(frame["feature"].fillna("(unannotated)"), frame["novel"])
        for col in (False, True):
            if col not in cross.columns:
                cross[col] = 0
        cross = cross.rename(columns={False: "supported", True: "novel"})
        cross["novel rate"] = cross["novel"] / (cross["novel"] + cross["supported"])
        print(cross.sort_values("novel", ascending=False).head(15).to_string(
            float_format=lambda x: f"{x:.3f}"))
        print()
        print("read the *rate* column against the leave-one-assembly-out table below, not against 0.")
        _sub("per-feature dropout rate: reference vs every other assembly")
        ref_col = _reference_column(run)
        if ref_col is not None:
            records = []
            for feature in pd.unique(top.dropna()):
                mask = (top == feature).to_numpy()
                if mask.sum() < 5:
                    continue
                rates = (~seen[mask]).mean(axis=0)
                rest = np.delete(rates, ref_col)
                records.append({
                    "feature": feature, "clusters": int(mask.sum()),
                    "reference": rates[ref_col], "others median": np.median(rest),
                    "others max": rest.max(),
                    "rank": f"{int((rates <= rates[ref_col]).sum())}/{run.n_assemblies}",
                })
            print(pd.DataFrame(records).sort_values("clusters", ascending=False).to_string(
                index=False, float_format=lambda x: f"{x:.3f}"))


def _cluster_feature(run: Run, reference_only: bool) -> pd.Series:
    """Modal annotated feature per cluster, from reference bins or assembly bins."""
    mask = run.feature >= 0
    mask &= run.is_ref if reference_only else ~run.is_ref
    out = pd.Series(pd.NA, index=range(run.n_clusters), dtype=object)
    if not mask.any():
        return out
    best, _ = _mode_per_group(run.cluster[mask], run.feature[mask], run.n_clusters)
    has = np.bincount(run.cluster[mask], minlength=run.n_clusters) > 0
    out.loc[np.flatnonzero(has)] = [str(run.feature_name[b]) for b in best[has]]
    return out


# --------------------------------------------------------------------------
# Q2 -- cross-chromosome sharing
# --------------------------------------------------------------------------


@dataclass
class Sharing:
    pair_cluster: np.ndarray
    pair_index: np.ndarray
    n_pairs_per_cluster: np.ndarray
    chrom_per_asm: np.ndarray
    frac_asm_multi: np.ndarray
    free_mean: np.ndarray
    free_z: np.ndarray
    free_p: np.ndarray
    free_multi: np.ndarray
    fixed_mean: np.ndarray
    fixed_z: np.ndarray
    mask_present: np.ndarray  # (n_pairs, n_chroms) observed chromosome presence


def _presence_per_pair(
    chrom: np.ndarray, pair_index: np.ndarray, n_pairs: int, n_chroms: int
) -> np.ndarray:
    out = np.zeros((n_pairs, n_chroms), dtype=bool)
    for c in range(n_chroms):
        out[:, c] = np.bincount(pair_index[chrom == c], minlength=n_pairs) > 0
    return out


def measure_sharing(run: Run, n_perm: int) -> Sharing:
    """Observed within-haplotype chromosome spread, and both permutation nulls."""
    key = run.cluster.astype(np.int64) * run.n_assemblies + run.asm
    uniq, pair_index = np.unique(key, return_inverse=True)
    pair_cluster = (uniq // run.n_assemblies).astype(np.int64)
    n_pairs = len(uniq)
    k = run.n_clusters
    n_per_cluster = np.bincount(pair_cluster, minlength=k).astype(np.float64)

    def totals(chrom: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        present = _presence_per_pair(chrom, pair_index, n_pairs, run.n_chroms)
        count = present.sum(axis=1)
        return (
            present,
            np.bincount(pair_cluster, weights=count, minlength=k),
            np.bincount(pair_cluster, weights=(count >= 2).astype(np.float64), minlength=k),
        )

    present, obs_sum, obs_multi = totals(run.chrom)

    rng = np.random.default_rng(11)
    scratch = run.chrom.copy()
    acc = {name: np.zeros(k) for name in ("f", "f2", "fm", "fge", "x", "x2")}
    a_order, a_bounds = _blocks(run.asm, run.n_assemblies)
    c_order, c_bounds = _blocks(run.cluster, k)
    for _ in range(n_perm):
        _shuffle_within(run.chrom, a_order, a_bounds, rng, scratch)
        _, s, m = totals(scratch)
        acc["f"] += s
        acc["f2"] += s * s
        acc["fm"] += m
        acc["fge"] += s >= obs_sum
        _shuffle_within(run.chrom, c_order, c_bounds, rng, scratch)
        _, s, _m = totals(scratch)
        acc["x"] += s
        acc["x2"] += s * s

    def moments(total: np.ndarray, square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = total / n_perm
        sd = np.sqrt(np.maximum(square / n_perm - mean**2, 0.0))
        return mean, sd

    free_mean, free_sd = moments(acc["f"], acc["f2"])
    fixed_mean, fixed_sd = moments(acc["x"], acc["x2"])
    return Sharing(
        pair_cluster=pair_cluster,
        pair_index=pair_index,
        n_pairs_per_cluster=n_per_cluster,
        chrom_per_asm=obs_sum / n_per_cluster,
        frac_asm_multi=obs_multi / n_per_cluster,
        free_mean=free_mean / n_per_cluster,
        free_z=np.divide(obs_sum - free_mean, free_sd, out=np.full(k, np.nan), where=free_sd > 0),
        free_p=(acc["fge"] + 1) / (n_perm + 1),
        free_multi=acc["fm"] / (n_perm * n_per_cluster),
        fixed_mean=fixed_mean / n_per_cluster,
        fixed_z=np.divide(obs_sum - fixed_mean, fixed_sd, out=np.full(k, np.nan), where=fixed_sd > 0),
        mask_present=present,
    )


def segment_ratio(run: Run) -> float:
    """Bins per maximal run of adjacent same-cluster bins on one contig.

    Near 1 means cluster membership is scattered, so permuting *bins* is the
    right null; far above 1 would mean bins come in blocks and the permutation
    would over-count independent observations.
    """
    order = np.lexsort((run.start, run.contig, run.cluster))
    cl, ct, st, en = run.cluster[order], run.contig[order], run.start[order], run.end[order]
    boundary = np.ones(len(order), dtype=bool)
    boundary[1:] = (cl[1:] != cl[:-1]) | (ct[1:] != ct[:-1]) | (st[1:] != en[:-1])
    return len(order) / max(int(boundary.sum()), 1)


def chromosome_composition(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster chromosome fractions and effective number of chromosomes."""
    counts = np.zeros((run.n_clusters, run.n_chroms), dtype=np.float64)
    for c in range(run.n_chroms):
        counts[:, c] = np.bincount(run.cluster[run.chrom == c], minlength=run.n_clusters)
    frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    keff = 1.0 / np.maximum((frac**2).sum(axis=1), 1e-12)
    return frac, keff


def proximal_fraction(run: Run) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fraction of a cluster's placed reference bins that lie on the short arm.

    The boundary is not a constant: it is the distal edge of that chromosome's
    own alpha-satellite array in the reference annotation, so it is derived from
    the run and moves with whatever chromosomes are in it.
    """
    n_ref_placed = np.zeros(run.n_clusters, dtype=np.int64)
    prox = np.full(run.n_clusters, np.nan)
    boundary: dict[str, float] = {}
    asat = [i for i, name in enumerate(run.feature_name) if str(name).startswith("asat_hor")]
    mask = run.is_ref & run.placed
    if not mask.any() or not asat:
        return n_ref_placed, prox, boundary
    edge = np.zeros(run.n_chroms, dtype=np.int64)
    is_asat = np.isin(run.feature, asat)
    for c in range(run.n_chroms):
        sel = mask & is_asat & (run.chrom == c)
        edge[c] = run.end[sel].max() if sel.any() else 0
        boundary[str(run.chrom_name[c])] = float(edge[c])
    is_prox = run.start < edge[run.chrom]
    n_ref_placed = np.bincount(run.cluster[mask], minlength=run.n_clusters)
    hits = np.bincount(run.cluster[mask & is_prox], minlength=run.n_clusters)
    with np.errstate(invalid="ignore"):
        prox = np.where(n_ref_placed > 0, hits / np.maximum(n_ref_placed, 1), np.nan)
    return n_ref_placed, prox, boundary


def chromosome_set_consistency(
    run: Run, sharing: Sharing
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal per-haplotype chromosome set, its share, and mean pairwise Jaccard.

    Answers "consistent between haplotypes, or idiosyncratic per individual?".
    Sets are bitmasks, and the Jaccard is computed over the *distinct* masks of a
    cluster weighted by how many haplotypes carry each, so the cost is the
    number of distinct sets squared, not the number of haplotypes squared.
    """
    bits = (sharing.mask_present * (1 << np.arange(run.n_chroms))).sum(axis=1).astype(np.int64)
    modal, share = _mode_per_group(sharing.pair_cluster.astype(np.int64), bits, run.n_clusters)
    jaccard = np.full(run.n_clusters, np.nan)
    order = np.argsort(sharing.pair_cluster, kind="stable")
    bounds = np.searchsorted(sharing.pair_cluster[order], np.arange(run.n_clusters + 1))
    for k in range(run.n_clusters):
        block = bits[order[bounds[k] : bounds[k + 1]]]
        if len(block) < 2:
            continue
        masks, counts = np.unique(block, return_counts=True)
        inter = _popcount(masks[:, None] & masks[None, :]).astype(np.float64)
        union = _popcount(masks[:, None] | masks[None, :]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            jac = np.where(union > 0, inter / union, 0.0)
        weight = counts[:, None] * counts[None, :]
        np.fill_diagonal(weight, counts * (counts - 1))
        total = weight.sum()
        jaccard[k] = float((jac * weight).sum() / total) if total else np.nan
    return modal, share, jaccard


def _set_label(mask: int, chrom_name: np.ndarray) -> str:
    names = [str(chrom_name[i]).replace("chr", "") for i in range(len(chrom_name)) if mask >> i & 1]
    return "+".join(names)


def report_sharing(run: Run, sharing: Sharing, table: pd.DataFrame, n_perm: int) -> None:
    _rule("Q2  CROSS-CHROMOSOME SHARING WITHIN A HAPLOTYPE")
    print(f"chromosomes in the run : {', '.join(str(c) for c in run.chrom_name)}")
    if run.n_chroms < 2:
        print("one chromosome in the run: there is nothing for a cluster to cross.  "
              "Re-run over several chromosomes for this half of the report.")
        return
    print(f"bins per contiguous cluster segment : {segment_ratio(run):.2f}  "
          "(near 1 = bins are scattered, so a bin-level permutation is the right null)")
    print()
    print("chrom/hap  = distinct chromosomes a haplotype carrying the cluster puts it on")
    print("free null  = chromosome labels permuted within assembly (no chromosome structure)")
    print("fixed null = chromosome labels permuted within cluster (its own mix, redistributed)")

    _sub("by reference-derived feature: which families cross chromosomes")
    grouped = table[table["ref_feature"].notna()].groupby("ref_feature")
    summary = grouped.agg(
        clusters=("size", "size"), bins=("size", "sum"),
        chrom_per_hap=("chrom_per_asm", "median"),
        free_null=("chrom_per_asm_null_free", "median"),
        fixed_null=("chrom_per_asm_null_fixed", "median"),
        z_within=("z_fixed", "median"),
        frac_hap_multi=("frac_asm_multi", "median"),
        keff=("chrom_keff", "median"),
        jaccard=("mean_jaccard", "median"),
    )
    summary = summary[summary["clusters"] >= 4].sort_values("chrom_per_hap", ascending=False)
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

    prox = table[(table["n_ref_placed"] >= MIN_REF_PLACED) & (table["ref_frac_proximal"] >= PROXIMAL_FRAC)]
    _sub(f"restricted to short-arm / pericentromeric clusters ({len(prox):,} clusters, "
         f"{_fmt_bp(prox['bp'].sum())})")
    if len(prox):
        sub = prox.groupby("ref_feature").agg(
            clusters=("size", "size"), bins=("size", "sum"),
            chrom_per_hap=("chrom_per_asm", "median"),
            free_null=("chrom_per_asm_null_free", "median"),
            fixed_null=("chrom_per_asm_null_fixed", "median"),
            z_within=("z_fixed", "median"),
            frac_hap_multi=("frac_asm_multi", "median"),
            haplotypes=("n_assemblies", "median"),
            jaccard=("mean_jaccard", "median"),
        )
        print(sub[sub["clusters"] >= 3].sort_values("chrom_per_hap", ascending=False).to_string(
            float_format=lambda x: f"{x:.2f}"))

    shared = table[table["sharing_class"] == "within-haplotype shared"]
    beyond = shared[~shared["ref_feature"].fillna("").isin(EXCLUDE_FEATURES)]
    on_arm = (table["n_ref_placed"] >= MIN_REF_PLACED) & (table["ref_frac_proximal"] >= PROXIMAL_FRAC)
    prox_beyond = beyond[on_arm.reindex(beyond.index).fillna(False)]
    _sub("the answer to 'what else, beyond rDNA'")
    print(f"clusters shared within a haplotype               : {len(shared):,}")
    print(f"...excluding {'/'.join(EXCLUDE_FEATURES)}: {len(beyond):,}  "
          f"({_fmt_bp(beyond['bp'].sum())}) -- but most of that is q-arm euchromatin, where")
    print("   LINE/SINE families are on every chromosome for reasons that are not recombination")
    if len(prox_beyond):
        print(f"...on the acrocentric short arm                  : {len(prox_beyond):,}  "
              f"({_fmt_bp(prox_beyond['bp'].sum())}, median "
              f"{prox_beyond['n_assemblies'].median():.0f} of {run.n_assemblies} assemblies) "
              "<-- this is the answer")
    else:
        print("...on the acrocentric short arm                  : not separable -- no reference "
              "annotation, so the p-arm boundary could not be derived")

    focus = prox_beyond if len(prox_beyond) >= 5 else beyond
    pair_set = focus.index.to_numpy()
    scope = ("short-arm, non-rDNA, multi-chromosomal clusters" if len(prox_beyond) >= 5
             else "all multi-chromosomal non-rDNA clusters (no short-arm set available)")

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    _sub(f"which chromosomes take part at all -- {scope}")
    marginal, pairs = _pair_table(run, n_perm, pair_set)
    print(marginal.to_string(float_format=fmt))
    _sub("recurrent chromosome pairs within one haplotype")
    print(pairs.to_string(float_format=fmt))
    print("the null is scoped to these clusters' own bins, so 1.00 is calibrated and the")
    print("deviations are the result; p_enriched is the permutation tail above the observation.")
    print("pair_only divides out both marginals: it is what is specific to the pair, and it is")
    print("the column a pseudo-homologous-region claim has to move.")

    # The known alpha-satellite suprachromosomal families would explain a pair
    # ordering on their own, so check the ordering survives without them.
    no_asat = table.loc[pair_set]
    no_asat = no_asat[~no_asat["ref_feature"].fillna("").str.startswith("asat")]
    if 5 <= len(no_asat) < len(pair_set):
        _sub(f"...the same, with every alpha-satellite cluster removed ({len(no_asat):,} clusters)")
        marginal, pairs = _pair_table(run, n_perm, no_asat.index.to_numpy())
        print(marginal.to_string(float_format=fmt))
        print()
        print(pairs.to_string(float_format=fmt))

    _sub("is the sharing consistent between haplotypes, or per individual?")
    _consistency_report(run, sharing, table)

    cols = ["size", "bp", "n_assemblies", "n_samples", "chrom_per_asm", "chrom_per_asm_null_free",
            "chrom_per_asm_null_fixed", "z_fixed", "frac_asm_multi", "modal_chrom_set",
            "frac_asm_modal", "mean_jaccard", "ref_feature", "n_sketch"]

    def show(frame: pd.DataFrame) -> None:
        out = frame[cols].copy()
        out["bp"] = (out["bp"] / 1e6).round(2)
        print(out.to_string(float_format=lambda x: f"{x:.2f}"))

    _sub("pan-acrocentric: clusters a haplotype puts on essentially every chromosome")
    pan = focus[focus["chrom_per_asm"] >= run.n_chroms - 1]
    if len(pan):
        show(pan.sort_values("bp", ascending=False).head(15))

    _sub("chromosome-restricted: clusters shared by a specific *subset*, ranked by "
         "how often the same subset recurs")
    subset = focus[(focus["chrom_keff"] <= run.n_chroms - 1.5)
                   & (focus["chrom_per_asm"] >= SHARED_CHROM_PER_ASM)]
    if len(subset):
        show(subset.sort_values(["frac_asm_modal", "bp"], ascending=False).head(20))
    else:
        print("none: on this run every shared short-arm cluster uses all the chromosomes.")


def _pair_table(
    run: Run, n_perm: int, selected: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chromosome-pair co-occurrence inside one haplotype, observed vs free null.

    The null permutes chromosome labels *within the selected clusters only*.
    Permuting over the whole run instead would confound the answer badly: the
    short arm is a much larger share of chr21 and chr22 than of chr13/14/15, so
    a genome-wide null under-predicts every chr21/chr22 pair and every short-arm
    cluster would look enriched for them whether or not anything is shared.
    """
    in_scope = np.isin(run.cluster_id[run.cluster], selected)
    if not in_scope.any():
        empty = pd.DataFrame(columns=["observed", "expected", "obs/exp"])
        return empty, empty
    chrom = run.chrom[in_scope]
    asm = run.asm[in_scope]
    key = run.cluster[in_scope].astype(np.int64) * run.n_assemblies + asm
    _, pair_index = np.unique(key, return_inverse=True)
    n_pairs = int(pair_index.max()) + 1

    def counts(present: np.ndarray) -> dict[tuple[str, str], int]:
        return {
            (str(run.chrom_name[a]), str(run.chrom_name[b])):
                int((present[:, a] & present[:, b]).sum())
            for a, b in combinations(range(run.n_chroms), 2)
        }

    present = _presence_per_pair(chrom, pair_index, n_pairs, run.n_chroms)
    observed = counts(present)
    obs_marginal = present.sum(axis=0).astype(np.float64)
    expected = dict.fromkeys(observed, 0.0)
    at_least = dict.fromkeys(observed, 0)
    exp_marginal = np.zeros(run.n_chroms)
    rng = np.random.default_rng(29)
    scratch = chrom.copy()
    order, bounds = _blocks(asm, run.n_assemblies)
    for _ in range(n_perm):
        _shuffle_within(chrom, order, bounds, rng, scratch)
        drawn_present = _presence_per_pair(scratch, pair_index, n_pairs, run.n_chroms)
        exp_marginal += drawn_present.sum(axis=0) / n_perm
        for name, value in counts(drawn_present).items():
            expected[name] += value / n_perm
            at_least[name] += value >= observed[name]

    marginal = pd.DataFrame({
        "observed": obs_marginal, "expected": exp_marginal,
        "obs/exp": obs_marginal / np.where(exp_marginal > 0, exp_marginal, np.nan),
    }, index=[str(c) for c in run.chrom_name])

    frame = pd.DataFrame({"observed": pd.Series(observed), "expected": pd.Series(expected)})
    frame["obs/exp"] = frame["observed"] / frame["expected"].replace(0, np.nan)
    frame["p_enriched"] = (pd.Series(at_least) + 1) / (n_perm + 1)
    # A chromosome that is over-represented on its own inflates every pair it is
    # in.  Dividing by the product of the two marginal ratios leaves only what
    # is specific to the *pair*, which is the quantity a PHR claim needs.
    ratio = marginal["obs/exp"].to_dict()
    frame["pair_only"] = [
        frame.loc[key, "obs/exp"] / (ratio[key[0]] * ratio[key[1]]) for key in frame.index
    ]
    frame["share"] = frame["observed"] / max(frame["observed"].sum(), 1)
    return marginal.sort_values("obs/exp", ascending=False), frame.sort_values(
        "obs/exp", ascending=False)


def _same_donor_pairs(run: Run, sharing: Sharing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cluster, left, right) index triples for haplotype pairs from one donor.

    Donors contribute two haplotypes almost always, so the size-2 case is
    vectorised and only larger sibships fall back to a loop.
    """
    sample_of_asm = np.zeros(run.n_assemblies, dtype=np.int64)
    sample_of_asm[run.asm] = run.smp
    pair_asm = (np.unique(run.cluster.astype(np.int64) * run.n_assemblies + run.asm)
                % run.n_assemblies)
    pair_sample = sample_of_asm[pair_asm]
    n_samples = len(run.smp_name)
    key = sharing.pair_cluster.astype(np.int64) * n_samples + pair_sample
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    sizes = np.diff(np.r_[starts, len(order)])
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    two = sizes == 2
    if two.any():
        left.append(order[starts[two]])
        right.append(order[starts[two] + 1])
    for s, n in zip(starts[sizes > 2], sizes[sizes > 2]):
        block = order[s : s + n]
        i, j = np.triu_indices(n, 1)
        left.append(block[i])
        right.append(block[j])
    if not left:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    lo = np.concatenate(left)
    hi = np.concatenate(right)
    return sharing.pair_cluster[lo], lo, hi


def _consistency_report(run: Run, sharing: Sharing, table: pd.DataFrame) -> None:
    """Do the two haplotypes of one donor agree more than two unrelated ones?

    ``mean_jaccard`` already holds the mean over *all* haplotype pairs of a
    cluster, so the same-donor pairs are enumerated explicitly and the
    different-donor mean is what is left over.  That avoids an O(haplotypes^2)
    pass at 463 assemblies.
    """
    selected = table.index[
        (table["chrom_per_asm"] >= SHARED_CHROM_PER_ASM)
        & (~table["ref_feature"].fillna("").isin(EXCLUDE_FEATURES))
        & table["mean_jaccard"].notna()
    ].to_numpy()
    if not len(selected):
        print("no cluster passes the sharing filter; nothing to compare.")
        return
    is_selected = np.zeros(run.n_clusters, dtype=bool)
    is_selected[np.searchsorted(run.cluster_id, selected)] = True

    n_hap = sharing.n_pairs_per_cluster
    all_den = np.where(is_selected, n_hap * (n_hap - 1) / 2.0, 0.0)
    mean_jac = np.nan_to_num(table["mean_jaccard"].to_numpy(dtype=np.float64))
    all_num = mean_jac * all_den

    bits = (sharing.mask_present * (1 << np.arange(run.n_chroms))).sum(axis=1).astype(np.int64)
    cl, lo, hi = _same_donor_pairs(run, sharing)
    same_num = same_den = 0.0
    if len(cl):
        take = is_selected[cl]
        inter = _popcount(bits[lo[take]] & bits[hi[take]]).astype(np.float64)
        union = _popcount(bits[lo[take]] | bits[hi[take]]).astype(np.float64)
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        same_num, same_den = float(jac.sum()), float(len(jac))

    diff_num, diff_den = float(all_num.sum()) - same_num, float(all_den.sum()) - same_den
    print(f"per-haplotype chromosome-set Jaccard over {len(selected):,} shared clusters")
    if same_den:
        print(f"  two haplotypes of the same donor : n={int(same_den):,}  mean {same_num / same_den:.3f}")
    if diff_den:
        print(f"  haplotypes of different donors   : n={int(diff_den):,}  mean {diff_num / diff_den:.3f}")
    if same_den and diff_den:
        verdict = ("the same-donor advantage is negligible: which chromosome a cluster sits on is a "
                   "property of the chromosome copy, not of the person")
        if same_num / same_den > diff_num / diff_den + 0.05:
            verdict = "same-donor haplotypes agree more: some of the pattern is individual-specific"
        print(f"  -> {verdict}")
    consistent = table.loc[selected, "frac_asm_modal"]
    print(f"  modal chromosome set is carried by a median {consistent.median():.0%} of haplotypes")


# --------------------------------------------------------------------------
# assembly of the per-cluster table
# --------------------------------------------------------------------------


def build_table(run: Run, seen: np.ndarray, q1: pd.DataFrame, sharing: Sharing) -> pd.DataFrame:
    size = np.bincount(run.cluster, minlength=run.n_clusters).astype(np.int64)
    bp = np.bincount(run.cluster, weights=(run.end - run.start), minlength=run.n_clusters)
    frac, keff = chromosome_composition(run)
    n_ref_placed, prox, _ = proximal_fraction(run)
    modal, modal_share, jaccard = chromosome_set_consistency(run, sharing)

    seen_smp = np.zeros(run.n_clusters * len(run.smp_name), dtype=bool)
    seen_smp[run.cluster.astype(np.int64) * len(run.smp_name) + run.smp] = True
    n_samples = seen_smp.reshape(run.n_clusters, -1).sum(axis=1)

    n_pop = np.zeros(run.n_clusters, dtype=np.int64)
    if run.pop_of_sample:
        pops = np.array([run.pop_of_sample.get(str(s), "") for s in run.smp_name], dtype=object)
        labels, pop_code = np.unique(pops, return_inverse=True)
        seen_pop = np.zeros(run.n_clusters * len(labels), dtype=bool)
        seen_pop[run.cluster.astype(np.int64) * len(labels) + pop_code[run.smp]] = True
        known = np.array([bool(x) for x in labels])
        n_pop = (seen_pop.reshape(run.n_clusters, -1) & known).sum(axis=1)

    ref_col = _reference_column(run)
    sole = seen.sum(axis=1) == run.n_assemblies - 1
    sole_of = np.full(run.n_clusters, "", dtype=object)
    if sole.any():
        who = np.argmax(~seen[sole], axis=1)
        sole_of[np.flatnonzero(sole)] = [str(run.asm_name[w]) for w in who]

    def median_per_cluster(values: np.ndarray) -> np.ndarray:
        return _median_per_group(run.cluster, values, run.n_clusters)

    table = pd.DataFrame({
        "cluster": run.cluster_id,
        "size": size,
        "bp": bp.astype(np.int64),
        "n_assemblies": seen.sum(axis=1).astype(np.int32),
        "n_samples": n_samples.astype(np.int32),
        "n_populations": n_pop.astype(np.int32),
        "n_ref_bins": q1["n_ref_bins"].to_numpy(),
        "novel": q1["novel"].to_numpy(),
        "p_novel_chance": q1.get("p_novel_chance", pd.Series(np.nan, index=q1.index)).to_numpy(),
        "n_missing_assemblies": (run.n_assemblies - seen.sum(axis=1)).astype(np.int32),
        "sole_absentee": pd.Series(sole_of, dtype="string"),
        "reference_is_sole_absentee": np.array(
            [ref_col is not None and s and s == str(run.asm_name[ref_col]) for s in sole_of], dtype=bool),
        "gc": median_per_cluster(run.gc).astype(np.float32),
        "n_sketch": median_per_cluster(run.n_sketch).astype(np.float32),
        "frac_placed": (np.bincount(run.cluster, weights=run.placed, minlength=run.n_clusters)
                        / np.maximum(size, 1)).astype(np.float32),
        "ref_feature": _cluster_feature(run, reference_only=True).astype("string").to_numpy(),
        "asm_feature": (_cluster_feature(run, reference_only=False).astype("string").to_numpy()
                        if run.has_asm_annotation else pd.array([pd.NA] * run.n_clusters, dtype="string")),
        "n_ref_placed": n_ref_placed.astype(np.int32),
        "ref_frac_proximal": prox.astype(np.float32),
        "n_chrom": (frac > 0).sum(axis=1).astype(np.int32),
        "chrom_top": pd.Series([str(run.chrom_name[i]) for i in frac.argmax(axis=1)], dtype="string"),
        "chrom_top_frac": frac.max(axis=1).astype(np.float32),
        "chrom_keff": keff.astype(np.float32),
        "chrom_per_asm": sharing.chrom_per_asm.astype(np.float32),
        "chrom_per_asm_null_free": sharing.free_mean.astype(np.float32),
        "chrom_per_asm_null_fixed": sharing.fixed_mean.astype(np.float32),
        "z_free": sharing.free_z.astype(np.float32),
        "z_fixed": sharing.fixed_z.astype(np.float32),
        "p_free_ge": sharing.free_p.astype(np.float32),
        "frac_asm_multi": sharing.frac_asm_multi.astype(np.float32),
        "frac_asm_multi_null": sharing.free_multi.astype(np.float32),
        "modal_chrom_set": pd.Series(
            [_set_label(int(m), run.chrom_name) for m in modal], dtype="string"),
        "frac_asm_modal": modal_share.astype(np.float32),
        "mean_jaccard": jaccard.astype(np.float32),
    })
    for c in range(run.n_chroms):
        table[f"frac_{run.chrom_name[c]}"] = frac[:, c].astype(np.float32)

    sharing_class = np.full(run.n_clusters, "mixed", dtype=object)
    sharing_class[sharing.chrom_per_asm < PURE_CHROM_PER_ASM] = "chromosome-specific"
    multi = sharing.chrom_per_asm >= SHARED_CHROM_PER_ASM
    within = multi & ~(sharing.fixed_z < WITHIN_Z)
    sharing_class[multi & (sharing.fixed_z < WITHIN_Z)] = "cohort-shared, haplotype-pure"
    sharing_class[within] = "within-haplotype shared"
    table["sharing_class"] = pd.Series(sharing_class, dtype="string")
    return table.set_index(pd.Index(run.cluster_id, name="cluster_id"))


def report_novel_ranking(run: Run, table: pd.DataFrame) -> None:
    novel = table[table["novel"]]
    if novel.empty:
        return
    _sub("recurrence of the novel clusters")
    floors = sorted({1, 2, 3, max(1, run.n_assemblies // 4), max(1, run.n_assemblies // 2),
                     max(1, (3 * run.n_assemblies) // 4), run.n_assemblies - 1})
    print(f"{'carried by >= N assemblies':>28}  {'clusters':>9}  {'bins':>9}  {'sequence':>12}")
    for floor in floors:
        m = novel["n_assemblies"] >= floor
        print(f"{floor:>28}  {int(m.sum()):>9,}  {int(novel.loc[m, 'size'].sum()):>9,}  "
              f"{_fmt_bp(novel.loc[m, 'bp'].sum()):>12}")

    _sub("largest novel clusters")
    cols = ["size", "bp", "n_assemblies", "n_samples", "n_populations", "p_novel_chance",
            "gc", "n_sketch", "frac_placed", "chrom_top", "chrom_top_frac", "ref_feature",
            "asm_feature"]
    for label, frame in (
        ("by sequence", novel.sort_values("bp", ascending=False).head(15)),
        ("by recurrence then size", novel.sort_values(
            ["n_samples", "n_assemblies", "bp"], ascending=False).head(15)),
    ):
        print(f"\n[{label}]")
        show = frame[cols].copy()
        show["bp"] = (show["bp"] / 1e6).round(2)
        print(show.to_string(float_format=lambda x: f"{x:.3f}"))

    private = novel[novel["n_samples"] <= 2]
    print(f"\nnovel clusters seen in <=2 donors (artefact candidates): {len(private):,}, "
          f"{_fmt_bp(private['bp'].sum())}")
    if len(private):
        show = private[cols].copy()
        show["bp"] = (show["bp"] / 1e6).round(2)
        print(show.to_string(float_format=lambda x: f"{x:.3f}"))


# --------------------------------------------------------------------------


def final_reading(run: Run, table: pd.DataFrame, findings: dict[str, object]) -> None:
    """Say what the numbers mean, from the numbers, not from a prior."""
    _rule("READING")
    novel = int(findings.get("novel_clusters", 0))
    line = f"Q1  {novel:,} novel clusters ({_fmt_bp(float(findings.get('novel_bp', 0)))})"
    if "expected_by_chance" in findings:
        line += (f"; a size-matched random draw already predicts "
                 f"{float(findings['expected_by_chance']):.0f} of them.")
    else:
        line += "; no reference in this run, so nothing is testable."
    print(line)
    rank = findings.get("loao_rank")
    if rank is not None:
        n = run.n_assemblies
        inside = int(rank) < n
        print(f"    Leaving out the reference removes {findings['loao_ref']:,} clusters; leaving out "
              f"a haplotype removes a median {findings['loao_median']:,.0f} (max {findings['loao_max']:,}).")
        print(f"    The reference ranks {rank}/{n}, empirical p = {findings['loao_p']:.3f} -- "
              + ("inside the haplotype distribution, so 'novel' here is presence/absence"
                 " polymorphism, not a reference gap."
                 if inside else
                 "outside it, so there is genuinely reference-specific material."))
    if "sole_ref" in findings:
        print(f"    As the *sole* absentee the reference accounts for {findings['sole_ref']:,} clusters "
              f"against a haplotype median of {findings['sole_median']:,.0f} "
              f"(rank {findings['sole_rank']}/{run.n_assemblies}).")
    a = float(findings.get("sketch_novel", np.nan))
    b = float(findings.get("sketch_supported", np.nan))
    if np.isfinite(a) and np.isfinite(b) and b > 0:
        print(f"    Novel bins carry a median {a:.0f} distinct 31-mers against {b:.0f} for "
              "reference-supported bins"
              + (": the novel set is satellite, not novel euchromatin."
                 if a < 0.75 * b else ", i.e. the same k-mer density."))

    if run.n_chroms < 2:
        print("Q2  one chromosome in the run: not asked.")
        return
    shared = table[(table["sharing_class"] == "within-haplotype shared")
                   & (~table["ref_feature"].fillna("").isin(EXCLUDE_FEATURES))]
    on_arm = shared[(shared["n_ref_placed"] >= MIN_REF_PLACED)
                    & (shared["ref_frac_proximal"] >= PROXIMAL_FRAC)]
    scope = "short-arm clusters" if len(on_arm) else "clusters (short arm not separable)"
    shared = on_arm if len(on_arm) else shared
    print(f"Q2  {len(shared):,} {scope} outside rDNA are shared *within* a haplotype "
          f"({_fmt_bp(shared['bp'].sum())}).")
    by_feature = shared.groupby("ref_feature")["chrom_per_asm"].agg(["size", "median"])
    by_feature = by_feature[by_feature["size"] >= 3].sort_values("median", ascending=False)
    if len(by_feature):
        head = ", ".join(f"{f} {v:.1f}" for f, v in by_feature["median"].head(4).items())
        print(f"    Chromosomes per haplotype, by family: {head} "
              f"(out of {run.n_chroms}).")
    hor = table[table["ref_feature"].fillna("").str.startswith("asat_hor")]
    mon = table[table["ref_feature"].fillna("") == "asat_mon"]
    if len(hor) >= 3 and len(mon) >= 3:
        a, b = hor["chrom_per_asm"].median(), mon["chrom_per_asm"].median()
        print(f"    Alpha-satellite splits by age: HOR {a:.2f} chromosomes per haplotype "
              f"({len(hor)} clusters), monomeric {b:.2f} ({len(mon)})."
              + ("  Same family, opposite answer." if b - a > 1.0 else ""))


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "results/acro3")
    n_perm = int(argv[2]) if len(argv) > 2 else 100
    if not (outdir / "cluster" / "clusters.parquet").exists():
        print(f"no clustering under {outdir}", file=sys.stderr)
        return 2

    print(f"kmer-dust :: novel material and cross-chromosome sharing :: {outdir}")
    run = load_run(outdir)
    print(f"{run.n_bins_total:,} bins, {run.n_assemblies:,} assemblies, "
          f"{len(run.smp_name):,} samples, {run.n_chroms} chromosomes, "
          f"{run.n_clusters:,} clusters, {n_perm} permutations")
    if not run.has_asm_annotation:
        print("per-assembly annotation absent (annotate.annotate_assemblies: false) -- "
              "the cross-check sections are skipped, everything else is unaffected")

    seen = cluster_presence(run)
    size = np.bincount(run.cluster, minlength=run.n_clusters).astype(np.int64)
    findings: dict[str, object] = {}
    q1 = report_novel(run, seen, size, n_perm, findings)
    sharing = measure_sharing(run, n_perm)
    table = build_table(run, seen, q1, sharing)

    describe_novel(run, q1, seen, findings)
    report_novel_ranking(run, table)
    report_sharing(run, sharing, table, n_perm)
    final_reading(run, table, findings)

    analysis = outdir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    path = analysis / "novel.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    table.reset_index(drop=True).to_parquet(tmp, index=False)
    tmp.replace(path)
    _rule("OUTPUT")
    print(f"{path}  ({len(table):,} clusters x {table.shape[1]} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
