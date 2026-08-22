"""Is the satellite vocabulary population-stratified?

Alpha-satellite higher-order repeat arrays vary between individuals, and some
HOR variants have ancestry-associated frequencies.  A kmer-dust run answers that
question without ever aligning anything: a cluster *is* a k-mer vocabulary, so
"which clusters does this haplotype occupy, and how much of each" is a direct,
reference-free measurement of which satellite dialects a genome speaks.  This
script turns a finished run into that measurement and tests it.

Four things make the naive version of this analysis wrong, and each one is
handled explicitly below.

**Completeness is the obvious confounder.**  Assemblies differ in how much
acrocentric sequence they resolved at all, and satellite arrays are exactly the
part that goes missing.  A haplotype with 10 % more bins would look "enriched"
for everything.  So the data are treated as *compositional*: counts are closed
to proportions and centred-log-ratio transformed, which removes the total by
construction, and the whole thing is re-run on abundances residualised on
log(total clustered bins) to check that nothing survives only because of it.

**The two haplotypes of one donor are not independent.**  They came from the
same reads, the same assembler and the same coverage.  The primary unit is
therefore the *sample*, with the two haplotypes summed; the haplotype-level test
is reported too, but with permutations restricted so a donor's haplotypes always
carry the same label.  Treating 2n haplotypes as 2n observations would roughly
halve every p-value for free.

**PERMANOVA R-squared is not zero under the null.**  With k groups and n units
its expectation is about (k-1)/(n-1) -- 0.27 for five superpopulations and
sixteen samples.  An R-squared of 0.3 is *nothing*.  Every effect size here is
therefore reported next to the mean of its own permutation null, and the number
to read is the excess over that null, not the raw value.

**A group difference can be dispersion, not location.**  PERMANOVA is sensitive
to both, so PERMDISP (homogeneity of within-group distance-to-centroid) is run
alongside it: a significant PERMANOVA with a significant PERMDISP means "the
groups differ in spread", which is a different claim.

The satellite-specific question needs one more control.  Satellite clusters are
fewer than non-satellite ones, and PERMANOVA's effect size depends on how many
features went in, so "satellites are significant and everything else is not"
could be an artefact of feature count.  The comparison is therefore made against
*size-matched random subsets* of non-satellite clusters -- same number of
clusters, matched bin-count distribution -- which gives an empirical null for
"any set of clusters that shape".

Everything is driven by permutation, because with a few dozen samples no
asymptotic null is credible.  Nothing here is hard-coded to a cohort size, a
chromosome set or a set of population labels; the group vocabulary comes from
the manifest.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/analysis/population.py [outdir]

Writes ``<outdir>/analysis/population.parquet`` (one row per cluster tested),
``<outdir>/analysis/population_units.parquet`` (per-unit ordination + covariates,
the figure data) and ``<outdir>/analysis/population_summary.json``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

# --------------------------------------------------------------------------
# knobs
# --------------------------------------------------------------------------

#: Feature classes counted as "satellite" for the satellite-vs-rest contrast.
#: Prefix match against ``annotate.dominant_feature`` -- ``asat_hor_active``,
#: ``hsat1a`` and friends all fold in without listing every variant.
SATELLITE_PREFIXES: Final[tuple[str, ...]] = ("asat", "hsat", "bsat", "gsat", "rdna")

#: HDBSCAN's "did not belong to any cluster".  Noise is excluded from the
#: composition -- it is the leftovers, not a vocabulary -- but its per-unit
#: fraction is kept as a covariate, because it is a completeness proxy.
NOISE_CLUSTER: Final[int] = -1

#: A cluster must be occupied by at least this fraction of units to enter the
#: log-ratio analysis.  CLR on a feature that is zero almost everywhere reports
#: the pseudocount, not the data.  The choice is unsupervised (it never looks at
#: the group labels), so it cannot manufacture a group difference.
MIN_UNIT_PREVALENCE: Final[float] = 0.25

#: Added to every count before closure.  Zeros are fatal to a log ratio and
#: there is no principled zero-replacement for count data this sparse; 0.5 is
#: the Jeffreys prior.  ``_sensitivity`` re-runs the headline test across a grid
#: of this and ``MIN_UNIT_PREVALENCE`` so the reader can see it does not matter.
PSEUDOCOUNT: Final[float] = 0.5

#: A group with fewer units than this cannot contribute a within-group variance
#: and is dropped from the test (with a note), rather than silently inflating it.
MIN_UNITS_PER_GROUP: Final[int] = 2

N_PERM: Final[int] = 4999
#: The Benjamini-Hochberg q of the best cluster can never fall below
#: ``n_clusters / (N_PERM_PER_CLUSTER + 1)``, because the permutation p has a
#: floor of ``1 / (N + 1)``.  With tens of thousands of clusters that floor is
#: the binding constraint on the FDR column, not the data -- the max-statistic
#: FWER column is the one that stays exact.
N_PERM_PER_CLUSTER: Final[int] = 9999
N_MATCHED_SUBSETS: Final[int] = 100
N_PERM_MATCHED: Final[int] = 199
MAX_JACKKNIFE_FITS: Final[int] = 40
SEED: Final[int] = 7

#: Rows per Arrow batch when streaming the big per-bin tables.  19 M bins x a
#: handful of int32 columns is the whole memory budget of this script.
BATCH_ROWS: Final[int] = 1 << 20


# --------------------------------------------------------------------------
# loading -- streaming, because the per-bin tables reach ~19 M rows
# --------------------------------------------------------------------------


def _stream_codes(path: Path, column: str) -> tuple[np.ndarray, list[str]]:
    """Read one string column as int32 codes plus its vocabulary.

    Materialising 19 M Python strings costs gigabytes; codes cost 4 bytes a bin.
    """
    pf = pq.ParquetFile(path)
    vocab: dict[str, int] = {}
    chunks: list[np.ndarray] = []
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=[column]):
        # dictionary_encode does the de-duplication in C; only the per-batch
        # vocabulary (tens of entries) ever crosses into Python.
        encoded = pc.dictionary_encode(pc.fill_null(batch.column(0), ""))
        local = encoded.dictionary.to_pylist()
        remap = np.empty(len(local), dtype=np.int32)
        for i, value in enumerate(local):
            key = str(value)
            remap[i] = vocab.setdefault(key, len(vocab))
        chunks.append(remap[encoded.indices.to_numpy(zero_copy_only=False)])
    codes = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)
    return codes, list(vocab)


def _stream_ints(path: Path, column: str, dtype: str = "int64") -> np.ndarray:
    pf = pq.ParquetFile(path)
    chunks = [
        batch.column(0).to_numpy(zero_copy_only=False).astype(dtype, copy=False)
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=[column])
    ]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)


def _same_order(path_a: Path, path_b: Path, column: str) -> bool:
    """Do two per-bin tables carry ``column`` in the same row order?

    ``annotate`` and ``cluster`` both promise one row per row of
    ``matrix/rows.parquet`` in the same order.  Promises about 19 M-row files
    are worth checking before joining on them -- and checking is far cheaper
    than the join it lets us skip.
    """
    a = pq.ParquetFile(path_a).iter_batches(batch_size=BATCH_ROWS, columns=[column])
    b = pq.ParquetFile(path_b).iter_batches(batch_size=BATCH_ROWS, columns=[column])
    left = np.empty(0, dtype=object)
    right = np.empty(0, dtype=object)
    while True:
        while left.size == 0:
            batch = next(a, None)
            if batch is None:
                break
            left = batch.column(0).to_numpy(zero_copy_only=False)
        while right.size == 0:
            batch = next(b, None)
            if batch is None:
                break
            right = batch.column(0).to_numpy(zero_copy_only=False)
        if left.size == 0 or right.size == 0:
            return left.size == right.size
        n = min(left.size, right.size)
        if not np.array_equal(left[:n], right[:n]):
            return False
        left, right = left[n:], right[n:]


@dataclass(frozen=True)
class Abundance:
    """A haplotype x cluster bin-count table and the covariates that go with it."""

    counts: np.ndarray  # (n_assemblies, n_clusters) int32
    assemblies: list[str]
    cluster_ids: np.ndarray  # (n_clusters,) int32, cluster label of each column
    noise_bins: np.ndarray  # (n_assemblies,) int64
    total_bins: np.ndarray  # (n_assemblies,) int64, clustered + noise
    bin_cluster_code: np.ndarray  # (n_bins,) int32 column index, -1 for noise


def load_abundance(outdir: Path) -> Abundance:
    """Assemble the haplotype x cluster count table from a finished run."""
    rows_path = outdir / "matrix" / "rows.parquet"
    clusters_path = outdir / "cluster" / "clusters.parquet"
    for path in (rows_path, clusters_path):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run the pipeline first")

    asm_codes, assemblies = _stream_codes(rows_path, "assembly")
    cluster = _stream_ints(clusters_path, "cluster", "int64")
    if cluster.size != asm_codes.size:
        raise ValueError(
            f"rows.parquet has {asm_codes.size} rows, clusters.parquet has {cluster.size}"
        )
    if not _same_order(rows_path, clusters_path, "row_idx"):
        raise ValueError(
            "rows.parquet and clusters.parquet are not in the same row order; "
            "the pipeline contract (one cluster row per matrix row) is broken"
        )

    n_asm = len(assemblies)
    total_bins = np.bincount(asm_codes, minlength=n_asm).astype(np.int64)
    is_noise = cluster == NOISE_CLUSTER
    noise_bins = np.bincount(asm_codes[is_noise], minlength=n_asm).astype(np.int64)

    keep = ~is_noise
    cluster_codes, cluster_ids = pd.factorize(cluster[keep], sort=True)
    n_clu = len(cluster_ids)
    # Built sparse and densified once: a (assembly, cluster) coordinate list is
    # O(bins), whereas a dense accumulator over 19 M bins would be O(bins) too
    # but with an int64 temporary three times the size of the answer.
    counts = (
        sparse.coo_matrix(
            (np.ones(cluster_codes.size, dtype=np.int32), (asm_codes[keep], cluster_codes)),
            shape=(n_asm, n_clu),
        )
        .tocsr()
        .toarray()
    )
    bin_cluster_code = np.full(cluster.size, -1, dtype=np.int32)
    bin_cluster_code[keep] = cluster_codes
    return Abundance(
        counts=counts,
        assemblies=assemblies,
        cluster_ids=np.asarray(cluster_ids, dtype=np.int32),
        noise_bins=noise_bins,
        total_bins=total_bins,
        bin_cluster_code=bin_cluster_code,
    )


def cluster_descriptions(outdir: Path, ab: Abundance) -> pd.DataFrame:
    """What each cluster is made of: modal feature, feature class, chromosome.

    The feature comes from ``annotate.dominant_feature`` over the cluster's own
    bins rather than from ``enrich``'s name, because ``enrich`` names a cluster
    after what it is *enriched* for -- a rare feature at high fold change can win
    there -- and the question here is what the bins mostly are.
    """
    rows_path = outdir / "matrix" / "rows.parquet"
    ann_path = outdir / "annotate" / "annotations.parquet"

    codes = ab.bin_cluster_code
    cluster_ids = ab.cluster_ids
    n_clu = len(cluster_ids)

    frame = pd.DataFrame({"cluster": cluster_ids})
    frame["feature"] = "unannotated"
    frame["feature_frac"] = np.nan

    if ann_path.exists() and _same_order(rows_path, ann_path, "bin_uid"):
        feat_codes, feat_vocab = _stream_codes(ann_path, "dominant_feature")
        frame["feature"], frame["feature_frac"] = _modal_label(
            codes, feat_codes, feat_vocab, n_clu, ignore={"unannotated", ""}
        )
    elif ann_path.exists():
        raise ValueError(
            "annotations.parquet is not in rows.parquet order; the annotate "
            "contract (one row per matrix row, same order) is broken"
        )

    chrom_codes, chrom_vocab = _stream_codes(rows_path, "chrom")
    frame["chrom"], frame["chrom_purity"] = _modal_label(
        codes, chrom_codes, chrom_vocab, n_clu, ignore={""}
    )

    feature = frame["feature"].astype(str)
    frame["is_satellite"] = feature.str.startswith(SATELLITE_PREFIXES)
    frame["feature_class"] = np.where(frame["is_satellite"], "satellite", "other")
    frame["family"] = feature.str.split("_").str[0]

    names_path = outdir / "enrich" / "cluster_names.parquet"
    if names_path.exists():
        names = pd.read_parquet(names_path, columns=["cluster", "name", "n_assemblies"])
        frame = frame.merge(names, on="cluster", how="left")
    return frame


def _modal_label(
    group_codes: np.ndarray,
    value_codes: np.ndarray,
    vocab: list[str],
    n_groups: int,
    ignore: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Per group: the commonest value and the fraction of the group it covers.

    ``ignore`` drops placeholder values (``unannotated``) from both numerator and
    denominator -- a cluster that is 90 % unannotated and 10 % hsat3 is an hsat3
    cluster with poor annotation coverage, not an unannotated one.
    """
    drop = np.zeros(len(vocab), dtype=bool)
    for i, name in enumerate(vocab):
        drop[i] = name in ignore
    usable = (group_codes >= 0) & ~drop[value_codes]
    if not usable.any():
        return np.full(n_groups, "unannotated" if "unannotated" in ignore else "", dtype=object), \
            np.full(n_groups, np.nan)
    table = np.bincount(
        group_codes[usable].astype(np.int64) * len(vocab) + value_codes[usable],
        minlength=n_groups * len(vocab),
    ).reshape(n_groups, len(vocab))
    totals = table.sum(axis=1)
    best = table.argmax(axis=1)
    labels = np.array([vocab[i] for i in best], dtype=object)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(totals > 0, table[np.arange(n_groups), best] / np.maximum(totals, 1), np.nan)
    labels[totals == 0] = "unannotated" if "unannotated" in ignore else ""
    return labels, frac


# --------------------------------------------------------------------------
# units -- a sample, not a haplotype
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Units:
    counts: np.ndarray  # (n_units, n_clusters)
    meta: pd.DataFrame  # one row per unit, index-aligned with counts
    level: str  # "sample" or "haplotype"
    block: np.ndarray  # (n_units,) int, sample index -- ties haplotypes together


def build_units(ab: Abundance, manifest: pd.DataFrame, level: str, group_col: str) -> Units:
    """Restrict to assemblies with a group label and aggregate to ``level``."""
    meta = pd.DataFrame({"assembly": ab.assemblies}).merge(manifest, on="assembly", how="left")
    meta["clustered_bins"] = ab.counts.sum(axis=1, dtype=np.int64)
    meta["noise_bins"] = ab.noise_bins
    meta["total_bins"] = ab.total_bins

    labelled = meta[group_col].notna() & (meta[group_col].astype(str).str.strip() != "")
    counts = ab.counts[labelled.to_numpy()]
    meta = meta[labelled].reset_index(drop=True)

    if level == "haplotype":
        block = pd.factorize(meta["sample"])[0]
        return Units(counts=counts, meta=meta, level=level, block=block)

    codes, samples = pd.factorize(meta["sample"], sort=True)
    agg = np.zeros((len(samples), counts.shape[1]), dtype=np.int64)
    for row, code in enumerate(codes):
        agg[code] += counts[row]
    first = meta.groupby("sample", sort=True).first().reset_index()
    first["n_haplotypes"] = meta.groupby("sample", sort=True).size().to_numpy()
    for col in ("clustered_bins", "noise_bins", "total_bins"):
        first[col] = meta.groupby("sample", sort=True)[col].sum().to_numpy()
    # The haplotype-specific columns describe one arbitrary haplotype of the
    # donor once the two are summed, so they must not survive the aggregation.
    first = first.drop(columns=[c for c in ("assembly", "haplotype") if c in first.columns])
    first = first.set_index("sample").reindex(samples).rename_axis("sample").reset_index()
    return Units(counts=agg, meta=first, level=level, block=np.arange(len(samples)))


def drop_small_groups(units: Units, group_col: str) -> tuple[Units, list[str]]:
    sizes = units.meta[group_col].value_counts()
    small = sorted(str(g) for g, n in sizes.items() if n < MIN_UNITS_PER_GROUP)
    if not small:
        return units, []
    keep = (~units.meta[group_col].astype(str).isin(small)).to_numpy()
    meta = units.meta[keep].reset_index(drop=True)
    return (
        Units(
            counts=units.counts[keep],
            meta=meta,
            level=units.level,
            block=pd.factorize(meta["sample"])[0],
        ),
        small,
    )


# --------------------------------------------------------------------------
# compositional transforms and distances
# --------------------------------------------------------------------------


def clr(
    counts: np.ndarray,
    min_prevalence: float = MIN_UNIT_PREVALENCE,
    pseudocount: float = PSEUDOCOUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Centred log ratio of the closed counts.  Returns ``(clr, kept_columns)``.

    Closure is what makes this immune to assembly completeness: two haplotypes
    that resolved 30 Mb and 33 Mb of the same sequence have the same composition.
    """
    if counts.shape[1] == 0:
        return np.zeros((counts.shape[0], 0)), np.zeros(0, dtype=bool)
    keep = (counts > 0).mean(axis=0) >= min_prevalence
    if not keep.any():
        keep = np.ones(counts.shape[1], dtype=bool)
    x = counts[:, keep].astype(np.float64) + pseudocount
    x /= x.sum(axis=1, keepdims=True)
    logx = np.log(x)
    return logx - logx.mean(axis=1, keepdims=True), keep


def euclidean(x: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance; on CLR coordinates this is Aitchison."""
    gram = x @ x.T
    diag = np.diag(gram)
    d2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * gram, 0.0)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def bray_curtis(counts: np.ndarray) -> np.ndarray:
    """Bray-Curtis on proportions -- a non-log check on the CLR pseudocount."""
    p = counts.astype(np.float64)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1.0)
    n = p.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        num = np.abs(p[i] - p).sum(axis=1)
        den = np.maximum((p[i] + p).sum(axis=1), 1e-300)
        d[i] = num / den
    np.fill_diagonal(d, 0.0)
    return d


def pcoa(d: np.ndarray, n_axes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Principal coordinates: eigen-decomposition of the Gower-centred -d^2/2."""
    n = d.shape[0]
    a = -0.5 * d**2
    j = np.eye(n) - 1.0 / n
    gram = j @ a @ j
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(-values)
    values, vectors = values[order], vectors[:, order]
    positive = values > 1e-9
    coords = vectors[:, positive] * np.sqrt(values[positive])
    explained = values[positive] / values[positive].sum() if positive.any() else np.zeros(0)
    return coords[:, :n_axes], explained[:n_axes]


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def _indicator(codes: np.ndarray, n_groups: int) -> np.ndarray:
    z = np.zeros((codes.size, n_groups))
    z[np.arange(codes.size), codes] = 1.0
    return z


def _pseudo_f(d2: np.ndarray, codes: np.ndarray, n_groups: int, ss_total: float) -> tuple[float, float]:
    """Anderson (2001) pseudo-F from squared distances, O(n^2 k) per call.

    ``SS_within = sum_g (1 / 2 n_g) * 1_g' D2 1_g`` -- the identity that makes a
    5000-permutation PERMANOVA cheap without ever forming a centred Gram matrix
    per permutation.
    """
    z = _indicator(codes, n_groups)
    sizes = z.sum(axis=0)
    within = np.einsum("ij,ji->i", z.T, d2 @ z)
    ss_within = float((within / (2.0 * np.maximum(sizes, 1))).sum())
    ss_between = ss_total - ss_within
    n = codes.size
    df_res = n - n_groups
    if df_res <= 0 or ss_within <= 0:
        return float("nan"), float("nan")
    f = (ss_between / (n_groups - 1)) / (ss_within / df_res)
    return float(f), float(ss_between / ss_total)


@dataclass(frozen=True)
class PermanovaResult:
    f: float
    r2: float
    p: float
    r2_null_mean: float
    r2_null_p95: float
    n_units: int
    n_groups: int

    @property
    def r2_excess(self) -> float:
        """The only effect size worth quoting: R^2 above its own null mean."""
        return self.r2 - self.r2_null_mean


def permanova(
    d: np.ndarray,
    labels: np.ndarray,
    *,
    block: np.ndarray | None = None,
    n_perm: int = N_PERM,
    rng: np.random.Generator | None = None,
) -> PermanovaResult:
    """PERMANOVA with optional block-restricted permutation.

    ``block`` names, for each unit, the exchangeable unit -- the *sample*.  Group
    labels are shuffled between blocks and copied to every unit in a block, so a
    donor's two haplotypes never end up in different groups.  Without that, two
    haplotypes of one person count as two independent observations and the test
    is anticonservative.
    """
    rng = rng or np.random.default_rng(SEED)
    codes, categories = pd.factorize(labels, sort=True)
    n_groups = len(categories)
    n = codes.size
    d2 = d**2
    ss_total = float(d2.sum() / (2.0 * n))
    f_obs, r2_obs = _pseudo_f(d2, codes, n_groups, ss_total)

    if block is None:
        block = np.arange(n)
    block_codes, _ = pd.factorize(block, sort=True)
    n_blocks = block_codes.max() + 1
    label_of_block = np.zeros(n_blocks, dtype=int)
    label_of_block[block_codes] = codes

    f_null = np.empty(n_perm)
    r2_null = np.empty(n_perm)
    for i in range(n_perm):
        permuted = label_of_block[rng.permutation(n_blocks)][block_codes]
        f_null[i], r2_null[i] = _pseudo_f(d2, permuted, n_groups, ss_total)
    p = float((1 + np.sum(f_null >= f_obs)) / (n_perm + 1))
    return PermanovaResult(
        f=f_obs,
        r2=r2_obs,
        p=p,
        r2_null_mean=float(np.nanmean(r2_null)),
        r2_null_p95=float(np.nanquantile(r2_null, 0.95)),
        n_units=n,
        n_groups=n_groups,
    )


def permdisp(
    d: np.ndarray,
    labels: np.ndarray,
    *,
    block: np.ndarray | None = None,
    n_perm: int = 999,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Homogeneity of multivariate dispersion (Anderson 2006), in PCoA space.

    A PERMANOVA can fire because one group is simply more variable.  That is a
    real finding but a different one, so it gets its own test.
    """
    rng = rng or np.random.default_rng(SEED)
    coords, _ = pcoa(d, n_axes=d.shape[0])
    codes, categories = pd.factorize(labels, sort=True)
    n_groups = len(categories)

    def spread(c: np.ndarray) -> np.ndarray:
        out = np.empty(c.size)
        for g in range(n_groups):
            m = c == g
            if not m.any():
                continue
            out[m] = np.sqrt(((coords[m] - coords[m].mean(axis=0)) ** 2).sum(axis=1))
        return out

    def f_of(c: np.ndarray) -> float:
        s = spread(c)
        grand = s.mean()
        sizes = np.bincount(c, minlength=n_groups).astype(float)
        means = np.bincount(c, weights=s, minlength=n_groups) / np.maximum(sizes, 1)
        ss_b = float((sizes * (means - grand) ** 2).sum())
        ss_w = float(((s - means[c]) ** 2).sum())
        df_res = c.size - n_groups
        if ss_w <= 0 or df_res <= 0:
            return float("nan")
        return (ss_b / (n_groups - 1)) / (ss_w / df_res)

    if block is None:
        block = np.arange(codes.size)
    block_codes, _ = pd.factorize(block, sort=True)
    n_blocks = block_codes.max() + 1
    label_of_block = np.zeros(n_blocks, dtype=int)
    label_of_block[block_codes] = codes

    f_obs = f_of(codes)
    null = np.array(
        [f_of(label_of_block[rng.permutation(n_blocks)][block_codes]) for _ in range(n_perm)]
    )
    p = float((1 + np.sum(null >= f_obs)) / (n_perm + 1))
    observed = spread(codes)
    means = {
        str(categories[g]): float(observed[codes == g].mean()) for g in range(n_groups)
    }
    return f_obs, p, means


def neighbour_order(d: np.ndarray) -> np.ndarray:
    """Rows of ``d`` argsorted with self excluded -- invariant to relabelling.

    Hoisted out of the permutation loop: the distances never change when the
    group labels are shuffled, only which label sits at each rank.
    """
    dm = d.copy()
    np.fill_diagonal(dm, np.inf)
    return np.argsort(dm, axis=1, kind="stable")


def knn_accuracy(d: np.ndarray, labels: np.ndarray, k: int, order: np.ndarray | None = None) -> float:
    """Leave-one-out k-NN accuracy.  Ties broken in favour of the nearer member."""
    if order is None:
        order = neighbour_order(d)
    codes, categories = pd.factorize(labels, sort=True)
    z = _indicator(codes, len(categories))
    # A vote weight that decays with rank, by less than the gap between two
    # whole votes, so it can only ever break a tie.
    weight = 1.0 + 1e-6 * np.arange(k, 0, -1)
    votes = (z[order[:, :k]] * weight[None, :, None]).sum(axis=1)
    return float((votes.argmax(axis=1) == codes).mean())


def silhouette(d: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette width of the group labelling, computed as one matmul."""
    codes, categories = pd.factorize(labels, sort=True)
    n_groups = len(categories)
    if n_groups < 2:
        return float("nan")
    z = _indicator(codes, n_groups)
    sums = d @ z  # (n, n_groups): total distance from each unit to each group
    sizes = z.sum(axis=0)
    rows = np.arange(codes.size)
    own_size = sizes[codes] - 1.0
    a = np.where(own_size > 0, sums[rows, codes] / np.maximum(own_size, 1.0), 0.0)
    other = sums / sizes
    other[rows, codes] = np.inf
    b = other.min(axis=1)
    scores = np.where(own_size > 0, (b - a) / np.maximum(np.maximum(a, b), 1e-300), 0.0)
    return float(scores.mean())


def _permute_labels(
    labels: np.ndarray, block: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    block_codes, _ = pd.factorize(block, sort=True)
    n_blocks = block_codes.max() + 1
    label_of_block = np.empty(n_blocks, dtype=labels.dtype)
    label_of_block[block_codes] = labels
    return label_of_block[rng.permutation(n_blocks)][block_codes]


def permutation_p(
    observed: float,
    statistic,
    labels: np.ndarray,
    block: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Empirical one-sided p (larger is better) and the null mean."""
    null = np.array([statistic(_permute_labels(labels, block, rng)) for _ in range(n_perm)])
    p = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    return p, float(np.nanmean(null))


# --------------------------------------------------------------------------
# per-cluster tests
# --------------------------------------------------------------------------


def _anova_f(z: np.ndarray, x: np.ndarray, ss_t: np.ndarray | None = None) -> np.ndarray:
    """One-way F for every column of ``x`` at once, given the indicator ``z``.

    ``ss_t`` does not depend on the labels, so the permutation loop passes it in
    and never touches the (n_units x n_clusters) array again -- that single hoist
    is the difference between seconds and minutes at 232 samples.
    """
    sizes = z.sum(axis=0)
    grand = x.mean(axis=0)
    if ss_t is None:
        ss_t = ((x - grand) ** 2).sum(axis=0)
    means = (z.T @ x) / sizes[:, None]
    ss_b = (sizes[:, None] * (means - grand) ** 2).sum(axis=0)
    ss_w = np.maximum(ss_t - ss_b, 0.0)
    k, n = z.shape[1], z.shape[0]
    return (ss_b / (k - 1)) / np.maximum(ss_w / (n - k), 1e-300)


def _bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.minimum(ranked, 1.0)
    return q


def per_cluster_tests(
    x: np.ndarray,
    labels: np.ndarray,
    block: np.ndarray,
    n_perm: int = N_PERM_PER_CLUSTER,
    rng: np.random.Generator | None = None,
    contrast: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, float] | None]:
    """F per cluster, with a permutation p, a max-statistic FWER p and a contrast.

    The max-statistic null is the honest correction here: with thousands of
    clusters and a few dozen samples, an uncorrected p of 1e-3 is expected
    several times over.  Comparing each observed F to the distribution of the
    *largest* F under permuted labels controls the family-wise error rate exactly
    and, unlike Bonferroni, accounts for the correlation between clusters.

    ``contrast`` is a boolean mask over clusters (here: satellite or not).  The
    same permutations that build the FWER null also build a null for "do the
    masked clusters carry more group signal than the rest", measured as the
    difference in mean eta-squared.  That is the right null for the question --
    it holds the clusters fixed and shuffles only the labels, so it cannot be
    fooled by satellite clusters simply being bigger, rarer or more variable.
    """
    rng = rng or np.random.default_rng(SEED)
    codes, _ = pd.factorize(labels, sort=True)
    z = _indicator(codes, codes.max() + 1)
    ss_t = ((x - x.mean(axis=0)) ** 2).sum(axis=0)
    f_obs = _anova_f(z, x, ss_t)
    k, n = z.shape[1], z.shape[0]

    def eta2_of(f: np.ndarray) -> np.ndarray:
        return (f * (k - 1)) / (f * (k - 1) + (n - k))

    usable_contrast = (
        contrast is not None and contrast.any() and (~contrast).any()
    )
    ge = np.zeros(f_obs.size)
    max_null = np.empty(n_perm)
    contrast_null = np.empty(n_perm) if usable_contrast else None
    # As a dot product rather than fancy indexing: no per-permutation allocation
    # of a mask-sized temporary, which matters at 40 k clusters x 10 k draws.
    weights = None
    if usable_contrast:
        inside = contrast.astype(np.float64)
        weights = inside / inside.sum() - (1.0 - inside) / (1.0 - inside).sum()
    for i in range(n_perm):
        zp = _indicator(pd.factorize(_permute_labels(labels, block, rng), sort=True)[0], z.shape[1])
        f_null = _anova_f(zp, x, ss_t)
        max_null[i] = f_null.max()
        ge += f_null >= f_obs
        if contrast_null is not None:
            contrast_null[i] = eta2_of(f_null) @ weights

    p_perm = (1.0 + ge) / (n_perm + 1)
    sorted_max = np.sort(max_null)
    p_fwer = (1.0 + (n_perm - np.searchsorted(sorted_max, f_obs, side="left"))) / (n_perm + 1)
    eta2 = eta2_of(f_obs)
    frame = pd.DataFrame(
        {"f": f_obs, "eta2": eta2, "p_perm": p_perm, "q_bh": _bh(p_perm), "p_fwer": p_fwer}
    )
    if contrast_null is None:
        return frame, None
    observed = float(eta2[contrast].mean() - eta2[~contrast].mean())
    return frame, {
        "mean_eta2_in": float(eta2[contrast].mean()),
        "mean_eta2_out": float(eta2[~contrast].mean()),
        "difference": observed,
        "null_mean": float(contrast_null.mean()),
        "null_sd": float(contrast_null.std(ddof=1)),
        "z": float((observed - contrast_null.mean()) / max(contrast_null.std(ddof=1), 1e-300)),
        "p": float((1 + np.sum(contrast_null >= observed)) / (n_perm + 1)),
    }


# --------------------------------------------------------------------------
# the satellite-vs-rest control
# --------------------------------------------------------------------------


def size_matched_subset(
    sizes: np.ndarray, pool: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one pool cluster per target cluster, matched on log bin count.

    Matching on size matters because PERMANOVA's effect size depends on how many
    features there are *and* how heavy-tailed they are; an unmatched draw would
    compare 431 satellite clusters against 431 unusually small ones.
    """
    pool_sizes = np.log1p(sizes[pool])
    order = np.argsort(pool_sizes)
    ordered_pool, ordered_sizes = pool[order], pool_sizes[order]
    used = np.zeros(ordered_pool.size, dtype=bool)
    chosen = np.empty(target.size, dtype=np.int64)
    for i, t in enumerate(rng.permutation(target)):
        want = np.log1p(sizes[t])
        candidates = np.flatnonzero(~used)
        if candidates.size == 0:
            chosen[i:] = rng.choice(pool, target.size - i, replace=True)
            break
        j = candidates[np.argmin(np.abs(ordered_sizes[candidates] - want))]
        used[j] = True
        chosen[i] = ordered_pool[j]
    return chosen


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _permanova_line(label: str, res: PermanovaResult) -> str:
    return (
        f"  {label:<26s} R2={res.r2:.3f}  null={res.r2_null_mean:.3f}  "
        f"excess={res.r2_excess:+.3f}  F={res.f:.3f}  p={res.p:.4f}"
    )


def main(argv: list[str]) -> int:
    outdir = Path(argv[1]) if len(argv) > 1 else Path("results/acro3")
    group_col = argv[2] if len(argv) > 2 else "superpopulation"
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary: dict[str, object] = {"outdir": str(outdir), "group_col": group_col, "seed": SEED}

    manifest = pd.read_csv(outdir / "manifest.tsv", sep="\t", dtype=str)
    ab = load_abundance(outdir)
    desc = cluster_descriptions(outdir, ab)
    sizes = ab.counts.sum(axis=0, dtype=np.int64)
    desc["bins"] = sizes

    _rule(f"kmer-dust population structure  --  {outdir}")
    print(f"  assemblies              {len(ab.assemblies)}")
    print(f"  bins                    {int(ab.total_bins.sum()):,}")
    print(
        f"  clustered / noise       {int(ab.counts.sum()):,} / {int(ab.noise_bins.sum()):,}"
        f"  ({ab.noise_bins.sum() / ab.total_bins.sum():.1%} noise)"
    )
    print(f"  clusters                {len(ab.cluster_ids):,}")
    sat_mask = desc["is_satellite"].to_numpy()
    print(
        f"  satellite clusters      {int(sat_mask.sum()):,}"
        f"  ({int(sizes[sat_mask].sum()):,} bins,"
        f" {sizes[sat_mask].sum() / max(sizes.sum(), 1):.1%} of clustered sequence)"
    )
    print(
        "  satellite families      "
        + ", ".join(
            f"{k}:{v}" for k, v in desc.loc[sat_mask, "feature"].value_counts().items()
        )
    )
    borderline = [
        f"{k}:{v}"
        for k, v in desc.loc[~sat_mask, "feature"].value_counts().items()
        if "sat" in str(k) or str(k) in {"subterminal", "mon"}
    ]
    if borderline:
        # Deliberately left in the comparison group: including them would only
        # make the satellite-vs-rest contrast easier to pass.
        print(f"  satellite-ish in 'other'  {', '.join(borderline)} (prefix rule is conservative)")

    samples = build_units(ab, manifest, "sample", group_col)
    samples, dropped = drop_small_groups(samples, group_col)
    haps = build_units(ab, manifest, "haplotype", group_col)
    haps, _ = drop_small_groups(haps, group_col)
    labels = samples.meta[group_col].astype(str).to_numpy()
    counts = samples.counts

    _rule("Units")
    print(f"  analysis unit           {samples.level} (haplotypes summed within donor)")
    print(f"  samples                 {counts.shape[0]}   haplotypes {haps.counts.shape[0]}")
    print(f"  groups ({group_col})")
    for g, n in samples.meta[group_col].value_counts().sort_index().items():
        n_hap = int((haps.meta[group_col] == g).sum())
        print(f"    {str(g):<8s} {n:3d} samples  {n_hap:3d} haplotypes")
    if dropped:
        print(f"  dropped (<{MIN_UNITS_PER_GROUP} units): {', '.join(dropped)}")
    summary["n_samples"] = int(counts.shape[0])
    summary["n_haplotypes"] = int(haps.counts.shape[0])
    summary["group_sizes"] = {
        str(k): int(v) for k, v in samples.meta[group_col].value_counts().items()
    }
    summary["dropped_groups"] = dropped

    if counts.shape[0] < 4 or len(set(labels)) < 2:
        print("\n  too few labelled units to test; stopping.")
        return 0

    # ---- confounder: completeness -----------------------------------------
    _rule("Confounder check: assembly completeness")
    total = samples.meta["total_bins"].to_numpy().astype(float)
    clustered = samples.meta["clustered_bins"].to_numpy().astype(float)
    noise_frac = samples.meta["noise_bins"].to_numpy() / np.maximum(total, 1)
    sat_frac = counts[:, sat_mask].sum(axis=1) / np.maximum(clustered, 1)
    for name, values in (
        ("total bins", total),
        ("noise fraction", noise_frac),
        ("satellite fraction", sat_frac),
    ):
        f_obs = _anova_f(_indicator(pd.factorize(labels, sort=True)[0], len(set(labels))),
                         values[:, None])[0]
        p, _ = permutation_p(
            f_obs,
            lambda lab, v=values: _anova_f(
                _indicator(pd.factorize(lab, sort=True)[0], len(set(labels))), v[:, None]
            )[0],
            labels,
            samples.block,
            999,
            rng,
        )
        by_group = pd.Series(values).groupby(labels).mean()
        print(f"  {name:<20s} F={f_obs:6.3f} p={p:.3f}   " +
              "  ".join(f"{g}={v:,.4g}" for g, v in by_group.items()))
        summary[f"confounder_{name.replace(' ', '_')}"] = {"F": float(f_obs), "p": float(p)}
    print(
        "  Completeness is removed by closure in what follows; the residualised\n"
        "  re-run below is the belt-and-braces version."
    )

    # ---- the three feature sets -------------------------------------------
    feature_sets: dict[str, np.ndarray] = {
        "all clusters": np.ones(counts.shape[1], dtype=bool),
        "satellite only": sat_mask,
        "non-satellite": ~sat_mask,
    }
    _rule(f"PERMANOVA on Aitchison distance  --  {samples.level} level, {N_PERM} permutations")
    print(
        f"  Under the null R2 ~ (k-1)/(n-1) = {(len(set(labels)) - 1) / (counts.shape[0] - 1):.3f};"
        " read the excess, not R2."
    )
    results: dict[str, PermanovaResult] = {}
    distances: dict[str, np.ndarray] = {}
    # The manifest carries a dozen S3 URLs per row; the figure data does not
    # need them.
    wanted = [
        "sample", "assembly", "haplotype", "source", "population", "superpopulation",
        "sex", "n_haplotypes", "clustered_bins", "noise_bins", "total_bins",
    ]
    unit_frame = samples.meta[[c for c in wanted if c in samples.meta.columns]].copy()
    unit_frame["level"] = samples.level
    unit_frame["noise_frac"] = noise_frac
    unit_frame["satellite_frac"] = sat_frac
    for name, cols in feature_sets.items():
        if cols.sum() == 0:
            print(f"  {name:<26s} no clusters in this set")
            continue
        x, kept = clr(counts[:, cols])
        d = euclidean(x)
        distances[name] = d
        res = permanova(d, labels, n_perm=N_PERM, rng=np.random.default_rng(SEED))
        results[name] = res
        print(_permanova_line(f"{name} ({kept.sum()})", res))
        coords, explained = pcoa(d)
        tag = name.split()[0]
        for axis in range(coords.shape[1]):
            unit_frame[f"pcoa{axis + 1}_{tag}"] = coords[:, axis]
        summary[f"permanova_{tag}"] = {
            "n_clusters": int(kept.sum()),
            "F": res.f,
            "R2": res.r2,
            "R2_null_mean": res.r2_null_mean,
            "R2_excess": res.r2_excess,
            "p": res.p,
            "pcoa_explained": [float(v) for v in explained],
        }

    # ---- k-NN, silhouette, dispersion --------------------------------------
    _rule("Classification, silhouette and dispersion")
    for name, d in distances.items():
        line = [f"  {name:<16s}"]
        order = neighbour_order(d)
        for k in (1, 3):
            acc = knn_accuracy(d, labels, k, order)
            p, null = permutation_p(
                acc,
                lambda lab, dd=d, kk=k, oo=order: knn_accuracy(dd, lab, kk, oo),
                labels, samples.block, 999, rng,
            )
            line.append(f"{k}NN={acc:.3f} (null {null:.3f}, p={p:.3f})")
        sil = silhouette(d, labels)
        p_sil, null_sil = permutation_p(
            sil, lambda lab, dd=d: silhouette(dd, lab), labels, samples.block, 999, rng
        )
        line.append(f"sil={sil:+.4f} (null {null_sil:+.4f}, p={p_sil:.3f})")
        print("  ".join(line))
        f_disp, p_disp, spread = permdisp(d, labels, n_perm=999, rng=np.random.default_rng(SEED))
        print(
            f"    PERMDISP F={f_disp:.3f} p={p_disp:.3f}  spread: "
            + "  ".join(f"{g}={v:.2f}" for g, v in sorted(spread.items()))
        )
        tag = name.split()[0]
        summary[f"tests_{tag}"] = {
            "knn1": float(knn_accuracy(d, labels, 1, order)),
            "silhouette": float(sil),
            "silhouette_p": float(p_sil),
            "permdisp_F": float(f_disp),
            "permdisp_p": float(p_disp),
            "dispersion_by_group": spread,
        }

    # ---- haplotype level, restricted permutation ---------------------------
    _rule("Haplotype level, permutations restricted within donor")
    hap_labels = haps.meta[group_col].astype(str).to_numpy()
    for name, cols in feature_sets.items():
        if cols.sum() == 0:
            continue
        x, _ = clr(haps.counts[:, cols])
        res = permanova(
            euclidean(x), hap_labels, block=haps.block, n_perm=N_PERM,
            rng=np.random.default_rng(SEED),
        )
        print(_permanova_line(name, res))
        summary[f"permanova_hap_{name.split()[0]}"] = {
            "F": res.f, "R2": res.r2, "R2_null_mean": res.r2_null_mean, "p": res.p
        }

    # ---- positive control: do a donor's two haplotypes pair up? ------------
    if haps.counts.shape[0] > samples.counts.shape[0]:
        _rule("Positive control: does a haplotype's nearest neighbour share its donor?")
        print(
            "  Homologous acrocentric short arms are two independently inherited\n"
            "  chromosomes, so strong pairing is NOT expected -- this measures how\n"
            "  much individual-level signal the composition carries at all."
        )
        sample_of_hap = haps.meta["sample"].astype(str).to_numpy()
        for name, cols in feature_sets.items():
            if cols.sum() == 0:
                continue
            x, _ = clr(haps.counts[:, cols])
            d = euclidean(x)
            np.fill_diagonal(d, np.inf)
            order = np.argsort(d, axis=1, kind="stable")
            ranks = []
            for i in range(len(sample_of_hap)):
                hit = np.flatnonzero(sample_of_hap[order[i]] == sample_of_hap[i])
                if hit.size:
                    ranks.append(hit[0] + 1)
            if not ranks:
                continue
            ranks_arr = np.asarray(ranks, dtype=float)
            n_other = len(sample_of_hap) - 1
            expected = (n_other + 1) / 2.0
            top1 = float(np.mean(ranks_arr == 1))
            print(
                f"  {name:<16s} mean partner rank {ranks_arr.mean():.2f} of {n_other}"
                f" (chance {expected:.2f});  partner is nearest neighbour {top1:.1%}"
                f" (chance {1 / n_other:.1%})"
            )
            summary[f"pairing_{name.split()[0]}"] = {
                "mean_rank": float(ranks_arr.mean()),
                "chance_rank": float(expected),
                "top1": top1,
            }

    # ---- distance decomposition -------------------------------------------
    if "satellite only" in distances and haps.counts.shape[0] > samples.counts.shape[0]:
        x, _ = clr(haps.counts[:, sat_mask])
        d = euclidean(x)
        s = haps.meta["sample"].astype(str).to_numpy()
        iu = np.triu_indices(len(s), 1)
        same_sample = s[iu[0]] == s[iu[1]]
        same_group = hap_labels[iu[0]] == hap_labels[iu[1]]
        pairs = d[iu]
        _rule("Aitchison distance decomposition (haplotype pairs, satellite clusters)")
        for label, mask in (
            ("same individual", same_sample),
            (f"same {group_col}, different individual", same_group & ~same_sample),
            (f"different {group_col}", ~same_group),
        ):
            if mask.any():
                print(f"  {label:<44s} n={int(mask.sum()):5d}  mean={pairs[mask].mean():.3f}")
        summary["distance_decomposition"] = {
            "same_individual": float(pairs[same_sample].mean()) if same_sample.any() else None,
            "same_group": float(pairs[same_group & ~same_sample].mean())
            if (same_group & ~same_sample).any() else None,
            "different_group": float(pairs[~same_group].mean()) if (~same_group).any() else None,
        }

    # ---- is the satellite result a feature-count artefact? -----------------
    if sat_mask.any() and (~sat_mask).any():
        _rule(f"Control: {N_MATCHED_SUBSETS} size-matched random non-satellite cluster sets")
        # Both sides skip the prevalence filter so that the matched sets keep
        # exactly the satellite cluster count -- comparing a filtered reference
        # against unfiltered draws would compare different numbers of features,
        # which is the very artefact this control exists to rule out.
        pool = np.flatnonzero(~sat_mask)
        target = np.flatnonzero(sat_mask)
        x_sat, _ = clr(counts[:, sat_mask], min_prevalence=0.0)
        sat_ref = permanova(
            euclidean(x_sat), labels, n_perm=N_PERM_MATCHED, rng=np.random.default_rng(SEED)
        )
        excesses = np.empty(N_MATCHED_SUBSETS)
        ps = np.empty(N_MATCHED_SUBSETS)
        for b in range(N_MATCHED_SUBSETS):
            sub = size_matched_subset(sizes, pool, target, rng)
            x, _ = clr(counts[:, sub], min_prevalence=0.0)
            res = permanova(
                euclidean(x), labels, n_perm=N_PERM_MATCHED, rng=np.random.default_rng(SEED + b)
            )
            excesses[b] = res.r2_excess
            ps[b] = res.p
        exceedance = float((1 + np.sum(excesses >= sat_ref.r2_excess)) / (N_MATCHED_SUBSETS + 1))
        print(
            f"  matched non-satellite excess R2: mean {excesses.mean():+.3f}"
            f"  p95 {np.quantile(excesses, 0.95):+.3f}  max {excesses.max():+.3f}"
        )
        print(f"  fraction of matched sets with p<0.05: {np.mean(ps < 0.05):.3f}")
        print(
            f"  satellite excess R2 {sat_ref.r2_excess:+.3f} on the same settings"
            f"  ->  exceeded by {np.mean(excesses >= sat_ref.r2_excess):.1%} of matched"
            f" sets  (empirical p = {exceedance:.3f})"
        )
        summary["matched_control"] = {
            "n_subsets": N_MATCHED_SUBSETS,
            "n_clusters": int(sat_mask.sum()),
            "excess_mean": float(excesses.mean()),
            "excess_max": float(excesses.max()),
            "frac_p_lt_05": float(np.mean(ps < 0.05)),
            "satellite_excess": sat_ref.r2_excess,
            "exceedance_p": exceedance,
        }

    # ---- robustness --------------------------------------------------------
    _rule("Robustness")
    if sat_mask.any():
        print("  satellite clusters, CLR grid (prevalence filter x pseudocount)")
        grid = []
        for prev in (0.0, 0.25, 0.5, 0.9):
            for pc in (0.5, 1.0):
                x, kept = clr(counts[:, sat_mask], min_prevalence=prev, pseudocount=pc)
                res = permanova(
                    euclidean(x), labels, n_perm=999, rng=np.random.default_rng(SEED)
                )
                print(
                    f"    prevalence>={prev:.2f} pseudocount={pc:.1f}: kept={kept.sum():5d}"
                    f"  excess R2={res.r2_excess:+.3f}  p={res.p:.4f}"
                )
                grid.append({"prevalence": prev, "pseudocount": pc,
                             "R2_excess": res.r2_excess, "p": res.p})
        summary["clr_grid"] = grid

    print("  Bray-Curtis on proportions (no log, no pseudocount)")
    for name, cols in feature_sets.items():
        if cols.sum() == 0:
            continue
        res = permanova(
            bray_curtis(counts[:, cols]), labels, n_perm=999, rng=np.random.default_rng(SEED)
        )
        print(_permanova_line(name, res))
        summary[f"braycurtis_{name.split()[0]}"] = {"R2_excess": res.r2_excess, "p": res.p}

    print("  residualised on log(total clustered bins)")
    logtot = np.log(np.maximum(clustered, 1.0))
    design = np.c_[np.ones(logtot.size), logtot - logtot.mean()]
    for name, cols in feature_sets.items():
        if cols.sum() == 0:
            continue
        x, _ = clr(counts[:, cols])
        resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        res = permanova(euclidean(resid), labels, n_perm=999, rng=np.random.default_rng(SEED))
        print(_permanova_line(name, res))
        summary[f"residualised_{name.split()[0]}"] = {"R2_excess": res.r2_excess, "p": res.p}

    if "satellite only" in distances:
        # Leave-one-out asks "is this driven by one donor?".  Past a few dozen
        # units no single one can be, and refitting 232 times is wasted work, so
        # a random subset of drops answers the same question.
        d = distances["satellite only"]
        drops = np.arange(len(labels))
        if drops.size > MAX_JACKKNIFE_FITS:
            drops = np.sort(rng.choice(drops, MAX_JACKKNIFE_FITS, replace=False))
        print(f"  leave-one-sample-out jackknife of the satellite excess R2 ({drops.size} fits)")
        jack = []
        for i in drops:
            m = np.ones(len(labels), dtype=bool)
            m[i] = False
            if pd.Series(labels[m]).value_counts().min() < MIN_UNITS_PER_GROUP:
                continue
            res = permanova(
                d[np.ix_(m, m)], labels[m], n_perm=199, rng=np.random.default_rng(SEED)
            )
            jack.append(res.r2_excess)
        if jack:
            jack_arr = np.asarray(jack)
            print(
                f"    excess R2 across {jack_arr.size} leave-one-out fits:"
                f"  min {jack_arr.min():+.3f}  mean {jack_arr.mean():+.3f}"
                f"  max {jack_arr.max():+.3f}  sd {jack_arr.std(ddof=1):.3f}"
            )
            summary["jackknife_satellite_excess_R2"] = {
                "min": float(jack_arr.min()), "mean": float(jack_arr.mean()),
                "max": float(jack_arr.max()), "sd": float(jack_arr.std(ddof=1)),
            }

    # ---- which groups differ from which -------------------------------------
    if "satellite only" in distances and len(set(labels)) > 2:
        _rule("Pairwise group contrasts (satellite clusters)")
        d = distances["satellite only"]
        cats = sorted(set(labels))
        pair_rows = []
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                m = np.isin(labels, [cats[i], cats[j]])
                if pd.Series(labels[m]).value_counts().min() < MIN_UNITS_PER_GROUP:
                    continue
                res = permanova(
                    d[np.ix_(m, m)], labels[m], n_perm=999, rng=np.random.default_rng(SEED)
                )
                print(
                    f"  {cats[i]} vs {cats[j]:<8s} n={int(m.sum()):3d}"
                    f"  excess R2={res.r2_excess:+.3f}  F={res.f:.2f}  p={res.p:.3f}"
                )
                pair_rows.append({"a": cats[i], "b": cats[j], "n": int(m.sum()),
                                  "R2_excess": res.r2_excess, "p": res.p})
        summary["pairwise"] = pair_rows
        print("  (uncorrected; with this many groups treat these as descriptive)")

    # ---- per satellite family ----------------------------------------------
    families = desc.loc[sat_mask, "feature"].value_counts()
    testable = [f for f, n in families.items() if n >= 8]
    if testable:
        _rule("Per satellite family")
        fam_rows = []
        for fam in testable:
            cols = (desc["feature"].to_numpy() == fam)
            x, kept = clr(counts[:, cols])
            res = permanova(euclidean(x), labels, n_perm=999, rng=np.random.default_rng(SEED))
            print(
                f"  {fam:<18s} clusters={int(cols.sum()):4d} kept={kept.sum():4d}"
                f"  excess R2={res.r2_excess:+.3f}  F={res.f:.3f}  p={res.p:.4f}"
            )
            fam_rows.append({"family": fam, "n_clusters": int(cols.sum()),
                             "R2_excess": res.r2_excess, "p": res.p})
        summary["families"] = fam_rows
        print("  (uncorrected across families; a Bonferroni threshold here is "
              f"{0.05 / len(testable):.4f})")

    # ---- per cluster --------------------------------------------------------
    _rule(f"Per-cluster tests ({N_PERM_PER_CLUSTER} permutations, max-statistic FWER)")
    x_all, kept_all = clr(counts)
    tests, contrast = per_cluster_tests(
        x_all, labels, samples.block, N_PERM_PER_CLUSTER, rng, contrast=sat_mask[kept_all]
    )
    table = desc.loc[kept_all].reset_index(drop=True).join(tests)
    present = counts[:, kept_all] > 0
    prop = counts[:, kept_all] / np.maximum(clustered[:, None], 1.0)
    groups = sorted(set(labels))
    for g in groups:
        m = labels == g
        table[f"mean_ppm_{g}"] = prop[m].mean(axis=0) * 1e6
        table[f"mean_clr_{g}"] = x_all[m].mean(axis=0)
        table[f"prevalence_{g}"] = present[m].mean(axis=0)
    ppm_cols = [f"mean_ppm_{g}" for g in groups]
    ppm = table[ppm_cols].to_numpy()
    # One bin, expressed in the same ppm units, is the smallest difference the
    # data can express; using it as the floor keeps the fold change finite
    # without inventing a 10^9 for a cluster that is simply absent in a group.
    floor = 1e6 / float(np.median(clustered))
    table["log2_max_over_min"] = np.log2((ppm.max(axis=1) + floor) / (ppm.min(axis=1) + floor))
    table["top_group"] = [ppm_cols[i].replace("mean_ppm_", "") for i in ppm.argmax(axis=1)]
    table["n_units_present"] = present.sum(axis=0)
    table = table.sort_values("f", ascending=False).reset_index(drop=True)

    print(f"  clusters tested         {len(table):,}")
    print(f"  smallest permutation p  {table['p_perm'].min():.5f}")
    print(f"  smallest FWER p         {table['p_fwer'].min():.4f}")
    print(f"  FWER < 0.05             {int((table['p_fwer'] < 0.05).sum())}")
    for thresh in (0.05, 0.10, 0.25):
        print(f"  BH q < {thresh:<4.2f}            {int((table['q_bh'] < thresh).sum())}")
    q_floor = len(table) / (N_PERM_PER_CLUSTER + 1)
    if q_floor >= 1.0:
        print(
            f"  (BH cannot reach any threshold here: with {len(table):,} clusters and"
            f" {N_PERM_PER_CLUSTER} permutations the\n   smallest possible q is"
            f" {q_floor:.1f}.  Read the FWER column, which has no such floor.)"
        )
    else:
        print(
            f"  (the smallest attainable BH q is {q_floor:.3f} at this permutation"
            " count; the FWER column has no such floor)"
        )
    print("\n  top clusters by F:")
    print(
        f"    {'cluster':<8s} {'feature':<17s} {'class':<10s} {'chrom':<7s} {'bins':>7s} "
        f"{'F':>7s} {'p':>8s} {'q_BH':>6s} {'FWER':>6s} {'top':<5s} {'log2fc':>7s}  present in top/rest"
    )
    n_units = counts.shape[0]
    for _, r in table.head(20).iterrows():
        top = str(r["top_group"])
        in_top = int(round(r[f"prevalence_{top}"] * (labels == top).sum()))
        n_top = int((labels == top).sum())
        rest = int(r["n_units_present"]) - in_top
        print(
            f"    C{int(r['cluster']):<7d} {str(r['feature']):<17s} {str(r['feature_class']):<10s}"
            f" {str(r['chrom']):<7s} {int(r['bins']):>7,d} {r['f']:>7.2f} {r['p_perm']:>8.4f}"
            f" {r['q_bh']:>6.3f} {r['p_fwer']:>6.3f} {top:<5s} {r['log2_max_over_min']:>+7.2f}"
            f"  {in_top}/{n_top} vs {rest}/{n_units - n_top}"
        )
    for topn in (50, 100, 200):
        if len(table) >= topn:
            frac = table.head(topn)["is_satellite"].mean()
            print(
                f"  satellite fraction of top {topn:>3d} by F: {frac:.3f}"
                f"  (background {table['is_satellite'].mean():.3f})"
            )
    if contrast is not None:
        print(
            "\n  Do satellite clusters carry more group signal than the rest?\n"
            f"    mean eta^2  satellite {contrast['mean_eta2_in']:.4f}"
            f"  vs other {contrast['mean_eta2_out']:.4f}"
            f"  difference {contrast['difference']:+.4f}\n"
            f"    permutation null {contrast['null_mean']:+.4f}"
            f" +/- {contrast['null_sd']:.4f} (sd)   z={contrast['z']:+.2f}"
            f"   p={contrast['p']:.4f}"
        )
        summary["satellite_contrast"] = contrast
    summary["per_cluster"] = {
        "n_tested": int(len(table)),
        "min_p_perm": float(table["p_perm"].min()),
        "min_p_fwer": float(table["p_fwer"].min()),
        "n_fwer_lt_05": int((table["p_fwer"] < 0.05).sum()),
        "n_q_lt_10": int((table["q_bh"] < 0.10).sum()),
        "n_q_lt_25": int((table["q_bh"] < 0.25).sum()),
    }

    # ---- write --------------------------------------------------------------
    table.to_parquet(analysis_dir / "population.parquet", index=False)
    unit_frame.to_parquet(analysis_dir / "population_units.parquet", index=False)
    (analysis_dir / "population_summary.json").write_text(json.dumps(summary, indent=2))

    _rule("Written")
    print(f"  {analysis_dir / 'population.parquet'}       one row per cluster tested")
    print(f"  {analysis_dir / 'population_units.parquet'} per-unit ordination + covariates")
    print(f"  {analysis_dir / 'population_summary.json'}  every statistic above")

    _rule("How to read this")
    n_s, n_g = counts.shape[0], len(set(labels))
    smallest = int(min(summary["group_sizes"].values()))
    print(f"  {n_s} donors in {n_g} groups; smallest group {smallest}.")
    if smallest < 10:
        print(
            "  A group of that size fixes what can be concluded.  The multivariate\n"
            "  tests can detect a shift in the *mean* composition of a whole cluster\n"
            "  set; a single cluster would need an enormous effect to clear a\n"
            f"  {len(table):,}-cluster family-wise threshold, so an empty FWER column\n"
            "  is a statement about power, not about biology.  Nothing here estimates\n"
            "  an allele frequency, and no individual cluster should be quoted as an\n"
            "  ancestry-associated variant on this evidence alone."
        )
    else:
        print(
            "  Group sizes are large enough that the per-cluster FWER column is\n"
            "  meaningful in both directions: a cluster that clears it is a real\n"
            "  candidate, and the multivariate excess R2 is estimated tightly enough\n"
            "  to compare between feature sets."
        )
    print(
        "  In every case the group label is confounded with everything else that\n"
        "  differs between donor cohorts -- sequencing batch, coverage, assembler\n"
        "  version, and which samples were chosen for assembly -- none of which this\n"
        "  design can separate from ancestry.  Satellite arrays are also the part of\n"
        "  the genome most sensitive to assembly quality, which is precisely why the\n"
        "  completeness conditioning above is not optional."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
