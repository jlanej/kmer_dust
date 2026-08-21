"""Give the clusters names by asking what they are made of.

HDBSCAN hands back integers.  A cluster is only interpretable once we can say
"cluster 7 is the live alpha-satellite HOR array" -- and say it with a number
attached, because "mostly satellite" is easy to believe and easy to be wrong
about.  So every (cluster, feature) pair gets a 2x2 contingency test against the
same background: *all* bins that reached the clustering stage, noise included.
Excluding noise from the background would inflate every enrichment, since noise
bins are exactly the ones that failed to look like anything.

A bin "carries" a feature when the annotated covered fraction reaches
``cfg.enrich.min_frac``; a bin can therefore carry several features at once
(a bin can be half satellite and half segdup), which is why each feature is
tested independently rather than as a multinomial.

The p-value is the hypergeometric survival function -- the probability of seeing
at least this many carrying bins when drawing ``cluster_size`` bins without
replacement from the background.  It is a descriptive ranking statistic, not an
inferential claim: bins are spatially autocorrelated along a chromosome, so the
independence assumption is generous and the p-values should be read as an
ordering, not as evidence.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from kmer_dust import schemas
from kmer_dust.config import Config
from kmer_dust.log import get_logger

logger = get_logger(__name__)

__all__ = ["enrich_clusters", "load_enrichment", "NOISE_CLUSTER", "NOISE_NAME"]

#: HDBSCAN's label for "did not belong to any cluster".
NOISE_CLUSTER = -1
#: Noise is never named after a feature: the whole point of the label is that
#: the bin did not join a coherent group, so an enrichment there is an accident
#: of the leftovers rather than a description of a class of sequence.
NOISE_NAME = "noise"

#: A feature needs at least this many carrying bins in a cluster before it may
#: name it -- a 2-bin coincidence at log2fc 6 is noise, not a discovery.
MIN_BINS_TO_NAME = 5


def _empty_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        schemas.empty_frame(schemas.ENRICHMENT_COLUMNS),
        schemas.empty_frame(schemas.CLUSTER_NAME_COLUMNS),
    )


def _hypergeom_neg_log10_sf(k: np.ndarray, N: int, K: np.ndarray, n: int) -> np.ndarray:
    """``-log10 P(X >= k)`` for X ~ Hypergeometric(N, K, n), underflow-safe.

    ``sf`` underflows to exactly 0 for the strongest enrichments -- which are the
    ones we most want to rank -- so fall back to ``logsf`` there instead of
    clamping everything interesting to the same number.
    """
    out = np.zeros(k.shape, dtype=np.float64)
    if N <= 0 or n <= 0 or k.size == 0:
        return out
    try:
        from scipy.stats import hypergeom
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        logger.warning("scipy unavailable; enrichment p-values reported as 0")
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = np.asarray(hypergeom.sf(k - 1, N, K, n), dtype=np.float64)
        sf = np.where(np.isfinite(sf), sf, 1.0)
        positive = sf > 0.0
        out[positive] = -np.log10(sf[positive])
        weak = ~positive
        if weak.any():
            logsf = np.asarray(hypergeom.logsf(k[weak] - 1, N, K[weak], n), dtype=np.float64)
            finite = np.isfinite(logsf)
            # -log10(min positive double) ~ 323.3; anything past that is "certain".
            out[np.flatnonzero(weak)[finite]] = -logsf[finite] / math.log(10.0)
            out[np.flatnonzero(weak)[~finite]] = -math.log10(np.finfo(np.float64).tiny)
    return np.clip(out, 0.0, None)


def enrich_clusters(
    rows: pd.DataFrame,
    clusters: pd.DataFrame,
    annotations: pd.DataFrame,
    cfg: Config,
    outdir: Path,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test every cluster against every feature and name the clusters.

    Returns ``(enrichment, cluster_names)`` matching
    :data:`schemas.ENRICHMENT_COLUMNS` and :data:`schemas.CLUSTER_NAME_COLUMNS`.
    Enrichment rows are produced for clusters of at least
    ``cfg.enrich.min_cluster_size`` bins (noise included); *every* cluster gets a
    name, so nothing downstream has to cope with a missing label.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    enrich_path = outdir / "enrichment.parquet"
    names_path = outdir / "cluster_names.parquet"
    if enrich_path.exists() and names_path.exists() and not force:
        logger.info("enrich: reusing %s", outdir)
        return load_enrichment(outdir)

    enrichment, names = _compute(rows, clusters, annotations, cfg)
    _write(enrichment, enrich_path)
    _write(names, names_path)
    logger.info(
        "enrich: %d cluster(s), %d enrichment row(s)", len(names), len(enrichment)
    )
    return enrichment, names


def _compute(
    rows: pd.DataFrame, clusters: pd.DataFrame, annotations: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if clusters is None or len(clusters) == 0:
        logger.warning("enrich: no clustered bins")
        return _empty_outputs()

    joined = _join(rows, clusters, annotations)
    if joined.empty:
        logger.warning("enrich: no bins survived the cluster/annotation join")
        return _empty_outputs()

    features = list(schemas.FEATURE_VOCAB)
    frac_cols = [schemas.feature_column(f) for f in features]
    min_frac = float(cfg.enrich.min_frac)
    carries = (
        joined[frac_cols].to_numpy(dtype=np.float32) >= np.float32(min_frac)
    ) & joined["annotated"].to_numpy(dtype=bool)[:, None]

    labels = joined["cluster"].to_numpy(dtype=np.int64)
    total = int(carries.shape[0])
    carried_total = carries.sum(axis=0).astype(np.int64)

    cluster_ids = np.unique(labels)
    enrich_records: list[dict[str, object]] = []
    name_records: list[dict[str, object]] = []
    # One bin's worth of frequency: keeps log2 finite for clusters that carry a
    # feature zero times without letting the pseudo-count dominate real ratios.
    eps = 1.0 / max(total, 1)

    for cid in cluster_ids:
        mask = labels == cid
        size = int(mask.sum())
        member_carries = carries[mask]
        counts = member_carries.sum(axis=0).astype(np.int64)
        annotated_here = int(joined["annotated"].to_numpy(dtype=bool)[mask].sum())

        top_feature = ""
        ranked: list[tuple[str, float]] = []
        if size >= int(cfg.enrich.min_cluster_size):
            testable = carried_total > 0
            if testable.any():
                idx = np.flatnonzero(testable)
                k = counts[idx]
                K = carried_total[idx]
                frac_cluster = k / size
                frac_background = K / total
                log2fc = np.log2((frac_cluster + eps) / (frac_background + eps))
                neg_log10_p = _hypergeom_neg_log10_sf(k, total, K, size)
                for j, fi in enumerate(idx):
                    enrich_records.append(
                        {
                            "cluster": int(cid),
                            "feature": features[fi],
                            "n_bins_cluster": int(k[j]),
                            "n_bins_total": int(K[j]),
                            "cluster_size": size,
                            "background_size": total,
                            "frac_cluster": float(frac_cluster[j]),
                            "frac_background": float(frac_background[j]),
                            "log2_enrichment": float(log2fc[j]),
                            "neg_log10_p": float(neg_log10_p[j]),
                        }
                    )
                # A cluster is named after what it is *enriched* for.  A
                # depletion is a real result, and it stays in the enrichment
                # table, but "C7 hsat2:-1.97" would read as an identity claim
                # when it means the exact opposite -- so naming only ever
                # considers log2fc > 0.
                qualifies = (k >= MIN_BINS_TO_NAME) & (log2fc > 0.0)
                if qualifies.any():
                    # Sort by enrichment, then feature name, so ties are stable.
                    ordered = sorted(
                        (-float(log2fc[j]), features[idx[j]]) for j in np.flatnonzero(qualifies)
                    )
                    ranked = [(feature, -neg) for neg, feature in ordered]
                    top_feature = ranked[0][0]

        if int(cid) == NOISE_CLUSTER:
            name = NOISE_NAME
            # Never let leftovers be named after a feature -- not in `name`, and
            # not through the `top_features` column either.
            top_feature = ""
            ranked = []
        elif top_feature:
            name = f"C{int(cid)} {top_feature}"
        else:
            name = f"C{int(cid)} unannotated"

        if top_feature:
            fi = features.index(top_feature)
            purity = float(counts[fi]) / annotated_here if annotated_here else 0.0
        else:
            purity = 0.0

        chroms = joined["chrom"].to_numpy(dtype=object)[mask]
        assemblies = joined["assembly"].to_numpy(dtype=object)[mask]
        name_records.append(
            {
                "cluster": int(cid),
                "name": name,
                "top_features": ";".join(
                    f"{f}:{v:.3f}" for f, v in ranked[: max(int(cfg.enrich.top_features), 0)]
                ),
                "size": size,
                "n_assemblies": int(pd.unique(assemblies).size),
                "n_chroms": int(len({c for c in chroms if c})),
                "purity": purity,
            }
        )

    if enrich_records:
        enrichment = (
            pd.DataFrame(enrich_records)
            .sort_values(["cluster", "log2_enrichment", "feature"], ascending=[True, False, True])
            .reset_index(drop=True)
        )
        enrichment = schemas.enforce(enrichment, schemas.ENRICHMENT_COLUMNS)
    else:
        enrichment = schemas.empty_frame(schemas.ENRICHMENT_COLUMNS)

    if name_records:
        names = pd.DataFrame(name_records).sort_values("cluster").reset_index(drop=True)
        names = schemas.enforce(names, schemas.CLUSTER_NAME_COLUMNS)
    else:
        names = schemas.empty_frame(schemas.CLUSTER_NAME_COLUMNS)
    return enrichment, names


def _join(
    rows: pd.DataFrame, clusters: pd.DataFrame, annotations: pd.DataFrame
) -> pd.DataFrame:
    """One frame with cluster label, assembly/chrom identity and feature fractions."""
    frac_cols = [schemas.feature_column(f) for f in schemas.FEATURE_VOCAB]
    left = clusters[["bin_uid", "cluster"]].copy()
    left["bin_uid"] = left["bin_uid"].astype("string")
    left["cluster"] = left["cluster"].astype("int32")

    if rows is not None and len(rows):
        meta = rows[["bin_uid", "assembly", "chrom"]].copy()
        meta["bin_uid"] = meta["bin_uid"].astype("string")
        left = left.merge(meta, on="bin_uid", how="left")
    if "assembly" not in left.columns:
        left["assembly"] = ""
    if "chrom" not in left.columns:
        left["chrom"] = ""
    left["assembly"] = left["assembly"].astype("string").fillna("")
    left["chrom"] = left["chrom"].astype("string").fillna("")

    if annotations is not None and len(annotations):
        ann = annotations.copy()
        ann["bin_uid"] = ann["bin_uid"].astype("string")
        keep = ["bin_uid", "annotated"] + [c for c in frac_cols if c in ann.columns]
        left = left.merge(ann[keep], on="bin_uid", how="left")
    if "annotated" not in left.columns:
        left["annotated"] = False
    left["annotated"] = left["annotated"].fillna(False).astype(bool)
    for col in frac_cols:
        if col not in left.columns:
            left[col] = np.float32(0.0)
        left[col] = left[col].astype("float32").fillna(np.float32(0.0))
    return left


def _write(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_enrichment(outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back the pair written by :func:`enrich_clusters`."""
    outdir = Path(outdir)
    enrich_path = outdir / "enrichment.parquet"
    names_path = outdir / "cluster_names.parquet"
    for path in (enrich_path, names_path):
        if not path.exists():
            raise FileNotFoundError(f"no enrichment output at {path}")
    enrichment = pd.read_parquet(enrich_path)
    names = pd.read_parquet(names_path)
    if enrichment.empty:
        enrichment = schemas.empty_frame(schemas.ENRICHMENT_COLUMNS)
    else:
        enrichment = schemas.enforce(enrichment, schemas.ENRICHMENT_COLUMNS)
    if names.empty:
        names = schemas.empty_frame(schemas.CLUSTER_NAME_COLUMNS)
    else:
        names = schemas.enforce(names, schemas.CLUSTER_NAME_COLUMNS)
    return enrichment, names
