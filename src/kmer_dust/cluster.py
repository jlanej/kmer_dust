"""Density-based clustering of the embedded (or component-space) bins.

The clusters are the product of this pipeline: each one is a set of 10 kb bins,
drawn from many haplotypes, that share a k-mer vocabulary.  Two deliberate
choices shape this module.

*HDBSCAN.*  Density-based, so it
does not have to be told how many clusters exist -- which matters because nobody
knows how many distinct satellite/repeat vocabularies a pangenome contains --
and it is allowed to say "noise" for the vast euchromatic middle instead of
forcing every bin into some cluster.  ``fast_hdbscan`` is used when it is
installed and ``sklearn.cluster.HDBSCAN`` is the fallback -- see
:func:`_hdbscan_backend` for why that order matters more than it sounds
(scikit-learn has no Boruvka MST, so it is quadratic in n even on a 2-D
embedding).  DBSCAN remains available for the case where the user genuinely
wants one global density threshold.

*Noise is reported, not hidden.*  A run where 95% of bins are noise is not a
result, it is a tuning failure (usually ``min_cluster_size`` far too large or an
embedding that never separated), and the user has to see it while there is still
time to change the config.  So the cluster-size histogram and the noise fraction
are logged at every run, and an implausible noise fraction is a warning.  A run
where *everything* is noise is still a valid, if sad, result: it produces a
well-formed table of ``-1`` labels rather than an exception.

``probability`` and ``outlier_score`` are HDBSCAN concepts.  DBSCAN has no soft
membership at all, so we synthesise them from what it does report: 1.0 for core
points, 0.5 for border points (assigned to a cluster but not dense themselves),
0.0 for noise, and an outlier score of 0.0 throughout -- a constant that is
honest about carrying no information rather than an invented score.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schemas
from .config import Config
from .log import get_logger, timed

logger = get_logger(__name__)

__all__ = ["CLUSTERS_FILE", "PARAMS_FILE", "cluster", "load_clusters", "load_cluster_params"]

CLUSTERS_FILE = "clusters.parquet"
PARAMS_FILE = "cluster_params.json"

#: Above this fraction of noise the clustering is almost certainly mistuned.
_NOISE_WARN = 0.5
#: How many cluster sizes to spell out in the log before summarising the tail.
_HISTOGRAM_HEAD = 20


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


def _save_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _resolve_coords(coords: Any, cfg: Config) -> np.ndarray:
    """Pick the array to cluster, honouring ``cfg.cluster.space``.

    ``pipeline.py`` normally passes the chosen array directly, but accepting a
    ``{"embedding": ..., "pcs": ...}`` mapping too means the caller cannot pick
    the wrong space by accident.
    """
    if isinstance(coords, Mapping):
        space = str(cfg.cluster.space)
        if space in coords:
            chosen = coords[space]
        else:  # the requested space was not computed; fall back to what exists
            available = [k for k in ("embedding", "pcs") if k in coords]
            if not available:
                raise ValueError(f"no coordinates provided for cluster.space={space!r}")
            logger.warning("cluster: space=%r not available, using %r instead", space, available[0])
            chosen = coords[available[0]]
    else:
        chosen = coords
    arr = np.asarray(chosen)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"cluster expects a 2-D coordinate array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _identity_frame(rows: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """The ``row_idx`` / ``bin_uid`` half of ``CLUSTER_COLUMNS``, in row order."""
    if len(rows) != n_rows:
        raise ValueError(
            f"rows has {len(rows)} entries but the coordinates have {n_rows}; "
            "the row table and the embedding must be in the same order"
        )
    if "bin_uid" not in rows.columns:
        raise ValueError("rows is missing 'bin_uid'; it must be schemas.BIN_COLUMNS + row_idx")
    if "row_idx" in rows.columns:
        row_idx = np.asarray(rows["row_idx"], dtype=np.int64)
    else:
        # A rows table straight out of a single shard has no global index yet.
        logger.warning("cluster: rows has no 'row_idx' column; numbering 0..%d", n_rows - 1)
        row_idx = np.arange(n_rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "row_idx": row_idx,
            "bin_uid": pd.Series(np.asarray(rows["bin_uid"]), dtype="string"),
        }
    )


def _assemble(
    ident: pd.DataFrame,
    labels: np.ndarray,
    probability: np.ndarray,
    outlier: np.ndarray,
) -> pd.DataFrame:
    out = ident.copy()
    out["cluster"] = np.asarray(labels, dtype=np.int32)
    out["probability"] = np.asarray(probability, dtype=np.float32)
    out["outlier_score"] = np.asarray(outlier, dtype=np.float32)
    return schemas.enforce(out, schemas.CLUSTER_COLUMNS)


def _log_histogram(labels: np.ndarray, min_cluster_size: int) -> tuple[int, float]:
    """Log the cluster-size histogram; return ``(n_clusters, noise_fraction)``."""
    n_rows = int(labels.size)
    if n_rows == 0:
        logger.warning("cluster: no rows to cluster")
        return 0, 0.0
    noise = int((labels < 0).sum())
    noise_frac = noise / n_rows
    assigned = labels[labels >= 0]
    ids, counts = np.unique(assigned, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    ids, counts = ids[order], counts[order]
    head = ", ".join(
        f"c{int(i)}:{int(c)}" for i, c in zip(ids[:_HISTOGRAM_HEAD], counts[:_HISTOGRAM_HEAD])
    )
    tail = f", (+{ids.size - _HISTOGRAM_HEAD} more)" if ids.size > _HISTOGRAM_HEAD else ""
    logger.info(
        "cluster: %d cluster(s) over %d rows; noise %d (%.1f%%); median size %d",
        int(ids.size),
        n_rows,
        noise,
        100.0 * noise_frac,
        int(np.median(counts)) if counts.size else 0,
    )
    if ids.size:
        logger.info("cluster: sizes %s%s", head, tail)
    if ids.size == 0:
        logger.warning(
            "cluster: every bin is noise -- valid, but almost certainly means "
            "min_cluster_size/min_samples are too large or the embedding did not separate"
        )
    elif noise_frac >= _NOISE_WARN:
        logger.warning(
            "cluster: %.1f%% of bins are noise; check min_cluster_size=%d and the embedding",
            100.0 * noise_frac,
            min_cluster_size,
        )
    return int(ids.size), float(noise_frac)




def _hdbscan_backend() -> tuple[Any, str]:
    """The HDBSCAN implementation to use, preferring the one that scales.

    scikit-learn's HDBSCAN has only two MST paths -- brute force and Prim's --
    and no Boruvka, so it is quadratic in n *regardless of dimensionality*.
    That is a property of the implementation, not of the algorithm, and on a
    2-D embedding it dominates everything else the pipeline does.  Measured on
    a real 1.3M-bin embedding:

        n =    50,000   sklearn    7.8 s   fast_hdbscan  0.1 s     71x
        n =   200,000   sklearn  126.0 s   fast_hdbscan  0.5 s    251x
        n = 1,303,159   sklearn 4819.4 s   fast_hdbscan  4.7 s   1025x

    and they agree on the answer where both are affordable: ARI 0.820 / AMI
    0.929 at 50k, ARI 0.832 / AMI 0.931 at 200k, with cluster counts within 1 %
    (809 vs 814 at 200k).  `fast_hdbscan` is by the same authors as the original
    hdbscan package, is pure numba with no compiler needed, and implements
    Boruvka over a KD-tree.

    It stays optional -- `pip install kmer-dust[fast]` -- so the package keeps
    working on a bare scikit-learn install, just slowly.
    """
    try:
        from fast_hdbscan import HDBSCAN as FastHDBSCAN
    except ImportError:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        logger.warning(
            "cluster: using scikit-learn's HDBSCAN, which is O(n^2) even on a 2-D "
            "embedding (1.3M rows took 80 minutes). `pip install fast_hdbscan` for the "
            "same clustering ~1000x faster."
        )
        return SklearnHDBSCAN, "sklearn"
    return FastHDBSCAN, "fast_hdbscan"


def _fit_subsample(x: np.ndarray, max_fit_rows: int, seed: int) -> np.ndarray | None:
    """Row indices to fit on, or ``None`` to fit on everything."""
    n_rows = int(x.shape[0])
    if max_fit_rows <= 0 or n_rows <= max_fit_rows:
        return None
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_rows, size=int(max_fit_rows), replace=False))


def _propagate_labels(
    x: np.ndarray,
    fit_idx: np.ndarray,
    fit_labels: np.ndarray,
    fit_prob: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give every row the label of its nearest fitted, non-noise neighbour.

    The fitted rows keep their own labels exactly -- they are their own nearest
    neighbour at distance zero -- so this only decides the rows that were left
    out of the fit.

    A row too far from any labelled point stays noise.  The cut-off is derived
    from the fit itself: the 99th percentile of the distance from each labelled
    fitted point to its nearest labelled neighbour, doubled.  Deriving it from
    the subsample is conservative in the right direction, because a subsample is
    sparser than the full set, so its neighbour distances are larger than the
    ones the full data would produce.
    """
    from sklearn.neighbors import NearestNeighbors

    labelled = fit_labels >= 0
    n_rows = int(x.shape[0])
    if not labelled.any():
        return (
            np.full(n_rows, -1, dtype=np.int32),
            np.zeros(n_rows, dtype=np.float32),
            np.zeros(n_rows, dtype=np.float32),
        )

    anchors = x[fit_idx][labelled]
    anchor_labels = fit_labels[labelled]
    anchor_prob = fit_prob[labelled]

    nn = NearestNeighbors(n_neighbors=1).fit(anchors)
    if anchors.shape[0] > 1:
        self_dist, _ = NearestNeighbors(n_neighbors=2).fit(anchors).kneighbors(anchors)
        cutoff = float(np.quantile(self_dist[:, 1], 0.99)) * 2.0
    else:
        cutoff = float("inf")
    if not np.isfinite(cutoff) or cutoff <= 0:
        cutoff = float("inf")

    dist, idx = nn.kneighbors(x)
    dist = dist[:, 0]
    idx = idx[:, 0]

    labels = anchor_labels[idx].astype(np.int32)
    probability = anchor_prob[idx].astype(np.float32)
    too_far = dist > cutoff
    labels[too_far] = -1
    probability[too_far] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        outlier = np.clip(dist / cutoff, 0.0, 1.0) if np.isfinite(cutoff) else np.zeros_like(dist)
    logger.info(
        "cluster: propagated %d fitted label(s) to %d row(s); %d left as noise beyond %.3g",
        int(labelled.sum()),
        n_rows,
        int(too_far.sum()),
        cutoff,
    )
    return labels, probability, np.asarray(outlier, dtype=np.float32)


def cluster(
    coords: np.ndarray,
    rows: pd.DataFrame,
    cfg: Config,
    outdir: Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Cluster ``coords`` and write ``clusters.parquet`` (``schemas.CLUSTER_COLUMNS``).

    ``rows`` supplies ``row_idx`` and ``bin_uid`` and must be in the same order
    as ``coords``.  The returned frame has exactly one row per coordinate row,
    in that same order, with ``cluster == -1`` for noise.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / CLUSTERS_FILE

    if not force and out_path.exists():
        existing = load_clusters(outdir)
        logger.info("cluster: reusing %s (%d rows)", out_path, len(existing))
        return existing

    ccfg = cfg.cluster
    x = _resolve_coords(coords, cfg)
    n_rows = int(x.shape[0])
    ident = _identity_frame(rows, n_rows)

    params: dict[str, Any] = {
        "method": str(ccfg.method),
        "space": str(ccfg.space),
        "n_rows": n_rows,
        "n_dims": int(x.shape[1]),
        "min_cluster_size_requested": int(ccfg.min_cluster_size),
        "min_samples_requested": int(ccfg.min_samples),
        "seed": int(ccfg.seed),
    }

    if n_rows < 2:
        # HDBSCAN refuses a single sample outright, and a lone bin cannot form a
        # cluster under any setting, so short-circuit to an all-noise result.
        if n_rows:
            logger.warning("cluster: only %d row(s); labelling everything noise", n_rows)
        result = _assemble(
            ident,
            np.full(n_rows, -1, dtype=np.int32),
            np.zeros(n_rows, dtype=np.float32),
            np.zeros(n_rows, dtype=np.float32),
        )
        params.update(
            {"n_clusters": 0, "noise_fraction": 1.0 if n_rows else 0.0, "degenerate": True}
        )
        _save_parquet(out_path, result)
        _save_json(outdir / PARAMS_FILE, params)
        logger.info("cluster: wrote %s (%d rows)", out_path, len(result))
        return result

    if not np.isfinite(x).all():
        raise ValueError("coordinates contain non-finite values; refusing to cluster them")

    # Both estimators need their size parameters <= n_rows or they simply never
    # find a cluster; clamp rather than let a small test run silently return all
    # noise for a reason the user cannot see.
    min_cluster_size = int(min(max(int(ccfg.min_cluster_size), 2), n_rows))
    min_samples = int(min(max(int(ccfg.min_samples), 1), n_rows))
    if min_cluster_size != int(ccfg.min_cluster_size):
        logger.warning(
            "cluster: min_cluster_size clamped %d -> %d for %d rows",
            int(ccfg.min_cluster_size),
            min_cluster_size,
            n_rows,
        )
    if min_samples != int(ccfg.min_samples):
        logger.warning(
            "cluster: min_samples clamped %d -> %d for %d rows",
            int(ccfg.min_samples),
            min_samples,
            n_rows,
        )
    params["min_cluster_size"] = min_cluster_size
    params["min_samples"] = min_samples
    n_jobs = max(int(getattr(cfg, "threads", 1) or 1), 1)

    method = str(ccfg.method).lower()
    if method == "dbscan":
        labels, probability, outlier = _run_dbscan(x, ccfg, min_samples, n_jobs, params)
    else:
        if method != "hdbscan":
            logger.warning("cluster: unknown method %r; using hdbscan", ccfg.method)
            params["method"] = "hdbscan"
        labels, probability, outlier = _run_hdbscan(
            x, ccfg, min_cluster_size, min_samples, n_jobs, params
        )

    n_clusters, noise_frac = _log_histogram(labels, min_cluster_size)
    params["n_clusters"] = n_clusters
    params["noise_fraction"] = noise_frac
    result = _assemble(ident, labels, probability, outlier)
    _save_parquet(out_path, result)
    _save_json(outdir / PARAMS_FILE, params)
    logger.info("cluster: wrote %s (%d rows)", out_path, len(result))
    return result


def _run_hdbscan(
    x: np.ndarray,
    ccfg: Any,
    min_cluster_size: int,
    min_samples: int,
    n_jobs: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    HDBSCAN, backend = _hdbscan_backend()
    params["backend"] = backend

    selection = str(ccfg.cluster_selection_method).lower()
    if selection not in {"eom", "leaf"}:
        logger.warning("cluster: unknown cluster_selection_method %r; using 'eom'", selection)
        selection = "eom"
    kwargs: dict[str, Any] = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_epsilon": float(ccfg.cluster_selection_epsilon),
        "cluster_selection_method": selection,
        "n_jobs": n_jobs,
    }
    # `copy` appeared after 1.4 and defaults to a FutureWarning; ask for it only
    # when it exists so we neither warn nor overwrite the caller's array.
    # Keep this before the kwargs are filtered: the diagnostic below needs it
    # even when the chosen backend does not accept the argument.
    epsilon = float(ccfg.cluster_selection_epsilon)
    accepted = set(inspect.signature(HDBSCAN.__init__).parameters)
    if "copy" in accepted:
        kwargs["copy"] = True
    # The two backends do not take the same arguments -- fast_hdbscan has no
    # n_jobs (it is numba-parallel throughout) and no copy. Drop anything the
    # chosen one will not accept rather than special-casing by name.
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        logger.debug("cluster: %s backend ignores %s", backend, ", ".join(dropped))
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    params["cluster_selection_method"] = selection
    params["cluster_selection_epsilon"] = float(ccfg.cluster_selection_epsilon)

    # sklearn's HDBSCAN is quadratic in n on 2-D input (measured: 2.0s/25k,
    # 8.0s/50k, 31.3s/100k, 124.5s/200k). Past a few hundred thousand rows the
    # only sane option is to fit a bounded subsample and propagate.
    fit_idx = _fit_subsample(x, int(getattr(ccfg, "max_fit_rows", 0) or 0), int(ccfg.seed))
    x_fit = x if fit_idx is None else np.ascontiguousarray(x[fit_idx])
    params["max_fit_rows"] = int(getattr(ccfg, "max_fit_rows", 0) or 0)
    params["fitted_rows"] = int(x_fit.shape[0])
    if fit_idx is not None:
        logger.warning(
            "cluster: fitting on %d of %d rows (cluster.max_fit_rows). This is NOT an "
            "approximation of the exact labelling -- measured against exact HDBSCAN on "
            "200k rows it gives roughly half the clusters and half the noise (ARI 0.17, "
            "AMI 0.74). Use it as a deliberately coarser view, and prefer "
            "cluster.method=dbscan if you need something that scales honestly.",
            x_fit.shape[0],
            x.shape[0],
        )

    with timed(
        logger,
        f"HDBSCAN on {x_fit.shape[0]}x{x_fit.shape[1]} "
        f"(min_cluster_size={min_cluster_size} min_samples={min_samples} selection={selection})",
    ):
        try:
            model = HDBSCAN(**kwargs).fit(x_fit)
        except TypeError as exc:
            # scikit-learn's Cython condensed-tree walk (`traverse_upwards` in
            # sklearn/cluster/_hdbscan/_tree.pyx) raises
            #   TypeError: only 0-dimensional arrays can be converted to Python scalars
            # when an epsilon search reaches the root of the tree.  It only ever
            # fires with cluster_selection_epsilon > 0, and it is size- and
            # data-dependent -- the same settings can work on a subsample and
            # fail on the full matrix, which makes it maddening to diagnose from
            # the bare TypeError.  Re-raise with the actual diagnosis attached
            # rather than silently changing the parameter: the embedding is
            # already on disk, so re-running just `cluster` costs seconds.
            if epsilon > 0 and "0-dimensional" in str(exc):
                raise RuntimeError(
                    "HDBSCAN failed inside scikit-learn's epsilon search "
                    f"(cluster_selection_epsilon={epsilon}). "
                    "This is an upstream bug in sklearn's condensed-tree traversal, not a "
                    "problem with your data, and it only triggers when the epsilon is "
                    "non-zero. Set cluster.cluster_selection_epsilon to 0.0 and control "
                    "granularity with cluster.min_cluster_size instead, or switch to "
                    "cluster.method: dbscan, which takes an eps directly. Re-running only "
                    "the `cluster` stage is cheap -- the embedding is already on disk."
                ) from exc
            raise

    labels = np.asarray(model.labels_, dtype=np.int32)
    probability = np.asarray(
        getattr(model, "probabilities_", np.ones(labels.size)), dtype=np.float32
    )
    scores = getattr(model, "outlier_scores_", None)
    if fit_idx is not None:
        fit_scores = (
            np.asarray(scores, dtype=np.float32)
            if scores is not None
            else np.zeros(labels.size, dtype=np.float32)
        )
        labels, probability, propagated_scores = _propagate_labels(x, fit_idx, labels, probability)
        # The fitted rows are their own nearest neighbour, so restore their real
        # outlier scores rather than the propagated distance proxy.
        propagated_scores[fit_idx] = fit_scores
        scores = propagated_scores
    if scores is None:
        # scikit-learn's HDBSCAN does not expose GLOSH outlier scores; the
        # complement of the membership probability is the closest honest stand-in
        # and keeps the column meaningful ("how marginal is this bin").
        outlier = (1.0 - probability).astype(np.float32)
        params["outlier_score_source"] = "1 - probability"
    else:
        outlier = np.asarray(scores, dtype=np.float32)
        params["outlier_score_source"] = "hdbscan.outlier_scores_"
    return labels, probability, outlier


def _run_dbscan(
    x: np.ndarray,
    ccfg: Any,
    min_samples: int,
    n_jobs: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import DBSCAN

    params["eps"] = float(ccfg.eps)
    with timed(
        logger,
        f"DBSCAN on {x.shape[0]}x{x.shape[1]} (eps={ccfg.eps} min_samples={min_samples})",
    ):
        model = DBSCAN(eps=float(ccfg.eps), min_samples=min_samples, n_jobs=n_jobs).fit(x)

    labels = np.asarray(model.labels_, dtype=np.int32)
    probability = np.zeros(labels.size, dtype=np.float32)
    probability[labels >= 0] = 0.5  # border: in a cluster, but not dense itself
    core = np.asarray(getattr(model, "core_sample_indices_", np.empty(0, dtype=np.int64)))
    if core.size:
        probability[core] = 1.0
    outlier = np.zeros(labels.size, dtype=np.float32)
    params["probability_source"] = "1.0 core / 0.5 border / 0.0 noise"
    params["outlier_score_source"] = "constant 0.0 (DBSCAN has none)"
    return labels, probability, outlier


def load_clusters(outdir: Path) -> pd.DataFrame:
    """Read ``clusters.parquet`` written by :func:`cluster`."""
    path = Path(outdir) / CLUSTERS_FILE
    if not path.exists():
        raise FileNotFoundError(f"no clusters at {path}; run the cluster stage first")
    return schemas.enforce(pd.read_parquet(path), schemas.CLUSTER_COLUMNS)


def load_cluster_params(outdir: Path) -> dict[str, Any]:
    """Read ``cluster_params.json``; an empty dict when the stage has not run."""
    path = Path(outdir) / PARAMS_FILE
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)
