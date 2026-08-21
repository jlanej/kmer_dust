"""HDBSCAN/DBSCAN stage: label table contract, determinism, all-noise input.

The one thing this stage must never do is drop or reorder rows: ``clusters``
is joined back onto ``rows`` by position and by ``bin_uid`` everywhere
downstream, so a shorter or permuted result silently mislabels the genome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kmer_dust import schemas
from kmer_dust.cluster import cluster, load_clusters


def blobs(n_per=40, k=4, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])[:k]
    pts = np.concatenate([c + rng.normal(scale=spread, size=(n_per, 2)) for c in centres])
    return pts.astype(np.float32), np.repeat(np.arange(k), n_per)


def make_rows(n: int) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "bin_idx": np.arange(n) % 50,
            "bin_uid": [schemas.bin_uid("A_pat", "chr21", i * 10_000) for i in range(n)],
            "assembly": "A_pat",
            "sample": "A",
            "haplotype": "pat",
            "source": "local",
            "contig": "chr21",
            "chrom": "chr21",
            "placed": True,
            "start": np.arange(n) * 10_000,
            "end": np.arange(n) * 10_000 + 10_000,
            "n_acgt": 10_000,
            "n_kmers": 9_970,
            "n_sketch": 50,
            "gc": 0.4,
            "nfrac": 0.0,
        }
    )
    df["row_idx"] = np.arange(n)
    return schemas.enforce(df, schemas.BIN_COLUMNS, subset=True)


@pytest.fixture
def coords_and_rows():
    coords, truth = blobs()
    return coords, make_rows(len(coords)), truth


def test_output_contract(coords_and_rows, make_config, run_dir):
    coords, rows, _ = coords_and_rows
    cfg = make_config(cluster={"min_cluster_size": 10, "min_samples": 5})
    out = cluster(coords, rows, cfg, run_dir)
    assert list(out.columns) == list(schemas.CLUSTER_COLUMNS)
    for col, dtype in schemas.CLUSTER_COLUMNS.items():
        assert str(out[col].dtype) == dtype, col
    assert len(out) == len(rows)
    assert out["row_idx"].tolist() == rows["row_idx"].tolist()
    assert out["bin_uid"].tolist() == rows["bin_uid"].tolist()
    assert (run_dir / "clusters.parquet").exists()
    pd.testing.assert_frame_equal(load_clusters(run_dir), out)


def test_probabilities_and_outlier_scores_are_in_range(coords_and_rows, make_config, run_dir):
    coords, rows, _ = coords_and_rows
    out = cluster(coords, rows, make_config(cluster={"min_cluster_size": 10}), run_dir)
    assert out["probability"].between(0.0, 1.0).all()
    assert np.isfinite(out["outlier_score"]).all()
    assert (out["cluster"] >= -1).all()
    noise = out[out["cluster"] == -1]
    if len(noise):
        assert (noise["probability"] == 0.0).all()


def test_well_separated_blobs_are_recovered(coords_and_rows, make_config, run_dir):
    coords, rows, truth = coords_and_rows
    cfg = make_config(cluster={"min_cluster_size": 10, "min_samples": 5})
    out = cluster(coords, rows, cfg, run_dir)
    labels = out["cluster"].to_numpy()
    assert (labels == -1).mean() < 0.1
    # the partition must agree with the planted one up to relabelling
    table = pd.crosstab(truth, labels)
    assert table.shape[0] == 4
    assert (table.max(axis=1) / table.sum(axis=1) > 0.9).all()


def test_cluster_labels_are_dense_and_start_at_zero(coords_and_rows, make_config, run_dir):
    coords, rows, _ = coords_and_rows
    out = cluster(coords, rows, make_config(cluster={"min_cluster_size": 10}), run_dir)
    labels = sorted(set(out["cluster"].tolist()) - {-1})
    assert labels == list(range(len(labels)))


def test_determinism(coords_and_rows, make_config, tmp_path):
    coords, rows, _ = coords_and_rows
    out = []
    for i in range(2):
        cfg = make_config(seed=31337, cluster={"min_cluster_size": 10})
        d = tmp_path / f"c{i}"
        d.mkdir()
        out.append(cluster(coords, rows, cfg, d))
    pd.testing.assert_frame_equal(out[0], out[1])


def test_rerun_short_circuits(coords_and_rows, make_config, run_dir):
    coords, rows, _ = coords_and_rows
    cfg = make_config(cluster={"min_cluster_size": 10})
    cluster(coords, rows, cfg, run_dir)
    before = (run_dir / "clusters.parquet").stat().st_mtime_ns
    cluster(coords, rows, cfg, run_dir)
    assert (run_dir / "clusters.parquet").stat().st_mtime_ns == before
    cluster(coords, rows, cfg, run_dir, force=True)
    assert (run_dir / "clusters.parquet").stat().st_mtime_ns != before


def test_dbscan_backend(coords_and_rows, make_config, run_dir):
    coords, rows, truth = coords_and_rows
    cfg = make_config(cluster={"method": "dbscan", "eps": 1.0, "min_samples": 5})
    out = cluster(coords, rows, cfg, run_dir)
    assert len(out) == len(rows)
    assert out["cluster"].nunique() >= 4
    assert list(out.columns) == list(schemas.CLUSTER_COLUMNS)


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------


def test_all_noise_is_labelled_not_crashed(make_config, run_dir):
    coords = np.random.default_rng(1).uniform(size=(60, 2)).astype(np.float32)
    rows = make_rows(60)
    cfg = make_config(cluster={"min_cluster_size": 55, "min_samples": 50})
    out = cluster(coords, rows, cfg, run_dir)
    assert len(out) == 60
    assert set(out["cluster"].tolist()) == {-1}
    assert (out["probability"] == 0.0).all()


def test_fewer_rows_than_min_cluster_size(make_config, run_dir):
    coords = np.random.default_rng(2).uniform(size=(3, 2)).astype(np.float32)
    out = cluster(coords, make_rows(3), make_config(cluster={"min_cluster_size": 50}), run_dir)
    assert len(out) == 3
    assert set(out["cluster"].tolist()) == {-1}


def test_single_row(make_config, run_dir):
    coords = np.zeros((1, 2), dtype=np.float32)
    out = cluster(coords, make_rows(1), make_config(cluster={"min_cluster_size": 5}), run_dir)
    assert len(out) == 1
    assert out.iloc[0]["cluster"] == -1


def test_empty_input(make_config, run_dir):
    coords = np.zeros((0, 2), dtype=np.float32)
    rows = make_rows(0)
    out = cluster(coords, rows, make_config(), run_dir)
    assert len(out) == 0
    assert list(out.columns) == list(schemas.CLUSTER_COLUMNS)
    for col, dtype in schemas.CLUSTER_COLUMNS.items():
        assert str(out[col].dtype) == dtype, col


def test_all_identical_points(make_config, run_dir):
    """Duplicate satellite bins collapse to one point; HDBSCAN can divide by 0."""
    coords = np.zeros((40, 2), dtype=np.float32)
    out = cluster(coords, make_rows(40), make_config(cluster={"min_cluster_size": 5}), run_dir)
    assert len(out) == 40
    assert np.isfinite(out["outlier_score"]).all()


def test_an_upstream_epsilon_crash_becomes_an_actionable_error(monkeypatch, make_config, run_dir):
    """scikit-learn's Cython epsilon search raises a bare, undiagnosable TypeError.

    `sklearn/cluster/_hdbscan/_tree.pyx::traverse_upwards` raises
    ``TypeError: only 0-dimensional arrays can be converted to Python scalars``
    when an epsilon search reaches the root of the condensed tree.  It fires
    only for ``cluster_selection_epsilon > 0`` and is size-dependent -- the same
    settings can succeed on a subsample and fail on the full matrix -- so the
    bare TypeError sends people looking for a bug in their own data.  We turn it
    into a message that names the cause and the fix.
    """
    import kmer_dust.cluster as kd_cluster
    from kmer_dust import cluster as cluster_mod

    class _Exploding:
        def __init__(self, **kwargs):
            pass

        def fit(self, x):
            raise TypeError("only 0-dimensional arrays can be converted to Python scalars")

    # Patch the backend selector, not sklearn: the stage prefers fast_hdbscan
    # when it is installed, so patching sklearn directly would be a no-op.
    monkeypatch.setattr(kd_cluster, "_hdbscan_backend", lambda: (_Exploding, "sklearn"))

    cfg = make_config()
    cfg.cluster.method = "hdbscan"
    cfg.cluster.cluster_selection_epsilon = 0.5
    coords = np.zeros((40, 2), dtype=np.float32)
    coords[20:] = 5.0
    rows = pd.DataFrame(
        {
            "row_idx": np.arange(40, dtype=np.int64),
            "bin_uid": [f"A|c|{i * 10000}" for i in range(40)],
        }
    )

    with pytest.raises(RuntimeError) as excinfo:
        cluster_mod.cluster(coords, rows, cfg, run_dir, force=True)
    message = str(excinfo.value)
    assert "cluster_selection_epsilon" in message
    assert "upstream" in message
    assert "min_cluster_size" in message  # the message must say what to do instead


def test_an_unrelated_typeerror_is_not_reinterpreted(monkeypatch, make_config, run_dir):
    import kmer_dust.cluster as kd_cluster
    from kmer_dust import cluster as cluster_mod

    class _Exploding:
        def __init__(self, **kwargs):
            pass

        def fit(self, x):
            raise TypeError("something else entirely")

    # Patch the backend selector, not sklearn: the stage prefers fast_hdbscan
    # when it is installed, so patching sklearn directly would be a no-op.
    monkeypatch.setattr(kd_cluster, "_hdbscan_backend", lambda: (_Exploding, "sklearn"))
    cfg = make_config()
    cfg.cluster.method = "hdbscan"
    cfg.cluster.cluster_selection_epsilon = 0.5
    rows = pd.DataFrame(
        {"row_idx": np.arange(8, dtype=np.int64), "bin_uid": [f"A|c|{i}" for i in range(8)]}
    )
    with pytest.raises(TypeError, match="something else entirely"):
        cluster_mod.cluster(np.zeros((8, 2), dtype=np.float32), rows, cfg, run_dir, force=True)


def test_max_fit_rows_is_off_by_default():
    """It changes the answer, so it has to be asked for.

    Measured against exact HDBSCAN on 200k rows of a real embedding (832
    clusters, 30.6 % noise), fitting on 100k gives 465 clusters and 14.3 %
    noise -- ARI 0.171. Scaling min_cluster_size by the sampling fraction does
    not rescue it (ARI 0.151, and 0.040 at 50k). A default that halves your
    cluster count is not an optimisation.
    """
    from kmer_dust.config import Config

    assert Config().cluster.max_fit_rows == 0


def test_subsample_fit_selection_is_seeded_and_sorted():
    import numpy as np

    from kmer_dust.cluster import _fit_subsample

    x = np.zeros((1000, 2))
    assert _fit_subsample(x, 0, 7) is None, "0 means fit on everything"
    assert _fit_subsample(x, 5000, 7) is None, "no subsample when it would be a no-op"
    a = _fit_subsample(x, 100, 7)
    b = _fit_subsample(x, 100, 7)
    c = _fit_subsample(x, 100, 8)
    assert a is not None and len(a) == 100
    assert np.array_equal(a, np.sort(a)), "indices must be sorted for stable slicing"
    assert np.array_equal(a, b), "same seed must give the same subsample"
    assert not np.array_equal(a, c), "a different seed must give a different subsample"


def test_propagated_labels_keep_the_fitted_rows_exact():
    """A fitted row is its own nearest neighbour, so its label must survive."""
    import numpy as np

    from kmer_dust.cluster import _propagate_labels

    rng = np.random.default_rng(3)
    blob_a = rng.normal(0, 0.1, size=(200, 2))
    blob_b = rng.normal(5, 0.1, size=(200, 2))
    x = np.vstack([blob_a, blob_b])
    fit_idx = np.sort(rng.choice(x.shape[0], 100, replace=False))
    fit_labels = np.where(fit_idx < 200, 0, 1).astype(np.int32)
    fit_prob = np.ones(fit_idx.size, dtype=np.float32)

    labels, prob, outlier = _propagate_labels(x, fit_idx, fit_labels, fit_prob)

    assert np.array_equal(labels[fit_idx], fit_labels)
    assert labels.shape == (x.shape[0],)
    # every point should land in the blob it belongs to
    assert (labels[:200] == 0).mean() > 0.95
    assert (labels[200:] == 1).mean() > 0.95
    assert prob.dtype == np.float32 and outlier.dtype == np.float32
    assert np.all((outlier >= 0) & (outlier <= 1))


def test_propagation_with_no_labelled_anchors_is_all_noise():
    import numpy as np

    from kmer_dust.cluster import _propagate_labels

    x = np.random.default_rng(0).normal(size=(50, 2))
    fit_idx = np.arange(10)
    labels, prob, outlier = _propagate_labels(
        x, fit_idx, np.full(10, -1, dtype=np.int32), np.zeros(10, dtype=np.float32)
    )
    assert (labels == -1).all()
    assert (prob == 0).all()


def test_the_fast_backend_is_preferred_when_installed():
    """scikit-learn's HDBSCAN has no Boruvka MST, so it is O(n^2) even in 2-D.

    Measured on a real 1.3M-bin embedding: sklearn 4819 s, fast_hdbscan 4.7 s --
    ~1000x -- for the same clustering (ARI 0.832 / AMI 0.931 at 200k, cluster
    counts within 1 %). The preference must not silently regress.
    """
    from kmer_dust.cluster import _hdbscan_backend

    klass, name = _hdbscan_backend()
    pytest.importorskip("fast_hdbscan")
    assert name == "fast_hdbscan"
    assert klass.__module__.startswith("fast_hdbscan")


def test_the_backend_falls_back_to_sklearn(monkeypatch):
    """A bare install must still work, just slowly."""
    import builtins

    import kmer_dust.cluster as kd_cluster

    real_import = builtins.__import__

    def _no_fast(name, *args, **kwargs):
        if name == "fast_hdbscan":
            raise ImportError("simulated: not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_fast)
    klass, name = kd_cluster._hdbscan_backend()
    assert name == "sklearn"
    assert klass.__module__.startswith("sklearn")


def test_backend_kwargs_are_filtered_to_what_it_accepts(make_config, run_dir):
    """The two backends take different arguments: fast_hdbscan has no n_jobs."""
    import numpy as np

    from kmer_dust.cluster import cluster as cluster_stage

    rng = np.random.default_rng(0)
    coords = np.vstack(
        [rng.normal(0, 0.2, size=(120, 2)), rng.normal(6, 0.2, size=(120, 2))]
    ).astype(np.float32)
    rows = make_rows(len(coords))
    cfg = make_config()
    cfg.cluster.min_cluster_size = 20
    cfg.cluster.min_samples = 5
    out = cluster_stage(coords, rows, cfg, run_dir, force=True)
    assert len(out) == len(coords)
    assert out["cluster"].nunique() >= 2, "two well-separated blobs must not merge"
