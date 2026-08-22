"""Cluster enrichment and naming.

A cluster only earns a name if the annotation it is enriched for is really
over-represented, so the tests plant a known enrichment and check both that it
is recovered and that the arithmetic in every column is internally consistent.
The negative case matters just as much: cluster ``-1`` is *noise* by
construction, and giving it a satellite name would put a confident label on the
bins the clustering explicitly failed to explain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kmer_dust import schemas
from kmer_dust.enrich import enrich_clusters, load_enrichment


def build_case(rng, n_noise=120):
    """Cluster 0 is 90 % hsat2, cluster 1 is 75 % line, noise is background.

    Returns ``(rows, clusters, annotations)`` with a hand-countable layout.
    """
    specs = []
    labels = []
    features = []
    for i in range(40):  # cluster 0
        specs.append(("C0_pat", "chr21", i))
        labels.append(0)
        features.append("hsat2" if i < 36 else None)
    for i in range(40):  # cluster 1
        specs.append(("C1_pat", "chr22", i))
        labels.append(1)
        features.append("line" if i < 30 else None)
    for i in range(n_noise):  # noise
        specs.append(("N_pat", "chrX", i))
        labels.append(-1)
        features.append("hsat2" if i < 6 else None)

    rows = []
    for idx, (assembly, chrom, i) in enumerate(specs):
        start = i * 10_000
        rows.append(
            {
                "bin_idx": i,
                "bin_uid": schemas.bin_uid(assembly, chrom, start),
                "assembly": assembly,
                "sample": assembly.split("_")[0],
                "haplotype": "pat",
                "source": "local",
                "contig": chrom,
                "chrom": chrom,
                "placed": True,
                "start": start,
                "end": start + 10_000,
                "n_acgt": 10_000,
                "n_kmers": 9_970,
                "n_sketch": 50,
                "gc": 0.4,
                "nfrac": 0.0,
                "row_idx": idx,
            }
        )
    rows = schemas.enforce(pd.DataFrame(rows), schemas.BIN_COLUMNS, subset=True)

    clusters = schemas.enforce(
        pd.DataFrame(
            {
                "row_idx": rows["row_idx"],
                "bin_uid": rows["bin_uid"],
                "cluster": labels,
                "probability": [0.0 if c == -1 else 0.9 for c in labels],
                "outlier_score": 0.1,
            }
        ),
        schemas.CLUSTER_COLUMNS,
    )

    ann = {col: np.zeros(len(rows), dtype=np.float32) for col in schemas.FEATURE_COLUMNS}
    dominant, dom_frac = [], []
    for i, feature in enumerate(features):
        if feature is None:
            dominant.append("unannotated")
            dom_frac.append(0.0)
        else:
            ann[schemas.feature_column(feature)][i] = 0.9
            dominant.append(feature)
            dom_frac.append(0.9)
    annotations = pd.DataFrame(
        {
            "bin_uid": rows["bin_uid"],
            "dominant_feature": dominant,
            "dominant_frac": dom_frac,
            "annotated": True,
            **ann,
        }
    )
    annotations = schemas.enforce(annotations, schemas.ANNOTATION_ID_COLUMNS, subset=True)
    return rows, clusters, annotations


@pytest.fixture
def case(rng):
    return build_case(rng)


def _cfg(make_config, **over):
    enrich = {"min_cluster_size": 10, "min_frac": 0.25, "top_features": 3}
    enrich.update(over)
    return make_config(enrich=enrich)


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


def test_output_contract(case, make_config, run_dir):
    rows, clusters, annotations = case
    enrichment, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    assert list(enrichment.columns) == list(schemas.ENRICHMENT_COLUMNS)
    for col, dtype in schemas.ENRICHMENT_COLUMNS.items():
        assert str(enrichment[col].dtype) == dtype, col
    assert list(names.columns) == list(schemas.CLUSTER_NAME_COLUMNS)
    for col, dtype in schemas.CLUSTER_NAME_COLUMNS.items():
        assert str(names[col].dtype) == dtype, col
    assert (run_dir / "enrichment.parquet").exists()
    assert (run_dir / "cluster_names.parquet").exists()
    e2, n2 = load_enrichment(run_dir)
    pd.testing.assert_frame_equal(e2, enrichment)
    pd.testing.assert_frame_equal(n2, names)


# --------------------------------------------------------------------------
# the planted enrichment
# --------------------------------------------------------------------------


def test_planted_enrichment_is_recovered(case, make_config, run_dir):
    rows, clusters, annotations = case
    enrichment, _ = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    hit = enrichment[(enrichment["cluster"] == 0) & (enrichment["feature"] == "hsat2")]
    assert len(hit) == 1
    hit = hit.iloc[0]
    assert int(hit.n_bins_cluster) == 36
    assert int(hit.cluster_size) == 40
    assert int(hit.n_bins_total) == 42  # 36 in the cluster + 6 in the noise bins
    assert hit.frac_cluster == pytest.approx(0.9)
    assert hit.log2_enrichment > 1.0
    assert hit.neg_log10_p > 5.0


def test_enrichment_columns_are_internally_consistent(case, make_config, run_dir):
    rows, clusters, annotations = case
    enrichment, _ = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    assert len(enrichment) > 0
    np.testing.assert_allclose(
        enrichment["frac_cluster"],
        enrichment["n_bins_cluster"] / enrichment["cluster_size"],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        enrichment["frac_background"],
        enrichment["n_bins_total"] / enrichment["background_size"],
        rtol=1e-9,
    )
    # log2_enrichment is regularised (a pseudocount keeps a 0/x ratio finite),
    # so only its sign and ordering are pinned, not its exact value.
    assert np.isfinite(enrichment["log2_enrichment"]).all()
    richer = enrichment["frac_cluster"] > enrichment["frac_background"]
    poorer = enrichment["frac_cluster"] < enrichment["frac_background"]
    assert (enrichment.loc[richer, "log2_enrichment"] > 0).all()
    assert (enrichment.loc[poorer, "log2_enrichment"] < 0).all()
    assert (enrichment["neg_log10_p"] >= 0).all()
    assert np.isfinite(enrichment["neg_log10_p"]).all()
    assert (enrichment["n_bins_cluster"] <= enrichment["cluster_size"]).all()
    assert (enrichment["n_bins_cluster"] <= enrichment["n_bins_total"]).all()


def test_min_frac_controls_what_counts_as_carrying_a_feature(case, make_config, run_dir):
    """The planted bins have frac 0.9; raise the bar past that and they vanish."""
    rows, clusters, annotations = case
    enrichment, _ = enrich_clusters(
        rows, clusters, annotations, _cfg(make_config, min_frac=0.95), run_dir
    )
    hit = enrichment[(enrichment["cluster"] == 0) & (enrichment["feature"] == "hsat2")]
    assert hit.empty or int(hit.iloc[0].n_bins_cluster) == 0


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------


def test_clusters_are_named_after_their_top_feature(case, make_config, run_dir):
    rows, clusters, annotations = case
    _, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    by_cluster = names.set_index("cluster")
    assert "hsat2" in by_cluster.loc[0, "name"]
    assert "line" in by_cluster.loc[1, "name"]
    assert "0" in by_cluster.loc[0, "name"]
    assert int(by_cluster.loc[0, "size"]) == 40
    assert int(by_cluster.loc[0, "n_assemblies"]) == 1
    assert int(by_cluster.loc[0, "n_chroms"]) == 1
    # 36 of the cluster's 40 annotated bins carry hsat2
    assert by_cluster.loc[0, "purity"] == pytest.approx(36 / 40)


def test_top_features_string_format(case, make_config, run_dir):
    rows, clusters, annotations = case
    _, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    top = names.set_index("cluster").loc[0, "top_features"]
    parts = [p for p in top.split(";") if p]
    assert 1 <= len(parts) <= 3
    assert parts[0].split(":")[0] == "hsat2"
    for part in parts:
        feature, value = part.rsplit(":", 1)
        assert feature in schemas.FEATURE_VOCAB
        float(value)  # must parse


def test_top_features_honours_the_cap(case, make_config, run_dir):
    rows, clusters, annotations = case
    _, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config, top_features=1),
                               run_dir)
    for top in names["top_features"]:
        assert len([p for p in top.split(";") if p]) <= 1


def test_the_noise_cluster_is_listed_but_never_given_an_identity(case, make_config, run_dir):
    """Noise keeps its rows -- what the unclustered material is made of is a
    genuinely useful diagnostic, and `backprop` needs a label for it -- but it
    must never be handed a satellite identity."""
    rows, clusters, annotations = case
    _enrichment, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    noise = names[names["cluster"] == -1]
    assert len(noise) == 1
    assert noise["name"].iloc[0] == "noise"
    assert noise["top_features"].iloc[0] == ""


def test_small_clusters_claim_no_feature(case, make_config, run_dir):
    """Below enrich.min_cluster_size a cluster may still be listed, but it must
    not be handed a satellite identity it has no statistical support for."""
    rows, clusters, annotations = case
    _, names = enrich_clusters(
        rows, clusters, annotations, _cfg(make_config, min_cluster_size=41), run_dir
    )
    assert list(names.columns) == list(schemas.CLUSTER_NAME_COLUMNS)
    for _, row in names.iterrows():
        tokens = set(str(row["name"]).replace(":", " ").replace(";", " ").split())
        assert not (tokens & set(schemas.FEATURE_VOCAB)), row["name"]
        assert row["purity"] == pytest.approx(0.0)


def test_names_are_unique_per_cluster(case, make_config, run_dir):
    rows, clusters, annotations = case
    _, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    assert names["cluster"].is_unique
    assert names["name"].is_unique


# --------------------------------------------------------------------------
# determinism and degenerate inputs
# --------------------------------------------------------------------------


def test_determinism_and_restart(case, make_config, run_dir):
    rows, clusters, annotations = case
    cfg = _cfg(make_config)
    e1, n1 = enrich_clusters(rows, clusters, annotations, cfg, run_dir)
    before = (run_dir / "enrichment.parquet").stat().st_mtime_ns
    e2, n2 = enrich_clusters(rows, clusters, annotations, cfg, run_dir)
    pd.testing.assert_frame_equal(e1, e2)
    pd.testing.assert_frame_equal(n1, n2)
    assert (run_dir / "enrichment.parquet").stat().st_mtime_ns == before
    e3, n3 = enrich_clusters(rows, clusters, annotations, cfg, run_dir, force=True)
    pd.testing.assert_frame_equal(e1, e3)
    pd.testing.assert_frame_equal(n1, n3)


def test_all_noise_input(rng, make_config, run_dir):
    rows, clusters, annotations = build_case(rng)
    clusters = clusters.copy()
    clusters["cluster"] = np.int32(-1)
    clusters = schemas.enforce(clusters, schemas.CLUSTER_COLUMNS)
    enrichment, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    # nothing was clustered, so no real cluster may be invented
    assert set(enrichment["cluster"].tolist()) <= {-1}
    assert set(names["cluster"].tolist()) <= {-1}
    assert list(enrichment.columns) == list(schemas.ENRICHMENT_COLUMNS)


def test_no_annotations_at_all(case, make_config, run_dir):
    rows, clusters, annotations = case
    blank = annotations.copy()
    for col in schemas.FEATURE_COLUMNS:
        blank[col] = np.float32(0.0)
    blank["dominant_feature"] = "unannotated"
    blank["annotated"] = False
    enrichment, names = enrich_clusters(rows, clusters, blank, _cfg(make_config), run_dir)
    # nothing can be enriched, and no cluster may claim a satellite identity
    assert enrichment.empty or (enrichment["n_bins_cluster"] == 0).all()
    for name in names["name"]:
        tokens = set(str(name).replace(":", " ").replace(";", " ").split())
        assert not (tokens & set(schemas.FEATURE_VOCAB))


def test_empty_inputs(make_config, run_dir):
    rows = schemas.enforce(
        schemas.empty_frame(schemas.BIN_COLUMNS).assign(row_idx=pd.Series([], dtype="int64")),
        schemas.BIN_COLUMNS,
        subset=True,
    )
    clusters = schemas.empty_frame(schemas.CLUSTER_COLUMNS)
    annotations = schemas.empty_frame(schemas.ANNOTATION_ID_COLUMNS)
    for col in schemas.FEATURE_COLUMNS:
        annotations[col] = pd.Series([], dtype="float32")
    enrichment, names = enrich_clusters(rows, clusters, annotations, _cfg(make_config), run_dir)
    assert len(enrichment) == 0 and len(names) == 0
    assert list(enrichment.columns) == list(schemas.ENRICHMENT_COLUMNS)
    assert list(names.columns) == list(schemas.CLUSTER_NAME_COLUMNS)


def test_enrichment_is_linear_in_bins_not_bins_times_clusters(make_config, run_dir):
    """The per-cluster mask was O(n_clusters x n_bins x n_features).

    Invisible at 3,021 clusters over 1.3 M bins; fatal at 53,130 over 18.3 M --
    32 trillion element operations, single-threaded, which stalled a real run
    for hours before it was caught. This pins the fix: many small clusters must
    cost about the same as few large ones for the same number of bins.
    """
    import time

    import numpy as np

    n_bins = 60_000
    rng = np.random.default_rng(0)

    def build(n_clusters: int):
        uids = [f"A|c|{i}" for i in range(n_bins)]
        rows = pd.DataFrame(
            {
                "bin_uid": uids,
                "assembly": "A",
                "chrom": "chr21",
                "source": "hprc",
                "haplotype": "pat",
            }
        )
        clusters = pd.DataFrame(
            {
                "bin_uid": uids,
                "cluster": (np.arange(n_bins) % n_clusters).astype(np.int32),
            }
        )
        ann = pd.DataFrame({"bin_uid": uids, "annotated": True})
        for f in schemas.FEATURE_VOCAB:
            ann[schemas.feature_column(f)] = np.float32(0.0)
        ann[schemas.feature_column("hsat3")] = rng.random(n_bins).astype(np.float32)
        ann["dominant_feature"] = "hsat3"
        ann["dominant_frac"] = np.float32(1.0)
        return rows, clusters, ann

    timings = {}
    for n_clusters in (10, 2_000):
        rows, clusters, ann = build(n_clusters)
        cfg = make_config()
        cfg.enrich.min_cluster_size = 1
        start = time.perf_counter()
        enrichment, names = enrich_clusters(rows, clusters, ann, cfg, run_dir / str(n_clusters))
        timings[n_clusters] = time.perf_counter() - start
        assert len(names) == n_clusters

    # 200x the clusters over the same bins must not cost anything like 200x.
    ratio = timings[2_000] / max(timings[10], 1e-6)
    assert ratio < 25, f"cost scales with n_clusters: {timings} (ratio {ratio:.0f}x)"


def test_grouped_counts_match_the_naive_per_cluster_masks(make_config, run_dir):
    """Correctness of the reduceat grouping against the loop it replaced."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 3_000
    uids = [f"A|c|{i}" for i in range(n)]
    labels = rng.integers(-1, 40, size=n).astype(np.int32)
    rows = pd.DataFrame(
        {"bin_uid": uids, "assembly": "A", "chrom": "chr21", "source": "hprc", "haplotype": "pat"}
    )
    clusters = pd.DataFrame({"bin_uid": uids, "cluster": labels})
    ann = pd.DataFrame({"bin_uid": uids, "annotated": True})
    for f in schemas.FEATURE_VOCAB:
        ann[schemas.feature_column(f)] = np.float32(0.0)
    frac = rng.random(n).astype(np.float32)
    ann[schemas.feature_column("bsat")] = frac
    ann["dominant_feature"] = "bsat"
    ann["dominant_frac"] = frac

    cfg = make_config()
    cfg.enrich.min_cluster_size = 1
    cfg.enrich.min_frac = 0.25
    enrichment, _names = enrich_clusters(rows, clusters, ann, cfg, run_dir)

    carries = frac >= 0.25
    got = enrichment[enrichment["feature"] == "bsat"].set_index("cluster")
    for cid in np.unique(labels):
        expected_k = int(carries[labels == cid].sum())
        expected_size = int((labels == cid).sum())
        if cid in got.index:
            assert int(got.loc[cid, "n_bins_cluster"]) == expected_k, cid
            assert int(got.loc[cid, "cluster_size"]) == expected_size, cid
