"""Offline end-to-end run over a synthetic six-haplotype pangenome.

This is the test CI runs on every commit, so it must never touch the network
and must finish in seconds.  It builds six 300 kb "assemblies" from one mutated
backbone plus two satellite regions tiled from a shared pool of monomer
variants (see ``conftest.build_synthetic_assemblies``), runs
``pipeline.run_all``, and then asks the only question that matters: did the
satellite bins -- which are scattered across two regions and six assemblies and
share nothing with their neighbours except a k-mer vocabulary -- end up in one
cluster?

The signal was verified independently before this test was written: with these
parameters the 42 planted satellite bins have pairwise Jaccard ~0.50 to each
other and 0.00 to backbone bins, while backbone bins match only the same
position in other haplotypes (~0.72).  So a failure here is a pipeline bug, not
a weak fixture.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import SYNTH_CONTIG, SYNTH_CONTIG_LEN, satellite_bin_indices
from scipy import sparse

pytestmark = pytest.mark.slow

N_ASSEMBLIES = 6
BIN_SIZE = 10_000
N_BINS_PER_ASSEMBLY = SYNTH_CONTIG_LEN // BIN_SIZE  # 30


@pytest.fixture(scope="module")
def run(smoke_run):
    cfg, manifest, _ = smoke_run
    rows = pd.read_parquet(cfg.path("matrix", "rows.parquet"))
    clusters = pd.read_parquet(cfg.path("cluster", "clusters.parquet"))
    return cfg, manifest, rows, clusters


def satellite_mask(rows: pd.DataFrame) -> np.ndarray:
    planted = satellite_bin_indices(BIN_SIZE)
    return np.array([int(s) // BIN_SIZE in planted for s in rows["start"]])


# --------------------------------------------------------------------------
# every stage produced its documented output
# --------------------------------------------------------------------------


def test_run_directory_layout(run):
    cfg, manifest, _, _ = run
    assert cfg.path("manifest.tsv").exists()
    assert cfg.path("config.resolved.yaml").exists()
    for assembly in manifest["assembly"]:
        assert cfg.path("sketch", f"{assembly}.done").exists()
        assert cfg.path("sketch", f"{assembly}.bins.parquet").exists()
        assert cfg.path("sketch", f"{assembly}.sketch.parquet").exists()
    for rel in (
        ("kmers", "kmers.parquet"),
        ("kmers", "prevalence.parquet"),
        ("matrix", "matrix.npz"),
        ("matrix", "rows.parquet"),
        ("decompose", "pcs.npy"),
        ("decompose", "svd.json"),
        ("embed", "umap.npy"),
        ("cluster", "clusters.parquet"),
        ("annotate", "annotations.parquet"),
        ("enrich", "enrichment.parquet"),
        ("enrich", "cluster_names.parquet"),
    ):
        assert cfg.path(*rel).exists(), "/".join(rel)


def test_shapes_line_up_across_stages(run):
    cfg, manifest, rows, clusters = run
    assert len(rows) > 0
    assert set(rows["assembly"]) == set(manifest["assembly"])
    assert set(rows["contig"]) == {SYNTH_CONTIG}
    # a few bins may fall below min_bin_sketch; most must survive
    assert 0.8 * N_ASSEMBLIES * N_BINS_PER_ASSEMBLY <= len(rows) <= N_ASSEMBLIES * N_BINS_PER_ASSEMBLY

    matrix = sparse.load_npz(cfg.path("matrix", "matrix.npz"))
    pcs = np.load(cfg.path("decompose", "pcs.npy"))
    coords = np.load(cfg.path("embed", "umap.npy"))
    assert matrix.shape[0] == len(rows)
    assert matrix.shape[1] > 0
    assert pcs.shape[0] == len(rows)
    assert coords.shape == (len(rows), 2)
    assert np.isfinite(coords).all()
    assert len(clusters) == len(rows)
    assert clusters["bin_uid"].tolist() == rows["bin_uid"].tolist()

    meta = json.loads(cfg.path("decompose", "svd.json").read_text())
    assert tuple(meta["shape"]) == tuple(matrix.shape)


def test_features_were_actually_selected(run):
    cfg, *_ = run
    kmers = pd.read_parquet(cfg.path("kmers", "kmers.parquet"))
    assert len(kmers) > 1_000
    assert kmers["hash"].is_monotonic_increasing
    assert kmers["col_idx"].tolist() == list(range(len(kmers)))


# --------------------------------------------------------------------------
# the point of the whole exercise
# --------------------------------------------------------------------------


def test_planted_satellite_bins_land_in_one_cluster(run):
    _, _, rows, clusters = run
    sat = satellite_mask(rows)
    assert sat.sum() >= 0.8 * N_ASSEMBLIES * len(satellite_bin_indices(BIN_SIZE))

    labels = clusters["cluster"].to_numpy()
    counts = pd.Series(labels[sat]).value_counts()
    modal = int(counts.index[0])
    coverage = counts.iloc[0] / sat.sum()
    assert modal != -1, "the satellite bins were called noise"
    assert coverage >= 0.8, f"only {coverage:.0%} of satellite bins share a cluster"

    members = labels == modal
    purity = sat[members].mean()
    assert purity >= 0.8, f"cluster {modal} is only {purity:.0%} satellite bins"
    assert rows.loc[members, "assembly"].nunique() >= N_ASSEMBLIES - 1


def test_backbone_bins_do_not_all_collapse_into_the_satellite_cluster(run):
    _, _, rows, clusters = run
    sat = satellite_mask(rows)
    labels = clusters["cluster"].to_numpy()
    modal = int(pd.Series(labels[sat]).value_counts().index[0])
    backbone_in_sat = (labels[~sat] == modal).mean()
    assert backbone_in_sat < 0.1
    assert len(set(labels.tolist()) - {-1}) >= 2, "everything ended up in one cluster"


def test_clusters_are_shared_across_assemblies(run):
    """A cluster that only ever contains one haplotype has learned nothing."""
    _, _, rows, clusters = run
    joined = rows.assign(cluster=clusters["cluster"].to_numpy())
    real = joined[joined["cluster"] != -1]
    per_cluster = real.groupby("cluster")["assembly"].nunique()
    assert len(per_cluster) > 0
    assert per_cluster.max() >= N_ASSEMBLIES - 1
    assert (per_cluster > 1).mean() > 0.5


# --------------------------------------------------------------------------
# annotation / enrichment / back-propagation on the same run
# --------------------------------------------------------------------------


def test_annotations_cover_every_row(run):
    from kmer_dust import schemas

    cfg, _, rows, _ = run
    ann = pd.read_parquet(cfg.path("annotate", "annotations.parquet"))
    assert len(ann) == len(rows)
    assert ann["bin_uid"].tolist() == rows["bin_uid"].tolist()
    frac_cols = [c for c in ann.columns if c.startswith("frac_")]
    assert set(frac_cols) == set(schemas.FEATURE_COLUMNS)
    sat = satellite_mask(rows)
    # the synthetic cenSat BED calls the planted regions hor(...)
    hor = ann[["frac_asat_hor_active", "frac_asat_hor"]].to_numpy().sum(axis=1)
    assert hor[sat].mean() > 0.8
    assert hor[~sat].mean() < 0.2


def test_the_satellite_cluster_is_named_after_its_annotation(run):
    cfg, _, rows, clusters = run
    names = pd.read_parquet(cfg.path("enrich", "cluster_names.parquet"))
    if names.empty:
        pytest.skip("no cluster passed enrich.min_cluster_size")
    sat = satellite_mask(rows)
    modal = int(pd.Series(clusters["cluster"].to_numpy()[sat]).value_counts().index[0])
    named = names[names["cluster"] == modal]
    assert len(named) == 1
    assert "asat_hor" in named.iloc[0]["name"]
    assert -1 not in set(names["cluster"].tolist())


def test_backprop_writes_one_valid_bed_per_assembly(run):
    cfg, manifest, rows, _ = run
    outdir = cfg.path("backprop")
    assert outdir.is_dir()
    total = 0
    for assembly in manifest["assembly"]:
        bed = outdir / f"{assembly}.clusters.bed"
        assert bed.exists(), assembly
        lines = [
            line.split("\t")
            for line in bed.read_text().splitlines()
            if line and not line.startswith(("track", "browser", "#"))
        ]
        assert lines
        total += len(lines)
        for fields in lines:
            assert len(fields) == 9
            assert 0 <= int(fields[1]) < int(fields[2]) <= SYNTH_CONTIG_LEN
            assert 0 <= int(fields[4]) <= 1000
            assert len(fields[8].split(",")) == 3
    assert total == len(rows)
    assert (outdir / "clusters.all.bed.gz").exists()


# --------------------------------------------------------------------------
# restartability
# --------------------------------------------------------------------------


def test_rerunning_the_pipeline_changes_nothing(smoke_run):
    """The second run must short-circuit every stage and keep the same labels."""
    from conftest import block_network

    from kmer_dust import pipeline

    cfg, _, _ = smoke_run
    before = pd.read_parquet(cfg.path("cluster", "clusters.parquet"))
    stamp = cfg.path("cluster", "clusters.parquet").stat().st_mtime_ns
    with pytest.MonkeyPatch.context() as mp:
        block_network(mp)
        pipeline.run_all(cfg)
    after = pd.read_parquet(cfg.path("cluster", "clusters.parquet"))
    pd.testing.assert_frame_equal(before, after)
    assert cfg.path("cluster", "clusters.parquet").stat().st_mtime_ns == stamp
