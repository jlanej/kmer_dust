"""The label-transfer score must only compare features both sides could carry.

This is the test for the bug that made a real 24-haplotype chr21 run report a
median transfer agreement of 0.000: the reference is annotated with a gene
track that the HPRC per-assembly track set does not have, so every euchromatic
reference bin came out ``gene`` while the same locus in an assembly came out
``line``.  The clustering was fine; the yardstick was broken.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kmer_dust import schemas
from kmer_dust.backprop import cluster_transfer_report


def _frame(cfg_outdir, n_ref=20, n_asm=60):
    """One cluster, gene-rich reference bins, LINE-rich assembly bins."""
    uids, source, hap, assembly = [], [], [], []
    for i in range(n_ref):
        uids.append(f"chm13v2.0|chr21|{i * 10000}")
        source.append("t2t")
        hap.append("ref")
        assembly.append("chm13v2.0")
    for i in range(n_asm):
        uids.append(f"HG1_pat|HG1#1#c|{i * 10000}")
        source.append("hprc")
        hap.append("pat")
        assembly.append("HG1_pat")
    rows = pd.DataFrame(
        {
            "bin_uid": uids,
            "assembly": assembly,
            "source": source,
            "haplotype": hap,
            "contig": ["chr21"] * n_ref + ["HG1#1#c"] * n_asm,
            "chrom": ["chr21"] * (n_ref + n_asm),
            "start": [0] * (n_ref + n_asm),
            "end": [10000] * (n_ref + n_asm),
        }
    )
    clusters = pd.DataFrame(
        {
            "row_idx": np.arange(len(uids), dtype=np.int64),
            "bin_uid": uids,
            "cluster": np.zeros(len(uids), dtype=np.int32),
            "probability": np.ones(len(uids), dtype=np.float32),
            "outlier_score": np.zeros(len(uids), dtype=np.float32),
        }
    )
    ann = pd.DataFrame({"bin_uid": uids, "annotated": True})
    for feature in schemas.FEATURE_VOCAB:
        ann[schemas.feature_column(feature)] = np.float32(0.0)
    # Reference: gene beats line.  Assemblies: line only, no gene track at all.
    ann.loc[: n_ref - 1, schemas.feature_column("gene")] = np.float32(0.9)
    ann.loc[: n_ref - 1, schemas.feature_column("line")] = np.float32(0.6)
    ann.loc[n_ref:, schemas.feature_column("line")] = np.float32(0.6)
    ann["dominant_feature"] = ["gene"] * n_ref + ["line"] * n_asm
    ann["dominant_frac"] = np.float32(0.9)
    names = pd.DataFrame(
        {
            "cluster": np.array([0], dtype=np.int32),
            "name": ["C0 line"],
            "top_features": ["line:2.0"],
            "size": [len(uids)],
            "n_assemblies": np.array([2], dtype=np.int32),
            "n_chroms": np.array([1], dtype=np.int32),
            "purity": [1.0],
        }
    )
    return rows, clusters, ann, names


def test_a_reference_only_track_cannot_score_as_disagreement(make_config, run_dir):
    cfg = make_config()
    cfg.annotate.reference_tracks = ["censat", "repeatmasker", "segdup", "telomere", "gene"]
    cfg.annotate.assembly_tracks = ["censat", "repeatmasker", "segdup"]
    rows, clusters, ann, names = _frame(run_dir)

    report = cluster_transfer_report(rows, clusters, ann, names, cfg, run_dir)
    row = report[report["cluster"] == 0].iloc[0]

    # `gene` is unreachable for the assemblies, so it must not be the yardstick.
    assert row["ref_top_feature"] == "line"
    assert row["asm_top_feature"] == "line"
    assert row["asm_agreement"] == pytest.approx(1.0)


def test_symmetric_track_sets_use_the_whole_vocabulary(make_config, run_dir):
    cfg = make_config()
    cfg.annotate.reference_tracks = ["censat", "repeatmasker", "segdup", "telomere", "gene"]
    cfg.annotate.assembly_tracks = list(cfg.annotate.reference_tracks)
    rows, clusters, ann, names = _frame(run_dir)

    report = cluster_transfer_report(rows, clusters, ann, names, cfg, run_dir)
    row = report[report["cluster"] == 0].iloc[0]

    # Now `gene` really is comparable, and the disagreement is real.
    assert row["ref_top_feature"] == "gene"
    assert row["asm_top_feature"] == "line"
    assert row["asm_agreement"] == pytest.approx(0.0)


def test_the_comparable_vocabulary_is_written_down(make_config, run_dir):
    cfg = make_config()
    cfg.annotate.reference_tracks = ["censat", "repeatmasker", "segdup", "telomere", "gene"]
    cfg.annotate.assembly_tracks = ["censat", "repeatmasker"]
    rows, clusters, ann, names = _frame(run_dir)
    cluster_transfer_report(rows, clusters, ann, names, cfg, run_dir)

    payload = json.loads((run_dir / "transfer_features.json").read_text())
    assert "gene" in payload["excluded_features"]
    assert "segdup" in payload["excluded_features"]
    assert "hsat3" in payload["comparable_features"]
    assert "line" in payload["comparable_features"]


def test_features_for_tracks_is_ordered_and_deduplicated():
    got = schemas.features_for_tracks(["repeatmasker", "censat", "censat"])
    assert list(got) == [f for f in schemas.FEATURE_VOCAB if f in set(got)]
    assert len(got) == len(set(got))
    assert schemas.features_for_tracks([]) == ()
    assert schemas.features_for_tracks(["not-a-track"]) == ()
