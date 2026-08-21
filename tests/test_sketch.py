"""Per-assembly sketching: bin construction, drop rules and restartability.

The sketch stage is the only place where sequence turns into rows, so its bin
rules define the entire coordinate system the rest of the pipeline reasons
about.  These tests pin: bins start at position 0 of every contig and never
overlap; a k-mer belongs to the bin containing its first base; dropped bins take
their hashes with them and leave a contiguous ``bin_idx``; and the ``.done``
marker really does short-circuit a re-run.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from conftest import random_sequence, write_fasta

from kmer_dust import schemas
from kmer_dust.hashing import encode_bases, sketch_contig
from kmer_dust.sketch import (
    load_sketch_shard,
    sketch_assembly,
    sketch_manifest,
    sketch_shard_paths,
)

#: The columns docs/API.md promises from sketch_manifest, in order.
REPORT_COLUMNS = ["assembly", "n_bins", "n_hashes", "n_contigs", "seconds", "status"]


def make_row(fasta, assembly="A_pat_syn_v1", **over):
    row = {
        "assembly": assembly,
        "sample": assembly.split("_")[0],
        "haplotype": "pat",
        "source": "local",
        "fasta": str(fasta),
        "fai": "",
        "gzi": "",
        "chrom_alias": "",
        "censat_bed": "",
        "repeatmasker_bed": "",
        "segdup_bed": "",
        "population": "",
        "superpopulation": "",
        "sex": "",
    }
    row.update(over)
    return row


@pytest.fixture
def simple_fasta(tmp_path):
    """Three clean 5 kb chr-named contigs, no ambiguity anywhere."""
    rng = np.random.default_rng(3)
    return write_fasta(
        tmp_path / "simple.fa",
        [
            ("chr21", random_sequence(5_000, rng)),
            ("chr22", random_sequence(5_000, rng)),
            ("chrX", random_sequence(5_000, rng)),
        ],
    )


@pytest.fixture
def sketch_cfg(make_config):
    def _make(**over):
        sketch = {"k": 31, "bin_size": 1_000, "scaled": 4, "min_bin_sketch": 1,
                  "include_unplaced": False}
        sketch.update(over.pop("sketch", {}))
        return make_config(
            sketch=sketch,
            manifest={"chroms": ["chr21", "chr22", "chrX"]},
            **over,
        )

    return _make


# --------------------------------------------------------------------------
# outputs, dtypes, ordering
# --------------------------------------------------------------------------


def test_sketch_assembly_writes_the_documented_files(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    out = sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    paths = sketch_shard_paths(run_dir, "A_pat_syn_v1")
    assert set(paths) >= {"bins", "sketch", "done"}
    for key in ("bins", "sketch", "done"):
        assert paths[key].exists(), key
    assert paths["bins"].name == "A_pat_syn_v1.bins.parquet"
    assert paths["sketch"].name == "A_pat_syn_v1.sketch.parquet"
    assert paths["done"].name == "A_pat_syn_v1.done"
    assert out is not None
    # no *.tmp litter is left behind by a successful run
    assert not list(run_dir.glob("*.tmp"))


def test_shard_columns_and_dtypes(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    bins, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert list(bins.columns) == list(schemas.BIN_COLUMNS)
    assert list(sketch.columns) == list(schemas.SKETCH_COLUMNS)
    for col, dtype in schemas.BIN_COLUMNS.items():
        assert str(bins[col].dtype) == dtype, col
    for col, dtype in schemas.SKETCH_COLUMNS.items():
        assert str(sketch[col].dtype) == dtype, col


def test_bin_geometry(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert len(bins) == 15  # 3 contigs x 5 exact 1 kb bins
    assert bins["bin_idx"].tolist() == list(range(15))
    assert (bins["end"] - bins["start"] == 1_000).all()
    assert (bins["start"] % 1_000 == 0).all()
    for _contig, group in bins.groupby("contig", sort=False):
        starts = group["start"].tolist()
        assert starts == sorted(starts)
        assert starts[0] == 0
        assert starts == list(range(0, 5_000, 1_000))
    expected_uid = [
        schemas.bin_uid("A_pat_syn_v1", c, s)
        for c, s in zip(bins["contig"], bins["start"])
    ]
    assert bins["bin_uid"].tolist() == expected_uid
    assert bins["bin_uid"].is_unique


def test_sketch_is_sorted_by_bin_then_hash(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    _, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    ordered = sketch.sort_values(["bin_idx", "hash"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(sketch.reset_index(drop=True), ordered)


def test_hashes_reference_existing_bins_and_match_n_sketch(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    bins, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert set(sketch["bin_idx"]) <= set(bins["bin_idx"])
    counts = sketch.groupby("bin_idx").size()
    for row in bins.itertuples():
        assert int(counts.get(row.bin_idx, 0)) == int(row.n_sketch)


def test_hashes_are_below_the_configured_threshold(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg(sketch={"k": 31, "bin_size": 1_000, "scaled": 16, "min_bin_sketch": 1})
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    _, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert len(sketch) > 0
    assert int(sketch["hash"].max()) <= cfg.max_hash


def test_hashes_match_a_direct_sketch_of_the_same_contig(simple_fasta, sketch_cfg, run_dir):
    """The shard must contain exactly what hashing.py produces, nothing else."""
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    bins, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    from kmer_dust.fasta import FastaSource

    src = FastaSource(str(simple_fasta))
    try:
        seq = src.fetch("chr21")
    finally:
        src.close()
    exp_bins, exp_hashes = sketch_contig(
        encode_bases(seq), k=31, bin_size=1_000, max_hash=cfg.max_hash
    )
    chr21 = bins[bins["contig"] == "chr21"]
    offset = int(chr21["bin_idx"].min())
    got = sketch[sketch["bin_idx"].isin(chr21["bin_idx"])]
    got_pairs = sorted(
        (int(b) - offset, int(h)) for b, h in zip(got["bin_idx"], got["hash"])
    )
    exp_pairs = sorted((int(b), int(h)) for b, h in zip(exp_bins, exp_hashes))
    assert got_pairs == exp_pairs


def test_per_bin_stats(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert (bins["n_acgt"] == 1_000).all()
    assert (bins["nfrac"] == 0.0).all()
    assert ((bins["gc"] > 0.2) & (bins["gc"] < 0.8)).all()
    # 1 kb bin, k=31: the last 30 k-mers of each bin start in the *next* bin,
    # so an interior bin holds exactly bin_size k-mer starts.
    assert bins["n_kmers"].iloc[0] == 1_000
    assert bins["n_kmers"].iloc[-1] == 1_000 - 30  # last bin of the last contig
    assert (bins["n_sketch"] <= bins["n_kmers"]).all()


# --------------------------------------------------------------------------
# drop rules
# --------------------------------------------------------------------------


def test_trailing_partial_bin_is_dropped_or_kept_per_config(tmp_path, sketch_cfg, run_dir):
    rng = np.random.default_rng(5)
    fasta = write_fasta(tmp_path / "partial.fa", [("chr21", random_sequence(2_500, rng))])

    cfg = sketch_cfg()
    assert cfg.sketch.drop_partial_terminal_bin
    sketch_assembly(make_row(fasta), cfg, run_dir)
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert bins["start"].tolist() == [0, 1_000]

    keep = sketch_cfg(sketch={"k": 31, "bin_size": 1_000, "scaled": 4, "min_bin_sketch": 1,
                              "drop_partial_terminal_bin": False, "min_bin_acgt_frac": 0.0})
    other = run_dir / "keep"
    other.mkdir()
    sketch_assembly(make_row(fasta), keep, other)
    bins2, _ = load_sketch_shard(other, "A_pat_syn_v1")
    assert bins2["start"].tolist() == [0, 1_000, 2_000]
    assert bins2["end"].tolist()[-1] == 2_500


def test_contig_shorter_than_one_bin_is_dropped(tmp_path, sketch_cfg, run_dir):
    rng = np.random.default_rng(6)
    fasta = write_fasta(
        tmp_path / "short.fa",
        [("chr21", random_sequence(5_000, rng)), ("chr22", random_sequence(400, rng))],
    )
    sketch_assembly(make_row(fasta), sketch_cfg(), run_dir)
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert set(bins["contig"]) == {"chr21"}


def test_contig_shorter_than_k_produces_no_bins(tmp_path, sketch_cfg, run_dir):
    fasta = write_fasta(tmp_path / "tiny.fa", [("chr21", b"ACGTACGTAC")])
    sketch_assembly(make_row(fasta), sketch_cfg(), run_dir)
    bins, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert len(bins) == 0
    assert len(sketch) == 0
    assert list(bins.columns) == list(schemas.BIN_COLUMNS)


def test_all_n_contig_produces_no_bins(tmp_path, sketch_cfg, run_dir):
    fasta = write_fasta(tmp_path / "alln.fa", [("chr21", b"N" * 5_000)])
    sketch_assembly(make_row(fasta), sketch_cfg(), run_dir)
    bins, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert len(bins) == 0
    assert len(sketch) == 0


def test_bins_failing_min_bin_acgt_frac_are_dropped(tmp_path, sketch_cfg, run_dir):
    rng = np.random.default_rng(7)
    seq = bytearray(random_sequence(4_000, rng))
    seq[1_000:1_900] = b"N" * 900  # bin 1 is 90 % N
    fasta = write_fasta(tmp_path / "gappy.fa", [("chr21", bytes(seq))])
    cfg = sketch_cfg(
        sketch={"k": 31, "bin_size": 1_000, "scaled": 4, "min_bin_sketch": 1,
                "min_bin_acgt_frac": 0.5}
    )
    sketch_assembly(make_row(fasta), cfg, run_dir)
    bins, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert 1_000 not in set(bins["start"])
    assert bins["start"].tolist() == [0, 2_000, 3_000]
    # dropped bins take their hashes with them, and bin_idx is renumbered
    assert bins["bin_idx"].tolist() == [0, 1, 2]
    assert set(sketch["bin_idx"]) <= {0, 1, 2}


def test_bins_failing_min_bin_sketch_are_dropped(tmp_path, sketch_cfg, run_dir):
    rng = np.random.default_rng(8)
    fasta = write_fasta(tmp_path / "sparse.fa", [("chr21", random_sequence(5_000, rng))])
    cfg = sketch_cfg(
        sketch={"k": 31, "bin_size": 1_000, "scaled": 4_000, "min_bin_sketch": 1_000_000}
    )
    sketch_assembly(make_row(fasta), cfg, run_dir)
    bins, sketch = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert len(bins) == 0
    assert len(sketch) == 0


# --------------------------------------------------------------------------
# contig selection
# --------------------------------------------------------------------------


def test_chroms_filter_selects_contigs(tmp_path, sketch_cfg, run_dir):
    rng = np.random.default_rng(9)
    fasta = write_fasta(
        tmp_path / "multi.fa",
        [
            ("chr21", random_sequence(3_000, rng)),
            ("chr22", random_sequence(3_000, rng)),
            ("chrX", random_sequence(3_000, rng)),
        ],
    )
    cfg = sketch_cfg()
    cfg.manifest.chroms = ["chr21"]
    sketch_assembly(make_row(fasta), cfg, run_dir)
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert set(bins["contig"]) == {"chr21"}
    assert set(bins["chrom"]) == {"chr21"}


def test_chrom_alias_maps_pansn_contigs(tmp_path, sketch_cfg, run_dir):
    rng = np.random.default_rng(10)
    contig = "HG00408#1#CM085953.1"
    fasta = write_fasta(tmp_path / "pansn.fa", [(contig, random_sequence(3_000, rng))])
    alias = tmp_path / "pansn.chromAlias.txt"
    alias.write_text(f"# assembly\tucsc\tgenbank\n{contig}\tchr2\tCM085953.1\n")
    cfg = sketch_cfg()
    cfg.manifest.chroms = ["chr2"]
    sketch_assembly(make_row(fasta, chrom_alias=str(alias)), cfg, run_dir)
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert set(bins["contig"]) == {contig}
    assert set(bins["chrom"]) == {"chr2"}


def test_unplaced_contigs_are_kept_only_when_requested(tmp_path, sketch_cfg, run_dir):
    rng = np.random.default_rng(11)
    fasta = write_fasta(
        tmp_path / "unplaced.fa",
        [("chr21", random_sequence(3_000, rng)), ("scaffold_7", random_sequence(3_000, rng))],
    )
    cfg = sketch_cfg()
    cfg.manifest.chroms = ["chr21"]
    cfg.sketch.include_unplaced = False
    sketch_assembly(make_row(fasta), cfg, run_dir)
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert set(bins["contig"]) == {"chr21"}

    other = run_dir / "with_unplaced"
    other.mkdir()
    cfg2 = sketch_cfg()
    cfg2.manifest.chroms = ["chr21"]
    cfg2.sketch.include_unplaced = True
    sketch_assembly(make_row(fasta), cfg2, other)
    bins2, _ = load_sketch_shard(other, "A_pat_syn_v1")
    assert set(bins2["contig"]) == {"chr21", "scaffold_7"}
    assert bins2.loc[bins2["contig"] == "scaffold_7", "chrom"].eq("").all()


# --------------------------------------------------------------------------
# restartability and determinism
# --------------------------------------------------------------------------


def test_rerun_is_bit_identical(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    first = load_sketch_shard(run_dir, "A_pat_syn_v1")
    first_bytes = sketch_shard_paths(run_dir, "A_pat_syn_v1")["sketch"].read_bytes()

    sketch_assembly(make_row(simple_fasta), cfg, run_dir, force=True)
    second = load_sketch_shard(run_dir, "A_pat_syn_v1")
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert sketch_shard_paths(run_dir, "A_pat_syn_v1")["sketch"].read_bytes() == first_bytes


def test_done_marker_short_circuits(simple_fasta, sketch_cfg, run_dir):
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    paths = sketch_shard_paths(run_dir, "A_pat_syn_v1")
    before = paths["bins"].stat().st_mtime_ns
    time.sleep(0.01)
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    assert paths["bins"].stat().st_mtime_ns == before, "shard was rewritten despite .done"

    sketch_assembly(make_row(simple_fasta), cfg, run_dir, force=True)
    assert paths["bins"].stat().st_mtime_ns != before, "force=True must rewrite the shard"


def test_missing_done_marker_forces_a_rebuild(simple_fasta, sketch_cfg, run_dir):
    """A shard whose .done is absent was interrupted mid-write; redo it."""
    cfg = sketch_cfg()
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    paths = sketch_shard_paths(run_dir, "A_pat_syn_v1")
    paths["done"].unlink()
    paths["bins"].write_bytes(b"corrupt")
    sketch_assembly(make_row(simple_fasta), cfg, run_dir)
    assert paths["done"].exists()
    bins, _ = load_sketch_shard(run_dir, "A_pat_syn_v1")
    assert len(bins) == 15


def test_missing_fasta_is_reported_not_silently_skipped(tmp_path, sketch_cfg, run_dir):
    row = make_row(tmp_path / "does_not_exist.fa")
    with pytest.raises((OSError, ValueError)):
        sketch_assembly(row, sketch_cfg(), run_dir)


# --------------------------------------------------------------------------
# sketch_manifest
# --------------------------------------------------------------------------


def test_sketch_manifest_returns_one_row_per_assembly(synthetic_assemblies, make_config):
    manifest = synthetic_assemblies(n_assemblies=3, contig_len=40_000)
    cfg = make_config(sketch={"k": 31, "bin_size": 10_000, "scaled": 50, "min_bin_sketch": 1})
    report = sketch_manifest(manifest, cfg, threads=1)
    # the contract columns come first; a stage may add diagnostics after them
    assert list(report.columns)[:6] == REPORT_COLUMNS
    assert report["assembly"].tolist() == manifest["assembly"].tolist()
    assert (report["n_bins"] == 4).all()
    assert (report["n_hashes"] > 0).all()
    assert (report["n_contigs"] == 1).all()
    assert set(report["status"]) <= {"ok", "cached", "skipped", "empty"}


def test_sketch_manifest_writes_into_the_sketch_stage_dir(synthetic_assemblies, make_config):
    manifest = synthetic_assemblies(n_assemblies=2, contig_len=30_000)
    cfg = make_config(sketch={"k": 31, "bin_size": 10_000, "scaled": 50, "min_bin_sketch": 1})
    sketch_manifest(manifest, cfg, threads=1)
    outdir = cfg.stage_dir("sketch")
    for assembly in manifest["assembly"]:
        assert (outdir / f"{assembly}.done").exists()


def test_sketch_manifest_on_an_empty_manifest(make_config):
    cfg = make_config()
    report = sketch_manifest(schemas.empty_frame(schemas.MANIFEST_COLUMNS), cfg, threads=1)
    assert len(report) == 0
    assert list(report.columns)[:6] == REPORT_COLUMNS


def test_sketch_manifest_is_deterministic_across_thread_counts(synthetic_assemblies, make_config):
    manifest = synthetic_assemblies(n_assemblies=3, contig_len=30_000)
    frames = []
    for threads in (1, 3):
        cfg = make_config(sketch={"k": 31, "bin_size": 10_000, "scaled": 50, "min_bin_sketch": 1})
        cfg.outdir = str(cfg.out.parent / f"out_t{threads}")
        sketch_manifest(manifest, cfg, threads=threads)
        outdir = cfg.stage_dir("sketch")
        frames.append(
            [load_sketch_shard(outdir, a) for a in manifest["assembly"]]
        )
    for (b1, s1), (b2, s2) in zip(frames[0], frames[1]):
        pd.testing.assert_frame_equal(b1, b2)
        pd.testing.assert_frame_equal(s1, s2)
