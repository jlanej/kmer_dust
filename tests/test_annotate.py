"""Annotation: name normalisation, interval coverage and per-bin annotation.

``bin_feature_fractions`` is the highest-stakes numerical routine outside
``hashing``: every enrichment number, every cluster name and the whole
label-transfer claim is downstream of it.  Interval-vs-bin overlap is also
exactly the kind of code that is right for the easy cases and off by one at the
edges, so the core test builds random intervals and compares against a
per-base boolean mask -- an implementation that is obviously correct and far too
slow to ship.
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest
from conftest import manifest_from_shards

from kmer_dust import schemas
from kmer_dust.annotate import (
    annotate_bins,
    bin_feature_fractions,
    load_annotations,
    normalize_censat_name,
    normalize_repeat_class,
    read_bed,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def make_bins(specs, assembly="A_pat") -> pd.DataFrame:
    """BIN_COLUMNS frame from ``(chrom, start, end)`` triples.

    ``contig`` is set equal to ``chrom`` so the test does not depend on which
    of the two ``bin_feature_fractions`` joins the intervals on (docs/API.md
    does not say; reference tracks are keyed by chrom, per-assembly tracks by
    contig).
    """
    rows = []
    for i, (chrom, start, end) in enumerate(specs):
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
                "start": start,
                "end": end,
                "n_acgt": end - start,
                "n_kmers": max(end - start - 30, 0),
                "n_sketch": 50,
                "gc": 0.4,
                "nfrac": 0.0,
            }
        )
    if not rows:
        return schemas.empty_frame(schemas.BIN_COLUMNS)
    return schemas.enforce(pd.DataFrame(rows), schemas.BIN_COLUMNS)


def make_intervals(records) -> pd.DataFrame:
    """``(chrom, start, end, feature)`` tuples -> the intervals frame."""
    df = pd.DataFrame(records, columns=["chrom", "start", "end", "feature"])
    return df.astype({"chrom": "string", "start": "int64", "end": "int64", "feature": "string"})


def brute_fractions(bins: pd.DataFrame, intervals: pd.DataFrame, features) -> np.ndarray:
    """Per-base reference implementation: paint a boolean mask, then average."""
    out = np.zeros((len(bins), len(features)), dtype=np.float64)
    for i, b in enumerate(bins.itertuples()):
        width = int(b.end) - int(b.start)
        if width <= 0:
            continue
        for j, feature in enumerate(features):
            mask = np.zeros(width, dtype=bool)
            sel = intervals[
                (intervals["feature"] == feature) & (intervals["chrom"] == b.chrom)
            ]
            for iv in sel.itertuples():
                s = max(int(iv.start), int(b.start))
                e = min(int(iv.end), int(b.end))
                if e > s:
                    mask[s - int(b.start) : e - int(b.start)] = True
            out[i, j] = mask.sum() / width
    return out


# --------------------------------------------------------------------------
# bin_feature_fractions -- against a per-base brute force
# --------------------------------------------------------------------------


def test_fractions_match_brute_force_on_random_intervals(rng):
    features = ["hsat2", "asat_hor_active", "segdup", "line"]
    bins = make_bins(
        [("chr21", s, s + 1_000) for s in range(0, 20_000, 1_000)]
        + [("chr22", s, s + 1_000) for s in range(0, 10_000, 1_000)]
    )
    records = []
    for _ in range(400):
        chrom = "chr21" if rng.random() < 0.6 else "chr22"
        start = int(rng.integers(-500, 21_000))
        length = int(rng.integers(1, 3_000))
        records.append((chrom, max(start, 0), max(start, 0) + length, features[rng.integers(0, 4)]))
    # a chromosome that is not in bins at all must be ignored
    records += [("chrX", 0, 1_000, "hsat2")] * 5
    intervals = make_intervals(records)

    got = bin_feature_fractions(bins, intervals, features)
    assert got.shape == (len(bins), len(features))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, brute_fractions(bins, intervals, features), atol=1e-6)


def test_fractions_match_brute_force_with_ragged_bins(rng):
    features = ["hsat2", "bsat"]
    edges = np.cumsum(rng.integers(300, 2_000, size=15))
    bins = make_bins(
        [("chr1", int(a), int(b)) for a, b in zip(np.r_[0, edges[:-1]], edges)]
    )
    records = [
        ("chr1", int(s), int(s + rng.integers(1, 1_500)), features[rng.integers(0, 2)])
        for s in rng.integers(0, int(edges[-1]), size=120)
    ]
    intervals = make_intervals(records)
    np.testing.assert_allclose(
        bin_feature_fractions(bins, intervals, features),
        brute_fractions(bins, intervals, features),
        atol=1e-6,
    )


def test_overlapping_same_feature_intervals_cap_at_one():
    bins = make_bins([("chr1", 0, 100)])
    intervals = make_intervals(
        [("chr1", 0, 100, "hsat2"), ("chr1", 0, 100, "hsat2"), ("chr1", 20, 80, "hsat2")]
    )
    got = bin_feature_fractions(bins, intervals, ["hsat2"])
    assert got[0, 0] == pytest.approx(1.0)


def test_adjacent_intervals_add_without_double_counting():
    bins = make_bins([("chr1", 0, 100)])
    intervals = make_intervals(
        [("chr1", 0, 40, "hsat2"), ("chr1", 40, 70, "hsat2"), ("chr1", 60, 90, "hsat2")]
    )
    # union is [0, 90) -> 0.9, not (40 + 30 + 30) / 100
    assert bin_feature_fractions(bins, intervals, ["hsat2"])[0, 0] == pytest.approx(0.9)


def test_partial_overlap_at_both_edges():
    bins = make_bins([("chr1", 1_000, 2_000)])
    intervals = make_intervals([("chr1", 500, 1_250, "line"), ("chr1", 1_900, 5_000, "line")])
    assert bin_feature_fractions(bins, intervals, ["line"])[0, 0] == pytest.approx(0.35)


def test_features_are_independent():
    bins = make_bins([("chr1", 0, 100)])
    intervals = make_intervals([("chr1", 0, 50, "hsat2"), ("chr1", 25, 100, "line")])
    got = bin_feature_fractions(bins, intervals, ["hsat2", "line", "bsat"])
    np.testing.assert_allclose(got[0], [0.5, 0.75, 0.0], atol=1e-6)


def test_intervals_on_another_chromosome_do_not_leak():
    bins = make_bins([("chr1", 0, 100), ("chr2", 0, 100)])
    intervals = make_intervals([("chr1", 0, 100, "hsat2")])
    got = bin_feature_fractions(bins, intervals, ["hsat2"])
    assert got[0, 0] == pytest.approx(1.0)
    assert got[1, 0] == pytest.approx(0.0)


def test_zero_length_and_inverted_intervals_are_ignored():
    bins = make_bins([("chr1", 0, 100)])
    intervals = make_intervals([("chr1", 50, 50, "hsat2"), ("chr1", 80, 20, "hsat2")])
    assert bin_feature_fractions(bins, intervals, ["hsat2"])[0, 0] == pytest.approx(0.0)


def test_unknown_feature_column_is_all_zero():
    bins = make_bins([("chr1", 0, 100)])
    intervals = make_intervals([("chr1", 0, 100, "hsat2")])
    got = bin_feature_fractions(bins, intervals, ["not_a_feature"])
    assert got.shape == (1, 1)
    assert got[0, 0] == 0.0


def test_empty_inputs():
    features = list(schemas.FEATURE_VOCAB)
    empty_iv = make_intervals([]).iloc[0:0]
    bins = make_bins([("chr1", 0, 100), ("chr1", 100, 200)])
    got = bin_feature_fractions(bins, empty_iv, features)
    assert got.shape == (2, len(features))
    assert got.dtype == np.float32
    assert not got.any()

    got = bin_feature_fractions(make_bins([]), empty_iv, features)
    assert got.shape == (0, len(features))

    got = bin_feature_fractions(bins, empty_iv, [])
    assert got.shape == (2, 0)


def test_full_feature_vocab_is_accepted(rng):
    bins = make_bins([("chr1", s, s + 500) for s in range(0, 5_000, 500)])
    intervals = make_intervals(
        [
            ("chr1", int(rng.integers(0, 5_000)), int(rng.integers(0, 5_000)), f)
            for f in schemas.FEATURE_VOCAB
        ]
    )
    got = bin_feature_fractions(bins, intervals, list(schemas.FEATURE_VOCAB))
    assert got.shape == (10, len(schemas.FEATURE_VOCAB))
    assert ((got >= 0.0) & (got <= 1.0)).all()


# --------------------------------------------------------------------------
# read_bed
# --------------------------------------------------------------------------


def test_read_bed_skips_the_track_line(toy_censat_bed):
    bed = read_bed(str(toy_censat_bed))
    assert len(bed) == 12
    assert list(bed.columns)[:6] == ["chrom", "start", "end", "name", "score", "strand"]
    assert "extra" in bed.columns
    assert str(bed["start"].dtype) == "int64"
    assert str(bed["end"].dtype) == "int64"
    assert bed["chrom"].iloc[0] == "chr21"
    assert bed["name"].iloc[1] == "hor_1_1(S3C1H2-A,B,C)"
    assert not bed["chrom"].str.startswith("track").any()
    assert (bed["end"] > bed["start"]).all()


def test_read_bed_keeps_extra_columns_as_a_list(toy_repeatmasker_bed):
    bed = read_bed(str(toy_repeatmasker_bed))
    assert len(bed) == 10
    extra = bed["extra"].iloc[0]
    assert isinstance(extra, (list, tuple))
    # RepeatMasker class/family live in columns 7 and 8 -> extra[0], extra[1]
    assert extra[0] == "SINE"
    assert extra[1] == "Alu"


def test_read_bed_handles_gzip(tmp_path, toy_segdup_bed):
    gz = tmp_path / "segdup.bed.gz"
    with gzip.open(gz, "wt") as handle:
        handle.write(toy_segdup_bed.read_text())
    bed = read_bed(str(gz))
    assert len(bed) == 3
    assert bed["chrom"].tolist() == ["chr21", "chr21", "chrX"]


def test_read_bed_of_a_missing_file_raises_a_catchable_error(tmp_path):
    """A manifest can name an annotation that was never published.

    ``read_bed`` reports that as FileNotFoundError rather than pretending the
    track was empty -- an empty track and a missing one mean different things.
    ``annotate_bins`` is the layer that has to tolerate it (see below).
    """
    with pytest.raises(FileNotFoundError):
        read_bed(str(tmp_path / "absent.bed"))


def test_read_bed_of_an_empty_file(tmp_path):
    path = tmp_path / "empty.bed"
    path.write_text("")
    assert len(read_bed(str(path))) == 0


def test_read_bed_skips_comment_and_browser_lines(tmp_path):
    path = tmp_path / "c.bed"
    path.write_text(
        "# a comment\nbrowser position chr1\ntrack name=x\nchr1\t0\t10\tfoo\t0\t+\n\n"
    )
    bed = read_bed(str(path))
    assert len(bed) == 1
    assert bed["name"].iloc[0] == "foo"


def test_read_bed_three_column_input(tmp_path):
    path = tmp_path / "b3.bed"
    path.write_text("chr1\t0\t10\nchr1\t20\t30\n")
    bed = read_bed(str(path))
    assert len(bed) == 2
    assert list(bed.columns)[:6] == ["chrom", "start", "end", "name", "score", "strand"]
    assert bed["name"].tolist() == ["", ""]


# --------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------


CENSAT_CASES = {
    # T2T cenSat v2.0 names (verified against the real chm13v2.0 track)
    "hsat1A": "hsat1a",
    "hsat1B": "hsat1b",
    "hsat1B_Y": "hsat1b",
    "hsat2": "hsat2",
    "hsat3": "hsat3",
    "hsat3_Y": "hsat3",
    "bsat": "bsat",
    "bsat_21_1": "bsat",
    "bsat_Y": "bsat",
    "gsat_1_1": "gsat",
    "gSat(TAR1)": "gsat",
    "rDNA": "rdna",
    "rDNA_1_1": "rdna",
    "ct": "ct",
    "ct_1_1(p_arm)": "ct",
    "ct_X_3": "ct",
    "censat_1_1(rnd-6_family-4384)": "censat_other",
    "censat_X_1": "censat_other",
    # HPRC per-assembly cenSat names
    "active_hor(S1C1H1L)": "asat_hor_active",
    "dhor_1_2(S3C1H1d)": "asat_hor",
    "dhor_X_1": "asat_hor",
    # non-satellite / unknown
    "": "",
    "track name=\"cenSat\"": "",
    "not_a_satellite_at_all": "",
}


@pytest.mark.parametrize("raw,expected", list(CENSAT_CASES.items()))
def test_normalize_censat_name(raw, expected):
    assert normalize_censat_name(raw) == expected


def test_normalize_censat_always_returns_a_vocabulary_member():
    for raw in list(CENSAT_CASES) + ["hor_1_1(S3C1H2-A,B,C)", "mon_1_1(S1C1)", "cenSat(HSAT5v1)"]:
        assert normalize_censat_name(raw) in set(schemas.CENSAT_CLASSES) | {""}


def test_normalize_censat_hor_variants():
    # Ambiguous by construction: T2T v2.0 writes live arrays as `hor` and
    # divergent ones as `dhor`, while the HPRC per-assembly BEDs write
    # `active_hor` vs `hor`.  Either reading of a bare `hor` is defensible;
    # what must hold is that it is alpha-satellite HOR of *some* kind and that
    # `dhor` and `active_hor` stay distinguishable.
    assert normalize_censat_name("hor_1_1(S3C1H2-A,B,C)") in {"asat_hor_active", "asat_hor"}
    assert normalize_censat_name("dhor_1_2(S3C1H1d)") == "asat_hor"
    assert normalize_censat_name("active_hor(S1C1H1L)") == "asat_hor_active"


def test_normalize_censat_monomeric_variants():
    # `mon` in the T2T track is monomeric *alpha* satellite; the vocabulary has
    # both `asat_mon` and a generic `mon`, so either is defensible.
    assert normalize_censat_name("mon_1_1(S1C1)") in {"asat_mon", "mon"}
    assert normalize_censat_name("mon_X_2") in {"asat_mon", "mon"}


def test_normalize_censat_is_case_insensitive():
    assert normalize_censat_name("HSAT2") == "hsat2"
    assert normalize_censat_name("HSat2_1_1") == "hsat2"
    assert normalize_censat_name("rdna") == "rdna"


REPEAT_CASES = {
    ("LINE", "L1"): "line",
    ("SINE", "Alu"): "sine",
    ("SINE", "MIR"): "sine",
    ("LTR", "ERVL-MaLR"): "ltr",
    ("DNA", "hAT-Charlie"): "dna",
    ("Satellite", "centr"): "satellite",
    ("Satellite", "telo"): "satellite",
    ("Simple_repeat", ""): "simple_repeat",
    ("Low_complexity", ""): "low_complexity",
    ("rRNA", ""): "rrna",
    ("tRNA", ""): "trna",
    ("snRNA", ""): "snrna",
    ("Retroposon", "SVA"): "retroposon",
    ("RC", "Helitron"): "rc",
    ("Unknown", "Unknown"): "repeat_unknown",
    ("", ""): "",
}


@pytest.mark.parametrize("args,expected", list(REPEAT_CASES.items()))
def test_normalize_repeat_class(args, expected):
    assert normalize_repeat_class(*args) == expected


def test_normalize_repeat_class_accepts_a_slash_joined_class():
    """RepeatMasker output is often 'LINE/L1' in a single field."""
    assert normalize_repeat_class("LINE/L1") == "line"
    assert normalize_repeat_class("DNA/hAT-Tip100") == "dna"
    assert normalize_repeat_class("Satellite/centr") == "satellite"


def test_normalize_repeat_class_always_returns_a_vocabulary_member():
    raws = list(REPEAT_CASES) + [("scRNA", ""), ("srpRNA", ""), ("nonsense", "x"), ("LINE/L1", "")]
    for cls, family in raws:
        assert normalize_repeat_class(cls, family) in set(schemas.REPEAT_CLASSES) | {""}


def test_normalize_repeat_class_family_only_is_not_enough():
    assert normalize_repeat_class("", "Alu") in {"", "sine"}


# --------------------------------------------------------------------------
# annotate_bins
# --------------------------------------------------------------------------


@pytest.fixture
def local_tracks(tmp_path):
    """One assembly with a cenSat BED covering the first bin only."""
    censat = tmp_path / "A_pat.censat.bed"
    censat.write_text(
        'track name="cenSat"\n'
        "chr21\t0\t10000\thsat2\t100\t.\t0\t10000\t153,0,0\n"
        "chr21\t10000\t12500\tbsat_21_1\t100\t.\t10000\t12500\t0,0,204\n"
    )
    manifest = manifest_from_shards([("A_pat", "A"), ("B_pat", "B")])
    manifest.loc[manifest["assembly"] == "A_pat", "censat_bed"] = str(censat)
    return manifest


def test_annotate_bins_contract(local_tracks, make_config, run_dir):
    rows = pd.concat(
        [
            make_bins([("chr21", s, s + 10_000) for s in range(0, 30_000, 10_000)], "A_pat"),
            make_bins([("chr21", 0, 10_000)], "B_pat"),
        ],
        ignore_index=True,
    )
    cfg = make_config(annotate={"reference_tracks": [], "assembly_tracks": ["censat"]})
    ann = annotate_bins(rows, local_tracks, cfg, run_dir)
    assert list(ann.columns) == list(schemas.ANNOTATION_ID_COLUMNS) + list(schemas.FEATURE_COLUMNS)
    for col, dtype in schemas.ANNOTATION_ID_COLUMNS.items():
        assert str(ann[col].dtype) == dtype, col
    for col in schemas.FEATURE_COLUMNS:
        assert str(ann[col].dtype) == "float32", col
    assert len(ann) == len(rows)
    assert ann["bin_uid"].tolist() == rows["bin_uid"].tolist()
    assert (run_dir / "annotations.parquet").exists()
    pd.testing.assert_frame_equal(load_annotations(run_dir), ann)


def test_annotate_bins_values(local_tracks, make_config, run_dir):
    rows = make_bins([("chr21", s, s + 10_000) for s in range(0, 30_000, 10_000)], "A_pat")
    cfg = make_config(annotate={"reference_tracks": [], "assembly_tracks": ["censat"]})
    ann = annotate_bins(rows, local_tracks, cfg, run_dir)
    assert ann.loc[0, "frac_hsat2"] == pytest.approx(1.0)
    assert ann.loc[1, "frac_bsat"] == pytest.approx(0.25)
    assert ann.loc[2, list(schemas.FEATURE_COLUMNS)].sum() == pytest.approx(0.0)
    assert ann.loc[0, "dominant_feature"] == "hsat2"
    assert ann.loc[0, "dominant_frac"] == pytest.approx(1.0)
    assert ann.loc[2, "dominant_feature"] == "unannotated"
    assert bool(ann.loc[0, "annotated"]) is True
    assert bool(ann.loc[2, "annotated"]) is True  # the assembly *had* a track


def test_min_frac_for_dominant(local_tracks, make_config, run_dir):
    rows = make_bins([("chr21", 10_000, 20_000)], "A_pat")
    cfg = make_config(
        annotate={"reference_tracks": [], "assembly_tracks": ["censat"],
                  "min_frac_for_dominant": 0.5}
    )
    ann = annotate_bins(rows, local_tracks, cfg, run_dir)
    assert ann.loc[0, "frac_bsat"] == pytest.approx(0.25)
    assert ann.loc[0, "dominant_feature"] == "unannotated"


def test_assembly_without_tracks_is_marked_unannotated(local_tracks, make_config, run_dir):
    rows = make_bins([("chr21", 0, 10_000)], "B_pat")
    cfg = make_config(annotate={"reference_tracks": [], "assembly_tracks": ["censat"]})
    ann = annotate_bins(rows, local_tracks, cfg, run_dir)
    assert bool(ann.loc[0, "annotated"]) is False
    assert ann.loc[0, "dominant_feature"] == "unannotated"
    assert ann.loc[0, list(schemas.FEATURE_COLUMNS)].sum() == pytest.approx(0.0)


def test_annotate_bins_on_empty_rows(local_tracks, make_config, run_dir):
    cfg = make_config(annotate={"reference_tracks": [], "assembly_tracks": ["censat"]})
    ann = annotate_bins(make_bins([]), local_tracks, cfg, run_dir)
    assert len(ann) == 0
    assert list(ann.columns) == list(schemas.ANNOTATION_ID_COLUMNS) + list(schemas.FEATURE_COLUMNS)


def test_annotate_bins_is_deterministic_and_restartable(local_tracks, make_config, run_dir):
    rows = make_bins([("chr21", s, s + 10_000) for s in range(0, 30_000, 10_000)], "A_pat")
    cfg = make_config(annotate={"reference_tracks": [], "assembly_tracks": ["censat"]})
    first = annotate_bins(rows, local_tracks, cfg, run_dir)
    before = (run_dir / "annotations.parquet").stat().st_mtime_ns
    again = annotate_bins(rows, local_tracks, cfg, run_dir)
    pd.testing.assert_frame_equal(first, again)
    assert (run_dir / "annotations.parquet").stat().st_mtime_ns == before
    forced = annotate_bins(rows, local_tracks, cfg, run_dir, force=True)
    pd.testing.assert_frame_equal(first, forced)


def test_annotate_bins_tolerates_a_broken_track_path(make_config, run_dir):
    manifest = manifest_from_shards([("A_pat", "A")])
    manifest.loc[:, "censat_bed"] = "/definitely/not/here.bed"
    rows = make_bins([("chr21", 0, 10_000)], "A_pat")
    cfg = make_config(annotate={"reference_tracks": [], "assembly_tracks": ["censat"]})
    ann = annotate_bins(rows, manifest, cfg, run_dir)
    assert len(ann) == 1
    assert ann.loc[0, "dominant_feature"] == "unannotated"
