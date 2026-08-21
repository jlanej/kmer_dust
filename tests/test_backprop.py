"""Back-propagation: per-assembly BED9 and the label-transfer report.

The BED files are the artefact a biologist actually loads, so they have to be
valid BED9 -- nine tab-separated fields, half-open coordinates that round-trip
to the bin they came from, a 0-1000 score and an ``r,g,b`` itemRgb.  Colours
have to be a pure function of the cluster label as well: a browser session
compared against last week's run is worthless if cluster 7 was orange then and
blue now.

``cluster_transfer_report`` is the honest test of the whole project: it asks
whether a label learned on CHM13 still means the same thing on bins from other
assemblies, so its agreement column must be computed from assembly bins only.
"""

from __future__ import annotations

import gzip
import re

import numpy as np
import pandas as pd
import pytest

from kmer_dust import schemas
from kmer_dust.backprop import cluster_transfer_report, write_cluster_beds

RGB = re.compile(r"^\d{1,3},\d{1,3},\d{1,3}$")


def build_case():
    """Reference + two assemblies; cluster 0 = hsat2, cluster 1 = line, plus noise."""
    specs = []
    for assembly, source, contig, chrom in (
        ("chm13v2.0", "t2t", "chr21", "chr21"),
        ("A_pat", "local", "A#1#CM000001.1", "chr21"),
        ("B_pat", "local", "B#1#CM000002.1", "chr21"),
    ):
        for i in range(12):
            specs.append((assembly, source, contig, chrom, i))

    rows = []
    for idx, (assembly, source, contig, chrom, i) in enumerate(specs):
        start = i * 10_000
        rows.append(
            {
                "bin_idx": i,
                "bin_uid": schemas.bin_uid(assembly, contig, start),
                "assembly": assembly,
                "sample": assembly.split("_")[0],
                "haplotype": "ref" if source == "t2t" else "pat",
                "source": source,
                "contig": contig,
                "chrom": chrom,
                "placed": True,
                "start": start,
                "end": start + 10_000,
                "n_acgt": 10_000,
                "n_kmers": 9_970,
                "n_sketch": 50,
                "gc": 0.42,
                "nfrac": 0.0,
                "row_idx": idx,
            }
        )
    rows = schemas.enforce(pd.DataFrame(rows), schemas.BIN_COLUMNS, subset=True)

    # bins 0-3 -> cluster 0, bins 4-7 -> cluster 1, bins 8-11 -> noise
    labels = [0 if i < 4 else (1 if i < 8 else -1) for _, _, _, _, i in specs]
    clusters = schemas.enforce(
        pd.DataFrame(
            {
                "row_idx": rows["row_idx"],
                "bin_uid": rows["bin_uid"],
                "cluster": labels,
                "probability": [0.0 if c == -1 else 0.85 for c in labels],
                "outlier_score": 0.2,
            }
        ),
        schemas.CLUSTER_COLUMNS,
    )

    names = schemas.enforce(
        pd.DataFrame(
            {
                "cluster": [0, 1],
                "name": ["C0 hsat2", "C1 line"],
                "top_features": ["hsat2:3.1", "line:2.4"],
                "size": [12, 12],
                "n_assemblies": [3, 3],
                "n_chroms": [1, 1],
                "purity": [1.0, 1.0],
            }
        ),
        schemas.CLUSTER_NAME_COLUMNS,
    )

    ann = {col: np.zeros(len(rows), dtype=np.float32) for col in schemas.FEATURE_COLUMNS}
    dominant = []
    for idx, label in enumerate(labels):
        feature = {0: "hsat2", 1: "line"}.get(label)
        if feature is None:
            dominant.append("unannotated")
        else:
            ann[schemas.feature_column(feature)][idx] = 0.8
            dominant.append(feature)
    annotations = schemas.enforce(
        pd.DataFrame(
            {
                "bin_uid": rows["bin_uid"],
                "dominant_feature": dominant,
                "dominant_frac": [0.0 if d == "unannotated" else 0.8 for d in dominant],
                "annotated": True,
                **ann,
            }
        ),
        schemas.ANNOTATION_ID_COLUMNS,
        subset=True,
    )
    return rows, clusters, names, annotations


@pytest.fixture
def case():
    return build_case()


def read_bed9(path):
    text = gzip.open(path, "rt").read() if str(path).endswith(".gz") else path.read_text()
    out = []
    for line in text.splitlines():
        if not line or line.startswith(("track", "browser", "#")):
            continue
        out.append(line.split("\t"))
    return out


# --------------------------------------------------------------------------
# file layout
# --------------------------------------------------------------------------


def test_one_bed_per_assembly_plus_a_combined_file(case, make_config, run_dir):
    rows, clusters, names, _ = case
    paths = write_cluster_beds(rows, clusters, names, make_config(), run_dir)
    assert isinstance(paths, list)
    assert all(p.exists() for p in paths)
    per_assembly = {p.name for p in run_dir.glob("*.clusters.bed")}
    assert per_assembly == {"chm13v2.0.clusters.bed", "A_pat.clusters.bed", "B_pat.clusters.bed"}
    combined = run_dir / "clusters.all.bed.gz"
    assert combined.exists()
    assert len(read_bed9(combined)) == sum(
        len(read_bed9(run_dir / n)) for n in per_assembly
    )


# --------------------------------------------------------------------------
# BED9 validity
# --------------------------------------------------------------------------


def test_bed9_is_well_formed(case, make_config, run_dir):
    rows, clusters, names, _ = case
    write_cluster_beds(rows, clusters, names, make_config(), run_dir)
    lines = read_bed9(run_dir / "A_pat.clusters.bed")
    assert lines
    for fields in lines:
        assert len(fields) == 9, fields
        chrom, start, end, name, score, strand, thick_start, thick_end, rgb = fields
        assert chrom
        assert 0 <= int(start) < int(end)
        assert name
        assert 0 <= int(score) <= 1000
        assert strand in {"+", "-", "."}
        assert int(thick_start) == int(start)
        assert int(thick_end) == int(end)
        assert RGB.match(rgb), rgb
        assert all(0 <= int(c) <= 255 for c in rgb.split(","))


def test_coordinates_round_trip_to_the_bins(case, make_config, run_dir):
    rows, clusters, names, _ = case
    write_cluster_beds(rows, clusters, names, make_config(), run_dir)
    lines = read_bed9(run_dir / "A_pat.clusters.bed")
    got = {(f[0], int(f[1]), int(f[2])) for f in lines}
    sub = rows[rows["assembly"] == "A_pat"]
    # The per-assembly BED has to be loadable against the assembly's own FASTA,
    # so the first column is the *contig* name, not the normalised chrom.
    expected = {(c, int(s), int(e)) for c, s, e in zip(sub["contig"], sub["start"], sub["end"])}
    assert got == expected


def test_lines_are_sorted_by_contig_then_start(case, make_config, run_dir):
    rows, clusters, names, _ = case
    write_cluster_beds(rows, clusters, names, make_config(), run_dir)
    lines = read_bed9(run_dir / "A_pat.clusters.bed")
    keys = [(f[0], int(f[1])) for f in lines]
    assert keys == sorted(keys)


def test_bed_names_carry_the_cluster_identity(case, make_config, run_dir):
    rows, clusters, names, _ = case
    write_cluster_beds(rows, clusters, names, make_config(), run_dir)
    lines = read_bed9(run_dir / "A_pat.clusters.bed")
    by_start = {int(f[1]): f[3] for f in lines}
    assert "hsat2" in by_start[0]
    assert "line" in by_start[40_000]


# --------------------------------------------------------------------------
# colour stability
# --------------------------------------------------------------------------


def test_colour_is_a_function_of_the_cluster_within_a_run(case, make_config, run_dir):
    rows, clusters, names, _ = case
    write_cluster_beds(rows, clusters, names, make_config(), run_dir)
    by_name: dict[str, set[str]] = {}
    for asm in ("chm13v2.0", "A_pat", "B_pat"):
        for fields in read_bed9(run_dir / f"{asm}.clusters.bed"):
            by_name.setdefault(fields[3], set()).add(fields[8])
    for name, rgbs in by_name.items():
        assert len(rgbs) == 1, f"{name} got {rgbs}"
    assert len({next(iter(v)) for v in by_name.values()}) == len(by_name)


def test_colours_are_stable_across_runs(case, make_config, tmp_path):
    rows, clusters, names, _ = case
    palettes = []
    for i in range(2):
        out = tmp_path / f"bp{i}"
        out.mkdir()
        write_cluster_beds(rows, clusters, names, make_config(), out)
        palettes.append(
            {f[3]: f[8] for f in read_bed9(out / "A_pat.clusters.bed")}
        )
    assert palettes[0] == palettes[1]


def test_files_are_byte_identical_across_runs(case, make_config, tmp_path):
    rows, clusters, names, _ = case
    blobs = []
    for i in range(2):
        out = tmp_path / f"det{i}"
        out.mkdir()
        write_cluster_beds(rows, clusters, names, make_config(), out)
        blobs.append((out / "A_pat.clusters.bed").read_bytes())
    assert blobs[0] == blobs[1]


# --------------------------------------------------------------------------
# noise and edge cases
# --------------------------------------------------------------------------


def test_noise_bins_are_either_omitted_or_consistently_grey(case, make_config, run_dir):
    rows, clusters, names, _ = case
    write_cluster_beds(rows, clusters, names, make_config(), run_dir)
    lines = read_bed9(run_dir / "A_pat.clusters.bed")
    noise_starts = {80_000, 90_000, 100_000, 110_000}
    noise = [f for f in lines if int(f[1]) in noise_starts]
    if noise:
        assert len({f[8] for f in noise}) == 1
        assert len({f[3] for f in noise}) == 1
    else:
        assert len(lines) == 8


def test_unnamed_cluster_still_gets_a_line(case, make_config, run_dir):
    """A cluster below the naming threshold must not vanish from the BED."""
    rows, clusters, names, _ = case
    only_one = names[names["cluster"] == 0].reset_index(drop=True)
    write_cluster_beds(rows, clusters, only_one, make_config(), run_dir)
    lines = read_bed9(run_dir / "A_pat.clusters.bed")
    assert len({f[3] for f in lines}) >= 2


def test_empty_input_writes_nothing_and_does_not_crash(make_config, run_dir):
    rows = schemas.enforce(
        schemas.empty_frame(schemas.BIN_COLUMNS).assign(row_idx=pd.Series([], dtype="int64")),
        schemas.BIN_COLUMNS,
        subset=True,
    )
    paths = write_cluster_beds(
        rows,
        schemas.empty_frame(schemas.CLUSTER_COLUMNS),
        schemas.empty_frame(schemas.CLUSTER_NAME_COLUMNS),
        make_config(),
        run_dir,
    )
    assert paths == [] or all(p.exists() for p in paths)
    assert not list(run_dir.glob("*.tmp"))


# --------------------------------------------------------------------------
# cluster_transfer_report
# --------------------------------------------------------------------------


def test_transfer_report_contract(case, make_config, run_dir):
    rows, clusters, names, annotations = case
    report = cluster_transfer_report(rows, clusters, annotations, names, make_config(), run_dir)
    assert list(report.columns) == [
        "cluster",
        "name",
        "n_ref_bins",
        "n_asm_bins",
        "ref_top_feature",
        "asm_top_feature",
        "asm_agreement",
        "asm_annotated_frac",
    ]
    # A row for cluster -1 is allowed (it is a diagnostic table); the named
    # clusters must all be there.
    assert set(report["cluster"]) >= {0, 1}
    # NaN is allowed where a cluster has no annotated assembly bin to compare
    assert report["asm_agreement"].dropna().between(0.0, 1.0).all()
    assert report["asm_annotated_frac"].dropna().between(0.0, 1.0).all()
    named = report[report["cluster"] >= 0]
    assert named["asm_agreement"].notna().all()


def test_transfer_report_separates_reference_from_assembly_bins(case, make_config, run_dir):
    rows, clusters, names, annotations = case
    report = cluster_transfer_report(rows, clusters, annotations, names, make_config(), run_dir)
    row = report[report["cluster"] == 0].iloc[0]
    assert int(row.n_ref_bins) == 4  # chm13 bins 0-3
    assert int(row.n_asm_bins) == 8  # A_pat + B_pat bins 0-3
    assert row.ref_top_feature == "hsat2"
    assert row.asm_top_feature == "hsat2"
    assert row.asm_agreement == pytest.approx(1.0)
    assert row.asm_annotated_frac == pytest.approx(1.0)


def test_transfer_report_detects_disagreement(case, make_config, run_dir):
    """If assembly bins in a reference-named cluster look different, say so."""
    rows, clusters, names, annotations = case
    broken = annotations.copy()
    asm_mask = (~rows["assembly"].eq("chm13v2.0")).to_numpy()
    c0_mask = (clusters["cluster"] == 0).to_numpy()
    target = asm_mask & c0_mask
    broken.loc[target, "frac_hsat2"] = np.float32(0.0)
    broken.loc[target, "frac_bsat"] = np.float32(0.8)
    broken.loc[target, "dominant_feature"] = "bsat"
    report = cluster_transfer_report(rows, clusters, broken, names, make_config(), run_dir)
    row = report[report["cluster"] == 0].iloc[0]
    assert row.ref_top_feature == "hsat2"
    assert row.asm_top_feature == "bsat"
    assert row.asm_agreement < 0.5


def test_transfer_report_with_no_reference_rows(case, make_config, run_dir):
    rows, clusters, names, annotations = case
    keep = (rows["assembly"] != "chm13v2.0").to_numpy()
    report = cluster_transfer_report(
        rows[keep].reset_index(drop=True),
        clusters[keep].reset_index(drop=True),
        annotations[keep].reset_index(drop=True),
        names,
        make_config(),
        run_dir,
    )
    assert (report["n_ref_bins"] == 0).all()
    assert np.isfinite(report["asm_agreement"].fillna(0.0)).all()


# --------------------------------------------------------------------------
# Reference-only inference: the actual product. Annotate CHM13, let cluster
# membership carry the labels onto assemblies that were never aligned.
# --------------------------------------------------------------------------


def _ref_and_asm(make_config, n_ref=6, n_asm=30, feature="hsat3"):
    """One cluster: a few annotated reference bins, many unannotated assembly bins."""
    import numpy as np

    uids, source, hap = [], [], []
    for i in range(n_ref):
        uids.append(f"chm13v2.0|chr21|{i * 10000}")
        source.append("t2t")
        hap.append("ref")
    for i in range(n_asm):
        uids.append(f"HG1_pat|HG1#1#c|{i * 10000}")
        source.append("hprc")
        hap.append("pat")
    rows = pd.DataFrame({"bin_uid": uids, "source": source, "haplotype": hap})
    clusters = pd.DataFrame(
        {
            "bin_uid": uids,
            "cluster": np.zeros(len(uids), dtype=np.int32),
        }
    )
    ann = pd.DataFrame(
        {
            "bin_uid": uids,
            # Only the reference is annotated -- annotate_assemblies is off.
            "dominant_feature": [feature] * n_ref + ["unannotated"] * n_asm,
        }
    )
    return rows, clusters, ann


def test_assembly_bins_inherit_the_reference_label(make_config, run_dir):
    from kmer_dust.backprop import infer_annotations

    rows, clusters, ann = _ref_and_asm(make_config)
    out = infer_annotations(rows, clusters, ann, make_config(), run_dir, force=True)

    assert list(out.columns) == list(__import__("kmer_dust.backprop", fromlist=["INFERRED_COLUMNS"]).INFERRED_COLUMNS)
    assert len(out) == len(rows)
    assert (out["inferred_feature"] == "hsat3").all()
    assert (out["support"] == 6).all()
    assert (out["cluster_ref_bins"] == 6).all()
    assert out["purity"].to_numpy() == pytest.approx(1.0)
    assert not out["novel"].any()


def test_a_cluster_with_no_reference_bin_is_flagged_novel(make_config, run_dir):
    """Sequence the assemblies share with each other and not with CHM13.

    That is a result, not a gap: it is the material a reference-based method
    cannot see at all.
    """
    import numpy as np

    from kmer_dust.backprop import infer_annotations

    rows, clusters, ann = _ref_and_asm(make_config)
    # Move the reference bins into their own cluster, leaving cluster 0 all-assembly.
    is_ref = rows["source"].eq("t2t").to_numpy()
    clusters["cluster"] = np.where(is_ref, 1, 0).astype(np.int32)

    out = infer_annotations(rows, clusters, ann, make_config(), run_dir, force=True)
    merged = out.merge(rows, on="bin_uid")

    asm = merged[merged["source"] != "t2t"]
    assert asm["novel"].all(), "a cluster with no reference bin must be flagged"
    assert (asm["inferred_feature"] == "").all(), "and must not be given a label"
    ref = merged[merged["source"] == "t2t"]
    assert not ref["novel"].any()


def test_noise_is_neither_labelled_nor_novel(make_config, run_dir):
    import numpy as np

    from kmer_dust.backprop import infer_annotations

    rows, clusters, ann = _ref_and_asm(make_config)
    clusters["cluster"] = np.full(len(rows), -1, dtype=np.int32)
    out = infer_annotations(rows, clusters, ann, make_config(), run_dir, force=True)
    assert (out["inferred_feature"] == "").all()
    assert not out["novel"].any(), "noise is not a cluster; it cannot be novel"
    assert (out["support"] == 0).all()


def test_only_reference_bins_vote(make_config, run_dir):
    """Even when assembly annotations exist, inference must ignore them.

    Otherwise the table is a lookup of what we already knew rather than an
    inference, and `cluster_transfer_report` would be scoring itself.
    """
    from kmer_dust.backprop import infer_annotations

    rows, clusters, ann = _ref_and_asm(make_config, n_ref=3, n_asm=30, feature="hsat3")
    # The assemblies loudly claim something else, and are outnumbered 10:1.
    ann.loc[ann["bin_uid"].str.startswith("HG1_"), "dominant_feature"] = "line"

    out = infer_annotations(rows, clusters, ann, make_config(), run_dir, force=True)
    assert (out["inferred_feature"] == "hsat3").all()
    assert (out["cluster_ref_bins"] == 3).all()


def test_purity_reports_a_split_reference_vote(make_config, run_dir):
    from kmer_dust.backprop import infer_annotations

    rows, clusters, ann = _ref_and_asm(make_config, n_ref=4, n_asm=10)
    ann.loc[:1, "dominant_feature"] = "hsat3"
    ann.loc[2:3, "dominant_feature"] = "hsat3"
    ann.loc[3, "dominant_feature"] = "bsat"  # 3 hsat3 : 1 bsat

    out = infer_annotations(rows, clusters, ann, make_config(), run_dir, force=True)
    assert (out["inferred_feature"] == "hsat3").all()
    assert (out["support"] == 3).all()
    assert (out["cluster_ref_bins"] == 4).all()
    assert out["purity"].iloc[0] == pytest.approx(0.75)


def test_inference_round_trips_through_disk(make_config, run_dir):
    from kmer_dust.backprop import infer_annotations

    rows, clusters, ann = _ref_and_asm(make_config)
    first = infer_annotations(rows, clusters, ann, make_config(), run_dir, force=True)
    assert (run_dir / "inferred.parquet").exists()
    again = infer_annotations(rows, clusters, ann, make_config(), run_dir, force=False)
    pd.testing.assert_frame_equal(first, again)
