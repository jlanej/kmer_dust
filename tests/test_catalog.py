"""Catalog: URL rewriting, T2T constants and manifest assembly/filters.

Everything here that touches the network is marked; the rest exercises the
offline half of the catalog -- the ``file`` manifest source, the filter
semantics in :class:`ManifestConfig`, and the TSV round trip that lets a manifest
be edited by hand and fed back in.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from kmer_dust import schemas
from kmer_dust.catalog import hprc, t2t
from kmer_dust.catalog.manifest import build_manifest, read_manifest, write_manifest

S3 = "https://s3-us-west-2.amazonaws.com/human-pangenomics"


# --------------------------------------------------------------------------
# hprc constants and helpers
# --------------------------------------------------------------------------


def test_s3_to_https():
    uri = "s3://human-pangenomics/working/HPRC/HG00408/assemblies/release2/x.fa.gz"
    assert hprc.s3_to_https(uri) == f"{S3}/working/HPRC/HG00408/assemblies/release2/x.fa.gz"


def test_s3_to_https_passes_through_https_and_empty():
    assert hprc.s3_to_https("") == ""
    url = f"{S3}/T2T/CHM13/assemblies/chm13v2.0.fa"
    assert hprc.s3_to_https(url) == url


def test_data_tables_base_points_at_the_hprc_repo():
    assert hprc.HPRC_DATA_TABLES_BASE.startswith("https://")
    assert "hprc_intermediate_assembly" in hprc.HPRC_DATA_TABLES_BASE
    assert hprc.HPRC_DATA_TABLES_BASE.endswith("/")


def test_population_to_superpopulation_is_sane():
    table = hprc.POPULATION_TO_SUPERPOPULATION
    assert set(table.values()) <= {"AFR", "AMR", "EAS", "EUR", "SAS"}
    # a handful of 1000G populations that HPRC release 2 definitely contains
    for pop, sup in (("YRI", "AFR"), ("CHS", "EAS"), ("GBR", "EUR"), ("PJL", "SAS")):
        assert table[pop] == sup


# --------------------------------------------------------------------------
# t2t constants
# --------------------------------------------------------------------------


def test_t2t_constants():
    assert t2t.T2T_ASSEMBLY == "chm13v2.0"
    assert t2t.T2T_FASTA.startswith("https://")
    assert t2t.T2T_FASTA.endswith(".fa"), "the bgzip build has no .fa.gz.fai; use the plain FASTA"
    assert t2t.T2T_FAI == t2t.T2T_FASTA + ".fai"


def test_t2t_tracks_cover_the_annotate_vocabulary():
    tracks = t2t.T2T_TRACKS
    assert set(tracks) >= {"censat", "repeatmasker", "telomere", "gene"}
    assert "segdup" in tracks  # may legitimately be "" if none was found
    for name, url in tracks.items():
        assert url == "" or url.startswith("https://"), name


def test_reference_manifest_row_matches_the_manifest_contract():
    row = t2t.reference_manifest_row()
    assert set(row) == set(schemas.MANIFEST_COLUMNS)
    assert row["assembly"] == t2t.T2T_ASSEMBLY
    assert row["source"] == "t2t"
    assert row["haplotype"] == "ref"
    assert row["fasta"] == t2t.T2T_FASTA
    assert all(isinstance(v, str) for v in row.values())
    # it must survive schema enforcement as a one-row frame
    df = schemas.enforce(pd.DataFrame([row]), schemas.MANIFEST_COLUMNS)
    assert len(df) == 1


# --------------------------------------------------------------------------
# manifest I/O
# --------------------------------------------------------------------------


def _toy_manifest(n_samples: int = 3) -> pd.DataFrame:
    rows = []
    for s in range(n_samples):
        sample = f"HG{s:05d}"
        for hap in ("pat", "mat"):
            rows.append(
                {
                    "assembly": f"{sample}_{hap}_hprc_r2_v1.0.1",
                    "sample": sample,
                    "haplotype": hap,
                    "source": "hprc",
                    "fasta": f"{S3}/{sample}_{hap}.fa.gz",
                    "fai": f"{S3}/{sample}_{hap}.fa.gz.fai",
                    "gzi": f"{S3}/{sample}_{hap}.fa.gz.gzi",
                    "chrom_alias": "",
                    "censat_bed": f"{S3}/{sample}_{hap}.censat.bed",
                    "repeatmasker_bed": "",
                    "segdup_bed": "",
                    "population": ["YRI", "GBR", "CHS"][s % 3],
                    "superpopulation": ["AFR", "EUR", "EAS"][s % 3],
                    "sex": "female" if s % 2 else "male",
                }
            )
    return schemas.enforce(pd.DataFrame(rows), schemas.MANIFEST_COLUMNS)


def test_manifest_tsv_round_trip(tmp_path):
    df = _toy_manifest()
    path = write_manifest(df, tmp_path / "manifest.tsv")
    assert path.exists()
    assert path.read_text().splitlines()[0].split("\t") == list(schemas.MANIFEST_COLUMNS)
    back = read_manifest(path)
    pd.testing.assert_frame_equal(back, df)


def test_manifest_round_trip_preserves_empty_strings(tmp_path):
    """Empty optional columns must come back as '' and not as NaN."""
    df = _toy_manifest(1)
    path = write_manifest(df, tmp_path / "m.tsv")
    back = read_manifest(path)
    assert back["repeatmasker_bed"].tolist() == ["", ""]
    assert back["repeatmasker_bed"].dtype == "string"
    assert back.notna().all().all()


def test_read_manifest_of_an_empty_table(tmp_path):
    path = tmp_path / "empty.tsv"
    write_manifest(schemas.empty_frame(schemas.MANIFEST_COLUMNS), path)
    back = read_manifest(path)
    assert len(back) == 0
    assert list(back.columns) == list(schemas.MANIFEST_COLUMNS)


# --------------------------------------------------------------------------
# build_manifest, source=file (fully offline)
# --------------------------------------------------------------------------


@pytest.fixture
def file_manifest(tmp_path):
    path = tmp_path / "input.tsv"
    write_manifest(_toy_manifest(), path)
    return path


def test_build_manifest_from_file(file_manifest, make_config, cache_dir):
    cfg = make_config(manifest={"source": "file", "path": str(file_manifest)})
    df = build_manifest(cfg, cache_dir)
    assert list(df.columns) == list(schemas.MANIFEST_COLUMNS)
    assert len(df) == 6
    assert df["assembly"].is_unique
    for col, dtype in schemas.MANIFEST_COLUMNS.items():
        assert str(df[col].dtype) == dtype, col


def test_build_manifest_is_deterministic(file_manifest, make_config, cache_dir):
    cfg = make_config(manifest={"source": "file", "path": str(file_manifest)})
    a = build_manifest(cfg, cache_dir)
    b = build_manifest(cfg, cache_dir)
    pd.testing.assert_frame_equal(a, b)


def test_reference_row_comes_first(file_manifest, make_config, cache_dir):
    cfg = make_config(
        manifest={"source": "file", "path": str(file_manifest), "include_reference": True}
    )
    df = build_manifest(cfg, cache_dir)
    assert df.iloc[0]["assembly"] == t2t.T2T_ASSEMBLY
    assert df.iloc[0]["source"] == "t2t"
    assert len(df) == 7


def test_max_samples_caps_samples_not_haplotypes(file_manifest, make_config, cache_dir):
    cfg = make_config(
        manifest={"source": "file", "path": str(file_manifest), "max_samples": 2}
    )
    df = build_manifest(cfg, cache_dir)
    assert df["sample"].nunique() == 2
    assert len(df) == 4


def test_max_assemblies_caps_rows(file_manifest, make_config, cache_dir):
    cfg = make_config(
        manifest={"source": "file", "path": str(file_manifest), "max_assemblies": 3}
    )
    assert len(build_manifest(cfg, cache_dir)) == 3


def test_sample_allow_and_deny_lists(file_manifest, make_config, cache_dir):
    cfg = make_config(
        manifest={"source": "file", "path": str(file_manifest), "samples": ["HG00000", "HG00002"]}
    )
    df = build_manifest(cfg, cache_dir)
    assert set(df["sample"]) == {"HG00000", "HG00002"}

    cfg = make_config(
        manifest={"source": "file", "path": str(file_manifest), "exclude_samples": ["HG00001"]}
    )
    df = build_manifest(cfg, cache_dir)
    assert "HG00001" not in set(df["sample"])
    assert len(df) == 4


def test_population_filter(file_manifest, make_config, cache_dir):
    cfg = make_config(
        manifest={"source": "file", "path": str(file_manifest), "populations": ["YRI"]}
    )
    df = build_manifest(cfg, cache_dir)
    assert set(df["population"]) == {"YRI"}


def test_require_annotations_drops_rows_without_the_track(
    tmp_path, file_manifest, make_config, cache_dir
):
    df = read_manifest(file_manifest)
    df.loc[df.index[:2], "censat_bed"] = ""
    path = tmp_path / "partial.tsv"
    write_manifest(df, path)
    cfg = make_config(
        manifest={"source": "file", "path": str(path), "require_annotations": ["censat"]}
    )
    out = build_manifest(cfg, cache_dir)
    assert len(out) == 4
    assert (out["censat_bed"] != "").all()


def test_build_manifest_missing_file_raises(make_config, cache_dir, tmp_path):
    cfg = make_config(manifest={"source": "file", "path": str(tmp_path / "nope.tsv")})
    with pytest.raises((OSError, ValueError)):
        build_manifest(cfg, cache_dir)


def test_build_manifest_from_a_local_dir(tmp_path, make_config, cache_dir, synthetic_assemblies):
    synthetic_assemblies(n_assemblies=2, contig_len=20_000)
    cfg = make_config(
        manifest={"source": "local_dir", "path": str(tmp_path / "assemblies"),
                  "include_reference": False, "require_annotations": []}
    )
    df = build_manifest(cfg, cache_dir)
    assert len(df) == 2
    assert list(df.columns) == list(schemas.MANIFEST_COLUMNS)
    assert all(p.endswith(".fa") for p in df["fasta"])
    assert df["assembly"].tolist() == sorted(df["assembly"].tolist())


def test_build_manifest_of_an_empty_dir_is_empty_not_a_crash(tmp_path, make_config, cache_dir):
    empty = tmp_path / "nothing"
    empty.mkdir()
    cfg = make_config(
        manifest={"source": "local_dir", "path": str(empty), "include_reference": False}
    )
    df = build_manifest(cfg, cache_dir)
    assert len(df) == 0
    assert list(df.columns) == list(schemas.MANIFEST_COLUMNS)


def test_committed_testdata_manifest_is_readable():
    """``tests/testdata.manifest.tsv`` names the real assemblies the fetched
    test slices come from (see ``data/README.md``).  It is committed, so it can
    rot silently unless something parses it."""
    path = pathlib.Path(__file__).parent / "testdata.manifest.tsv"
    df = read_manifest(path)
    assert len(df) >= 6
    assert list(df.columns) == list(schemas.MANIFEST_COLUMNS)
    assert df["assembly"].is_unique
    assert df["sample"].nunique() >= 3
    assert (df["source"] == "hprc").all()
    for col in ("fasta", "fai", "gzi"):
        assert df[col].str.startswith("https://").all(), col
    assert df["fasta"].str.endswith(".fa.gz").all()
    assert (df["fai"] == df["fasta"] + ".fai").all()
    assert (df["gzi"] == df["fasta"] + ".gzi").all()


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------


@pytest.mark.network
def test_release2_index_is_two_haplotypes_per_sample(cache_dir):
    """The accessor returns a *normalised* frame, not the upstream CSV.

    The upstream table also carries two ``haplotype == 0`` rows (GRCh38 and a
    masked CHM13) which are references rather than haplotypes; kmer-dust brings
    its own reference row, so those are dropped. The count is asserted as a
    relationship rather than a magic number, because HPRC keeps adding samples.
    """
    df = hprc.release2_index(cache_dir)
    assert list(df.columns) == ["sample", "haplotype", "assembly", "fasta", "fai", "gzi"]
    assert len(df) > 400
    assert len(df) == 2 * df["sample"].nunique(), "every sample should have both haplotypes"
    assert set(df["haplotype"]) <= {"pat", "mat", "hap1", "hap2"}
    # Not every row is a release-2 assembly: the upstream index also carries a
    # handful of HPRC_PLUS entries (HG002 Q100 as `hg002v1.1.pat`, HG06807 as
    # `HG06807_{mat,pat}_v1`).  Verified live 2026-08-20: 460 of 464.
    hprc_r2 = df["assembly"].str.contains("hprc_r2").fillna(False)
    assert hprc_r2.sum() >= len(df) - 8
    assert df["assembly"].is_unique
    assert df["fasta"].str.startswith("https://").all()
    assert df["fasta"].str.startswith("https://").all()
    assert df["fai"].str.endswith(".fa.gz.fai").all()
    assert df["gzi"].str.endswith(".fa.gz.gzi").all()
    assert df["assembly"].is_unique


@pytest.mark.network
def test_release2_index_is_cached(cache_dir):
    first = hprc.release2_index(cache_dir)
    cached = sorted(p.name for p in cache_dir.rglob("*") if p.is_file())
    assert cached, "nothing was written to the cache directory"
    # a second call must not need the network at all
    again = hprc.release2_index(cache_dir)
    assert len(again) == len(first)


@pytest.mark.network
@pytest.mark.parametrize("kind", ["censat", "repeatmasker", "segdup", "chrom_alias"])
def test_annotation_indexes(cache_dir, kind):
    df = hprc.annotation_index(kind, cache_dir)
    assert list(df.columns) == ["sample", "assembly", "url"]
    assert len(df) > 400
    assert df["url"].str.startswith("https://").all()
    assert df["assembly"].is_unique


@pytest.mark.network
def test_sample_metadata_survives_quoted_commas(cache_dir):
    """One column is free text containing quoted commas; a naive split shears
    the table sideways and every downstream population label becomes garbage."""
    df = hprc.sample_metadata(cache_dir)
    assert list(df.columns) == [
        "sample",
        "population",
        "superpopulation",
        "sex",
        "trio_available",
    ]
    assert len(df) >= 230
    assert df["sample"].is_unique
    assert set(df["sex"]) <= {"male", "female", ""}
    assert set(df["superpopulation"]) <= {"AFR", "AMR", "EAS", "EUR", "SAS", ""}
    # the shear failure mode shows up as populations that are really fragments
    # of the free-text descriptor, so demand that most rows resolved cleanly
    assert (df["superpopulation"] != "").mean() > 0.8


@pytest.mark.network
def test_t2t_sequence_table_points_at_per_assembly_tsvs(cache_dir):
    df = hprc.t2t_sequence_table(cache_dir)
    assert list(df.columns) == ["sample", "assembly", "url"]
    assert len(df) > 400
    assert df["url"].str.endswith(".t2t_chromosomes.tsv").all()


@pytest.mark.network
def test_complete_chromosomes_reads_one_haplotype(cache_dir):
    """The T2T-completeness filter is only as good as this parse."""
    index = hprc.t2t_sequence_table(cache_dir)
    url = index.iloc[0]["url"]
    complete = hprc.complete_chromosomes(url, cache_dir)
    assert isinstance(complete, frozenset)
    assert all(c.startswith("chr") for c in complete)
    # a release-2 haplotype has at least a handful of gapless chromosomes
    assert len(complete) >= 1
