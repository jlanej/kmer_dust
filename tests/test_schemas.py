"""On-disk contract helpers: dtype coercion, empty frames, bin identifiers.

Every stage hands its output through :func:`schemas.enforce` before writing, so
a dtype bug here becomes a parquet file that a later stage silently misreads.
The ``bin_uid`` tests matter for a specific real-world reason: HPRC contigs are
PanSN (``HG00408#1#CM085953.1``) and fetched slices carry region suffixes
(``...:1000000-3000000``), so the identifier must survive ``#``, ``.``, ``:``
and ``-`` in the contig name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kmer_dust import schemas

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------


def test_feature_vocab_is_ordered_and_unique():
    assert schemas.FEATURE_VOCAB == (
        schemas.CENSAT_CLASSES + schemas.REPEAT_CLASSES + schemas.EXTRA_FEATURES
    )
    assert len(set(schemas.FEATURE_VOCAB)) == len(schemas.FEATURE_VOCAB)
    assert all(f == f.lower() for f in schemas.FEATURE_VOCAB)


def test_feature_columns_track_the_vocab():
    assert schemas.feature_column("hsat2") == "frac_hsat2"
    assert schemas.FEATURE_COLUMNS == tuple(f"frac_{f}" for f in schemas.FEATURE_VOCAB)
    assert len(schemas.FEATURE_COLUMNS) == len(schemas.FEATURE_VOCAB)


def test_hash_constants():
    assert schemas.HASH_DTYPE is np.uint64
    assert schemas.HASH_MAX == (1 << 64) - 1
    assert np.uint64(schemas.HASH_MAX) == np.iinfo(np.uint64).max


def test_manifest_required_is_a_subset_of_manifest_columns():
    assert set(schemas.MANIFEST_REQUIRED) <= set(schemas.MANIFEST_COLUMNS)


# --------------------------------------------------------------------------
# bin_uid
# --------------------------------------------------------------------------


BIN_UID_CASES = [
    ("chm13v2.0", "chr21", 0),
    ("HG00408_pat_hprc_r2_v1.0.1", "HG00408#1#CM085953.1", 10_000),
    # a fetched slice keeps its region suffix in the contig name
    ("HG00408_pat_hprc_r2_v1.0.1", "HG00408#1#CM085953.1:1000000-3000000", 250_000),
    ("weird", "contig.with.dots_and-dashes", 999_999_999),
    ("weird", "colon:in:the:middle", 42),
]


@pytest.mark.parametrize("assembly,contig,start", BIN_UID_CASES)
def test_bin_uid_round_trip(assembly, contig, start):
    uid = schemas.bin_uid(assembly, contig, start)
    assert uid.count("|") == 2
    assert schemas.parse_bin_uid(uid) == (assembly, contig, start)


def test_bin_uid_is_unique_per_bin():
    uids = {schemas.bin_uid("a", "chr1", s) for s in (0, 10_000, 20_000)}
    assert len(uids) == 3
    assert schemas.bin_uid("a", "chr1", 0) != schemas.bin_uid("b", "chr1", 0)


def test_parse_bin_uid_rejects_garbage():
    with pytest.raises(ValueError):
        schemas.parse_bin_uid("no-separators-here")


# --------------------------------------------------------------------------
# enforce / empty_frame
# --------------------------------------------------------------------------


def _bin_row(**over):
    row = {
        "bin_idx": 0,
        "bin_uid": "a|chr1|0",
        "assembly": "a",
        "sample": "a",
        "haplotype": "pat",
        "source": "local",
        "contig": "chr1",
        "chrom": "chr1",
        "start": 0,
        "end": 10_000,
        "n_acgt": 9_000,
        "n_kmers": 8_970,
        "n_sketch": 45,
        "gc": 0.41,
        "nfrac": 0.1,
    }
    row.update(over)
    return row


def _assert_dtypes(df: pd.DataFrame, columns: dict[str, str]) -> None:
    for col, dtype in columns.items():
        if dtype == "binary":
            continue
        assert str(df[col].dtype) == dtype, f"{col}: {df[col].dtype} != {dtype}"


@pytest.mark.parametrize(
    "columns",
    [
        schemas.MANIFEST_COLUMNS,
        schemas.BIN_COLUMNS,
        schemas.SKETCH_COLUMNS,
        schemas.KMER_COLUMNS,
        schemas.PREVALENCE_HIST_COLUMNS,
        schemas.CLUSTER_COLUMNS,
        schemas.ANNOTATION_ID_COLUMNS,
        schemas.ENRICHMENT_COLUMNS,
        schemas.CLUSTER_NAME_COLUMNS,
        schemas.BUCKET_COLUMNS,
    ],
    ids=lambda c: ",".join(list(c)[:2]),
)
def test_empty_frame_has_contract_dtypes_and_columns(columns):
    df = schemas.empty_frame(columns)
    assert list(df.columns) == list(columns)
    assert len(df) == 0
    _assert_dtypes(df, columns)


def test_empty_frame_round_trips_through_parquet(tmp_path):
    """An empty stage output must still be readable by the next stage."""
    for name, columns in (
        ("bins", schemas.BIN_COLUMNS),
        ("kmers", schemas.KMER_COLUMNS),
        ("clusters", schemas.CLUSTER_COLUMNS),
    ):
        path = tmp_path / f"{name}.parquet"
        schemas.empty_frame(columns).to_parquet(path, index=False)
        back = pd.read_parquet(path)
        assert list(back.columns) == list(columns)
        _assert_dtypes(back, columns)


def test_enforce_coerces_dtypes_and_orders_columns():
    df = pd.DataFrame([_bin_row()])
    scrambled = df[list(reversed(df.columns))]
    # everything arriving as object/float, the way a hand-built frame does
    scrambled = scrambled.astype(object)
    out = schemas.enforce(scrambled, schemas.BIN_COLUMNS)
    assert list(out.columns) == list(schemas.BIN_COLUMNS)
    _assert_dtypes(out, schemas.BIN_COLUMNS)


def test_enforce_uses_pandas_string_dtype_not_object():
    out = schemas.enforce(pd.DataFrame([_bin_row()]), schemas.BIN_COLUMNS)
    assert out["contig"].dtype == "string"
    assert out["contig"].dtype != object


def test_enforce_fills_missing_text_with_empty_string():
    out = schemas.enforce(pd.DataFrame([_bin_row(chrom=None)]), schemas.BIN_COLUMNS)
    assert out.loc[0, "chrom"] == ""
    assert out["chrom"].notna().all()


def test_enforce_drops_extra_columns_by_default():
    df = pd.DataFrame([_bin_row()])
    df["scratch"] = 1
    out = schemas.enforce(df, schemas.BIN_COLUMNS)
    assert "scratch" not in out.columns


def test_enforce_subset_keeps_extras_after_the_contract_columns():
    df = pd.DataFrame([_bin_row()])
    df["row_idx"] = 7
    out = schemas.enforce(df, schemas.BIN_COLUMNS, subset=True)
    assert list(out.columns) == list(schemas.BIN_COLUMNS) + ["row_idx"]
    assert out.loc[0, "row_idx"] == 7


def test_enforce_raises_on_missing_columns():
    df = pd.DataFrame([_bin_row()]).drop(columns=["gc", "nfrac"])
    with pytest.raises(ValueError, match="missing required columns"):
        schemas.enforce(df, schemas.BIN_COLUMNS)


def test_enforce_does_not_mutate_its_input():
    df = pd.DataFrame([_bin_row()]).astype({"bin_idx": "int64"})
    before = df.copy(deep=True)
    schemas.enforce(df, schemas.BIN_COLUMNS)
    pd.testing.assert_frame_equal(df, before)


def test_enforce_on_an_empty_frame_keeps_the_dtypes():
    out = schemas.enforce(schemas.empty_frame(schemas.CLUSTER_COLUMNS), schemas.CLUSTER_COLUMNS)
    assert len(out) == 0
    _assert_dtypes(out, schemas.CLUSTER_COLUMNS)


def test_enforce_preserves_full_uint64_hashes():
    """A hash near 2**64 must not be silently truncated or turned into a float."""
    big = schemas.HASH_MAX
    df = pd.DataFrame({"bin_idx": [0], "hash": pd.array([big], dtype="uint64")})
    out = schemas.enforce(df, schemas.SKETCH_COLUMNS)
    assert out["hash"].dtype == "uint64"
    assert int(out.loc[0, "hash"]) == big


def test_cluster_noise_label_fits_int32():
    df = pd.DataFrame(
        {
            "row_idx": [0, 1],
            "bin_uid": ["a|chr1|0", "a|chr1|10000"],
            "cluster": [-1, 3],
            "probability": [0.0, 0.9],
            "outlier_score": [1.0, 0.1],
        }
    )
    out = schemas.enforce(df, schemas.CLUSTER_COLUMNS)
    assert out["cluster"].tolist() == [-1, 3]
    assert str(out["cluster"].dtype) == "int32"
