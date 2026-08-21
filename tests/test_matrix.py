"""Assembling the bin x k-mer matrix: row order, weighting and normalisation.

The weighting tests compute the answer densely with NumPy from first principles
and compare against the sparse result, because an off-by-one in the IDF
denominator or a norm applied along the wrong axis produces a matrix that is
still *plausible* -- SVD and UMAP will happily consume it and give a wrong map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import manifest_from_shards, write_sketch_shard
from scipy import sparse

from kmer_dust import schemas
from kmer_dust.matrix import build_matrix, load_matrix

K0, K1, K2 = 0x1111111111111111, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA
NOISE = 0xDEADBEEFDEADBEEF  # present in a shard but not in the feature set


def make_kmers(hashes, n_bins) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "hash": pd.array(list(hashes), dtype="uint64"),
            "col_idx": np.arange(len(hashes)),
            "n_samples": np.ones(len(hashes)),
            "n_assemblies": np.ones(len(hashes)),
            "n_bins": list(n_bins),
        }
    )
    return schemas.enforce(df.sort_values("hash"), schemas.KMER_COLUMNS)


@pytest.fixture
def toy(tmp_path):
    """Four non-empty bins over two assemblies; df = [3, 3, 1] for [K0, K1, K2]."""
    sketch_dir = tmp_path / "sketch"
    write_sketch_shard(sketch_dir, "A_pat", [[K0, K1], [K0, K2, NOISE]], sample="A")
    write_sketch_shard(sketch_dir, "B_pat", [[K0, K1], [K1]], sample="B")
    manifest = manifest_from_shards([("A_pat", "A"), ("B_pat", "B")])
    kmers = make_kmers([K0, K1, K2], [3, 3, 1])
    return sketch_dir, manifest, kmers


def dense_reference(weighting: str, row_norm: str) -> np.ndarray:
    """Independent, dense implementation of the documented value pipeline."""
    binary = np.array(
        [
            [1.0, 1.0, 0.0],  # A_pat bin 0
            [1.0, 0.0, 1.0],  # A_pat bin 1
            [1.0, 1.0, 0.0],  # B_pat bin 0
            [0.0, 1.0, 0.0],  # B_pat bin 1
        ]
    )
    values = binary.copy()
    if weighting == "idf":
        df = np.array([3.0, 3.0, 1.0])
        values *= np.log(binary.shape[0] / df)
    elif weighting == "log":
        values *= np.log1p(1.0)
    if row_norm == "l2":
        norms = np.sqrt((values**2).sum(axis=1, keepdims=True))
    elif row_norm == "l1":
        norms = np.abs(values).sum(axis=1, keepdims=True)
    else:
        norms = np.ones((values.shape[0], 1))
    norms[norms == 0] = 1.0
    return values / norms


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def test_matrix_shape_dtype_and_row_table(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "none", "row_norm": "none"})
    mat, rows = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    assert sparse.isspmatrix_csr(mat)
    assert mat.dtype == np.float32
    assert mat.shape == (4, 3)
    assert list(rows.columns) == list(schemas.BIN_COLUMNS) + ["row_idx"]
    assert str(rows["row_idx"].dtype) == "int64"
    for col, dtype in schemas.BIN_COLUMNS.items():
        assert str(rows[col].dtype) == dtype, col
    assert rows["row_idx"].tolist() == [0, 1, 2, 3]
    assert (run_dir / "matrix.npz").exists()
    assert (run_dir / "rows.parquet").exists()


def test_rows_follow_manifest_order_then_bin_idx(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "none", "row_norm": "none"})
    _, rows = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    assert rows["assembly"].tolist() == ["A_pat", "A_pat", "B_pat", "B_pat"]
    assert rows["bin_idx"].tolist() == [0, 1, 0, 1]

    reversed_manifest = manifest.iloc[::-1].reset_index(drop=True)
    other = run_dir / "rev"
    other.mkdir()
    _, rows2 = build_matrix(sketch_dir, kmers, reversed_manifest, cfg, other)
    assert rows2["assembly"].tolist() == ["B_pat", "B_pat", "A_pat", "A_pat"]
    assert rows2["row_idx"].tolist() == [0, 1, 2, 3]


def test_binary_incidence_ignores_unselected_hashes(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "none", "row_norm": "none"})
    mat, _ = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    assert np.array_equal(mat.toarray(), dense_reference("none", "none").astype(np.float32))
    assert mat.nnz == 7  # NOISE must not appear anywhere


def test_column_order_follows_kmer_col_idx(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "none", "row_norm": "none"})
    mat, _ = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    dense = mat.toarray()
    order = {int(h): int(c) for h, c in zip(kmers["hash"], kmers["col_idx"])}
    assert dense[1, order[K2]] == 1.0
    assert dense[1, order[K1]] == 0.0


# --------------------------------------------------------------------------
# weighting and normalisation, against a dense hand computation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weighting", ["none", "idf", "log"])
@pytest.mark.parametrize("row_norm", ["none", "l1", "l2"])
def test_values_match_the_dense_reference(toy, make_config, tmp_path, weighting, row_norm):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": weighting, "row_norm": row_norm})
    out = tmp_path / f"m_{weighting}_{row_norm}"
    out.mkdir()
    mat, _ = build_matrix(sketch_dir, kmers, manifest, cfg, out)
    expected = dense_reference(weighting, row_norm)
    np.testing.assert_allclose(mat.toarray(), expected, rtol=1e-6, atol=1e-7)


def test_idf_specific_values(toy, make_config, run_dir):
    """log(n_rows / df): the two df=3 columns must be lighter than the df=1 one."""
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "idf", "row_norm": "none"})
    mat, _ = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    dense = mat.toarray()
    order = {int(h): int(c) for h, c in zip(kmers["hash"], kmers["col_idx"])}
    assert dense[0, order[K0]] == pytest.approx(np.log(4 / 3), rel=1e-6)
    assert dense[1, order[K2]] == pytest.approx(np.log(4 / 1), rel=1e-6)
    assert dense[1, order[K2]] > dense[0, order[K0]]


def test_l2_rows_are_unit_length(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "idf", "row_norm": "l2"})
    mat, _ = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1))).ravel()
    np.testing.assert_allclose(norms, np.ones(4), rtol=1e-6)


def test_l1_rows_sum_to_one(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "none", "row_norm": "l1"})
    mat, _ = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    sums = np.asarray(abs(mat).sum(axis=1)).ravel()
    np.testing.assert_allclose(sums, np.ones(4), rtol=1e-6)


# --------------------------------------------------------------------------
# empty rows
# --------------------------------------------------------------------------


@pytest.fixture
def toy_with_empty_bin(tmp_path):
    sketch_dir = tmp_path / "sketch_empty"
    write_sketch_shard(sketch_dir, "A_pat", [[K0, K1], [NOISE], [K0, K2]], sample="A")
    write_sketch_shard(sketch_dir, "B_pat", [[K1]], sample="B")
    manifest = manifest_from_shards([("A_pat", "A"), ("B_pat", "B")])
    return sketch_dir, manifest, make_kmers([K0, K1, K2], [2, 2, 1])


def test_empty_rows_are_dropped_and_row_idx_renumbered(toy_with_empty_bin, make_config, run_dir):
    sketch_dir, manifest, kmers = toy_with_empty_bin
    cfg = make_config(matrix={"weighting": "none", "row_norm": "l2", "drop_empty_rows": True})
    mat, rows = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    assert mat.shape == (3, 3)
    assert len(rows) == 3
    assert rows["row_idx"].tolist() == [0, 1, 2]
    assert rows["bin_idx"].tolist() == [0, 2, 0]
    assert mat.getnnz(axis=1).min() > 0
    # row_idx must still index the matrix after renumbering
    assert rows["bin_uid"].tolist() == [
        schemas.bin_uid("A_pat", "chr21", 0),
        schemas.bin_uid("A_pat", "chr21", 20_000),
        schemas.bin_uid("B_pat", "chr21", 0),
    ]


def test_empty_rows_can_be_kept(toy_with_empty_bin, make_config, run_dir):
    sketch_dir, manifest, kmers = toy_with_empty_bin
    cfg = make_config(matrix={"weighting": "idf", "row_norm": "l2", "drop_empty_rows": False})
    mat, rows = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    assert mat.shape == (4, 3)
    assert len(rows) == 4
    dense = mat.toarray()
    assert not np.isnan(dense).any(), "normalising an all-zero row must not divide by zero"
    assert dense[1].sum() == 0.0


# --------------------------------------------------------------------------
# persistence, determinism, edge cases
# --------------------------------------------------------------------------


def test_load_matrix_round_trip(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "idf", "row_norm": "l2"})
    mat, rows = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    mat2, rows2 = load_matrix(run_dir)
    assert (mat != mat2).nnz == 0
    assert mat2.dtype == np.float32
    pd.testing.assert_frame_equal(rows, rows2)


def test_rebuild_is_identical(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config(matrix={"weighting": "idf", "row_norm": "l2"})
    a, rows_a = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir, force=True)
    b, rows_b = build_matrix(sketch_dir, kmers, manifest, cfg, run_dir, force=True)
    assert (a != b).nnz == 0
    pd.testing.assert_frame_equal(rows_a, rows_b)


def test_rerun_short_circuits(toy, make_config, run_dir):
    sketch_dir, manifest, kmers = toy
    cfg = make_config()
    build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    before = (run_dir / "matrix.npz").stat().st_mtime_ns
    build_matrix(sketch_dir, kmers, manifest, cfg, run_dir)
    assert (run_dir / "matrix.npz").stat().st_mtime_ns == before


def test_no_features_gives_a_zero_width_matrix(toy, make_config, run_dir):
    sketch_dir, manifest, _ = toy
    empty = schemas.empty_frame(schemas.KMER_COLUMNS)
    cfg = make_config(matrix={"drop_empty_rows": False})
    mat, rows = build_matrix(sketch_dir, empty, manifest, cfg, run_dir)
    assert mat.shape[1] == 0
    assert mat.nnz == 0
    assert len(rows) == mat.shape[0]


def test_no_bins_gives_a_zero_height_matrix(tmp_path, make_config, run_dir):
    sketch_dir = tmp_path / "no_bins"
    write_sketch_shard(sketch_dir, "A_pat", [], sample="A")
    manifest = manifest_from_shards([("A_pat", "A")])
    kmers = make_kmers([K0, K1], [1, 1])
    mat, rows = build_matrix(sketch_dir, kmers, manifest, make_config(), run_dir)
    assert mat.shape[0] == 0
    assert len(rows) == 0
    assert list(rows.columns) == list(schemas.BIN_COLUMNS) + ["row_idx"]
