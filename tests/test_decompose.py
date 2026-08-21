"""Randomized SVD stage: shapes, determinism and degenerate matrices.

``decompose`` is where a 465-haplotype run turns an out-of-core sparse matrix
into something UMAP can hold, so it must (a) be reproducible from ``cfg.seed``
alone -- randomized SVD draws a random test matrix, and an unseeded draw makes
every rerun of the pipeline produce a different map -- and (b) survive the
matrices that real filters can leave behind: one row, one column, all zeros.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from kmer_dust.decompose import decompose, load_pcs


def block_matrix(n_blocks=5, rows_per_block=40, cols_per_block=12, rng=None, density=0.6):
    """A block-structured incidence matrix: the SVD should recover the blocks."""
    rng = rng or np.random.default_rng(0)
    n_rows = n_blocks * rows_per_block
    n_cols = n_blocks * cols_per_block
    dense = np.zeros((n_rows, n_cols), dtype=np.float32)
    for b in range(n_blocks):
        r0, c0 = b * rows_per_block, b * cols_per_block
        block = (rng.random((rows_per_block, cols_per_block)) < density).astype(np.float32)
        dense[r0 : r0 + rows_per_block, c0 : c0 + cols_per_block] = block
    dense += (rng.random(dense.shape) < 0.01).astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return sparse.csr_matrix(dense / norms)


@pytest.fixture
def mat(rng):
    return block_matrix(rng=rng)


def test_shape_dtype_and_files(mat, make_config, run_dir):
    cfg = make_config(decompose={"n_components": 8, "n_iter": 4, "keep_components": True})
    pcs = decompose(mat, cfg, run_dir)
    assert isinstance(pcs, np.ndarray)
    assert pcs.dtype == np.float32
    assert pcs.shape == (mat.shape[0], 8)
    assert (run_dir / "pcs.npy").exists()
    assert (run_dir / "components.npy").exists()
    assert (run_dir / "svd.json").exists()
    components = np.load(run_dir / "components.npy")
    assert components.shape == (8, mat.shape[1])
    assert components.dtype == np.float32


def test_components_are_optional(mat, make_config, run_dir):
    cfg = make_config(decompose={"n_components": 6, "keep_components": False})
    decompose(mat, cfg, run_dir)
    assert not (run_dir / "components.npy").exists()


def test_svd_json_contract(mat, make_config, run_dir):
    cfg = make_config(decompose={"n_components": 8, "n_iter": 4})
    decompose(mat, cfg, run_dir)
    meta = json.loads((run_dir / "svd.json").read_text())
    assert set(meta) >= {"singular_values", "explained_variance_ratio", "n_components", "shape"}
    assert meta["n_components"] == 8
    assert tuple(meta["shape"]) == tuple(mat.shape)
    sv = meta["singular_values"]
    assert len(sv) == 8
    assert sv == sorted(sv, reverse=True), "singular values must be descending"
    evr = meta["explained_variance_ratio"]
    assert len(evr) == 8
    assert all(0.0 <= v <= 1.0 for v in evr)
    assert sum(evr) <= 1.0 + 1e-6
    json.dumps(meta)  # must be plain JSON, not numpy scalars


def test_pcs_are_u_times_s(mat, make_config, run_dir):
    """pcs.npy is U*S, so column norms must track the singular values."""
    cfg = make_config(decompose={"n_components": 6, "n_iter": 7})
    pcs = decompose(mat, cfg, run_dir)
    sv = np.array(json.loads((run_dir / "svd.json").read_text())["singular_values"])
    np.testing.assert_allclose(np.linalg.norm(pcs, axis=0), sv, rtol=1e-3)


def test_block_structure_is_recovered(make_config, run_dir, rng):
    """Rows from the same block must be closer in PC space than across blocks."""
    m = block_matrix(n_blocks=4, rows_per_block=30, cols_per_block=10, rng=rng)
    cfg = make_config(decompose={"n_components": 6, "n_iter": 7})
    pcs = decompose(m, cfg, run_dir)
    labels = np.repeat(np.arange(4), 30)
    centroids = np.stack([pcs[labels == b].mean(axis=0) for b in range(4)])
    within = np.mean([np.linalg.norm(pcs[labels == b] - centroids[b], axis=1).mean() for b in range(4)])
    between = np.mean(
        [
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ]
    )
    assert between > 2 * within


def test_determinism_under_a_fixed_seed(mat, make_config, tmp_path):
    out = []
    for i in range(2):
        cfg = make_config(seed=4242, decompose={"n_components": 8, "n_iter": 4})
        d = tmp_path / f"d{i}"
        d.mkdir()
        out.append(decompose(mat, cfg, d))
    np.testing.assert_array_equal(out[0], out[1])


def test_a_different_seed_changes_the_random_projection(mat, make_config, tmp_path):
    """Not a correctness requirement, but proof the seed is actually used."""
    out = []
    for seed in (1, 2):
        cfg = make_config(seed=seed, decompose={"n_components": 8, "n_iter": 0})
        d = tmp_path / f"s{seed}"
        d.mkdir()
        out.append(decompose(mat, cfg, d))
    assert not np.array_equal(out[0], out[1])


def test_drop_first_discards_leading_components(mat, make_config, run_dir):
    # Reading of docs/API.md: drop_first removes leading components from the
    # returned array, so its width is n_components - drop_first.
    cfg = make_config(decompose={"n_components": 8, "drop_first": 2, "n_iter": 4})
    pcs = decompose(mat, cfg, run_dir)
    assert pcs.shape == (mat.shape[0], 6)


def test_load_pcs_round_trip(mat, make_config, run_dir):
    cfg = make_config(decompose={"n_components": 5})
    pcs = decompose(mat, cfg, run_dir)
    np.testing.assert_array_equal(load_pcs(run_dir), pcs)


def test_rerun_short_circuits(mat, make_config, run_dir):
    cfg = make_config(decompose={"n_components": 5})
    decompose(mat, cfg, run_dir)
    before = (run_dir / "pcs.npy").stat().st_mtime_ns
    decompose(mat, cfg, run_dir)
    assert (run_dir / "pcs.npy").stat().st_mtime_ns == before
    decompose(mat, cfg, run_dir, force=True)
    assert (run_dir / "pcs.npy").stat().st_mtime_ns != before


# --------------------------------------------------------------------------
# degenerate inputs -- all of these can come out of a real filter cascade
# --------------------------------------------------------------------------


def test_more_components_than_rank_is_clamped_not_crashed(make_config, run_dir):
    m = sparse.csr_matrix(np.eye(4, dtype=np.float32))
    cfg = make_config(decompose={"n_components": 32})
    pcs = decompose(m, cfg, run_dir)
    assert pcs.shape[0] == 4
    assert pcs.shape[1] <= 32
    assert np.isfinite(pcs).all()


def test_single_row_matrix(make_config, run_dir):
    m = sparse.csr_matrix(np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    cfg = make_config(decompose={"n_components": 4})
    pcs = decompose(m, cfg, run_dir)
    assert pcs.shape[0] == 1
    assert np.isfinite(pcs).all()


def test_all_zero_matrix(make_config, run_dir):
    m = sparse.csr_matrix((10, 5), dtype=np.float32)
    cfg = make_config(decompose={"n_components": 3})
    pcs = decompose(m, cfg, run_dir)
    assert pcs.shape[0] == 10
    assert np.isfinite(pcs).all()
    assert not pcs.any()


def test_empty_matrix(make_config, run_dir):
    m = sparse.csr_matrix((0, 0), dtype=np.float32)
    cfg = make_config(decompose={"n_components": 3})
    pcs = decompose(m, cfg, run_dir)
    assert pcs.shape[0] == 0
    assert pcs.ndim == 2
