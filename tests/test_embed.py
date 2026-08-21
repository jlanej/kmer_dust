"""UMAP stage: shapes, seeded determinism and the fit-subsample path.

``embed`` fits on at most ``cfg.embed.max_fit_rows`` rows and transforms the
rest, which is the only way a 15-million-bin run is tractable.  The subsample
must be seeded, and it must not change the *shape* of the answer -- every row of
``pcs`` gets coordinates, fitted or transformed.
"""

from __future__ import annotations

import numpy as np
import pytest

from kmer_dust.embed import embed, load_embedding


def blobs(n_per=40, k=4, dim=8, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4.0, size=(k, dim))
    pts = np.concatenate([centres[i] + rng.normal(scale=spread, size=(n_per, dim)) for i in range(k)])
    return pts.astype(np.float32)


@pytest.fixture
def pcs():
    return blobs()


def test_shape_dtype_and_file(pcs, make_config, run_dir):
    cfg = make_config(embed={"n_components": 2, "n_neighbors": 10})
    coords = embed(pcs, cfg, run_dir)
    assert coords.shape == (pcs.shape[0], 2)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()
    assert (run_dir / "umap.npy").exists()


def test_three_dimensional_embedding(pcs, make_config, run_dir):
    cfg = make_config(embed={"n_components": 3, "n_neighbors": 10})
    assert embed(pcs, cfg, run_dir).shape == (pcs.shape[0], 3)


def test_determinism_under_a_fixed_seed(pcs, make_config, tmp_path):
    out = []
    for i in range(2):
        cfg = make_config(seed=99, embed={"n_components": 2, "n_neighbors": 10})
        d = tmp_path / f"e{i}"
        d.mkdir()
        out.append(embed(pcs, cfg, d))
    np.testing.assert_array_equal(out[0], out[1])


def test_load_embedding_round_trip(pcs, make_config, run_dir):
    cfg = make_config(embed={"n_neighbors": 10})
    coords = embed(pcs, cfg, run_dir)
    np.testing.assert_array_equal(load_embedding(run_dir), coords)


def test_rerun_short_circuits(pcs, make_config, run_dir):
    cfg = make_config(embed={"n_neighbors": 10})
    embed(pcs, cfg, run_dir)
    before = (run_dir / "umap.npy").stat().st_mtime_ns
    embed(pcs, cfg, run_dir)
    assert (run_dir / "umap.npy").stat().st_mtime_ns == before
    embed(pcs, cfg, run_dir, force=True)
    assert (run_dir / "umap.npy").stat().st_mtime_ns != before


def test_fit_subsample_still_places_every_row(pcs, make_config, run_dir):
    cfg = make_config(embed={"n_components": 2, "n_neighbors": 10, "max_fit_rows": 60})
    coords = embed(pcs, cfg, run_dir)
    assert coords.shape == (pcs.shape[0], 2)
    assert np.isfinite(coords).all()


def test_fit_subsample_is_seeded(pcs, make_config, tmp_path):
    out = []
    for i in range(2):
        cfg = make_config(seed=7, embed={"n_neighbors": 10, "max_fit_rows": 60})
        d = tmp_path / f"f{i}"
        d.mkdir()
        out.append(embed(pcs, cfg, d))
    np.testing.assert_array_equal(out[0], out[1])


def test_structure_survives_the_embedding(pcs, make_config, run_dir):
    """Well-separated blobs must stay separated in 2-D."""
    cfg = make_config(embed={"n_components": 2, "n_neighbors": 10, "min_dist": 0.0})
    coords = embed(pcs, cfg, run_dir)
    labels = np.repeat(np.arange(4), 40)
    centroids = np.stack([coords[labels == b].mean(axis=0) for b in range(4)])
    within = np.mean(
        [np.linalg.norm(coords[labels == b] - centroids[b], axis=1).mean() for b in range(4)]
    )
    between = np.min(
        [
            np.linalg.norm(centroids[i] - centroids[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ]
    )
    assert between > 2 * within


# --------------------------------------------------------------------------
# degenerate inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_rows", [1, 2, 3, 6])
def test_tiny_inputs_do_not_crash(n_rows, make_config, tmp_path):
    """n_neighbors=30 against 3 rows is an immediate UMAP error; clamp it."""
    pcs = np.random.default_rng(0).normal(size=(n_rows, 5)).astype(np.float32)
    cfg = make_config(embed={"n_components": 2, "n_neighbors": 30})
    d = tmp_path / f"tiny{n_rows}"
    d.mkdir()
    coords = embed(pcs, cfg, d)
    assert coords.shape == (n_rows, 2)
    assert np.isfinite(coords).all()


def test_empty_input(make_config, run_dir):
    coords = embed(np.zeros((0, 5), dtype=np.float32), make_config(), run_dir)
    assert coords.shape[0] == 0
    assert coords.ndim == 2


def test_identical_rows_do_not_produce_nans(make_config, run_dir):
    """Duplicate bins are common (identical satellite arrays); UMAP hates them."""
    pcs = np.ones((30, 4), dtype=np.float32)
    coords = embed(pcs, make_config(embed={"n_neighbors": 5}), run_dir)
    assert coords.shape == (30, 2)
    assert np.isfinite(coords).all()
