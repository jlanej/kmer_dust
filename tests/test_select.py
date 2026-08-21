"""Feature selection: prevalence maths and order-independent sub-sampling.

Two properties matter here.  First, prevalence is counted over *samples*, not
over haplotype assemblies -- a k-mer in both haplotypes of one donor is present
in one sample, and getting that wrong doubles the apparent frequency of
everything heterozygous.  Second, the cut to ``max_features`` must be a second
FracMinHash pass rather than a head/sample of a list, so that the feature set
does not depend on the order the shards happened to be read in (which on a
cluster is whatever order the Slurm array finished in).
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import manifest_from_shards, write_sketch_shard

from kmer_dust import schemas
from kmer_dust.select import load_kmers, select_kmers

# Six hashes, ascending, with well-separated high bits so the radix partition
# actually spreads them over buckets.
H = [
    0x0123456789ABCDEF,
    0x1FEDCBA987654321,
    0x3000000000000001,
    0x5555555555555555,
    0x9999999999999999,
    0xF0F0F0F0F0F0F0F0,
]

# hash -> (n_samples, n_assemblies, n_bins) for the toy layout below
EXPECTED = {
    H[0]: (2, 3, 4),
    H[1]: (1, 2, 2),
    H[2]: (2, 2, 3),
    H[3]: (1, 1, 1),
    H[4]: (1, 1, 1),
    H[5]: (1, 1, 1),
}


@pytest.fixture
def toy_shards(tmp_path):
    """Two samples, three haplotype assemblies, two bins each."""
    sketch_dir = tmp_path / "sketch"
    write_sketch_shard(sketch_dir, "S1_pat", [[H[0], H[1], H[2]], [H[0], H[3]]], sample="S1")
    write_sketch_shard(sketch_dir, "S1_mat", [[H[0], H[1]], [H[4]]], sample="S1")
    write_sketch_shard(sketch_dir, "S2_pat", [[H[0], H[2]], [H[2], H[5]]], sample="S2")
    manifest = manifest_from_shards([("S1_pat", "S1"), ("S1_mat", "S1"), ("S2_pat", "S2")])
    return sketch_dir, manifest


def _cfg(make_config, **select):
    base = {
        "min_sample_prevalence": 0.4,
        "max_sample_prevalence": 1.0,
        "min_bins": 2,
        "max_features": 0,
        "n_buckets": 4,
    }
    base.update(select)
    return make_config(select=base)


# --------------------------------------------------------------------------
# prevalence maths on an enumerable case
# --------------------------------------------------------------------------


def test_counts_are_exactly_the_hand_enumerated_answer(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config, min_sample_prevalence=0.0, max_sample_prevalence=1.0, min_bins=1)
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    got = {
        int(r.hash): (int(r.n_samples), int(r.n_assemblies), int(r.n_bins))
        for r in kmers.itertuples()
    }
    assert got == EXPECTED


def test_prevalence_is_over_samples_not_haplotypes(toy_shards, make_config, run_dir):
    """H[1] is in both haplotypes of S1 -- that is one sample, not two."""
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config, min_sample_prevalence=0.0, max_sample_prevalence=1.0, min_bins=1)
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    row = kmers[kmers["hash"] == H[1]].iloc[0]
    assert int(row.n_assemblies) == 2
    assert int(row.n_samples) == 1


def test_default_band_selects_the_expected_three(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    kmers = select_kmers(sketch_dir, manifest, _cfg(make_config), run_dir)
    assert kmers["hash"].tolist() == [H[0], H[1], H[2]]


def test_max_prevalence_removes_the_ubiquitous_kmers(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config, min_sample_prevalence=0.0, max_sample_prevalence=0.9, min_bins=1)
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    assert kmers["hash"].tolist() == [H[1], H[3], H[4], H[5]]


def test_min_prevalence_removes_the_private_kmers(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config, min_sample_prevalence=0.6, max_sample_prevalence=1.0, min_bins=1)
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    assert kmers["hash"].tolist() == [H[0], H[2]]


def test_min_bins_removes_singletons(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config, min_sample_prevalence=0.0, max_sample_prevalence=1.0, min_bins=3)
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    assert kmers["hash"].tolist() == [H[0], H[2]]


def test_prevalence_bounds_are_inclusive(toy_shards, make_config, run_dir):
    # Assumption, not pinned by docs/API.md: "min"/"max" read as >= and <=.
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config, min_sample_prevalence=0.5, max_sample_prevalence=0.5, min_bins=1)
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    assert kmers["hash"].tolist() == [H[1], H[3], H[4], H[5]]


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------


def test_kmers_table_contract(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    kmers = select_kmers(sketch_dir, manifest, _cfg(make_config), run_dir)
    assert list(kmers.columns) == list(schemas.KMER_COLUMNS)
    for col, dtype in schemas.KMER_COLUMNS.items():
        assert str(kmers[col].dtype) == dtype, col
    assert kmers["hash"].is_monotonic_increasing
    assert kmers["col_idx"].tolist() == list(range(len(kmers)))
    assert (run_dir / "kmers.parquet").exists()
    assert (run_dir / "prevalence.parquet").exists()
    pd.testing.assert_frame_equal(load_kmers(run_dir), kmers)


def test_prevalence_histogram_contract(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    kmers = select_kmers(sketch_dir, manifest, _cfg(make_config), run_dir)
    hist = pd.read_parquet(run_dir / "prevalence.parquet")
    assert list(hist.columns) == list(schemas.PREVALENCE_HIST_COLUMNS)
    for col, dtype in schemas.PREVALENCE_HIST_COLUMNS.items():
        assert str(hist[col].dtype) == dtype, col
    assert hist["n_kmers"].sum() == len(EXPECTED)
    # the histogram may carry empty buckets (including n_samples == 0)
    assert set(hist.loc[hist["n_kmers"] > 0, "n_samples"]) == {1, 2}
    assert hist["n_samples"].is_monotonic_increasing
    # `selected` marks the prevalence band, which is a superset of the final
    # feature set (min_bins and max_features cut further).  Every k-mer that
    # made it into kmers.parquet must sit in a `selected` prevalence bucket.
    selected_prevalences = set(hist.loc[hist["selected"], "n_samples"].tolist())
    assert selected_prevalences
    assert int(hist.loc[hist["selected"], "n_kmers"].sum()) >= len(kmers)
    assert set(kmers["n_samples"].tolist()) <= selected_prevalences


def test_radix_buckets_do_not_change_the_answer(toy_shards, make_config, tmp_path):
    """Pass 1 partitions by the high bits; the partition count must not matter.

    (The intermediates under ``buckets/`` are deleted once pass 2 finishes, so
    there is nothing on disk to inspect -- the observable contract is that the
    result is invariant.)
    """
    sketch_dir, manifest = toy_shards
    results = []
    for n_buckets in (1, 2, 4, 16):
        out = tmp_path / f"b{n_buckets}"
        out.mkdir()
        cfg = _cfg(make_config, n_buckets=n_buckets)
        results.append(select_kmers(sketch_dir, manifest, cfg, out))
        leftover = sorted((out / "buckets").glob("bucket_*.parquet"))
        assert len(leftover) <= n_buckets
    for other in results[1:]:
        pd.testing.assert_frame_equal(results[0], other)


# --------------------------------------------------------------------------
# order independence and sub-sampling
# --------------------------------------------------------------------------


@pytest.fixture
def many_shards(tmp_path, rng):
    """Three assemblies with a few thousand shared/private hashes each."""
    sketch_dir = tmp_path / "sketch_many"
    shared = rng.integers(0, 1 << 63, size=3_000, dtype="uint64")
    specs = []
    for assembly, sample in [
        ("A_pat", "A"), ("A_mat", "A"), ("B_pat", "B"), ("B_mat", "B"), ("C_pat", "C")
    ]:
        private = rng.integers(0, 1 << 63, size=1_000, dtype="uint64")
        pool = list(shared) + list(private)
        bins = [pool[j::8] for j in range(8)]
        write_sketch_shard(sketch_dir, assembly, bins, sample=sample)
        specs.append((assembly, sample))
    return sketch_dir, manifest_from_shards(specs)


def test_subsampling_is_order_independent(many_shards, make_config, tmp_path):
    sketch_dir, manifest = many_shards
    cfg = _cfg(
        make_config,
        min_sample_prevalence=0.0,
        max_sample_prevalence=1.0,
        min_bins=1,
        max_features=500,
        n_buckets=8,
    )
    a_dir, b_dir = tmp_path / "sel_a", tmp_path / "sel_b"
    a_dir.mkdir()
    b_dir.mkdir()
    a = select_kmers(sketch_dir, manifest, cfg, a_dir)
    shuffled = manifest.iloc[[4, 0, 3, 1, 2]].reset_index(drop=True)
    assert shuffled["assembly"].tolist() != manifest["assembly"].tolist()
    b = select_kmers(sketch_dir, shuffled, cfg, b_dir)
    pd.testing.assert_frame_equal(a, b)


def test_max_features_is_respected_and_deterministic(many_shards, make_config, tmp_path):
    sketch_dir, manifest = many_shards
    cfg = _cfg(
        make_config,
        min_sample_prevalence=0.0,
        max_sample_prevalence=1.0,
        min_bins=1,
        max_features=500,
        n_buckets=8,
    )
    first_dir, second_dir = tmp_path / "s1", tmp_path / "s2"
    first_dir.mkdir()
    second_dir.mkdir()
    first = select_kmers(sketch_dir, manifest, cfg, first_dir)
    second = select_kmers(sketch_dir, manifest, cfg, second_dir)
    pd.testing.assert_frame_equal(first, second)
    # A FracMinHash threshold cannot hit the target exactly; it must not
    # massively overshoot either.
    assert 0 < len(first) <= 1.5 * cfg.select.max_features
    assert first["hash"].is_monotonic_increasing
    assert first["col_idx"].tolist() == list(range(len(first)))


def test_a_different_seed_gives_a_different_subsample(many_shards, make_config, tmp_path):
    sketch_dir, manifest = many_shards
    out = []
    for seed in (1, 2):
        cfg = _cfg(
            make_config,
            min_sample_prevalence=0.0,
            max_sample_prevalence=1.0,
            min_bins=1,
            max_features=500,
            n_buckets=8,
            seed=seed,
        )
        d = tmp_path / f"seed{seed}"
        d.mkdir()
        out.append(set(select_kmers(sketch_dir, manifest, cfg, d)["hash"].tolist()))
    assert out[0] != out[1]


def test_max_features_zero_keeps_everything(many_shards, make_config, run_dir):
    sketch_dir, manifest = many_shards
    cfg = _cfg(
        make_config,
        min_sample_prevalence=0.0,
        max_sample_prevalence=1.0,
        min_bins=1,
        max_features=0,
        n_buckets=8,
    )
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    assert len(kmers) == 3_000 + 5 * 1_000


# --------------------------------------------------------------------------
# edge cases and restartability
# --------------------------------------------------------------------------


def test_no_shards_yields_an_empty_but_typed_frame(tmp_path, make_config, run_dir):
    empty_sketch = tmp_path / "no_shards"
    empty_sketch.mkdir()
    manifest = manifest_from_shards([])
    kmers = select_kmers(empty_sketch, manifest, _cfg(make_config), run_dir)
    assert len(kmers) == 0
    assert list(kmers.columns) == list(schemas.KMER_COLUMNS)
    assert (run_dir / "kmers.parquet").exists()


def test_shards_with_no_hashes_yield_an_empty_frame(tmp_path, make_config, run_dir):
    sketch_dir = tmp_path / "sketch_empty"
    write_sketch_shard(sketch_dir, "A_pat", [[], []], sample="A")
    manifest = manifest_from_shards([("A_pat", "A")])
    kmers = select_kmers(sketch_dir, manifest, _cfg(make_config), run_dir)
    assert len(kmers) == 0
    assert list(kmers.columns) == list(schemas.KMER_COLUMNS)


def test_everything_filtered_out_is_not_an_error(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config, min_bins=99)
    kmers = select_kmers(sketch_dir, manifest, cfg, run_dir)
    assert len(kmers) == 0
    assert list(kmers.columns) == list(schemas.KMER_COLUMNS)


def test_manifest_rows_without_a_shard_are_tolerated(toy_shards, make_config, run_dir):
    """A Slurm array can leave a hole; select must not invent counts for it."""
    sketch_dir, manifest = toy_shards
    extended = pd.concat(
        [manifest, manifest_from_shards([("S3_pat", "S3")])], ignore_index=True
    )
    # min_sample_prevalence=0 keeps the assertion independent of whether the
    # denominator is "samples in the manifest" or "samples with a shard".
    cfg = _cfg(make_config, min_sample_prevalence=0.0, min_bins=2)
    kmers = select_kmers(
        sketch_dir, schemas.enforce(extended, schemas.MANIFEST_COLUMNS), cfg, run_dir
    )
    assert kmers["hash"].tolist() == [H[0], H[1], H[2]]


def test_rerun_short_circuits_unless_forced(toy_shards, make_config, run_dir):
    sketch_dir, manifest = toy_shards
    cfg = _cfg(make_config)
    first = select_kmers(sketch_dir, manifest, cfg, run_dir)
    before = (run_dir / "kmers.parquet").stat().st_mtime_ns
    again = select_kmers(sketch_dir, manifest, cfg, run_dir)
    pd.testing.assert_frame_equal(first, again)
    assert (run_dir / "kmers.parquet").stat().st_mtime_ns == before
    select_kmers(sketch_dir, manifest, cfg, run_dir, force=True)
    assert (run_dir / "kmers.parquet").stat().st_mtime_ns != before
