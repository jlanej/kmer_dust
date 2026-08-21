"""Regression net for the k-mer core.

``hashing.py`` is the one module whose output every other stage inherits: if the
canonical encoding or the FracMinHash threshold ever drifts, every sketch on
disk silently becomes incomparable with every other.  These tests therefore
check the *properties* (canonicality, retention rate, N-handling, bin
attribution) rather than golden values, plus an independent pure-Python
implementation to check the two fast paths against.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import random_sequence, revcomp

from kmer_dust.hashing import (
    NUMBA_AVAILABLE,
    bin_base_stats,
    bin_valid_kmer_counts,
    canonical_code,
    code_to_kmer,
    encode_bases,
    kmer_to_code,
    max_hash_for_scaled,
    sketch_contig,
    splitmix64,
    splitmix64_array,
)

K = 31
ALL_HASHES = (1 << 64) - 1


def brute_sketch(seq: str, k: int, bin_size: int, max_hash: int) -> list[tuple[int, int]]:
    """Independent reference implementation -- slow, obvious, no NumPy tricks."""
    out: list[tuple[int, int]] = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k].upper()
        if any(base not in "ACGT" for base in kmer):
            continue
        h = int(splitmix64(canonical_code(kmer)))
        if h <= max_hash:
            out.append((i // bin_size, h))
    return out


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------


def test_encode_bases_handles_case_bytes_and_ambiguity():
    codes = encode_bases("AcGtNnXU-")
    assert codes[:4].tolist() == [0, 1, 2, 3]
    assert codes[4] == 255 and codes[5] == 255  # N/n
    assert codes[6] == 255  # X
    assert codes[7] == 3  # U reads as T
    assert codes[8] == 255  # gap
    assert np.array_equal(encode_bases(b"ACGT"), encode_bases("ACGT"))


def test_kmer_code_round_trip(rng):
    for k in (1, 5, 21, 31):
        for _ in range(50):
            kmer = random_sequence(k, rng).decode()
            code = kmer_to_code(kmer)
            assert 0 <= code < (1 << (2 * k))
            assert code_to_kmer(code, k) == kmer


def test_kmer_to_code_rejects_ambiguous_bases():
    with pytest.raises(ValueError):
        kmer_to_code("ACGTN")


def test_canonical_code_is_invariant_under_reverse_complement(rng):
    """The whole cross-assembly comparison rests on this one identity."""
    for k in (5, 21, 31):
        for _ in range(200):
            kmer = random_sequence(k, rng).decode()
            rc = revcomp(kmer)
            assert canonical_code(kmer) == canonical_code(rc)
            # ...and the canonical form really is one of the two strands.
            assert canonical_code(kmer) in {kmer_to_code(kmer), kmer_to_code(rc)}
            assert canonical_code(kmer) == min(kmer_to_code(kmer), kmer_to_code(rc))


def test_canonical_code_of_a_palindrome_is_itself():
    # An even-length palindrome is its own reverse complement; k is odd in the
    # pipeline precisely so this degenerate case cannot arise for real k-mers.
    assert canonical_code("ACGT") == kmer_to_code("ACGT")


# --------------------------------------------------------------------------
# splitmix64
# --------------------------------------------------------------------------


def test_splitmix64_is_injective_on_a_sample(rng):
    values = rng.integers(0, 1 << 62, size=20_000, dtype=np.uint64)
    hashed = splitmix64_array(values)
    assert np.unique(hashed).size == np.unique(values).size


def test_splitmix64_scalar_matches_array():
    values = np.array([0, 1, 2, 12345, (1 << 64) - 1], dtype=np.uint64)
    scalar = np.array([splitmix64(int(v)) for v in values], dtype=np.uint64)
    assert np.array_equal(scalar, splitmix64_array(values))


def test_splitmix64_known_vectors():
    # SplitMix64 finalizer of 0 is a widely published constant; it pins the
    # exact mixing constants so a "harmless" refactor cannot change hashes.
    assert int(splitmix64(0)) == 0xE220A8397B1DCDAF


def test_max_hash_for_scaled():
    assert max_hash_for_scaled(1) == ALL_HASHES
    assert max_hash_for_scaled(2) == ALL_HASHES // 2
    with pytest.raises(ValueError):
        max_hash_for_scaled(0)


# --------------------------------------------------------------------------
# sketch_contig
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [3, 15, 31])
def test_numba_numpy_and_bruteforce_agree(rng, k):
    seq = random_sequence(5_000, rng).decode()
    max_hash = max_hash_for_scaled(20)
    codes = encode_bases(seq)
    expected = brute_sketch(seq, k, 500, max_hash)

    for force_numpy in (True, False):
        bins, hashes = sketch_contig(
            codes, k=k, bin_size=500, max_hash=max_hash, force_numpy=force_numpy
        )
        got = list(zip(bins.tolist(), [int(h) for h in hashes]))
        assert got == expected


def test_numpy_chunking_is_transparent(rng):
    """Internal chunking must not change a single emitted hash."""
    from kmer_dust.hashing import _sketch_numpy

    codes = encode_bases(random_sequence(20_000, rng))
    max_hash = max_hash_for_scaled(10)
    ref = _sketch_numpy(codes, K, max_hash, 1_000, chunk=4_000_000)
    for chunk in (K, 37, 1_000, 9_999):
        got = _sketch_numpy(codes, K, max_hash, 1_000, chunk=chunk)
        assert np.array_equal(got[0], ref[0])
        assert np.array_equal(got[1], ref[1])


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not importable")
def test_numba_path_is_actually_exercised(rng):
    codes = encode_bases(random_sequence(2_000, rng))
    a = sketch_contig(codes, k=K, bin_size=100, max_hash=max_hash_for_scaled(5))
    b = sketch_contig(codes, k=K, bin_size=100, max_hash=max_hash_for_scaled(5), force_numpy=True)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_soft_masking_does_not_change_the_sketch(rng):
    seq = random_sequence(3_000, rng)
    masked = seq[:1000] + seq[1000:2000].lower() + seq[2000:]
    plain = sketch_contig(encode_bases(seq), k=K, bin_size=500, max_hash=max_hash_for_scaled(10))
    soft = sketch_contig(encode_bases(masked), k=K, bin_size=500, max_hash=max_hash_for_scaled(10))
    assert np.array_equal(plain[0], soft[0])
    assert np.array_equal(plain[1], soft[1])


def test_kmers_spanning_n_runs_are_excluded(rng):
    left = random_sequence(200, rng)
    right = random_sequence(200, rng)
    seq = (left + b"N" * 7 + right).decode()
    codes = encode_bases(seq)
    for force_numpy in (True, False):
        bins, hashes = sketch_contig(
            codes, k=K, bin_size=1, max_hash=ALL_HASHES, force_numpy=force_numpy
        )
        starts = set(bins.tolist())
        # Every start whose window touches the N run must be absent, and every
        # other start must be present.
        forbidden = set(range(200 - K + 1, 200 + 7))
        assert not (starts & forbidden)
        assert starts == set(range(len(seq) - K + 1)) - forbidden
        assert len(hashes) == len(starts)


def test_all_n_contig_yields_nothing():
    codes = encode_bases("N" * 500)
    bins, hashes = sketch_contig(codes, k=K, bin_size=100, max_hash=ALL_HASHES)
    assert bins.size == 0 and hashes.size == 0
    assert bins.dtype == np.int32 and hashes.dtype == np.uint64


def test_contig_shorter_than_k_yields_nothing():
    for length in (0, 1, K - 1):
        codes = encode_bases("A" * length)
        bins, hashes = sketch_contig(codes, k=K, bin_size=100, max_hash=ALL_HASHES)
        assert bins.size == 0 and hashes.size == 0


def test_bin_attribution_uses_the_first_base(rng):
    """A k-mer belongs to the bin holding its first base, never its last."""
    bin_size = 64
    seq = random_sequence(1_000, rng)
    bins, _ = sketch_contig(encode_bases(seq), k=K, bin_size=bin_size, max_hash=ALL_HASHES)
    expected = np.arange(len(seq) - K + 1) // bin_size
    assert np.array_equal(bins, expected.astype(np.int32))
    # The last k-mer starts at len-K; with K=31 and bin_size=64 its final base
    # lives in the next bin, which must not change the answer.
    assert bins[-1] == (len(seq) - K) // bin_size


def test_sketch_contig_rejects_impossible_k():
    codes = encode_bases("A" * 100)
    for k in (0, 32, 64):
        with pytest.raises(ValueError):
            sketch_contig(codes, k=k, bin_size=10, max_hash=ALL_HASHES)


def test_fracminhash_retention_rate_is_within_binomial_tolerance(rng):
    """~1/scaled of k-mer space, with no per-bin or per-assembly coordination."""
    n = 400_000
    scaled = 100
    codes = encode_bases(random_sequence(n, rng))
    _, hashes = sketch_contig(codes, k=K, bin_size=10_000, max_hash=max_hash_for_scaled(scaled))
    trials = n - K + 1
    p = 1.0 / scaled
    mean = trials * p
    sd = np.sqrt(trials * p * (1 - p))
    assert abs(hashes.size - mean) < 5 * sd, f"kept {hashes.size}, expected {mean:.0f} +/- {sd:.0f}"


def test_retained_hashes_are_below_the_threshold(rng):
    max_hash = max_hash_for_scaled(37)
    codes = encode_bases(random_sequence(50_000, rng))
    _, hashes = sketch_contig(codes, k=K, bin_size=1_000, max_hash=max_hash)
    assert hashes.size > 0
    assert int(hashes.max()) <= max_hash


def test_sketch_is_reverse_complement_symmetric(rng):
    """Sketching a contig or its reverse complement yields the same hash set."""
    seq = random_sequence(20_000, rng)
    max_hash = max_hash_for_scaled(10)
    _, fwd = sketch_contig(encode_bases(seq), k=K, bin_size=20_000, max_hash=max_hash)
    _, rev = sketch_contig(encode_bases(revcomp(seq)), k=K, bin_size=20_000, max_hash=max_hash)
    assert set(fwd.tolist()) == set(rev.tolist())


# --------------------------------------------------------------------------
# per-bin summaries
# --------------------------------------------------------------------------


def test_bin_base_stats():
    codes = encode_bases("AACCGGTTNN" + "GCGCGCGCGC")
    n_acgt, n_gc = bin_base_stats(codes, bin_size=10, n_bins=2)
    assert n_acgt.tolist() == [8, 10]
    assert n_gc.tolist() == [4, 10]
    assert n_acgt.dtype == np.int64 and n_gc.dtype == np.int64


def test_bin_base_stats_on_empty_and_short_input():
    n_acgt, n_gc = bin_base_stats(encode_bases(""), bin_size=10, n_bins=3)
    assert n_acgt.tolist() == [0, 0, 0]
    assert n_gc.tolist() == [0, 0, 0]


def test_bin_valid_kmer_counts_matches_sketch_bins(rng):
    codes = encode_bases(random_sequence(500, rng) + b"N" * 10 + random_sequence(490, rng))
    bin_size, k = 100, 11
    n_bins = -(-codes.size // bin_size)
    counts = bin_valid_kmer_counts(codes, k, bin_size, n_bins)
    bins, _ = sketch_contig(codes, k=k, bin_size=bin_size, max_hash=ALL_HASHES)
    expected = np.bincount(bins, minlength=n_bins)[:n_bins]
    assert counts.tolist() == expected.tolist()


def test_bin_valid_kmer_counts_short_contig():
    assert bin_valid_kmer_counts(encode_bases("ACGT"), 31, 10, 2).tolist() == [0, 0]
