"""Canonical k-mer encoding and FracMinHash selection.

The whole pipeline hangs off one idea: a *bin* is represented by the set of
64-bit hashes of its canonical k-mers that fall below a threshold
``max_hash = 2**64 / scaled``.  Because ``splitmix64`` is a bijection on
``uint64``, that threshold keeps a uniform ~1/``scaled`` random sample of k-mer
space -- the same sample in every bin of every assembly, with no coordination
and no reference.  This is the FracMinHash / "scaled MinHash" construction used
by sourmash, and it is what makes the bin x k-mer matrix comparable across
haplotypes without any alignment.

A numba kernel does the rolling encode when numba is importable; an equivalent
chunked NumPy implementation is used otherwise.  Both are exercised by the test
suite and must agree exactly.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = [
    "BASE_CODES",
    "encode_bases",
    "splitmix64",
    "splitmix64_array",
    "canonical_code",
    "kmer_to_code",
    "code_to_kmer",
    "max_hash_for_scaled",
    "sketch_contig",
    "bin_base_stats",
    "bin_valid_kmer_counts",
    "NUMBA_AVAILABLE",
]

# --------------------------------------------------------------------------
# base encoding
# --------------------------------------------------------------------------

#: 256-entry lookup: A/a->0 C/c->1 G/g->2 T/t/U/u->3, everything else -> 255.
BASE_CODES: np.ndarray = np.full(256, 255, dtype=np.uint8)
for _base, _code in (("A", 0), ("C", 1), ("G", 2), ("T", 3), ("U", 3)):
    BASE_CODES[ord(_base)] = _code
    BASE_CODES[ord(_base.lower())] = _code


def encode_bases(seq: bytes | bytearray | str | np.ndarray) -> np.ndarray:
    """Map a nucleotide sequence to ``uint8`` codes (255 == not ACGT).

    Soft-masked (lower-case) bases are treated exactly like upper-case ones:
    repeat-masking is an annotation, not a property of the sequence, and we want
    satellite bins to keep their k-mers.
    """
    if isinstance(seq, str):
        seq = seq.encode("ascii", "replace")
    arr = np.frombuffer(bytes(seq), dtype=np.uint8) if not isinstance(seq, np.ndarray) else seq
    return BASE_CODES[arr]


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

_SM64_A = np.uint64(0x9E3779B97F4A7C15)
_SM64_B = np.uint64(0xBF58476D1CE4E5B9)
_SM64_C = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)


def splitmix64(x: int | np.uint64) -> np.uint64:
    """SplitMix64 finalizer -- a bijection on the 64-bit integers.

    Overflow *is* the algorithm here: every step is mod 2**64, so NumPy's
    scalar overflow warning is silenced rather than avoided.
    """
    with np.errstate(over="ignore"):
        z = np.uint64(x) + _SM64_A
        z = (z ^ (z >> _S30)) * _SM64_B
        z = (z ^ (z >> _S27)) * _SM64_C
        return np.uint64(z ^ (z >> _S31))


def splitmix64_array(x: np.ndarray) -> np.ndarray:
    """Vectorised :func:`splitmix64`."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _SM64_A
        z = (z ^ (z >> _S30)) * _SM64_B
        z = (z ^ (z >> _S27)) * _SM64_C
        return z ^ (z >> _S31)


def max_hash_for_scaled(scaled: int) -> int:
    """Inclusive hash threshold retaining ~1/``scaled`` of k-mer space."""
    if scaled < 1:
        raise ValueError("scaled must be >= 1")
    return ((1 << 64) - 1) // int(scaled)


# --------------------------------------------------------------------------
# k-mer codes
# --------------------------------------------------------------------------


def kmer_to_code(kmer: str) -> int:
    """2-bit pack a k-mer string (raises on non-ACGT)."""
    codes = encode_bases(kmer)
    if np.any(codes > 3):
        raise ValueError(f"non-ACGT base in k-mer {kmer!r}")
    value = 0
    for code in codes:
        value = (value << 2) | int(code)
    return value


def code_to_kmer(code: int, k: int) -> str:
    """Inverse of :func:`kmer_to_code`."""
    letters = "ACGT"
    out = []
    for i in range(k - 1, -1, -1):
        out.append(letters[(code >> (2 * i)) & 3])
    return "".join(out)


def canonical_code(kmer: str) -> int:
    """The smaller of a k-mer's 2-bit code and its reverse complement's."""
    k = len(kmer)
    fwd = kmer_to_code(kmer)
    rev = 0
    for i in range(k):
        rev = (rev << 2) | (3 - ((fwd >> (2 * i)) & 3))
    return min(fwd, rev)


# --------------------------------------------------------------------------
# numba kernel (with a NumPy twin)
# --------------------------------------------------------------------------

_DISABLE_NUMBA = os.environ.get("KMER_DUST_NO_NUMBA", "").lower() in {"1", "true", "yes"}

try:  # pragma: no cover - exercised implicitly
    if _DISABLE_NUMBA:
        raise ImportError("numba disabled via KMER_DUST_NO_NUMBA")
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in supporting both ``@njit`` and ``@njit(...)`` forms."""

        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap


@njit(cache=True, nogil=True)
def _sm64_scalar(x):  # pragma: no cover - trivial, covered via kernel
    m = np.uint64(0xFFFFFFFFFFFFFFFF)
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & m
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & m
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & m
    return z ^ (z >> np.uint64(31))


@njit(cache=True, nogil=True)
def _sketch_kernel(codes, k, mask, shift, max_hash, bin_size, out_bin, out_hash, collect):
    """Single rolling pass; returns the number of retained hashes.

    Called twice -- once with ``collect=False`` to size the output, once to fill
    it -- so the caller never has to guess or reallocate.
    """
    n = codes.shape[0]
    two = np.uint64(2)
    three = np.uint64(3)
    fwd = np.uint64(0)
    rev = np.uint64(0)
    run = 0
    count = 0
    for i in range(n):
        c = codes[i]
        if c > 3:
            run = 0
            fwd = np.uint64(0)
            rev = np.uint64(0)
            continue
        cu = np.uint64(c)
        fwd = ((fwd << two) | cu) & mask
        rev = (rev >> two) | ((three - cu) << shift)
        run += 1
        if run >= k:
            canon = fwd if fwd < rev else rev
            h = _sm64_scalar(canon)
            if h <= max_hash:
                if collect:
                    out_bin[count] = (i - k + 1) // bin_size
                    out_hash[count] = h
                count += 1
    return count


def _sketch_numpy(codes: np.ndarray, k: int, max_hash: int, bin_size: int, chunk: int = 4_000_000):
    """NumPy twin of :func:`_sketch_kernel`; identical output, more memory."""
    n = codes.shape[0]
    if n < k:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint64)
    mask = np.uint64((1 << (2 * k)) - 1)
    thr = np.uint64(max_hash)
    bins_out: list[np.ndarray] = []
    hash_out: list[np.ndarray] = []
    step = max(chunk, k)
    for begin in range(0, n - k + 1, step):
        end = min(begin + step + k - 1, n)
        block = codes[begin:end]
        m = block.shape[0] - k + 1
        if m <= 0:
            break
        valid = block <= 3
        # k-mer starting at j is usable only if the whole window is ACGT.
        cs = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
        usable = (cs[k:] - cs[:-k]) == 0
        safe = np.where(valid, block, 0).astype(np.uint64)
        fwd = np.zeros(m, dtype=np.uint64)
        rev = np.zeros(m, dtype=np.uint64)
        for j in range(k):
            col = safe[j : j + m]
            fwd |= col << np.uint64(2 * (k - 1 - j))
            rev |= (np.uint64(3) - col) << np.uint64(2 * j)
        fwd &= mask
        rev &= mask
        canon = np.minimum(fwd, rev)
        h = splitmix64_array(canon)
        keep = usable & (h <= thr)
        if not keep.any():
            continue
        idx = np.nonzero(keep)[0]
        starts = idx + begin
        bins_out.append((starts // bin_size).astype(np.int32))
        hash_out.append(h[idx])
    if not bins_out:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint64)
    return np.concatenate(bins_out), np.concatenate(hash_out)


def sketch_contig(
    codes: np.ndarray,
    *,
    k: int,
    bin_size: int,
    max_hash: int,
    force_numpy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """FracMinHash every canonical k-mer of an encoded contig.

    Parameters
    ----------
    codes:
        ``uint8`` array from :func:`encode_bases` (255 marks non-ACGT).
    k:
        k-mer length, ``1 <= k <= 31``.
    bin_size:
        A k-mer is attributed to the bin containing its **first** base.
    max_hash:
        Inclusive threshold from :func:`max_hash_for_scaled`.

    Returns
    -------
    (bin_index, hash) arrays of equal length, in contig order.
    """
    if not 1 <= k <= 31:
        raise ValueError("k must be in 1..31")
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    if codes.shape[0] < k:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.uint64)
    if force_numpy or not NUMBA_AVAILABLE:
        return _sketch_numpy(codes, k, max_hash, bin_size)
    mask = np.uint64((1 << (2 * k)) - 1)
    shift = np.uint64(2 * (k - 1))
    empty_i = np.empty(0, dtype=np.int32)
    empty_h = np.empty(0, dtype=np.uint64)
    total = _sketch_kernel(
        codes, k, mask, shift, np.uint64(max_hash), bin_size, empty_i, empty_h, False
    )
    out_bin = np.empty(total, dtype=np.int32)
    out_hash = np.empty(total, dtype=np.uint64)
    _sketch_kernel(
        codes, k, mask, shift, np.uint64(max_hash), bin_size, out_bin, out_hash, True
    )
    return out_bin, out_hash


# --------------------------------------------------------------------------
# per-bin summaries (pure NumPy, cheap next to the sketching pass)
# --------------------------------------------------------------------------


def bin_base_stats(codes: np.ndarray, bin_size: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin (n_acgt, n_gc) counts."""
    idx = np.arange(codes.shape[0], dtype=np.int64) // bin_size
    acgt = codes <= 3
    n_acgt = np.bincount(idx[acgt], minlength=n_bins)[:n_bins]
    gc_mask = (codes == 1) | (codes == 2)
    n_gc = np.bincount(idx[gc_mask], minlength=n_bins)[:n_bins]
    return n_acgt.astype(np.int64), n_gc.astype(np.int64)


def bin_valid_kmer_counts(codes: np.ndarray, k: int, bin_size: int, n_bins: int) -> np.ndarray:
    """Per-bin count of canonical k-mers whose first base falls in the bin."""
    n = codes.shape[0]
    if n < k:
        return np.zeros(n_bins, dtype=np.int64)
    bad = (codes > 3).astype(np.int64)
    cs = np.concatenate(([0], np.cumsum(bad)))
    usable = (cs[k:] - cs[:-k]) == 0
    starts = np.nonzero(usable)[0]
    if starts.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    return np.bincount(starts // bin_size, minlength=n_bins)[:n_bins].astype(np.int64)
