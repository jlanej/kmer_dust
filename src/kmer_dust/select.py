"""Choosing the k-mer columns of the matrix, out of core.

The sketch stage emits one ``(bin_idx, hash)`` table per haplotype.  Across a few
hundred assemblies that is easily 10^8-10^9 rows -- far too many to sort or
group in memory, and far too many to use as matrix columns.  This module turns
that pile into a modest, ordered, reproducible feature set.

The trick is a **radix partition on the high bits of the hash**.  Because
``splitmix64`` output is uniform, the top ``log2(n_buckets)`` bits split the
hashes into equal-sized, *disjoint* groups: a given hash lands in exactly one
bucket, so every bucket can be counted independently and the results simply
concatenated.  Pass 1 streams each shard in row-group batches and appends to the
bucket files; pass 2 loads one bucket, sorts it once, and derives per-hash
counts.  Peak memory is therefore ``total_rows / n_buckets``, tunable from the
config without touching the code.

Two counting subtleties matter biologically:

* **Prevalence is over samples, not haplotypes.**  A k-mer carried by both
  haplotypes of one donor is *one* observation of "who has this sequence";
  counting it twice would make heterozygous variation look twice as common as it
  is.  Distinct-sample counting here is exact (a group-by on sorted
  ``(hash, sample)`` pairs), never estimated.
* **A repeated k-mer inside one bin is one observation of that bin.**  Alpha
  satellite bins contain the same k-mer hundreds of times; letting that inflate
  the document frequency would make every HOR array look like a universal k-mer.
  Duplicate ``(bin_idx, hash)`` rows are therefore collapsed on the way into the
  buckets.

The final sub-sample to ``max_features`` is a *second* FracMinHash, over the
already-selected hashes: keep ``h`` iff ``splitmix64(h ^ seed) <= threshold``.
The threshold is one global scalar derived from the observed survivor count, so
the decision for a hash depends on nothing but the hash -- order-independent,
streamable, and identical on a re-run or on a different bucket count.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import schemas
from .config import Config
from .hashing import splitmix64_array
from .log import get_logger, timed

logger = get_logger(__name__)

__all__ = [
    "select_kmers",
    "load_kmers",
    "load_prevalence",
    "bucket_index",
    "subsample_threshold",
    "KMERS_FILENAME",
    "PREVALENCE_FILENAME",
    "BUCKET_DIRNAME",
]

KMERS_FILENAME = "kmers.parquet"
PREVALENCE_FILENAME = "prevalence.parquet"
BUCKET_DIRNAME = "buckets"

#: Rows pulled out of a shard at a time.  Large enough that the per-batch NumPy
#: overhead disappears, small enough that a batch is a few tens of MB.
_READ_BATCH_ROWS = 1 << 20

_MASK64 = (1 << 64) - 1

#: Bucket files are a private intermediate, not part of the public contract:
#: ``schemas.BUCKET_COLUMNS`` describes a reserved bitset layout that the default
#: path does not use.  Explicit assembly *and* sample codes are cheaper here --
#: both columns are highly repetitive and compress to almost nothing.
_BUCKET_SCHEMA = pa.schema(
    [
        pa.field("hash", pa.uint64()),
        pa.field("sample_code", pa.uint32()),
        pa.field("assembly_code", pa.uint32()),
    ]
)

_BUCKET_COMPRESSION = "zstd" if pa.Codec.is_available("zstd") else "snappy"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def bucket_index(hashes: np.ndarray, n_buckets: int) -> np.ndarray:
    """Radix bucket for each hash, keyed by its ``log2(n_buckets)`` high bits."""
    if n_buckets < 1 or n_buckets & (n_buckets - 1):
        raise ValueError("n_buckets must be a power of two")
    hashes = np.asarray(hashes, dtype=np.uint64)
    if n_buckets == 1:
        return np.zeros(hashes.shape[0], dtype=np.int64)
    shift = np.uint64(64 - int(math.log2(n_buckets)))
    return (hashes >> shift).astype(np.int64)


def subsample_threshold(n_survivors: int, max_features: int) -> int:
    """Inclusive FracMinHash threshold whose *expected* yield is ``max_features``.

    Computed in Python ints so the 2**64 scaling is exact; returns ``-1`` when
    nothing can be kept and ``2**64 - 1`` when everything should be.
    """
    if max_features <= 0 or n_survivors <= 0 or n_survivors <= max_features:
        return _MASK64
    return max(-1, min(_MASK64, (int(max_features) << 64) // int(n_survivors) - 1))


def _prevalence_window(cfg: Config, n_samples_total: int) -> tuple[int, int]:
    """Resolve the fractional prevalence window to inclusive sample counts.

    The epsilon absorbs binary-floating-point error: ``0.7 * 10`` is
    ``6.999999999999999``, and a k-mer in 7 of 10 samples must not be dropped by
    a ``max_sample_prevalence`` of ``0.7``.
    """
    eps = 1e-9
    lo = int(math.ceil(cfg.select.min_sample_prevalence * n_samples_total - eps))
    hi = int(math.floor(cfg.select.max_sample_prevalence * n_samples_total + eps))
    lo = max(lo, 1)  # every observed k-mer is in at least one sample
    if hi < lo:
        # e.g. a single sample with max_sample_prevalence < 1.0: the window is
        # empty by construction and would silently discard the whole run.
        logger.warning(
            "prevalence window [%d, %d] over %d sample(s) is empty; widening the "
            "upper bound to %d so the run stays usable",
            lo,
            hi,
            n_samples_total,
            lo,
        )
        hi = lo
    return lo, hi


def _dedupe_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per assembly, manifest order preserved."""
    if "assembly" not in manifest.columns:
        raise ValueError("manifest must have an 'assembly' column")
    dup = manifest["assembly"].duplicated()
    if bool(dup.any()):
        logger.warning(
            "manifest has %d duplicate assembly row(s); keeping the first", int(dup.sum())
        )
        manifest = manifest.loc[~dup]
    return manifest.reset_index(drop=True)


def _sample_codes(manifest: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per-manifest-row sample code plus the code -> sample name table.

    Codes are assigned by first appearance in manifest order, which is itself
    deterministic, so the same manifest always yields the same codes.
    """
    if "sample" in manifest.columns:
        samples = manifest["sample"].fillna("").astype(str).to_list()
    else:
        logger.warning("manifest has no 'sample' column; treating every assembly as its own sample")
        samples = manifest["assembly"].astype(str).to_list()
    order: dict[str, int] = {}
    codes = np.empty(len(samples), dtype=np.uint32)
    for i, name in enumerate(samples):
        key = name if name else f"__unnamed__{i}"
        codes[i] = order.setdefault(key, len(order))
    return codes, list(order)


def _atomic_write(table: pa.Table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, path)


def _column_array(batch: pa.RecordBatch, name: str, dtype: np.dtype | str) -> np.ndarray:
    column = batch.column(name)
    if column.null_count:
        raise ValueError(f"column {name!r} contains nulls; the sketch shard is corrupt")
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


# --------------------------------------------------------------------------
# pass 1 -- partition every shard into hash-keyed buckets
# --------------------------------------------------------------------------


def _iter_shard_batches(path: Path) -> Iterator[tuple[np.ndarray | None, np.ndarray]]:
    """Yield ``(bin_idx, hash)`` batches from a sketch shard, streaming."""
    handle = pq.ParquetFile(path)
    names = set(handle.schema_arrow.names)
    if "hash" not in names:
        raise ValueError(f"{path} is not a sketch shard: no 'hash' column")
    columns = ["bin_idx", "hash"] if "bin_idx" in names else ["hash"]
    for batch in handle.iter_batches(batch_size=_READ_BATCH_ROWS, columns=columns):
        if batch.num_rows == 0:
            continue
        hashes = _column_array(batch, "hash", np.uint64)
        bins = _column_array(batch, "bin_idx", np.int64) if "bin_idx" in columns else None
        yield bins, hashes


def _pass1_partition(
    sketch_dir: Path,
    manifest: pd.DataFrame,
    sample_codes: np.ndarray,
    bucket_dir: Path,
    n_buckets: int,
) -> tuple[int, int]:
    """Stream every shard into ``n_buckets`` parquet files; return (rows, shards)."""
    bucket_dir.mkdir(parents=True, exist_ok=True)
    for stale in sorted(bucket_dir.glob("bucket_*.parquet")):
        stale.unlink()

    writers: list[pq.ParquetWriter] = []
    total_rows = 0
    n_shards = 0
    try:
        for i in range(n_buckets):
            writers.append(
                pq.ParquetWriter(
                    bucket_dir / f"bucket_{i:03d}.parquet",
                    _BUCKET_SCHEMA,
                    compression=_BUCKET_COMPRESSION,
                )
            )
        for pos, assembly in enumerate(manifest["assembly"].astype(str).to_list()):
            path = sketch_dir / f"{assembly}.sketch.parquet"
            if not path.exists():
                logger.warning("no sketch shard for %s (%s); skipping", assembly, path.name)
                continue
            if not (sketch_dir / f"{assembly}.done").exists():
                logger.warning("%s has no .done marker; the shard may be incomplete", assembly)
            n_shards += 1
            shard_rows = 0
            unsorted_warned = False
            prev_bin = np.int64(-1)
            prev_hash = np.uint64(0)
            have_prev = False
            for bins, hashes in _iter_shard_batches(path):
                if bins is not None:
                    if not unsorted_warned and not _is_sorted(bins, hashes):
                        logger.warning(
                            "%s is not sorted by (bin_idx, hash); duplicate k-mers within a "
                            "bin may inflate n_bins",
                            assembly,
                        )
                        unsorted_warned = True
                    keep = np.ones(hashes.shape[0], dtype=bool)
                    keep[1:] = (bins[1:] != bins[:-1]) | (hashes[1:] != hashes[:-1])
                    if have_prev:
                        keep[0] = bool(bins[0] != prev_bin or hashes[0] != prev_hash)
                    prev_bin, prev_hash, have_prev = bins[-1], hashes[-1], True
                    hashes = hashes[keep]
                if hashes.size == 0:
                    continue
                shard_rows += int(hashes.size)
                _write_batch(writers, hashes, int(sample_codes[pos]), pos, n_buckets)
            total_rows += shard_rows
            logger.debug("%s -> %d bucketed rows", assembly, shard_rows)
    finally:
        for writer in writers:
            writer.close()
    return total_rows, n_shards


def _is_sorted(bins: np.ndarray, hashes: np.ndarray) -> bool:
    if bins.shape[0] < 2:
        return True
    forward = bins[1:] > bins[:-1]
    same = bins[1:] == bins[:-1]
    return bool(np.all(forward | (same & (hashes[1:] >= hashes[:-1]))))


def _write_batch(
    writers: Sequence[pq.ParquetWriter],
    hashes: np.ndarray,
    sample_code: int,
    assembly_code: int,
    n_buckets: int,
) -> None:
    """Scatter one batch of hashes across the bucket writers."""
    if n_buckets == 1:
        groups = [hashes]
    else:
        which = bucket_index(hashes, n_buckets)
        counts = np.bincount(which, minlength=n_buckets)
        order = np.argsort(which, kind="stable")
        ordered = hashes[order]
        bounds = np.concatenate(([0], np.cumsum(counts)))
        groups = [ordered[bounds[i] : bounds[i + 1]] for i in range(n_buckets)]
    for i, group in enumerate(groups):
        if group.size == 0:
            continue
        writers[i].write_table(
            pa.table(
                {
                    "hash": pa.array(group, type=pa.uint64()),
                    "sample_code": pa.array(
                        np.full(group.size, sample_code, dtype=np.uint32), type=pa.uint32()
                    ),
                    "assembly_code": pa.array(
                        np.full(group.size, assembly_code, dtype=np.uint32), type=pa.uint32()
                    ),
                },
                schema=_BUCKET_SCHEMA,
            )
        )


# --------------------------------------------------------------------------
# pass 2 -- count one bucket at a time
# --------------------------------------------------------------------------


def _count_bucket(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-hash ``(hash, n_samples, n_assemblies, n_bins)`` for one bucket.

    One ``lexsort`` on ``(hash, sample, assembly)`` answers all three questions:
    within a hash group the sample codes are non-decreasing, so a sample change
    starts a new distinct sample, and because an assembly belongs to exactly one
    sample a ``(sample, assembly)`` change starts a new distinct assembly.
    """
    empty_h = np.empty(0, dtype=np.uint64)
    empty_i = np.empty(0, dtype=np.int64)
    if not path.exists():
        return empty_h, empty_i, empty_i, empty_i
    table = pq.read_table(path, columns=["hash", "sample_code", "assembly_code"])
    n = table.num_rows
    if n == 0:
        return empty_h, empty_i, empty_i, empty_i

    hashes = np.asarray(table.column("hash").to_numpy(zero_copy_only=False), dtype=np.uint64)
    samples = np.asarray(table.column("sample_code").to_numpy(zero_copy_only=False), dtype=np.int64)
    assemblies = np.asarray(
        table.column("assembly_code").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    del table

    order = np.lexsort((assemblies, samples, hashes))
    hashes = hashes[order]
    samples = samples[order]
    assemblies = assemblies[order]
    del order

    new_hash = np.empty(n, dtype=bool)
    new_hash[0] = True
    np.not_equal(hashes[1:], hashes[:-1], out=new_hash[1:])
    starts = np.flatnonzero(new_hash)
    uniq_hash = hashes[starts]
    n_bins = np.diff(np.append(starts, n))

    group_id = np.cumsum(new_hash) - 1
    n_groups = starts.shape[0]

    new_sample = new_hash.copy()
    new_sample[1:] |= samples[1:] != samples[:-1]
    n_samples = np.bincount(group_id[new_sample], minlength=n_groups)

    # Deliberate aliasing: the sample counts are already extracted, and a
    # bucket-sized boolean array is worth not duplicating.
    new_assembly = new_sample
    new_assembly[1:] |= assemblies[1:] != assemblies[:-1]
    n_assemblies = np.bincount(group_id[new_assembly], minlength=n_groups)

    return (
        uniq_hash,
        n_samples.astype(np.int64),
        n_assemblies.astype(np.int64),
        n_bins.astype(np.int64),
    )


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def select_kmers(
    sketch_dir: Path,
    manifest: pd.DataFrame,
    cfg: Config,
    outdir: Path,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Pick the k-mer feature set; write ``kmers.parquet`` + ``prevalence.parquet``."""
    sketch_dir = Path(sketch_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    kmers_path = outdir / KMERS_FILENAME
    prevalence_path = outdir / PREVALENCE_FILENAME

    if kmers_path.exists() and prevalence_path.exists() and not force:
        logger.info("k-mer selection already present in %s; reusing", outdir)
        return load_kmers(outdir)

    manifest = _dedupe_manifest(manifest)
    sample_codes, sample_names = _sample_codes(manifest)
    n_samples_total = len(sample_names)
    n_buckets = int(cfg.select.n_buckets)
    bucket_dir = outdir / BUCKET_DIRNAME

    logger.info(
        "selecting k-mers from %d assemblies / %d samples into %d bucket(s)",
        len(manifest),
        n_samples_total,
        n_buckets,
    )

    with timed(logger, "pass 1: partitioning sketches by hash prefix"):
        total_rows, n_shards = _pass1_partition(
            sketch_dir, manifest, sample_codes, bucket_dir, n_buckets
        )
    logger.info("pass 1 wrote %d (bin, k-mer) rows from %d shard(s)", total_rows, n_shards)

    lo_count, hi_count = _prevalence_window(cfg, n_samples_total)
    min_bins = max(int(cfg.select.min_bins), 1)
    logger.info(
        "filters: n_bins >= %d, %d <= n_samples <= %d (of %d samples)",
        min_bins,
        lo_count,
        hi_count,
        n_samples_total,
    )

    hist_len = max(n_samples_total, 1) + 1
    histogram = np.zeros(hist_len, dtype=np.int64)
    keep_hash: list[np.ndarray] = []
    keep_samples: list[np.ndarray] = []
    keep_assemblies: list[np.ndarray] = []
    keep_bins: list[np.ndarray] = []
    n_distinct = 0
    n_after_bins = 0

    with timed(logger, "pass 2: counting per-hash prevalence"):
        for i in range(n_buckets):
            path = bucket_dir / f"bucket_{i:03d}.parquet"
            hashes, n_samples, n_assemblies, n_bins = _count_bucket(path)
            if hashes.size == 0:
                continue
            n_distinct += int(hashes.size)
            # The histogram covers *every* distinct k-mer seen, before any
            # filter, so users can see what the window is throwing away.
            observed_max = int(n_samples.max())
            if observed_max >= histogram.shape[0]:
                histogram = np.pad(histogram, (0, observed_max + 1 - histogram.shape[0]))
            histogram += np.bincount(n_samples, minlength=histogram.shape[0])

            keep = n_bins >= min_bins
            n_after_bins += int(keep.sum())
            keep &= (n_samples >= lo_count) & (n_samples <= hi_count)
            if not keep.any():
                continue
            keep_hash.append(hashes[keep])
            keep_samples.append(n_samples[keep])
            keep_assemblies.append(n_assemblies[keep])
            keep_bins.append(n_bins[keep])
            logger.debug("bucket %03d: %d distinct, %d kept", i, hashes.size, int(keep.sum()))

    if keep_hash:
        hashes = np.concatenate(keep_hash)
        n_samples = np.concatenate(keep_samples)
        n_assemblies = np.concatenate(keep_assemblies)
        n_bins = np.concatenate(keep_bins)
    else:
        hashes = np.empty(0, dtype=np.uint64)
        n_samples = np.empty(0, dtype=np.int64)
        n_assemblies = np.empty(0, dtype=np.int64)
        n_bins = np.empty(0, dtype=np.int64)
    del keep_hash, keep_samples, keep_assemblies, keep_bins

    # Buckets are keyed by the high bits and each bucket is emitted in ascending
    # hash order, so concatenating them in bucket order is already globally
    # sorted -- but a cheap check beats a silently broken searchsorted later.
    if hashes.size > 1 and not bool(np.all(hashes[1:] > hashes[:-1])):
        order = np.argsort(hashes, kind="stable")
        hashes, n_samples = hashes[order], n_samples[order]
        n_assemblies, n_bins = n_assemblies[order], n_bins[order]
        logger.warning("bucket concatenation was not sorted; re-sorted %d hashes", hashes.size)

    n_prevalent = int(hashes.size)
    max_features = int(cfg.select.max_features)
    threshold = subsample_threshold(n_prevalent, max_features)
    if threshold < _MASK64 and n_prevalent:
        seed64 = np.uint64(int(cfg.select.seed) & _MASK64)
        keys = splitmix64_array(hashes ^ seed64)
        # threshold < 0 means "the target is so much smaller than the survivor
        # count that not even one hash is expected"; keep nothing rather than
        # letting a hash of exactly 0 slip through.
        keep = (
            keys <= np.uint64(threshold)
            if threshold >= 0
            else np.zeros(hashes.shape[0], dtype=bool)
        )
        hashes = hashes[keep]
        n_samples = n_samples[keep]
        n_assemblies = n_assemblies[keep]
        n_bins = n_bins[keep]
        logger.info(
            "sub-sampled to ~%d features (threshold %d, kept %d)",
            max_features,
            threshold,
            int(hashes.size),
        )

    logger.info(
        "funnel: %d distinct k-mers -> %d after min_bins -> %d after prevalence -> %d selected",
        n_distinct,
        n_after_bins,
        n_prevalent,
        int(hashes.size),
    )
    if hashes.size == 0:
        logger.warning(
            "no k-mers survived selection; loosen select.min_sample_prevalence / "
            "select.min_bins or sketch more assemblies"
        )

    kmers = pd.DataFrame(
        {
            "hash": hashes,
            "col_idx": np.arange(hashes.size, dtype=np.int32),
            "n_samples": n_samples.astype(np.int32),
            "n_assemblies": n_assemblies.astype(np.int32),
            "n_bins": n_bins.astype(np.int64),
        }
    )
    kmers = schemas.enforce(kmers, schemas.KMER_COLUMNS)

    selected_flag = np.zeros(histogram.shape[0], dtype=bool)
    upper = min(hi_count, histogram.shape[0] - 1)
    if upper >= lo_count:
        selected_flag[lo_count : upper + 1] = True
    prevalence = pd.DataFrame(
        {
            "n_samples": np.arange(histogram.shape[0], dtype=np.int32),
            "n_kmers": histogram,
            "selected": selected_flag,
        }
    )
    prevalence = schemas.enforce(prevalence, schemas.PREVALENCE_HIST_COLUMNS)

    _atomic_write(pa.Table.from_pandas(kmers, preserve_index=False), kmers_path)
    _atomic_write(pa.Table.from_pandas(prevalence, preserve_index=False), prevalence_path)

    if _env_flag("KMER_DUST_KEEP_BUCKETS"):
        logger.info("KMER_DUST_KEEP_BUCKETS set; leaving %s in place", bucket_dir)
    else:
        _remove_buckets(bucket_dir)
    return kmers


def _remove_buckets(bucket_dir: Path) -> None:
    if not bucket_dir.exists():
        return
    for path in sorted(bucket_dir.glob("bucket_*.parquet")):
        path.unlink()
    if not any(bucket_dir.iterdir()):
        bucket_dir.rmdir()


def load_kmers(outdir: Path) -> pd.DataFrame:
    """Read back the feature set written by :func:`select_kmers`."""
    path = Path(outdir) / KMERS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no k-mer table at {path}; run the select stage first")
    frame = pq.read_table(path).to_pandas()
    if frame.empty:
        return schemas.empty_frame(schemas.KMER_COLUMNS)
    return schemas.enforce(frame, schemas.KMER_COLUMNS)


def load_prevalence(outdir: Path) -> pd.DataFrame:
    """Read back the prevalence histogram (diagnostics only)."""
    path = Path(outdir) / PREVALENCE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"no prevalence table at {path}; run the select stage first")
    frame = pq.read_table(path).to_pandas()
    if frame.empty:
        return schemas.empty_frame(schemas.PREVALENCE_HIST_COLUMNS)
    return schemas.enforce(frame, schemas.PREVALENCE_HIST_COLUMNS)
