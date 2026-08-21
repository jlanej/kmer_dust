"""Turn one assembly into two parquet shards: its bins and their k-mer hashes.

This is the only stage that touches sequence, and the only one whose cost scales
with genome size rather than with the number of bins, so everything here is
arranged around streaming.  A haplotype is read contig by contig in fixed-size
blocks; nothing bigger than one block of sequence is ever resident, which is
what lets a 3 Gb assembly sketch inside a few hundred megabytes.

The subtle part is the block boundary.  ``FastaSource.iter_contigs`` hands out
*contiguous, non-overlapping* blocks, so the k-mers that straddle a boundary --
the ``k-1`` of them whose first base is in one block and whose last base is in
the next -- exist in neither block on its own.  :class:`_ContigAccumulator`
carries the trailing ``k-1`` bases of every block onto the front of the next one
so that exactly those k-mers are emitted, exactly once.  The carried bases are
in turn preceded by enough ``N`` padding to put the buffer's first base on a bin
boundary, which means the bin indices that :func:`kmer_dust.hashing.sketch_contig`
computes (as ``start // bin_size`` within the buffer it was given) are correct
after a single constant shift.  ``N`` never produces a k-mer, so the padding is
invisible to the result.  The upshot is the property the test suite asserts:
the shard is byte-identical for any block size, including one larger than the
whole contig.

A bin is a *set* of hashes, not a multiset -- a satellite array that repeats one
31-mer ten thousand times must not out-vote a unique region -- so duplicates
within a bin are collapsed before the shard is written.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import schemas
from .config import Config
from .fasta import FastaSource, load_chrom_alias, normalize_chrom
from .hashing import (
    bin_base_stats,
    bin_valid_kmer_counts,
    encode_bases,
    max_hash_for_scaled,
    sketch_contig,
)
from .log import get_logger

logger = get_logger(__name__)

__all__ = [
    "sketch_assembly",
    "sketch_manifest",
    "load_sketch_shard",
    "sketch_shard_paths",
    "SUMMARY_COLUMNS",
    "SKETCH_BLOCK",
]

#: Sequence bytes pulled per read.  Large enough that per-block NumPy and HTTP
#: overheads vanish, small enough that the O(8n) temporaries inside the per-bin
#: helpers stay well under a gigabyte per worker process.
SKETCH_BLOCK: int = 8_000_000

#: Columns of the frame returned by :func:`sketch_manifest`.  ``cached`` and
#: ``error`` are additive: ``status`` is exactly ``ok`` or ``failed`` so that
#: ``summary[summary.status != "ok"]`` means what it looks like.
SUMMARY_COLUMNS: dict[str, str] = {
    "assembly": "string",
    "n_bins": "int64",
    "n_hashes": "int64",
    "n_contigs": "int64",
    "seconds": "float64",
    "status": "string",
    "cached": "bool",
    "error": "string",
}

#: Override the parallelism strategy without touching the config:
#: ``process`` | ``thread`` | ``serial``.
EXECUTOR_ENV: str = "KMER_DUST_SKETCH_EXECUTOR"
#: Processes by default -- the numba kernel only drops the GIL inside itself and
#: the surrounding FASTA parsing and pandas work is pure Python.
DEFAULT_EXECUTOR: str = "process"

#: Bumped when a change would make an existing shard incompatible.
_SHARD_FORMAT: int = 1


# --------------------------------------------------------------------------
# shard paths and IO
# --------------------------------------------------------------------------


def sketch_shard_paths(outdir: Path, assembly: str) -> dict[str, Path]:
    """The three files that make up one assembly's shard."""
    base = Path(outdir)
    return {
        "bins": base / f"{assembly}.bins.parquet",
        "sketch": base / f"{assembly}.sketch.parquet",
        "done": base / f"{assembly}.done",
    }


def load_sketch_shard(outdir: Path, assembly: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read back one shard as ``(bins, sketch)`` with contract dtypes restored.

    Parquet round-trips lose the ``string`` extension dtype under some pandas
    builds, so both frames go back through :func:`schemas.enforce`; every
    downstream stage can then assume the contract holds regardless of who wrote
    the file.
    """
    paths = sketch_shard_paths(outdir, assembly)
    missing = [name for name in ("bins", "sketch") if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(
            f"sketch shard for {assembly!r} is incomplete: missing {missing} in {outdir}"
        )
    bins = schemas.enforce(pd.read_parquet(paths["bins"]), schemas.BIN_COLUMNS)
    sketch = schemas.enforce(pd.read_parquet(paths["sketch"]), schemas.SKETCH_COLUMNS)
    return bins, sketch


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomic parquet write.  Snappy is pinned so bytes are reproducible."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _shard_params(cfg: Config) -> dict[str, int]:
    """The parameters a shard's contents depend on."""
    return {
        "format": _SHARD_FORMAT,
        "k": int(cfg.sketch.k),
        "bin_size": int(cfg.sketch.bin_size),
        "scaled": int(cfg.sketch.scaled),
        "min_bin_acgt_frac": float(cfg.sketch.min_bin_acgt_frac),
        "min_bin_sketch": int(cfg.sketch.min_bin_sketch),
        "drop_partial_terminal_bin": bool(cfg.sketch.drop_partial_terminal_bin),
        "include_unplaced": bool(cfg.sketch.include_unplaced),
        "chroms": sorted(str(c) for c in cfg.manifest.chroms),
    }


def _read_done(path: Path) -> dict[str, Any] | None:
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("unreadable marker %s (%s); re-sketching", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _completed_shard(paths: Mapping[str, Path], cfg: Config) -> dict[str, Any] | None:
    """Return the recorded stats when the shard on disk is usable, else ``None``.

    A marker written under different sketch parameters is *not* reusable: mixing
    k=21 and k=31 shards in one matrix would produce a plausible-looking, wrong
    answer, which is worse than redoing the work.
    """
    if not all(paths[name].exists() for name in ("bins", "sketch", "done")):
        return None
    data = _read_done(paths["done"])
    if data is None:
        return None
    if data.get("params") != _shard_params(cfg):
        logger.warning(
            "%s was sketched with different parameters; re-sketching", paths["done"].name
        )
        return None
    return data


# --------------------------------------------------------------------------
# per-contig streaming accumulator
# --------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class _ContigAccumulator:
    """Fold a stream of contiguous blocks into per-bin counters and hashes.

    The counters grow as the stream reveals the contig's length, because the
    streaming (unindexed) FASTA path cannot tell us the length in advance.
    """

    __slots__ = (
        "k",
        "bin_size",
        "max_hash",
        "length",
        "n_acgt",
        "n_gc",
        "n_kmers",
        "_carry",
        "_hash_bins",
        "_hashes",
    )

    def __init__(self, k: int, bin_size: int, max_hash: int):
        self.k = int(k)
        self.bin_size = int(bin_size)
        self.max_hash = int(max_hash)
        self.length = 0
        self.n_acgt = np.zeros(0, dtype=np.int64)
        self.n_gc = np.zeros(0, dtype=np.int64)
        self.n_kmers = np.zeros(0, dtype=np.int64)
        self._carry = b""
        self._hash_bins: list[np.ndarray] = []
        self._hashes: list[np.ndarray] = []

    def _grow(self, n_bins: int) -> None:
        if n_bins <= self.n_acgt.size:
            return
        size = max(n_bins, 2 * self.n_acgt.size, 64)
        for name in ("n_acgt", "n_gc", "n_kmers"):
            old = getattr(self, name)
            new = np.zeros(size, dtype=np.int64)
            new[: old.size] = old
            setattr(self, name, new)

    def add_block(self, block: bytes) -> None:
        if not block:
            return
        bin_size = self.bin_size
        offset = self.length

        # -- base composition: the block alone, shifted onto the bin grid -----
        stats_base = (offset // bin_size) * bin_size
        stats_pad = offset - stats_base
        codes = encode_bases(block)
        if stats_pad:
            codes = np.concatenate((np.full(stats_pad, 255, dtype=np.uint8), codes))
        n_local = _ceil_div(codes.size, bin_size)
        first = stats_base // bin_size
        self._grow(first + n_local)
        acgt, gc = bin_base_stats(codes, bin_size, n_local)
        self.n_acgt[first : first + n_local] += acgt
        self.n_gc[first : first + n_local] += gc

        # -- k-mers: the carried k-1 bases, this block, and grid padding ------
        # Every k-mer whose first base lies in [offset - len(carry), end - k] is
        # produced here and nowhere else, so the union over blocks is exactly
        # the set of k-mers of the contig, with no gaps and no repeats.
        carry = self._carry
        start_pos = offset - len(carry)
        base = (start_pos // bin_size) * bin_size
        pad = start_pos - base
        buffer = (b"N" * pad + carry + block) if (pad or carry) else block
        kcodes = encode_bases(buffer)
        n_local = _ceil_div(kcodes.size, bin_size)
        first = base // bin_size
        self._grow(first + n_local)
        self.n_kmers[first : first + n_local] += bin_valid_kmer_counts(
            kcodes, self.k, bin_size, n_local
        )
        bin_idx, hashes = sketch_contig(
            kcodes, k=self.k, bin_size=bin_size, max_hash=self.max_hash
        )
        if bin_idx.size:
            self._hash_bins.append(bin_idx.astype(np.int64) + first)
            self._hashes.append(hashes)

        self.length = offset + len(block)
        self._carry = (carry + block)[-(self.k - 1) :] if self.k > 1 else b""

    def finish(self, *, drop_partial: bool) -> dict[str, np.ndarray]:
        """Per-bin arrays for the whole contig, before the drop rules apply."""
        length = self.length
        n_bins = (length // self.bin_size) if drop_partial else _ceil_div(length, self.bin_size)
        if n_bins <= 0:
            return {
                "start": np.zeros(0, dtype=np.int64),
                "end": np.zeros(0, dtype=np.int64),
                "n_acgt": np.zeros(0, dtype=np.int64),
                "n_gc": np.zeros(0, dtype=np.int64),
                "n_kmers": np.zeros(0, dtype=np.int64),
                "n_sketch": np.zeros(0, dtype=np.int64),
                "hash_bin": np.zeros(0, dtype=np.int64),
                "hash": np.zeros(0, dtype=np.uint64),
            }
        self._grow(n_bins)
        starts = np.arange(n_bins, dtype=np.int64) * self.bin_size
        ends = np.minimum(starts + self.bin_size, length)

        if self._hash_bins:
            hash_bin = np.concatenate(self._hash_bins)
            hash_val = np.concatenate(self._hashes)
        else:
            hash_bin = np.zeros(0, dtype=np.int64)
            hash_val = np.zeros(0, dtype=np.uint64)
        keep = hash_bin < n_bins  # a dropped partial terminal bin takes its hashes with it
        hash_bin, hash_val = hash_bin[keep], hash_val[keep]
        if hash_bin.size:
            # Sort by (bin, hash) and collapse repeats: a bin is a set.
            order = np.lexsort((hash_val, hash_bin))
            hash_bin, hash_val = hash_bin[order], hash_val[order]
            unique = np.ones(hash_bin.size, dtype=bool)
            np.not_equal(hash_bin[1:], hash_bin[:-1], out=unique[1:])
            unique[1:] |= hash_val[1:] != hash_val[:-1]
            hash_bin, hash_val = hash_bin[unique], hash_val[unique]
        n_sketch = np.bincount(hash_bin, minlength=n_bins)[:n_bins].astype(np.int64)

        return {
            "start": starts,
            "end": ends,
            "n_acgt": self.n_acgt[:n_bins].copy(),
            "n_gc": self.n_gc[:n_bins].copy(),
            "n_kmers": self.n_kmers[:n_bins].copy(),
            "n_sketch": n_sketch,
            "hash_bin": hash_bin,
            "hash": hash_val,
        }


# --------------------------------------------------------------------------
# contig selection
# --------------------------------------------------------------------------


def _chrom_of(contig: str, alias: Mapping[str, str]) -> str:
    """Canonical chromosome for a contig, or ``""`` when it is unplaced.

    The alias map is consulted first and its answer is authoritative: PanSN
    names like ``HG00408#1#CM085953.1`` carry a GenBank accession that only the
    assembly's own chromAlias file can resolve, and guessing is not an option.
    """
    aliased = alias.get(contig, "")
    return normalize_chrom(aliased) if aliased else normalize_chrom(contig)


def _contig_filter(cfg: Config, alias: Mapping[str, str]):
    """Build the ``contig -> bool`` predicate implied by the config."""
    raw = [str(c).strip() for c in cfg.manifest.chroms if str(c).strip()]
    if not raw:
        return lambda contig: True  # empty chroms list means "everything"
    wanted = {normalize_chrom(c) for c in raw} - {""}
    literal = set(raw)  # lets a local FASTA be selected by its own contig names
    include_unplaced = bool(cfg.sketch.include_unplaced)

    def keep(contig: str) -> bool:
        chrom = _chrom_of(contig, alias)
        if chrom and chrom in wanted:
            return True
        if contig in literal:
            return True
        return include_unplaced and not chrom

    return keep


# --------------------------------------------------------------------------
# the sketching itself
# --------------------------------------------------------------------------


def _empty_shard_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return schemas.empty_frame(schemas.BIN_COLUMNS), schemas.empty_frame(schemas.SKETCH_COLUMNS)


def _build_frames(
    row: Mapping[str, Any], cfg: Config, per_contig: Sequence[tuple[str, str, dict[str, np.ndarray]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the two shard frames from the surviving bins of every contig."""
    assembly = str(row.get("assembly", ""))
    min_frac = float(cfg.sketch.min_bin_acgt_frac)
    min_sketch = int(cfg.sketch.min_bin_sketch)

    bin_chunks: list[dict[str, np.ndarray]] = []
    contigs: list[str] = []
    chroms: list[str] = []
    hash_bin_chunks: list[np.ndarray] = []
    hash_chunks: list[np.ndarray] = []
    next_idx = 0

    for contig, chrom, data in per_contig:
        span = data["end"] - data["start"]
        n_acgt = data["n_acgt"]
        with np.errstate(invalid="ignore", divide="ignore"):
            acgt_frac = np.where(span > 0, n_acgt / np.maximum(span, 1), 0.0)
        keep = (acgt_frac >= min_frac) & (data["n_sketch"] >= min_sketch)
        n_keep = int(keep.sum())
        if n_keep == 0:
            continue
        kept = np.nonzero(keep)[0]
        # Old local bin index -> new shard-local bin_idx; -1 marks a dropped bin.
        remap = np.full(keep.size, -1, dtype=np.int64)
        remap[kept] = np.arange(next_idx, next_idx + n_keep, dtype=np.int64)

        hb = data["hash_bin"]
        if hb.size:
            mapped = remap[hb]
            alive = mapped >= 0
            if alive.any():
                hash_bin_chunks.append(mapped[alive])
                hash_chunks.append(data["hash"][alive])

        bin_chunks.append(
            {
                "bin_idx": remap[kept],
                "start": data["start"][kept],
                "end": data["end"][kept],
                "n_acgt": n_acgt[kept],
                "n_gc": data["n_gc"][kept],
                "n_kmers": data["n_kmers"][kept],
                "n_sketch": data["n_sketch"][kept],
            }
        )
        contigs.extend([contig] * n_keep)
        chroms.extend([chrom] * n_keep)
        next_idx += n_keep

    if not bin_chunks:
        return _empty_shard_frames()

    def stack(name: str) -> np.ndarray:
        return np.concatenate([chunk[name] for chunk in bin_chunks])

    starts = stack("start")
    ends = stack("end")
    n_acgt = stack("n_acgt")
    n_gc = stack("n_gc")
    span = ends - starts
    with np.errstate(invalid="ignore", divide="ignore"):
        gc = np.where(n_acgt > 0, n_gc / np.maximum(n_acgt, 1), np.nan)
        nfrac = 1.0 - np.where(span > 0, n_acgt / np.maximum(span, 1), 0.0)

    bins = pd.DataFrame(
        {
            "bin_idx": stack("bin_idx").astype(np.int32),
            "bin_uid": pd.array(
                [schemas.bin_uid(assembly, c, int(s)) for c, s in zip(contigs, starts)],
                dtype="string",
            ),
            "assembly": assembly,
            "sample": str(row.get("sample", "") or ""),
            "haplotype": str(row.get("haplotype", "") or ""),
            "source": str(row.get("source", "") or ""),
            "contig": pd.array(contigs, dtype="string"),
            "chrom": pd.array(chroms, dtype="string"),
            "start": starts,
            "end": ends,
            "n_acgt": n_acgt,
            "n_kmers": stack("n_kmers"),
            "n_sketch": stack("n_sketch").astype(np.int32),
            "gc": gc.astype(np.float32),
            "nfrac": nfrac.astype(np.float32),
        }
    )
    if hash_bin_chunks:
        sketch = pd.DataFrame(
            {
                "bin_idx": np.concatenate(hash_bin_chunks).astype(np.int32),
                "hash": np.concatenate(hash_chunks).astype(np.uint64),
            }
        )
    else:
        sketch = schemas.empty_frame(schemas.SKETCH_COLUMNS)
    return (
        schemas.enforce(bins, schemas.BIN_COLUMNS),
        schemas.enforce(sketch, schemas.SKETCH_COLUMNS),
    )


def sketch_assembly(
    row: Mapping[str, Any],
    cfg: Config,
    outdir: Path,
    *,
    force: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Sketch one manifest row into ``<outdir>/<assembly>.{bins,sketch}.parquet``.

    Returns the path of the ``.done`` marker, which is written last: its
    existence (with matching parameters) is the definition of "this shard is
    complete", and is what makes the stage restartable after a killed job.
    """
    assembly = str(row.get("assembly", "") or "")
    if not assembly:
        raise ValueError("manifest row has no 'assembly'")
    fasta = str(row.get("fasta", "") or "")
    if not fasta:
        raise ValueError(f"manifest row {assembly!r} has no 'fasta'")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = sketch_shard_paths(outdir, assembly)
    if not force and _completed_shard(paths, cfg) is not None:
        logger.info("sketch %s: already done, skipping", assembly)
        return paths["done"]

    k = int(cfg.sketch.k)
    bin_size = int(cfg.sketch.bin_size)
    max_hash = max_hash_for_scaled(int(cfg.sketch.scaled))
    alias = load_chrom_alias(str(row.get("chrom_alias", "") or ""), cache_dir)
    keep_contig = _contig_filter(cfg, alias)

    per_contig: list[tuple[str, str, dict[str, np.ndarray]]] = []
    seen: set[str] = set()
    source = FastaSource(
        fasta,
        str(row.get("fai", "") or ""),
        str(row.get("gzi", "") or ""),
        cache_dir=cache_dir,
    )
    stream: Iterator[tuple[str, int, bytes]] | None = None
    try:
        if source.indexed:
            selected: list[str] | None = [c for c in source.contigs if keep_contig(c)]
            logger.info(
                "sketch %s: %d/%d contigs selected",
                assembly,
                len(selected),
                len(source.contigs),
            )
            stream = source.iter_contigs(selected, block=SKETCH_BLOCK)
        else:
            selected = None  # filter on the fly; a stream cannot be seeked
            stream = source.iter_contigs(None, block=SKETCH_BLOCK)

        current = ""
        accumulator: _ContigAccumulator | None = None

        def flush() -> None:
            if accumulator is None:
                return
            data = accumulator.finish(drop_partial=bool(cfg.sketch.drop_partial_terminal_bin))
            if data["start"].size:
                per_contig.append((current, _chrom_of(current, alias), data))

        for name, _length, block in stream:
            if selected is None and not keep_contig(name):
                continue
            if name != current or accumulator is None:
                flush()
                if name in seen:
                    logger.warning(
                        "%s: contig %r appears more than once; bin ids will collide",
                        assembly,
                        name,
                    )
                seen.add(name)
                current = name
                accumulator = _ContigAccumulator(k, bin_size, max_hash)
            accumulator.add_block(block)
        flush()
    finally:
        if stream is not None:
            stream.close()  # release the handle if we bailed out mid-contig
        source.close()

    bins, hashes = _build_frames(row, cfg, per_contig)
    _write_parquet(bins, paths["bins"])
    _write_parquet(hashes, paths["sketch"])

    marker = {
        "assembly": assembly,
        "n_bins": int(len(bins)),
        "n_hashes": int(len(hashes)),
        "n_contigs": int(bins["contig"].nunique()) if len(bins) else 0,
        "params": _shard_params(cfg),
    }
    tmp = paths["done"].with_name(f"{paths['done'].name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as handle:
            json.dump(marker, handle, sort_keys=True, indent=2)
        os.replace(tmp, paths["done"])
    finally:
        tmp.unlink(missing_ok=True)
    logger.info(
        "sketch %s: %d bins, %d hashes, %d contigs",
        assembly,
        marker["n_bins"],
        marker["n_hashes"],
        marker["n_contigs"],
    )
    return paths["done"]


# --------------------------------------------------------------------------
# manifest-level driver
# --------------------------------------------------------------------------


def _row_result(
    row: Mapping[str, Any],
    cfg: Config,
    outdir: Path,
    force: bool,
    cache_dir: Path | None,
) -> dict[str, Any]:
    """Sketch one row, returning a summary dict.  Never raises for data reasons."""
    assembly = str(row.get("assembly", "") or "")
    started = time.perf_counter()
    paths = sketch_shard_paths(Path(outdir), assembly)
    cached = _completed_shard(paths, cfg) if (assembly and not force) else None
    if cached is not None:
        return {
            "assembly": assembly,
            "n_bins": int(cached.get("n_bins", 0)),
            "n_hashes": int(cached.get("n_hashes", 0)),
            "n_contigs": int(cached.get("n_contigs", 0)),
            "seconds": 0.0,
            "status": "ok",
            "cached": True,
            "error": "",
        }
    try:
        done = sketch_assembly(row, cfg, outdir, force=force, cache_dir=cache_dir)
        marker = _read_done(done) or {}
        return {
            "assembly": assembly,
            "n_bins": int(marker.get("n_bins", 0)),
            "n_hashes": int(marker.get("n_hashes", 0)),
            "n_contigs": int(marker.get("n_contigs", 0)),
            "seconds": time.perf_counter() - started,
            "status": "ok",
            "cached": False,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - one bad assembly must not kill the run
        logger.exception("sketch %s failed", assembly or "<unnamed>")
        failed = _failure(assembly, exc)
        failed["seconds"] = time.perf_counter() - started
        return failed


def _sketch_worker(
    row: dict[str, Any], cfg: Config, outdir: str, force: bool, cache_dir: str | None
) -> dict[str, Any]:
    """Module-level entry point so a spawned process can unpickle it."""
    return _row_result(row, cfg, Path(outdir), force, Path(cache_dir) if cache_dir else None)


def _scalar(value: Any) -> Any:
    """Manifest cells arrive as ``pd.NA``/``NaN``/``None``; workers want ``""``."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):  # arrays and other non-scalars
        return value
    return value


def _executor_kind(threads: int) -> str:
    override = os.environ.get(EXECUTOR_ENV, "").strip().lower()
    if override in {"process", "thread", "serial"}:
        return override
    # One thread means "I am debugging"; run inline so tracebacks are real.
    return "serial" if threads <= 1 else DEFAULT_EXECUTOR


def _failure(assembly: str, exc: BaseException) -> dict[str, Any]:
    return {
        "assembly": assembly,
        "n_bins": 0,
        "n_hashes": 0,
        "n_contigs": 0,
        "seconds": 0.0,
        "status": "failed",
        "cached": False,
        "error": f"{type(exc).__name__}: {exc}"[:500].replace("\n", " "),
    }


def _run_pool(
    kind: str,
    n_threads: int,
    rows: Sequence[Mapping[str, Any]],
    cfg: Config,
    outdir: Path,
    force: bool,
    cache_dir: Path | None,
) -> list[dict[str, Any]]:
    """Submit every row to a pool and collect results back in *submission* order."""
    pool_cls = cf.ProcessPoolExecutor if kind == "process" else cf.ThreadPoolExecutor
    cache_str = str(cache_dir) if cache_dir is not None else None
    results: list[dict[str, Any] | None] = [None] * len(rows)
    try:
        executor = pool_cls(max_workers=n_threads)
    except (OSError, ValueError) as exc:  # pragma: no cover - platform dependent
        logger.warning("could not start a %s pool (%s); falling back to threads", kind, exc)
        executor = cf.ThreadPoolExecutor(max_workers=n_threads)
    with executor:
        pending = {
            executor.submit(_sketch_worker, dict(row), cfg, str(outdir), force, cache_str): i
            for i, row in enumerate(rows)
        }
        for future in cf.as_completed(pending):
            i = pending[future]
            try:
                results[i] = future.result()
            except Exception as exc:  # noqa: BLE001 - includes BrokenProcessPool
                logger.error("sketch worker for %s died: %s", rows[i].get("assembly"), exc)
                results[i] = _failure(str(rows[i].get("assembly", "") or ""), exc)
    return [r for r in results if r is not None]


def sketch_manifest(
    manifest: pd.DataFrame,
    cfg: Config,
    *,
    threads: int | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Sketch every row of ``manifest``, in parallel, tolerating failures.

    Returns one row per assembly in *manifest order* (never completion order, so
    the summary is deterministic) with columns ``assembly, n_bins, n_hashes,
    n_contigs, seconds, status`` plus ``cached`` and ``error``.  A row that
    raises is reported as ``status == "failed"`` and the run continues: losing
    one of 464 haplotypes to a flaky S3 read should not cost the other 463.
    """
    outdir = cfg.stage_dir("sketch")
    empty = schemas.empty_frame(SUMMARY_COLUMNS)
    if manifest is None or len(manifest) == 0:
        logger.warning("sketch: empty manifest, nothing to do")
        return empty
    missing = [c for c in ("assembly", "fasta") if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing required column(s): {missing}")

    rows: list[dict[str, Any]] = [
        {str(key): _scalar(value) for key, value in record.items()}
        for record in manifest.to_dict(orient="records")
    ]
    n_threads = int(threads if threads is not None else (cfg.sketch.threads or cfg.threads or 1))
    n_threads = max(1, min(n_threads, len(rows)))
    kind = _executor_kind(n_threads)
    logger.info("sketch: %d assemblies, %d worker(s), executor=%s", len(rows), n_threads, kind)

    if kind == "serial":
        results = [_row_result(row, cfg, outdir, force, cache_dir) for row in rows]
    else:
        results = _run_pool(kind, n_threads, rows, cfg, outdir, force, cache_dir)
        if kind == "process" and results and all(
            r["error"].startswith("BrokenProcessPool") for r in results
        ):
            # Every worker died before doing anything: almost always a spawn
            # start-method problem in the caller (no ``if __name__ ==
            # "__main__"`` guard), not a data problem.  Threads still work.
            logger.warning("process pool unusable; retrying every assembly with threads")
            results = _run_pool("thread", n_threads, rows, cfg, outdir, force, cache_dir)

    filled = [r for r in results if r is not None]
    if not filled:
        return empty
    summary = schemas.enforce(pd.DataFrame(filled), SUMMARY_COLUMNS)
    n_failed = int((summary["status"] != "ok").sum())
    logger.info(
        "sketch: %d ok, %d failed, %d bins, %d hashes",
        len(summary) - n_failed,
        n_failed,
        int(summary["n_bins"].sum()),
        int(summary["n_hashes"].sum()),
    )
    return summary
