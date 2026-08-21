"""Assembling the sparse bin x k-mer matrix.

Everything upstream is per-assembly and embarrassingly parallel; this is the
first stage that has to see the whole run at once, so it is the stage where the
memory budget is decided.  Two choices keep it cheap:

* **The columns are found by ``searchsorted``, not by a dict.**  ``kmers.hash``
  is sorted, so a whole shard's hashes are mapped to column indices with one
  binary search per hash and no Python-level hashing.  The insertion point that
  ``searchsorted`` returns is only a *candidate*; it is confirmed by an equality
  test, because a hash that was filtered out in the select stage would otherwise
  silently steal its neighbour's column.
* **The rows arrive already grouped.**  Sketch shards are sorted by ``bin_idx``
  and read in manifest order, so CSR ``indptr`` can be accumulated directly from
  per-bin counts and no ``(row, col)`` COO triple ever has to be materialised
  and sorted.

Weighting is applied *after* empty rows are dropped, and the document frequency
behind ``idf`` is recomputed from the matrix that actually exists.  Reusing
``kmers.n_bins`` would be subtly wrong: it was counted over all sketched bins,
including bins that this matrix does not contain, so the ``log(n_rows / df)``
would not be a log of a fraction of anything.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from . import schemas
from .config import Config
from .log import get_logger, timed

logger = get_logger(__name__)

__all__ = ["build_matrix", "load_matrix", "MATRIX_FILENAME", "ROWS_FILENAME"]

MATRIX_FILENAME = "matrix.npz"
ROWS_FILENAME = "rows.parquet"

_READ_BATCH_ROWS = 1 << 20


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _dedupe_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if "assembly" not in manifest.columns:
        raise ValueError("manifest must have an 'assembly' column")
    dup = manifest["assembly"].duplicated()
    if bool(dup.any()):
        logger.warning(
            "manifest has %d duplicate assembly row(s); keeping the first", int(dup.sum())
        )
        manifest = manifest.loc[~dup]
    return manifest.reset_index(drop=True)


def _feature_index(kmers: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """``(sorted hashes, matching col_idx, n_cols)`` for the membership test."""
    if kmers is None or len(kmers) == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int32), 0
    for column in ("hash", "col_idx"):
        if column not in kmers.columns:
            raise ValueError(f"k-mer table is missing the {column!r} column")
    hashes = np.asarray(kmers["hash"].to_numpy(), dtype=np.uint64)
    cols = np.asarray(kmers["col_idx"].to_numpy(), dtype=np.int64)
    n_cols = int(cols.max()) + 1 if cols.size else 0
    if n_cols != len(kmers):
        logger.warning(
            "k-mer col_idx is not a dense 0..n-1 range (max %d, %d rows); using %d columns",
            n_cols - 1,
            len(kmers),
            max(n_cols, len(kmers)),
        )
        n_cols = max(n_cols, len(kmers))
    if cols.min(initial=0) < 0:
        raise ValueError("k-mer col_idx contains negative values")
    if hashes.size > 1 and not bool(np.all(hashes[1:] > hashes[:-1])):
        order = np.argsort(hashes, kind="stable")
        hashes, cols = hashes[order], cols[order]
        logger.warning("k-mer table was not sorted by hash; sorted a local copy")
    return hashes, cols.astype(np.int32), n_cols


def _iter_sketch_batches(path: Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    handle = pq.ParquetFile(path)
    names = set(handle.schema_arrow.names)
    missing = {"bin_idx", "hash"} - names
    if missing:
        raise ValueError(f"{path} is missing sketch column(s) {sorted(missing)}")
    for batch in handle.iter_batches(batch_size=_READ_BATCH_ROWS, columns=["bin_idx", "hash"]):
        if batch.num_rows == 0:
            continue
        bins = batch.column("bin_idx")
        hashes = batch.column("hash")
        if bins.null_count or hashes.null_count:
            raise ValueError(f"{path} contains null bin_idx/hash; the shard is corrupt")
        yield (
            np.asarray(bins.to_numpy(zero_copy_only=False), dtype=np.int64),
            np.asarray(hashes.to_numpy(zero_copy_only=False), dtype=np.uint64),
        )


def _read_bins(path: Path) -> pd.DataFrame:
    frame = pq.read_table(path).to_pandas()
    if frame.empty:
        return schemas.empty_frame(schemas.BIN_COLUMNS)
    frame = schemas.enforce(frame, schemas.BIN_COLUMNS)
    frame = frame.sort_values("bin_idx", kind="stable").reset_index(drop=True)
    if bool(frame["bin_idx"].duplicated().any()):
        logger.warning(
            "%s has duplicate bin_idx values; k-mers will be attributed to the first copy",
            path.name,
        )
    return frame


def _row_sums(values: np.ndarray, indptr: np.ndarray, n_rows: int) -> np.ndarray:
    """Per-row sums of a CSR value array without materialising row indices.

    ``np.add.reduceat`` returns ``values[start]`` instead of 0 for an empty
    segment, so empty rows are zeroed afterwards.
    """
    if n_rows == 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(n_rows, dtype=np.float64)
    starts = np.minimum(indptr[:-1], values.size - 1)
    sums = np.add.reduceat(values.astype(np.float64, copy=False), starts)
    sums[np.diff(indptr) == 0] = 0.0
    return sums


def _apply_weighting(matrix: sparse.csr_matrix, weighting: str) -> None:
    """In-place value weighting.  ``matrix.data`` is float32 throughout."""
    n_rows, n_cols = matrix.shape
    if weighting == "none" or matrix.nnz == 0:
        return
    if weighting == "log":
        matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
        return
    if weighting != "idf":
        raise ValueError(f"unknown matrix.weighting {weighting!r}")
    df = np.bincount(matrix.indices, minlength=n_cols).astype(np.float64)
    idf = np.zeros(n_cols, dtype=np.float64)
    present = df > 0
    idf[present] = np.log(n_rows / df[present])
    # A k-mer in literally every row gets idf 0: it separates nothing.  The
    # entry is kept as an explicit zero so the sparsity pattern still records
    # "this bin has this k-mer" for anything that wants to inspect it.
    matrix.data *= idf.astype(np.float32)[matrix.indices]


def _apply_row_norm(matrix: sparse.csr_matrix, row_norm: str) -> None:
    if row_norm == "none" or matrix.nnz == 0:
        return
    values = matrix.data.astype(np.float64)
    if row_norm == "l1":
        norms = _row_sums(np.abs(values), matrix.indptr, matrix.shape[0])
    elif row_norm == "l2":
        norms = np.sqrt(_row_sums(np.square(values), matrix.indptr, matrix.shape[0]))
    else:
        raise ValueError(f"unknown matrix.row_norm {row_norm!r}")
    del values
    # Rows that are empty, or all-zero after idf, keep their zeros.
    norms[norms == 0.0] = 1.0
    matrix.data *= np.repeat((1.0 / norms).astype(np.float32), np.diff(matrix.indptr))


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="snappy")
    os.replace(tmp, path)


def _empty_rows_frame() -> pd.DataFrame:
    frame = schemas.empty_frame(schemas.BIN_COLUMNS)
    frame["row_idx"] = pd.Series([], dtype="int64")
    return frame


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------



#: Below this many selected k-mers per bin the cosine geometry stops meaning
#: anything: most pairs of bins share no feature at all, so their similarity is
#: exactly zero and the embedding is dominated by ties.  Healthy runs observed
#: so far sit at 40-45.
MIN_HEALTHY_NNZ_PER_ROW = 10.0


def _warn_if_too_sparse(matrix: sparse.csr_matrix, cfg: Config) -> None:
    """Catch the failure mode where the feature budget did not scale with the run.

    ``select.max_features`` is an absolute cap, but the number of *eligible*
    k-mers grows with the amount of sequence in the run.  Take a config tuned on
    60k bins to 1.3M bins and the same 200k cap now keeps 11 % of the eligible
    vocabulary instead of most of it, the matrix thins from ~44 non-zeros per row
    to ~5, and the clustering quietly degenerates -- with nothing in the log
    saying so, because every stage still "succeeded".
    """
    n_rows = int(matrix.shape[0])
    if n_rows == 0:
        return
    per_row = matrix.nnz / n_rows
    if per_row >= MIN_HEALTHY_NNZ_PER_ROW:
        return
    expected = cfg.sketch.bin_size / max(cfg.sketch.scaled, 1)
    logger.warning(
        "matrix averages only %.1f selected k-mers per bin (a %d bp bin sketched at "
        "scaled=%d holds ~%.0f). Most bin pairs now share nothing, so the cosine "
        "geometry is mostly ties and the clustering will be poor. Raise "
        "select.max_features (currently %s) so it scales with the %d bins in this "
        "run, or lower sketch.scaled.",
        per_row,
        cfg.sketch.bin_size,
        cfg.sketch.scaled,
        expected,
        cfg.select.max_features or "unlimited",
        n_rows,
    )


def build_matrix(
    sketch_dir: Path,
    kmers: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: Config,
    outdir: Path,
    *,
    force: bool = False,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Build, weight and persist the bin x k-mer CSR matrix."""
    sketch_dir = Path(sketch_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matrix_path = outdir / MATRIX_FILENAME
    rows_path = outdir / ROWS_FILENAME

    if matrix_path.exists() and rows_path.exists() and not force:
        logger.info("matrix already present in %s; reusing", outdir)
        return load_matrix(outdir)

    manifest = _dedupe_manifest(manifest)
    hashes, hash_cols, n_cols = _feature_index(kmers)
    if n_cols == 0:
        logger.warning("the k-mer table is empty; the matrix will have no columns")

    bin_frames: list[pd.DataFrame] = []
    col_chunks: list[np.ndarray] = []
    row_counts: list[np.ndarray] = []
    n_rows = 0
    n_shards = 0

    with timed(logger, "building the bin x k-mer matrix"):
        for assembly in manifest["assembly"].astype(str).to_list():
            bins_path = sketch_dir / f"{assembly}.bins.parquet"
            sketch_path = sketch_dir / f"{assembly}.sketch.parquet"
            if not bins_path.exists():
                logger.warning("no bin table for %s (%s); skipping", assembly, bins_path.name)
                continue
            bins = _read_bins(bins_path)
            if bins.empty:
                logger.debug("%s has no bins", assembly)
                continue
            n_shards += 1
            bin_ids = bins["bin_idx"].to_numpy(dtype=np.int64)
            counts, cols = _shard_entries(sketch_path, bin_ids, hashes, hash_cols, assembly)
            bin_frames.append(bins)
            row_counts.append(counts)
            col_chunks.append(cols)
            n_rows += len(bins)

    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    if row_counts:
        np.cumsum(np.concatenate(row_counts), out=indptr[1:])
    indices = np.concatenate(col_chunks) if col_chunks else np.empty(0, dtype=np.int32)
    del row_counts, col_chunks
    nnz = int(indices.size)
    if nnz < np.iinfo(np.int32).max:
        indptr = indptr.astype(np.int32)
        indices = indices.astype(np.int32, copy=False)
    else:  # pragma: no cover - needs a >2e9 nnz run
        indices = indices.astype(np.int64, copy=False)

    data = np.ones(nnz, dtype=np.float32)
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=np.float32)
    matrix.sort_indices()

    rows = (
        pd.concat(bin_frames, ignore_index=True)
        if bin_frames
        else schemas.empty_frame(schemas.BIN_COLUMNS)
    )
    del bin_frames

    if cfg.matrix.drop_empty_rows:
        keep = np.diff(matrix.indptr) > 0
        n_dropped = int((~keep).sum())
        if n_dropped:
            matrix = matrix[keep]
            rows = rows.loc[keep].reset_index(drop=True)
            logger.info("dropped %d bin(s) with no selected k-mer", n_dropped)

    logger.info(
        "matrix %d x %d, nnz=%d, density=%.3g, %.1f MiB",
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        _density(matrix),
        _nbytes(matrix) / 2**20,
    )
    if matrix.shape[0] == 0:
        logger.warning("the matrix has no rows; every bin was empty or no shard was found")
    _warn_if_too_sparse(matrix, cfg)

    _apply_weighting(matrix, cfg.matrix.weighting)
    _apply_row_norm(matrix, cfg.matrix.row_norm)
    if matrix.dtype != np.float32:  # pragma: no cover - defensive
        matrix = matrix.astype(np.float32)

    rows = rows.reset_index(drop=True)
    if len(rows) != matrix.shape[0]:  # pragma: no cover - internal invariant
        raise RuntimeError(
            f"row table ({len(rows)}) and matrix ({matrix.shape[0]}) disagree; refusing to write "
            "a table that would mislabel every downstream cluster"
        )
    rows["row_idx"] = np.arange(len(rows), dtype=np.int64)
    rows = schemas.enforce(rows, schemas.BIN_COLUMNS, subset=True)
    rows["row_idx"] = rows["row_idx"].astype("int64")
    if rows.empty:
        rows = _empty_rows_frame()

    _save_matrix(matrix, matrix_path)
    _atomic_write_parquet(rows, rows_path)
    logger.info(
        "wrote %s (%d rows from %d shard(s)) and %s",
        matrix_path.name,
        len(rows),
        n_shards,
        rows_path.name,
    )
    return matrix, rows


def _shard_entries(
    sketch_path: Path,
    bin_ids: np.ndarray,
    hashes: np.ndarray,
    hash_cols: np.ndarray,
    assembly: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``(per-bin nnz, column indices)`` for one shard, in bin order."""
    n_bins = bin_ids.shape[0]
    empty_counts = np.zeros(n_bins, dtype=np.int64)
    if not sketch_path.exists():
        logger.warning("no sketch shard for %s; its %d bin(s) will be empty", assembly, n_bins)
        return empty_counts, np.empty(0, dtype=np.int32)
    if hashes.size == 0 or n_bins == 0:
        return empty_counts, np.empty(0, dtype=np.int32)

    pos_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    n_unknown_bin = 0
    for batch_bins, batch_hashes in _iter_sketch_batches(sketch_path):
        # searchsorted gives an insertion point; only an exact match is a hit.
        slot = np.searchsorted(hashes, batch_hashes)
        slot_clipped = np.minimum(slot, hashes.size - 1)
        hit = hashes[slot_clipped] == batch_hashes
        if not hit.any():
            continue
        rows_of_hits = batch_bins[hit]
        # Map shard-local bin_idx onto a row position the same way, so a bin
        # dropped from bins.parquet cannot shift every later row.
        bin_slot = np.searchsorted(bin_ids, rows_of_hits)
        bin_slot_clipped = np.minimum(bin_slot, n_bins - 1)
        known = bin_ids[bin_slot_clipped] == rows_of_hits
        n_unknown_bin += int((~known).sum())
        pos_parts.append(bin_slot_clipped[known])
        col_parts.append(hash_cols[slot_clipped[hit]][known])

    if n_unknown_bin:
        logger.warning(
            "%s: %d sketch row(s) referenced a bin_idx absent from its bin table",
            assembly,
            n_unknown_bin,
        )
    if not pos_parts:
        return empty_counts, np.empty(0, dtype=np.int32)

    pos = np.concatenate(pos_parts)
    cols = np.concatenate(col_parts)
    del pos_parts, col_parts

    if pos.size > 1 and not bool(np.all(pos[1:] >= pos[:-1])):
        logger.warning("%s sketch is not in bin order; sorting %d entries", assembly, pos.size)
        order = np.lexsort((cols, pos))
        pos, cols = pos[order], cols[order]
    # The same k-mer can occur many times inside one satellite bin; the matrix
    # is a presence/absence sketch, so collapse the repeats.
    if pos.size > 1:
        keep = np.ones(pos.size, dtype=bool)
        keep[1:] = (pos[1:] != pos[:-1]) | (cols[1:] != cols[:-1])
        if not keep.all():
            pos, cols = pos[keep], cols[keep]

    counts = np.bincount(pos, minlength=n_bins).astype(np.int64)
    return counts, cols.astype(np.int32, copy=False)


def _density(matrix: sparse.csr_matrix) -> float:
    cells = int(matrix.shape[0]) * int(matrix.shape[1])
    return float(matrix.nnz) / cells if cells else 0.0


def _nbytes(matrix: sparse.csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _save_matrix(matrix: sparse.csr_matrix, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    # save_npz would append ".npz" to a string path; a handle keeps the name.
    with open(tmp, "wb") as handle:
        sparse.save_npz(handle, matrix)
    os.replace(tmp, path)


def load_matrix(outdir: Path) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Read back the matrix and its row table."""
    outdir = Path(outdir)
    matrix_path = outdir / MATRIX_FILENAME
    rows_path = outdir / ROWS_FILENAME
    for path in (matrix_path, rows_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run the matrix stage first")
    matrix = sparse.load_npz(matrix_path)
    matrix = sparse.csr_matrix(matrix)
    frame = pq.read_table(rows_path).to_pandas()
    if frame.empty:
        return matrix, _empty_rows_frame()
    rows = schemas.enforce(frame, schemas.BIN_COLUMNS, subset=True)
    rows["row_idx"] = rows["row_idx"].astype("int64")
    return matrix, rows
