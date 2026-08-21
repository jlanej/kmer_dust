"""Truncated SVD of the sparse bin x k-mer matrix -- latent semantic analysis.

Why an *uncentered, truncated* SVD rather than PCA
-------------------------------------------------
The matrix out of :mod:`kmer_dust.matrix` is a presence matrix: row = 10 kb bin,
column = one FracMinHash-selected canonical 31-mer, value = an IDF weight on the
non-zero entries, then L2-normalised per row.  Two properties follow.

1.  It is *extremely* sparse -- a bin holds a few hundred of a few hundred
    thousand possible k-mers.  Mean-centering would subtract a dense column mean
    from every entry, turning a 0.01%-dense matrix into a fully dense one.  For
    a few million bins that is not a tuning preference, it is the difference
    between a laptop job and an impossible one.  So we never centre, and we call
    the outputs "components" rather than "principal components".

2.  IDF weighting + L2 row normalisation + truncated SVD *is* latent semantic
    analysis, which is exactly the right model for the question we are asking.
    A bin is a document, a k-mer is a word, and we want to know which latent
    "vocabulary" a bin draws on -- alpha-satellite HOR, HSat2/3, LINE-rich
    euchromatin.  LSA groups documents that share rare-ish words, which is what
    makes an active HOR array in HG00408 land next to one in CHM13 without any
    alignment ever being computed.  Uncentered SVD on L2-normalised rows also
    keeps cosine geometry, which is the metric UMAP then uses downstream.

A consequence worth remembering when reading ``svd.json``: because the data is
not centered, ``explained_variance_ratio`` is not a variance ratio in the PCA
sense.  It is the fraction of the matrix's total squared Frobenius norm captured
by each singular triplet -- computed from the sparse values themselves so the
number is honest rather than a proxy.  The leading component of an uncentered
non-negative matrix is nearly always a "mean document" direction that mostly
encodes how many k-mers survived in a bin; ``decompose.drop_first`` exists to
throw it away when it drowns out the structure we care about.

Randomness comes from ``cfg.decompose.seed`` alone, and ``randomized_svd`` is
called with ``flip_sign=True`` (its default), so sign conventions -- and hence
the bytes of ``pcs.npy`` -- are reproducible.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .config import Config
from .log import get_logger, timed

logger = get_logger(__name__)

__all__ = [
    "PCS_FILE",
    "COMPONENTS_FILE",
    "SVD_FILE",
    "decompose",
    "load_pcs",
    "load_components",
    "load_svd_info",
]

PCS_FILE = "pcs.npy"
COMPONENTS_FILE = "components.npy"
SVD_FILE = "svd.json"

#: Frobenius norm is accumulated in float64 over blocks of this many non-zeros:
#: bounded memory (no float64 copy of a billion-element ``.data``) and a fixed
#: block size keeps the summation order -- and therefore the result -- identical
#: from run to run.
_NORM_BLOCK = 1 << 22


# --------------------------------------------------------------------------
# small IO helpers (kept local: every stage writes tmp-then-rename on its own)
# --------------------------------------------------------------------------


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """Atomically write ``arr`` to ``path``.

    ``np.save`` would helpfully append ``.npy`` to a ``*.tmp`` name, which is
    exactly not what we want, so the temporary file is written through an open
    handle and then renamed into place.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, arr, allow_pickle=False)
    os.replace(tmp, path)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def _as_float_matrix(matrix: Any) -> Any:
    """Coerce the caller's matrix to something ``randomized_svd`` accepts."""
    if sparse.issparse(matrix):
        mat = matrix.tocsr() if matrix.format != "csr" else matrix
        if mat.dtype.kind != "f":
            mat = mat.astype(np.float32)
        return mat
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"decompose expects a 2-D matrix, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False) if arr.dtype.kind != "f" else arr


def _matrix_values(mat: Any) -> np.ndarray:
    return mat.data if sparse.issparse(mat) else np.asarray(mat).ravel()


def _squared_frobenius(values: np.ndarray) -> float:
    total = 0.0
    for start in range(0, values.size, _NORM_BLOCK):
        block = values[start : start + _NORM_BLOCK].astype(np.float64, copy=False)
        total += float(np.dot(block, block))
    return total


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def decompose(
    matrix: sparse.csr_matrix,
    cfg: Config,
    outdir: Path,
    *,
    force: bool = False,
) -> np.ndarray:
    """Factor ``matrix`` and write ``pcs.npy`` / ``components.npy`` / ``svd.json``.

    Returns the ``(n_rows, n_components - drop_first)`` float32 array that was
    written, so :mod:`kmer_dust.pipeline` can hand it straight to
    :func:`kmer_dust.embed.embed` without a round-trip through disk.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pcs_path = outdir / PCS_FILE

    if not force and pcs_path.exists():
        pcs = load_pcs(outdir)
        logger.info("decompose: reusing %s (%d x %d)", pcs_path, pcs.shape[0], pcs.shape[1])
        return pcs

    mat = _as_float_matrix(matrix)
    n_rows, n_cols = int(mat.shape[0]), int(mat.shape[1])
    values = _matrix_values(mat)
    nnz = int(values.size)
    logger.info(
        "decompose: matrix %d bins x %d k-mers, %d non-zeros (%.4f%% dense)",
        n_rows,
        n_cols,
        nnz,
        100.0 * nnz / max(n_rows * n_cols, 1),
    )
    if nnz and not np.isfinite(values).all():
        # NaN/inf here means the weighting step upstream divided by zero; the SVD
        # would happily return an all-NaN factorisation and poison every later
        # stage, so refuse rather than propagate.
        raise ValueError("matrix contains non-finite values; refusing to factor it")

    dcfg = cfg.decompose
    requested = int(dcfg.n_components)
    # randomized_svd can return at most min(shape) triplets; we stop one short so
    # the last, numerically worst-determined direction never reaches the
    # embedding.  For a real run (millions of bins) this clamp never fires --
    # it exists for toy runs and single-assembly smoke tests.
    max_components = max(min(n_rows, n_cols) - 1, 0)
    n_components = min(requested, max_components)
    if n_components < requested:
        logger.warning(
            "decompose: n_components clamped %d -> %d (matrix is %d x %d; "
            "at most min(shape) - 1 components are well defined)",
            requested,
            n_components,
            n_rows,
            n_cols,
        )

    total_sq = _squared_frobenius(values) if nnz else 0.0

    if n_components <= 0:
        logger.error(
            "decompose: matrix %d x %d is too small to factor; writing an empty "
            "decomposition (downstream stages will fall back to trivial layouts)",
            n_rows,
            n_cols,
        )
        pcs = np.zeros((n_rows, 0), dtype=np.float32)
        components = np.zeros((0, n_cols), dtype=np.float32)
        singular = np.zeros(0, dtype=np.float64)
        drop_first = 0
    else:
        drop_first = int(dcfg.drop_first)
        if drop_first >= n_components:
            # Config validation guarantees drop_first < n_components as configured,
            # but the clamp above can invalidate that; keep at least one column
            # rather than handing UMAP a zero-width array.
            new_drop = max(n_components - 1, 0)
            logger.warning(
                "decompose: drop_first %d >= available components %d; dropping %d instead",
                drop_first,
                n_components,
                new_drop,
            )
            drop_first = new_drop

        from sklearn.utils.extmath import randomized_svd  # slow import, keep it lazy

        with timed(
            logger,
            f"randomized SVD k={n_components} n_iter={dcfg.n_iter} "
            f"n_oversamples={dcfg.n_oversamples} seed={dcfg.seed}",
        ):
            u, singular, components = randomized_svd(
                mat,
                n_components=n_components,
                n_oversamples=int(dcfg.n_oversamples),
                n_iter=int(dcfg.n_iter),
                # LU is the stable-but-cheap normaliser: QR at every power
                # iteration costs more than the extra accuracy is worth here,
                # and no normaliser at all loses the small singular values to
                # round-off after 7 iterations.
                power_iteration_normalizer="LU",
                random_state=int(dcfg.seed),
            )
        pcs = (u * singular).astype(np.float32, copy=False)
        pcs = np.ascontiguousarray(pcs[:, drop_first:])
        components = np.ascontiguousarray(components.astype(np.float32, copy=False))

    ratios = (
        (np.square(singular.astype(np.float64)) / total_sq).tolist()
        if total_sq > 0.0
        else [0.0] * int(singular.size)
    )
    _save_npy(pcs_path, pcs)
    components_path = outdir / COMPONENTS_FILE
    if dcfg.keep_components:
        # Full Vt, including any rows the pcs dropped, so that row i of
        # components.npy always pairs with singular_values[i].
        _save_npy(components_path, components)
    elif components_path.exists():
        components_path.unlink()  # never leave a stale Vt from an earlier run

    info: dict[str, Any] = {
        "n_components": int(singular.size),
        "n_components_requested": requested,
        "n_components_kept": int(pcs.shape[1]),
        "drop_first": int(drop_first),
        "shape": [n_rows, n_cols],
        "nnz": nnz,
        "singular_values": [float(x) for x in singular],
        "explained_variance_ratio": [float(x) for x in ratios],
        "cumulative_explained_variance_ratio": float(sum(ratios)),
        "total_squared_frobenius_norm": float(total_sq),
        "n_oversamples": int(dcfg.n_oversamples),
        "n_iter": int(dcfg.n_iter),
        "power_iteration_normalizer": "LU",
        "keep_components": bool(dcfg.keep_components),
        "seed": int(dcfg.seed),
    }
    _save_json(outdir / SVD_FILE, info)

    if singular.size:
        logger.info(
            "decompose: wrote %s (%d x %d); sigma_0=%.4g sigma_last=%.4g; "
            "captured %.1f%% of squared Frobenius norm",
            pcs_path,
            pcs.shape[0],
            pcs.shape[1],
            float(singular[0]),
            float(singular[-1]),
            100.0 * sum(ratios),
        )
    else:
        logger.info("decompose: wrote %s (%d x 0)", pcs_path, pcs.shape[0])
    return pcs


def load_pcs(outdir: Path) -> np.ndarray:
    """Read ``pcs.npy`` written by :func:`decompose`."""
    path = Path(outdir) / PCS_FILE
    if not path.exists():
        raise FileNotFoundError(f"no decomposition at {path}; run the decompose stage first")
    return np.load(path, allow_pickle=False)


def load_components(outdir: Path) -> np.ndarray | None:
    """Read ``components.npy``; ``None`` when ``keep_components`` was false."""
    path = Path(outdir) / COMPONENTS_FILE
    if not path.exists():
        return None
    return np.load(path, allow_pickle=False)


def load_svd_info(outdir: Path) -> dict[str, Any]:
    """Read ``svd.json``; an empty dict when the stage has not run."""
    path = Path(outdir) / SVD_FILE
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)
