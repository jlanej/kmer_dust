"""UMAP embedding of the SVD component scores.

Why UMAP at all, given we already have 64 SVD dimensions?  Because the thing we
ultimately want is *density-based clusters* of bins, and HDBSCAN's notion of
density is meaningless in 64 dimensions where every point is equidistant from
every other.  UMAP's job here is not to make a pretty picture (though the report
uses it for exactly that); it is to turn a cosine neighbourhood graph in
component space into a low-dimensional layout where "these bins share a k-mer
vocabulary" becomes "these bins are close", which is the only kind of statement
HDBSCAN can act on.

Two engineering choices are worth spelling out.

*Fit on a subsample, transform the rest.*  UMAP's fit is superlinear in memory
and cost; a full HPRC release-2 run is a few million 10 kb bins.  Fitting on a
seeded subsample of at most ``cfg.embed.max_fit_rows`` rows and then projecting
everything through the fitted model keeps a big run tractable and, because the
subsample is drawn from ``np.random.default_rng(cfg.embed.seed)``, keeps it
reproducible.  The whole matrix is transformed in one call: UMAP's ``transform``
picks its epoch count from the batch size, so splitting the projection into
chunks would silently give the last, smaller chunk a different optimisation
budget from the others.

*``random_state`` is set by default.*  This forces UMAP (and pynndescent under
it) onto a single thread -- the parallel code paths race on the negative-sample
RNG and are not reproducible.  A run that cannot be reproduced cannot be
debugged, and this pipeline exists to make claims about biology, so that is the
default and ``n_jobs=1`` is passed explicitly to say so out loud instead of
letting UMAP override it with a warning.

It is also, at 10^5-10^6 bins, comfortably the slowest stage in the pipeline.
``embed.deterministic: false`` drops ``random_state`` and lets UMAP use
``embed.n_jobs`` (or ``threads``) cores.  The layout then differs between runs
while the structure does not, so it is the right choice for exploration and the
wrong one for a figure you intend to publish.  Whichever was used is recorded
in ``umap_params.json`` so a plot can always be traced back.

Degenerate inputs are handled before UMAP sees them.  Fewer than four rows, zero
components, or a component matrix whose columns are all constant will variously
make UMAP raise, emit NaNs, or fail its spectral initialisation; in those cases
we write a deterministic trivial layout (points evenly spaced on a circle) and
say so loudly in the log and in ``umap_params.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .log import get_logger, timed

logger = get_logger(__name__)

__all__ = ["EMBED_FILE", "PARAMS_FILE", "embed", "load_embedding", "load_embed_params"]

EMBED_FILE = "umap.npy"
PARAMS_FILE = "umap_params.json"

#: UMAP itself refuses ``n_neighbors <= 1``; a k-NN graph needs at least two.
_MIN_NEIGHBORS = 2
#: Below this many rows UMAP's spectral init / graph construction is either
#: undefined or numerically meaningless, so we lay the points out by hand.
_MIN_ROWS_FOR_UMAP = 4


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """Atomically write ``arr`` (``np.save`` would rename a ``*.tmp`` target)."""
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


def _trivial_layout(n_rows: int, n_components: int) -> np.ndarray:
    """Points evenly spaced on the unit circle -- deterministic and non-degenerate.

    Used when the input cannot support a manifold: it keeps every point distinct
    (so plots and distance-based code do not divide by zero) without pretending
    the arrangement means anything.
    """
    coords = np.zeros((max(n_rows, 0), max(n_components, 1)), dtype=np.float32)
    if n_rows > 0:
        angles = 2.0 * np.pi * np.arange(n_rows, dtype=np.float64) / float(n_rows)
        coords[:, 0] = np.cos(angles)
        if coords.shape[1] > 1:
            coords[:, 1] = np.sin(angles)
    return coords


def _sanitise(x: np.ndarray, what: str) -> tuple[np.ndarray, int]:
    """Replace non-finite entries with 0.0, returning how many were replaced.

    A single NaN anywhere would make UMAP (or, later, HDBSCAN) fail with an
    opaque message hours into a run.  Zeroing it is a lie, but a loud, counted
    one that leaves the rest of the run interpretable.
    """
    bad_mask = ~np.isfinite(x)
    n_bad = int(bad_mask.sum())
    if n_bad:
        logger.warning("embed: %d non-finite value(s) in %s replaced with 0.0", n_bad, what)
        x = np.where(bad_mask, np.float32(0.0), x)
    return x, n_bad


def embed(pcs: np.ndarray, cfg: Config, outdir: Path, *, force: bool = False) -> np.ndarray:
    """Embed ``pcs`` with UMAP, writing ``umap.npy`` and ``umap_params.json``.

    Returns the ``(n_rows, cfg.embed.n_components)`` float32 layout it wrote.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / EMBED_FILE

    if not force and out_path.exists():
        coords = load_embedding(outdir)
        logger.info("embed: reusing %s (%d x %d)", out_path, coords.shape[0], coords.shape[1])
        return coords

    ecfg = cfg.embed
    x = np.asarray(pcs)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"embed expects a 2-D array of component scores, got shape {x.shape}")
    x = np.ascontiguousarray(x, dtype=np.float32)
    x, n_bad = _sanitise(x, "the component scores")

    n_rows, n_dims = int(x.shape[0]), int(x.shape[1])
    n_components = max(int(ecfg.n_components), 1)

    params: dict[str, Any] = {
        "method": "umap",
        "n_rows": n_rows,
        "n_input_dims": n_dims,
        "n_components": n_components,
        "n_neighbors_requested": int(ecfg.n_neighbors),
        "n_neighbors": int(ecfg.n_neighbors),
        "min_dist": float(ecfg.min_dist),
        "metric": str(ecfg.metric),
        "max_fit_rows": int(ecfg.max_fit_rows),
        "n_fit_rows": n_rows,
        "subsampled": False,
        "seed": int(ecfg.seed),
        "n_nonfinite_input": n_bad,
        "n_nonfinite_output": 0,
    }

    reason = _degenerate_reason(x, n_rows, n_dims)
    if reason:
        logger.warning(
            "embed: %s -- writing a deterministic trivial layout instead of UMAP", reason
        )
        coords = _trivial_layout(n_rows, n_components)
        params["method"] = "trivial"
        params["trivial_reason"] = reason
        _save_npy(out_path, coords)
        _save_json(outdir / PARAMS_FILE, params)
        return coords

    rng = np.random.default_rng(int(ecfg.seed))
    max_fit = int(ecfg.max_fit_rows)
    if 0 < max_fit < _MIN_ROWS_FOR_UMAP:
        logger.warning(
            "embed: max_fit_rows=%d is below the %d rows UMAP needs; fitting on %d",
            max_fit,
            _MIN_ROWS_FOR_UMAP,
            _MIN_ROWS_FOR_UMAP,
        )
        max_fit = _MIN_ROWS_FOR_UMAP
    if 0 < max_fit < n_rows:
        # Sorted so the fit set is a stable, contiguous-ish read regardless of
        # how Generator.choice happens to order its draw.
        fit_idx = np.sort(rng.choice(n_rows, size=max_fit, replace=False))
        fit_x = np.ascontiguousarray(x[fit_idx])
        params["subsampled"] = True
    else:
        fit_idx = None
        fit_x = x
    n_fit = int(fit_x.shape[0])
    params["n_fit_rows"] = n_fit

    n_neighbors = int(ecfg.n_neighbors)
    if n_neighbors >= n_fit:
        # UMAP truncates this itself, but only after warning; doing it here keeps
        # the number we record in umap_params.json equal to the number used.
        clamped = max(min(n_fit - 1, n_neighbors), _MIN_NEIGHBORS)
        logger.warning(
            "embed: n_neighbors %d >= fit rows %d; using %d", n_neighbors, n_fit, clamped
        )
        n_neighbors = clamped
    n_neighbors = max(n_neighbors, _MIN_NEIGHBORS)
    params["n_neighbors"] = n_neighbors

    min_dist = float(ecfg.min_dist)
    if not 0.0 <= min_dist <= 1.0:  # UMAP requires 0 <= min_dist <= spread (1.0)
        clamped_dist = float(min(max(min_dist, 0.0), 1.0))
        logger.warning("embed: min_dist %g out of range; using %g", min_dist, clamped_dist)
        min_dist = clamped_dist
        params["min_dist"] = min_dist

    import umap  # heavy (numba) import: keep the CLI's --help fast

    params["umap_version"] = getattr(umap, "__version__", "")
    deterministic = bool(getattr(ecfg, "deterministic", True))
    if deterministic:
        random_state: int | None = int(ecfg.seed)
        n_jobs = 1  # implied by random_state; stated explicitly to avoid the warning
    else:
        random_state = None
        n_jobs = int(getattr(ecfg, "n_jobs", 0) or 0) or int(getattr(cfg, "threads", 1) or 1)
        logger.warning(
            "embed.deterministic is false: UMAP will use %d thread(s) and the layout "
            "will NOT be reproducible (the structure will be)",
            n_jobs,
        )
    params["deterministic"] = deterministic
    params["n_jobs"] = n_jobs
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=str(ecfg.metric),
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=False,
    )

    with timed(
        logger,
        f"UMAP fit on {n_fit}/{n_rows} rows "
        f"(n_neighbors={n_neighbors} min_dist={min_dist} metric={ecfg.metric})",
    ):
        if fit_idx is None:
            coords = reducer.fit_transform(fit_x)
        else:
            reducer.fit(fit_x)
    if fit_idx is not None:
        with timed(logger, f"UMAP transform of {n_rows} rows"):
            coords = reducer.transform(x)

    coords = np.ascontiguousarray(np.asarray(coords), dtype=np.float32)
    coords, n_bad_out = _sanitise(coords, "the UMAP layout")
    params["n_nonfinite_output"] = n_bad_out
    if coords.shape != (n_rows, n_components):  # defensive: UMAP should never do this
        raise ValueError(f"UMAP returned shape {coords.shape}, expected {(n_rows, n_components)}")

    _save_npy(out_path, coords)
    _save_json(outdir / PARAMS_FILE, params)
    logger.info("embed: wrote %s (%d x %d)", out_path, coords.shape[0], coords.shape[1])
    return coords


def _degenerate_reason(x: np.ndarray, n_rows: int, n_dims: int) -> str:
    """Return why UMAP cannot be run on this input, or ``""`` if it can."""
    if n_rows == 0:
        return "no rows to embed"
    if n_dims == 0:
        return "component matrix has zero columns (the decomposition was empty)"
    if n_rows < _MIN_ROWS_FOR_UMAP:
        return f"only {n_rows} row(s); UMAP needs at least {_MIN_ROWS_FOR_UMAP}"
    # All-identical rows give a disconnected/degenerate k-NN graph; with the
    # cosine metric an all-zero matrix additionally has no defined direction.
    if bool(np.all(x.max(axis=0) == x.min(axis=0))):
        return "every row of the component matrix is identical"
    return ""


def load_embedding(outdir: Path) -> np.ndarray:
    """Read ``umap.npy`` written by :func:`embed`."""
    path = Path(outdir) / EMBED_FILE
    if not path.exists():
        raise FileNotFoundError(f"no embedding at {path}; run the embed stage first")
    return np.load(path, allow_pickle=False)


def load_embed_params(outdir: Path) -> dict[str, Any]:
    """Read ``umap_params.json``; an empty dict when the stage has not run."""
    path = Path(outdir) / PARAMS_FILE
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)
