"""Assemble the one artefact a human actually reads: the run report.

Every other stage writes a table.  This module joins those tables back into one
bin-level frame and renders a single self-contained HTML file around it.

Three decisions shape the code:

**Partial runs are the normal case.**  A run is restartable and stages are
separate Slurm jobs, so the report is routinely built while ``annotate`` has not
finished or ``backprop`` has not started.  Nothing here raises on a missing
stage; each loader returns ``None``, the corresponding section renders an
explanatory placeholder, and the headline tiles say "not run" rather than "0".

**The payload, not the pixels, is the size budget.**  A 464-haplotype chr21 run
is ~2.1 M bins; even after sub-sampling to ``report.max_points`` the coordinates
plus eight colourings dominate the file.  So we do not serialise a plotly figure
per colouring, and we do not serialise hover strings at all.  Python emits one
columnar blob of base64 little-endian typed arrays (see :func:`_pack`) and
``report.js`` builds the traces, the colour arrays and the hover text from it.
That is roughly a third of the size of the equivalent figure JSON and it means
the cross-plot linking code reads the *same* arrays plotly is drawing.

**Determinism.**  Sub-sampling is seeded from ``cfg.report.seed``; level orders
are explicit, never dictionary order; and the "generated" timestamp comes from
the newest input file (or ``KMER_DUST_REPORT_TIMESTAMP``) rather than the clock,
so rebuilding a report over an unchanged run directory is byte-identical.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ..config import Config
from ..log import get_logger
from ..schemas import (
    ANNOTATION_ID_COLUMNS,
    BIN_COLUMNS,
    CLUSTER_COLUMNS,
    FEATURE_VOCAB,
    MANIFEST_COLUMNS,
)
from . import palette

log = get_logger(__name__)

__all__ = ["build_report", "collect_report_frame", "REPORT_COLUMNS"]

_HERE = Path(__file__).resolve().parent

#: Columns of the joined bin-level frame returned by :func:`collect_report_frame`.
REPORT_COLUMNS: dict[str, str] = {
    "row_idx": "int64",
    "bin_uid": "string",
    "assembly": "string",
    "sample": "string",
    "haplotype": "string",
    "source": "string",
    "contig": "string",
    "chrom": "string",
    "start": "int64",
    "end": "int64",
    "n_acgt": "int64",
    "n_kmers": "int64",
    "n_sketch": "int32",
    "gc": "float32",
    "nfrac": "float32",
    "population": "string",
    "superpopulation": "string",
    "sex": "string",
    "cluster": "int32",
    "probability": "float32",
    "outlier_score": "float32",
    "cluster_name": "string",
    "dominant_feature": "string",
    "dominant_frac": "float32",
    "annotated": "bool",
    "is_reference": "bool",
    "x": "float32",
    "y": "float32",
}

#: Superpopulations in their conventional reporting order.
_SUPERPOP_ORDER: tuple[str, ...] = ("AFR", "AMR", "EAS", "EUR", "SAS", "OCE")

#: Above this the HTML gets unwieldy for a browser to open from disk.
_SIZE_WARN_BYTES = 25 * 1024 * 1024

#: The ribbon is a picture, not a table: past this many bins it is thinned.
_RIBBON_MAX = 200_000

#: Rows of the enrichment heatmap.  More than this and the labels are unreadable.
_HEATMAP_MAX_CLUSTERS = 60


# --------------------------------------------------------------------------
# tolerant loaders
# --------------------------------------------------------------------------


def _read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:  # ArrowInvalid subclasses ValueError
        log.warning("could not read %s: %s", path, exc)
        return None


def _parquet_row_count(path: Path) -> int | None:
    """Row count without materialising the table."""
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except (OSError, ValueError, ImportError) as exc:
        log.warning("could not read metadata of %s: %s", path, exc)
        return None


def _read_npy(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.load(path)
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _read_manifest(path: Path) -> pd.DataFrame | None:
    """Read ``manifest.tsv`` without importing the catalog package.

    Everything is read as text with NA detection off, because a sample called
    ``NA18939`` is not a missing value and neither is an empty optional URL.
    """
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return None
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[list(MANIFEST_COLUMNS)].astype("string").fillna("")


# --------------------------------------------------------------------------
# run assembly
# --------------------------------------------------------------------------


@dataclass
class _Run:
    """Everything the renderer needs, with explicit presence flags."""

    bins: pd.DataFrame
    manifest: pd.DataFrame
    enrichment: pd.DataFrame | None = None
    names: pd.DataFrame | None = None
    transfer: pd.DataFrame | None = None
    svd: dict[str, Any] | None = None
    n_features: int | None = None
    n_bins_total: int = 0
    reference: str = ""
    stages: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timestamp: str = ""
    embed_label: str = "UMAP"


def _merge_left(base: pd.DataFrame, other: pd.DataFrame, key: str) -> pd.DataFrame:
    """Left-join ``other`` onto ``base`` without ever producing ``_x``/``_y``.

    A stage that starts writing a column the bin table already has would
    otherwise silently rename both copies and break every consumer downstream,
    so the incoming table wins and the stale copy is dropped.
    """
    clash = [c for c in other.columns if c != key and c in base.columns]
    if clash:
        log.debug("overwriting %s from the joined table", clash)
        base = base.drop(columns=clash)
    return base.merge(other, on=key, how="left")


def _dedupe(df: pd.DataFrame | None, key: str) -> pd.DataFrame | None:
    if df is None or key not in df.columns:
        return None
    if df[key].duplicated().any():
        log.warning("duplicate %s values in a stage table; keeping the first of each", key)
        return df.drop_duplicates(subset=key, keep="first")
    return df


def _empty_report_frame() -> pd.DataFrame:
    data = {}
    for col, dtype in REPORT_COLUMNS.items():
        data[col] = pd.Series([], dtype="string" if dtype == "string" else dtype)
    return pd.DataFrame(data)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in REPORT_COLUMNS.items():
        if col not in df.columns:
            continue
        if dtype == "string":
            df[col] = df[col].fillna("").astype("string")
        elif dtype == "bool":
            df[col] = df[col].fillna(False).astype(bool)
        elif dtype.startswith("int"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    return df[[c for c in REPORT_COLUMNS if c in df.columns]]


def _load_run(cfg: Config, outdir: Path) -> _Run:
    root = Path(outdir)
    stages: dict[str, bool] = {}
    notes: list[str] = []
    inputs: list[Path] = []

    def seen(path: Path) -> Path:
        if path.is_file():
            inputs.append(path)
        return path

    manifest = _read_manifest(seen(root / "manifest.tsv"))
    stages["manifest"] = manifest is not None
    if manifest is None:
        manifest = pd.DataFrame({c: pd.Series([], dtype="string") for c in MANIFEST_COLUMNS})

    rows = _read_parquet(seen(root / "matrix" / "rows.parquet"))
    stages["matrix"] = rows is not None
    if rows is None or rows.empty:
        if rows is None:
            notes.append("The <b>matrix</b> stage has not run, so there is nothing to plot yet.")
        else:
            notes.append("The bin table is empty: every bin was filtered out upstream.")
        run = _Run(
            bins=_empty_report_frame(),
            manifest=manifest,
            stages=stages,
            notes=notes,
            timestamp=_timestamp(inputs),
        )
        return run

    for col, dtype in BIN_COLUMNS.items():
        if col not in rows.columns:
            rows[col] = pd.Series([""] * len(rows), dtype="string") if dtype == "string" else 0
    if "row_idx" not in rows.columns:
        rows["row_idx"] = np.arange(len(rows), dtype=np.int64)
    bins = rows.reset_index(drop=True)

    # --- per-assembly metadata -------------------------------------------
    meta_cols = ["assembly", "population", "superpopulation", "sex"]
    if not manifest.empty:
        meta = manifest[[c for c in meta_cols if c in manifest.columns]].copy()
        meta = _dedupe(meta, "assembly")
        if meta is not None:
            bins = _merge_left(bins, meta, "assembly")
    for col in ("population", "superpopulation", "sex"):
        if col not in bins.columns:
            bins[col] = ""

    # --- clusters ---------------------------------------------------------
    clusters = _dedupe(_read_parquet(seen(root / "cluster" / "clusters.parquet")), "bin_uid")
    stages["cluster"] = clusters is not None
    if clusters is not None:
        keep = [c for c in CLUSTER_COLUMNS if c in clusters.columns and c != "row_idx"]
        bins = _merge_left(bins, clusters[keep], "bin_uid")
    for col, fill in (("cluster", -1), ("probability", np.nan), ("outlier_score", np.nan)):
        if col not in bins.columns:
            bins[col] = fill
    bins["cluster"] = pd.to_numeric(bins["cluster"], errors="coerce").fillna(-1).astype("int32")

    # --- annotations ------------------------------------------------------
    ann = _dedupe(_read_parquet(seen(root / "annotate" / "annotations.parquet")), "bin_uid")
    stages["annotate"] = ann is not None
    if ann is not None:
        keep = [c for c in ANNOTATION_ID_COLUMNS if c in ann.columns]
        bins = _merge_left(bins, ann[keep], "bin_uid")
    if "dominant_feature" not in bins.columns:
        bins["dominant_feature"] = ""
    bins["dominant_feature"] = bins["dominant_feature"].fillna("").astype("string")
    if "dominant_frac" not in bins.columns:
        bins["dominant_frac"] = np.nan
    if "annotated" not in bins.columns:
        bins["annotated"] = False

    # --- cluster names ----------------------------------------------------
    names = _dedupe(_read_parquet(seen(root / "enrich" / "cluster_names.parquet")), "cluster")
    enrichment = _read_parquet(seen(root / "enrich" / "enrichment.parquet"))
    stages["enrich"] = names is not None or enrichment is not None
    if names is not None and "name" in names.columns:
        lut = names[["cluster", "name"]].copy()
        lut["cluster"] = pd.to_numeric(lut["cluster"], errors="coerce").fillna(-1).astype("int32")
        bins = _merge_left(bins, lut.rename(columns={"name": "cluster_name"}), "cluster")
    if "cluster_name" not in bins.columns:
        bins["cluster_name"] = ""
    bins["cluster_name"] = bins["cluster_name"].fillna("").astype("string")

    transfer = _read_parquet(seen(root / "backprop" / "cluster_transfer.parquet"))
    stages["backprop"] = transfer is not None

    # --- coordinates ------------------------------------------------------
    embed_label = "UMAP"
    coords = _read_npy(seen(root / "embed" / "umap.npy"))
    stages["embed"] = coords is not None
    pcs = _read_npy(seen(root / "decompose" / "pcs.npy"))
    stages["decompose"] = pcs is not None
    if coords is None or coords.ndim != 2 or len(coords) != len(bins):
        if coords is not None:
            log.warning(
                "embed/umap.npy has %s rows but the bin table has %d; ignoring it",
                getattr(coords, "shape", "?"),
                len(bins),
            )
            notes.append(
                "The stored embedding does not match the bin table and was ignored; "
                "re-run <b>embed</b>."
            )
        if pcs is not None and pcs.ndim == 2 and len(pcs) == len(bins) and pcs.shape[1] >= 2:
            coords = pcs[:, :2]
            embed_label = "PC"
            notes.append(
                "No UMAP embedding was found, so the map shows the first two SVD components."
            )
        else:
            coords = None
    if coords is not None and coords.shape[1] > 2:
        notes.append(
            f"The embedding has {coords.shape[1]} dimensions; the map shows the first two."
        )
    if coords is None:
        bins["x"] = np.nan
        bins["y"] = np.nan
        notes.append("Neither <b>embed</b> nor <b>decompose</b> has produced usable coordinates.")
    else:
        bins["x"] = np.asarray(coords[:, 0], dtype=np.float32)
        bins["y"] = np.asarray(coords[:, 1], dtype=np.float32)

    svd = _read_json(seen(root / "decompose" / "svd.json"))
    n_features = _parquet_row_count(seen(root / "kmers" / "kmers.parquet"))
    stages["select"] = n_features is not None
    if n_features is None and svd and isinstance(svd.get("shape"), (list, tuple)):
        shape = svd["shape"]
        if len(shape) == 2:
            n_features = int(shape[1])

    reference = _pick_reference(bins)
    bins["is_reference"] = (bins["assembly"] == reference) if reference else False
    bins = _coerce(bins)

    return _Run(
        bins=bins,
        manifest=manifest,
        enrichment=enrichment,
        names=names,
        transfer=transfer,
        svd=svd,
        n_features=n_features,
        n_bins_total=len(bins),
        reference=reference,
        stages=stages,
        notes=notes,
        timestamp=_timestamp(inputs),
        embed_label=embed_label,
    )


def _pick_reference(bins: pd.DataFrame) -> str:
    """The assembly whose bins carry the genome ribbon.

    T2T-CHM13 if it is in the run (that is the point of including it); otherwise
    whichever assembly has the most bins with a resolved chromosome, so the
    ribbon still works for reference-free runs.
    """
    if bins.empty:
        return ""
    placed = bins[bins["chrom"].astype("string").fillna("") != ""]
    if placed.empty:
        return ""
    t2t = placed[placed["source"].astype("string").str.lower() == "t2t"]
    pool = t2t if not t2t.empty else placed
    counts = pool.groupby("assembly", observed=True).size()
    # Ties broken by name so the choice is reproducible.
    best = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return str(best[0][0]) if best else ""


def _timestamp(inputs: Sequence[Path]) -> str:
    """UTC stamp derived from the inputs, not the clock, so rebuilds are stable."""
    override = os.environ.get("KMER_DUST_REPORT_TIMESTAMP")
    if override:
        return override
    mtimes = [p.stat().st_mtime for p in inputs if p.is_file()]
    when = max(mtimes) if mtimes else 0.0
    return _dt.datetime.fromtimestamp(when, tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def collect_report_frame(cfg: Config, outdir: Path) -> pd.DataFrame:
    """Join every completed stage into one row-per-bin frame.

    Always returns :data:`REPORT_COLUMNS`; missing stages leave their columns at
    the neutral value (``cluster == -1``, ``dominant_feature == ''``, NaN
    coordinates) rather than being absent.
    """
    return _load_run(cfg, Path(outdir)).bins


# --------------------------------------------------------------------------
# payload encoding
# --------------------------------------------------------------------------


def _pack(values: np.ndarray, dtype: str) -> dict[str, Any]:
    """One column as base64 little-endian bytes plus its element count."""
    arr = np.ascontiguousarray(values, dtype=np.dtype("<" + dtype))
    return {"d": dtype, "n": int(arr.size), "b": base64.b64encode(arr.tobytes()).decode("ascii")}


def _pack_positions(values: np.ndarray) -> dict[str, Any]:
    """Genomic coordinates: int32 is plenty for a chromosome, but do not assume."""
    values = np.asarray(values, dtype=np.int64)
    if values.size and int(values.max()) > 2**31 - 1:
        return _pack(values.astype(np.float64), "f8")
    return _pack(values, "i4")



def _pack_quantised(values: np.ndarray, *, bits: int = 16) -> dict[str, Any]:
    """A continuous column as integer codes plus a scale and offset.

    These columns exist to be turned into pixels and colours, and nothing
    downstream reads them back as numbers, so full float precision is wasted
    bytes.  16 bits across the observed range is ~65k distinguishable levels --
    two orders of magnitude finer than any screen -- and halves the payload
    against float32.  For a colour ramp 8 bits is already more than the eye
    resolves.

    NaN is mapped to the low end and paired with plotly's own handling by the
    caller; the alternative (a sentinel level) costs a branch in the decoder for
    no visible benefit.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return _pack(np.zeros(arr.size, dtype=np.int16), "i2") | {"s": 1.0, "o": 0.0}
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    span = hi - lo
    levels = (1 << bits) - 1
    scale = (span / levels) if span > 0 else 1.0
    codes = np.zeros(arr.size, dtype=np.float64)
    codes[finite] = np.round((arr[finite] - lo) / scale)
    codes = np.clip(codes, 0, levels)
    dtype = "u2" if bits > 8 else "u1"
    packed = _pack(codes.astype(np.uint16 if bits > 8 else np.uint8), dtype)
    packed["s"] = scale
    packed["o"] = lo
    return packed


def _pack_codes(codes: np.ndarray) -> dict[str, Any]:
    """Integer codes in the narrowest type that holds them."""
    codes = np.asarray(codes)
    hi = int(codes.max()) if codes.size else 0
    lo = int(codes.min()) if codes.size else 0
    if lo >= -128 and hi <= 127:
        return _pack(codes, "i1")
    if lo >= -32768 and hi <= 32767:
        return _pack(codes, "i2")
    return _pack(codes, "i4")


def _levels(
    values: pd.Series,
    *,
    order: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    noise_labels: Sequence[str] = (),
    relabel: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode a text column as codes + an explicit level table."""
    text = values.astype("string").fillna("")
    observed = pd.Index(pd.unique(text.to_numpy()))
    if order is None:
        counts = text.value_counts()
        order = [str(v) for v in counts.index]
    else:
        order = [str(v) for v in order if v in set(observed.tolist())]
        order += sorted(str(v) for v in observed if str(v) not in set(order))
    cat = pd.Categorical(text, categories=order)
    codes = np.asarray(cat.codes, dtype=np.int64)
    if (codes < 0).any():  # defensive: a level slipped through
        codes = np.where(codes < 0, 0, codes)
    counts = text.value_counts().reindex(order, fill_value=0)
    labels = [relabel.get(v, v) if relabel else (v if v else "unassigned") for v in order]
    if colors is None:
        colors = palette.categorical_colors(order, noise_labels=noise_labels)
    return codes, {
        "labels": labels,
        "colors": list(colors),
        "counts": [int(c) for c in counts.to_numpy()],
    }


def _chrom_key(name: str) -> tuple[int, int, str]:
    stem = name[3:] if name.lower().startswith("chr") else name
    if stem.isdigit():
        return (0, int(stem), stem)
    special = {"X": 1, "Y": 2, "M": 3, "MT": 3}
    if stem.upper() in special:
        return (1, special[stem.upper()], stem)
    if not name:
        return (3, 0, "")
    return (2, 0, name)


def _cluster_levels(bins: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], list[int]]:
    """Cluster codes ordered by size, with noise pinned last and grey."""
    ids = bins["cluster"].to_numpy()
    sizes = pd.Series(ids).value_counts()
    real = sorted(
        (int(c) for c in sizes.index if int(c) >= 0),
        key=lambda c: (-int(sizes.loc[c]), c),
    )
    has_noise = bool((ids < 0).any())
    order = real + ([-1] if has_noise else [])
    lut = {cid: i for i, cid in enumerate(order)}
    codes = np.array([lut.get(int(c), len(order) - 1) for c in ids], dtype=np.int64)

    name_by_id: dict[int, str] = {}
    if "cluster_name" in bins.columns:
        pairs = bins[["cluster", "cluster_name"]].drop_duplicates()
        for cid, name in zip(pairs["cluster"].tolist(), pairs["cluster_name"].tolist()):
            if name:
                name_by_id[int(cid)] = str(name)
    labels = [name_by_id.get(c, f"C{c}") if c >= 0 else "noise" for c in order]
    colors = palette.qualitative_palette(len(real)) + ([palette.NOISE_COLOR] if has_noise else [])
    counts = [int(sizes.get(c, 0)) for c in order]
    return codes, {"labels": labels, "colors": colors, "counts": counts}, order


def _feature_levels(bins: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    vocab = list(FEATURE_VOCAB) + ["unannotated", ""]
    text = bins["dominant_feature"].astype("string").fillna("")
    present = set(text.unique().tolist())
    order = [f for f in vocab if f in present]
    order += sorted(f for f in present if f not in set(order))
    colors = [palette.feature_color(f) for f in order]
    codes, level = _levels(
        text, order=order, colors=colors, relabel={"": "unannotated"}
    )
    return codes, level


def _finite_range(values: np.ndarray) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    if not math.isfinite(lo) or not math.isfinite(hi):
        return None
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _subsample(bins: pd.DataFrame, cfg: Config, notes: list[str]) -> pd.DataFrame:
    """Seeded thinning that always keeps the reference bins.

    The reference is what the ribbon and every cluster name are built from, so
    dropping its bins to make room for the 400th haplotype would be exactly
    backwards.
    """
    budget = int(cfg.report.max_points)
    total = len(bins)
    if budget <= 0 or total <= budget:
        return bins
    budget = max(1000, budget)
    rng = np.random.default_rng(int(cfg.report.seed))
    ref = np.flatnonzero(bins["is_reference"].to_numpy())
    other = np.flatnonzero(~bins["is_reference"].to_numpy())
    if ref.size >= budget:
        keep = np.sort(rng.choice(ref, size=budget, replace=False))
        notes.append(
            f"The map shows a seeded random sample of <b>{budget:,}</b> of "
            f"<b>{total:,}</b> bins (seed {cfg.report.seed})."
        )
    else:
        take = budget - ref.size
        keep = np.sort(np.concatenate([ref, rng.choice(other, size=take, replace=False)]))
        notes.append(
            f"The map shows <b>{len(keep):,}</b> of <b>{total:,}</b> bins: every reference bin "
            f"plus a seeded random sample of the rest (seed {cfg.report.seed}). Counts in the "
            "tiles, the table and the heatmap are computed over all bins."
        )
    return bins.iloc[keep].reset_index(drop=True)


def _build_payload(run: _Run, cfg: Config, notes: list[str]) -> dict[str, Any]:
    bins = _subsample(run.bins, cfg, notes)
    n = len(bins)
    arrays: dict[str, Any] = {}
    levels: dict[str, Any] = {}
    colorings: list[dict[str, Any]] = []

    if n == 0:
        return {
            "version": __version__,
            "runName": cfg.run_name,
            "n": 0,
            "nTotal": run.n_bins_total,
            "binSize": int(cfg.sketch.bin_size),
            "pointSize": float(cfg.report.point_size),
            "missingColor": palette.MISSING_COLOR,
            "embedLabel": run.embed_label,
            "hasCoords": False,
            "arrays": {},
            "levels": {},
            "colorings": [],
            "ribbon": None,
            "heatmap": None,
            "table": None,
            "spans": False,
        }

    # Coordinates are quantised: see _pack_quantised. This is the single
    # biggest line in the payload, so halving it is what lets the map show
    # millions of points rather than a sample of them.
    arrays["x"] = _pack_quantised(bins["x"].to_numpy(dtype=np.float64))
    arrays["y"] = _pack_quantised(bins["y"].to_numpy(dtype=np.float64))
    arrays["start"] = _pack_positions(bins["start"].to_numpy(dtype=np.int64))
    spans = (bins["end"] - bins["start"]).to_numpy(dtype=np.int64)
    uniform = bool(spans.size and np.all(spans == spans[0]) and spans[0] == cfg.sketch.bin_size)
    if not uniform:
        arrays["span"] = _pack_positions(spans)

    gc = bins["gc"].to_numpy(dtype=np.float32)
    arrays["gc"] = _pack_quantised(gc, bits=8)  # a colour ramp; 256 levels is plenty
    with np.errstate(divide="ignore", invalid="ignore"):
        logns = np.log10(np.maximum(bins["n_sketch"].to_numpy(dtype=np.float64), 1.0))
    arrays["logns"] = _pack_quantised(logns, bits=8)

    # --- categorical columns ---------------------------------------------
    chrom_codes, levels["chrom"] = _levels(
        bins["chrom"],
        order=sorted({str(v) for v in bins["chrom"].tolist()}, key=_chrom_key),
        relabel={"": "unplaced"},
        noise_labels=("",),
    )
    arrays["chrom"] = _pack_codes(chrom_codes)

    contig_codes, levels["contig"] = _levels(
        bins["contig"], order=sorted({str(v) for v in bins["contig"].tolist()})
    )
    arrays["contig"] = _pack_codes(contig_codes)

    asm_order = [a for a in run.manifest["assembly"].tolist()] if not run.manifest.empty else None
    asm_codes, levels["assembly"] = _levels(bins["assembly"], order=asm_order)
    arrays["assembly"] = _pack_codes(asm_codes)

    sample_codes, levels["sample"] = _levels(bins["sample"])
    arrays["sample"] = _pack_codes(sample_codes)

    spop_colors_order = list(_SUPERPOP_ORDER)
    spop_present = sorted(
        {str(v) for v in bins["superpopulation"].tolist()},
        key=lambda v: (spop_colors_order.index(v) if v in spop_colors_order else 99, v),
    )
    spop_codes, levels["superpop"] = _levels(
        bins["superpopulation"],
        order=spop_present,
        colors=[palette.SUPERPOP_COLORS.get(v, palette.NOISE_COLOR) for v in spop_present],
        relabel={"": "unknown"},
    )
    arrays["superpop"] = _pack_codes(spop_codes)

    src_present = sorted(
        {str(v) for v in bins["source"].tolist()}, key=lambda v: (v.lower() != "t2t", v)
    )
    src_codes, levels["source"] = _levels(bins["source"], order=src_present)
    arrays["source"] = _pack_codes(src_codes)

    cluster_codes, levels["cluster"], cluster_ids = _cluster_levels(bins)
    if run.stages.get("cluster"):
        arrays["cluster"] = _pack_codes(cluster_codes)

    feat_codes, levels["feature"] = _feature_levels(bins)
    if run.stages.get("annotate"):
        arrays["feature"] = _pack_codes(feat_codes)
        arrays["domfrac"] = _pack_quantised(
            bins["dominant_frac"].to_numpy(dtype=np.float64), bits=8
        )

    # --- colourings -------------------------------------------------------
    def add_cat(key: str, label: str, level_key: str) -> None:
        if len(levels[level_key]["labels"]) < 2:
            return
        colorings.append(
            {"key": key, "label": label, "kind": "cat", "array": key, "levels": level_key}
        )

    if run.stages.get("cluster"):
        add_cat("cluster", "cluster", "cluster")
    if run.stages.get("annotate"):
        add_cat("feature", "dominant feature", "feature")
    add_cat("chrom", "chromosome", "chrom")
    add_cat("sample", "sample", "sample")
    add_cat("superpop", "superpopulation", "superpop")
    add_cat("source", "source", "source")

    for key, label, colors, unit, dp in (
        ("gc", "GC fraction", palette.CIVIDIS, "G+C / unambiguous bases", 2),
        ("logns", "log₁₀ sketch size", palette.VIRIDIS, "hashes retained per bin", 2),
    ):
        span = _finite_range(np.asarray(logns if key == "logns" else gc, dtype=np.float64))
        if span is None:
            continue
        colorings.append(
            {
                "key": key,
                "label": label,
                "kind": "num",
                "array": key,
                "scale": palette.scale_stops(colors),
                "cmin": round(span[0], 6),
                "cmax": round(span[1], 6),
                "unit": unit,
                "dp": dp,
            }
        )

    if not colorings:
        colorings.append(
            {"key": "source", "label": "source", "kind": "cat", "array": "source",
             "levels": "source"}
        )

    ribbon = _build_ribbon(bins, chrom_codes, levels, run, cfg, notes)

    return {
        "version": __version__,
        "runName": cfg.run_name,
        "n": n,
        "nTotal": run.n_bins_total,
        "binSize": int(cfg.sketch.bin_size),
        "pointSize": float(cfg.report.point_size),
        "missingColor": palette.MISSING_COLOR,
        "embedLabel": run.embed_label,
        "hasCoords": bool(np.isfinite(bins["x"].to_numpy(dtype=np.float64)).any()),
        "spans": not uniform,
        "arrays": arrays,
        "levels": levels,
        "colorings": colorings,
        "ribbon": ribbon,
        "heatmap": _build_heatmap(run, cluster_ids, levels["cluster"], notes),
        "table": _build_table(run, cluster_ids, levels["cluster"]),
    }


def _build_ribbon(
    bins: pd.DataFrame,
    chrom_codes: np.ndarray,
    levels: dict[str, Any],
    run: _Run,
    cfg: Config,
    notes: list[str],
) -> dict[str, Any] | None:
    """Reference bins laid out in genomic order, one row per chromosome."""
    if not run.reference:
        return None
    mask = (bins["assembly"].to_numpy() == run.reference) & (
        bins["chrom"].astype("string").fillna("").to_numpy() != ""
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    if idx.size > _RIBBON_MAX:
        rng = np.random.default_rng(int(cfg.report.seed) + 1)
        idx = np.sort(rng.choice(idx, size=_RIBBON_MAX, replace=False))
        notes.append(
            f"The genome ribbon is thinned to {_RIBBON_MAX:,} bins so it stays responsive."
        )
    starts = bins["start"].to_numpy(dtype=np.int64)[idx]
    ends = bins["end"].to_numpy(dtype=np.int64)[idx]
    codes = chrom_codes[idx]

    chrom_labels = levels["chrom"]["labels"]
    present = sorted({int(c) for c in codes}, key=lambda c: _chrom_key(str(chrom_labels[c])))
    row_of = {code: i for i, code in enumerate(present)}
    rows = []
    for code in present:
        sel = codes == code
        rows.append({"name": str(chrom_labels[code]), "length": int(ends[sel].max())})

    row_index = np.array([row_of[int(c)] for c in codes], dtype=np.int64)
    order = np.lexsort((starts, row_index))
    per_row = np.bincount(row_index, minlength=len(present))

    row_of_chrom = [-1] * len(chrom_labels)
    for code, row in row_of.items():
        row_of_chrom[code] = row

    return {
        "subject": run.reference,
        "rows": rows,
        "rowOfChrom": row_of_chrom,
        "maxBinsPerRow": int(per_row.max()) if per_row.size else 1,
        "order": _pack(idx[order].astype(np.int64), "i4"),
    }


def _build_heatmap(
    run: _Run,
    cluster_ids: Sequence[int],
    cluster_level: dict[str, Any],
    notes: list[str],
) -> dict[str, Any] | None:
    enr = run.enrichment
    if enr is None or enr.empty:
        return None
    needed = {"cluster", "feature", "log2_enrichment"}
    if not needed.issubset(enr.columns):
        log.warning("enrichment.parquet is missing %s; skipping the heatmap", sorted(needed))
        return None
    enr = enr.copy()
    enr["cluster"] = pd.to_numeric(enr["cluster"], errors="coerce").fillna(-1).astype(int)
    enr["feature"] = enr["feature"].astype("string").fillna("")

    shown = [c for c in cluster_ids if c >= 0]
    if len(shown) > _HEATMAP_MAX_CLUSTERS:
        notes.append(
            f"The heatmap shows the {_HEATMAP_MAX_CLUSTERS} largest of {len(shown):,} clusters."
        )
        shown = shown[:_HEATMAP_MAX_CLUSTERS]
    if not shown:
        return None
    label_of = {cid: cluster_level["labels"][i] for i, cid in enumerate(cluster_ids)}

    sub = enr[enr["cluster"].isin(shown)]
    if sub.empty:
        return None
    used = set(sub["feature"].tolist())
    features = [f for f in FEATURE_VOCAB if f in used]
    features += sorted(f for f in used if f not in set(features))
    if not features:
        return None

    def col(name: str) -> pd.Series:
        return sub[name] if name in sub.columns else pd.Series(np.nan, index=sub.index)

    frame = pd.DataFrame(
        {
            "cluster": sub["cluster"].to_numpy(),
            "feature": sub["feature"].to_numpy(),
            "z": pd.to_numeric(sub["log2_enrichment"], errors="coerce").to_numpy(),
            "nc": pd.to_numeric(col("n_bins_cluster"), errors="coerce").to_numpy(),
            "cs": pd.to_numeric(col("cluster_size"), errors="coerce").to_numpy(),
            "nt": pd.to_numeric(col("n_bins_total"), errors="coerce").to_numpy(),
            "bs": pd.to_numeric(col("background_size"), errors="coerce").to_numpy(),
            "p": pd.to_numeric(col("neg_log10_p"), errors="coerce").to_numpy(),
        }
    )
    lookup = {(int(r.cluster), str(r.feature)): r for r in frame.itertuples()}

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for cid in shown:
        zrow: list[float | None] = []
        trow: list[str] = []
        for feat in features:
            rec = lookup.get((cid, feat))
            if rec is None or not np.isfinite(rec.z):
                zrow.append(None)
                trow.append(
                    f"<b>{html.escape(label_of.get(cid, str(cid)))}</b>"
                    f"<br>{html.escape(feat)}<br>&mdash;"
                )
                continue
            zrow.append(round(float(rec.z), 4))
            trow.append(
                f"<b>{html.escape(label_of.get(cid, str(cid)))}</b> · {html.escape(feat)}"
                f"<br>log2 enrichment {float(rec.z):+.2f}"
                f"<br>{_ratio(rec.nc, rec.cs)} of the cluster"
                f"<br>{_ratio(rec.nt, rec.bs)} genome-wide"
                f"<br>{_pvalue(rec.p)}"
            )
        z.append(zrow)
        text.append(trow)

    flat = np.array([v for row in z for v in row if v is not None], dtype=np.float64)
    zabs = float(np.percentile(np.abs(flat), 98)) if flat.size else 1.0
    zabs = max(1.0, round(zabs, 3))

    return {
        "x": features,
        "y": [label_of.get(c, f"C{c}") for c in shown],
        "z": z,
        "text": text,
        "zabs": zabs,
        "scales": {
            "dark": palette.scale_stops(palette.DIVERGING_DARK),
            "light": palette.scale_stops(palette.DIVERGING_LIGHT),
        },
    }


def _ratio(num: float, den: float) -> str:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return "n/a"
    return f"{int(num):,}/{int(den):,} bins ({100 * num / den:.1f}%)"


def _pvalue(neg_log10: float) -> str:
    if not np.isfinite(neg_log10):
        return "p n/a"
    if neg_log10 <= 0:
        return "p = 1"
    if neg_log10 > 300:
        return "p &lt; 1e-300"
    return f"p = {10 ** -float(neg_log10):.1e}"


def _build_table(
    run: _Run, cluster_ids: Sequence[int], cluster_level: dict[str, Any]
) -> dict[str, Any] | None:
    names = run.names
    if names is None or names.empty or "cluster" not in names.columns:
        return None
    names = names.copy()
    names["cluster"] = pd.to_numeric(names["cluster"], errors="coerce").fillna(-1).astype(int)
    by_id = {int(r["cluster"]): r for _, r in names.iterrows()}

    transfer: dict[int, Any] = {}
    if run.transfer is not None and "cluster" in run.transfer.columns:
        tf = run.transfer.copy()
        tf["cluster"] = pd.to_numeric(tf["cluster"], errors="coerce").fillna(-1).astype(int)
        transfer = {int(r["cluster"]): r for _, r in tf.iterrows()}

    columns = [
        {"key": "name", "label": "cluster", "type": "name"},
        {"key": "size", "label": "bins", "type": "int"},
        {"key": "n_assemblies", "label": "assemblies", "type": "int"},
        {"key": "n_chroms", "label": "chroms", "type": "int"},
        {"key": "top_features", "label": "top features", "type": "text"},
        {"key": "purity", "label": "purity", "type": "pct"},
    ]
    if transfer:
        columns += [
            {"key": "n_ref_bins", "label": "ref bins", "type": "int"},
            {"key": "n_asm_bins", "label": "asm bins", "type": "int"},
            {"key": "asm_agreement", "label": "transfer", "type": "pct"},
        ]

    rows: list[list[Any]] = []
    for level, cid in enumerate(cluster_ids):
        if cid < 0:
            continue
        rec = by_id.get(cid)
        if rec is None:
            continue
        values: list[Any] = [
            cluster_level["labels"][level],
            _int_or_none(rec.get("size")),
            _int_or_none(rec.get("n_assemblies")),
            _int_or_none(rec.get("n_chroms")),
            str(rec.get("top_features") or ""),
            _float_or_none(rec.get("purity")),
        ]
        if transfer:
            tr = transfer.get(cid)
            values += [
                _int_or_none(tr.get("n_ref_bins")) if tr is not None else None,
                _int_or_none(tr.get("n_asm_bins")) if tr is not None else None,
                _float_or_none(tr.get("asm_agreement")) if tr is not None else None,
            ]
        values.append(level)  # trailing level index, consumed by report.js
        rows.append(values)

    if not rows:
        return None
    return {"columns": columns, "rows": rows}


def _int_or_none(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return round(out, 6) if math.isfinite(out) else None


# --------------------------------------------------------------------------
# static HTML fragments
# --------------------------------------------------------------------------


def _fmt_int(value: Any) -> str:
    if value is None:
        return "–"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "–"


def _human_bp(value: int) -> str:
    if value >= 1_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000} Mb"
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000} kb"
    return f"{value:,} bp"


def _tile(label: str, value: str, sub: str = "", empty: bool = False) -> str:
    cls = "kd-tile is-empty" if empty else "kd-tile"
    sub_html = f'<div class="s">{html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="{cls}"><div class="k">{html.escape(label)}</div>'
        f'<div class="v">{value}</div>{sub_html}</div>'
    )



def _findings_html(outdir: Path) -> str:
    """Render `analysis/findings.json` if an analysis wrote one.

    The explorer below is a tool; this band is the argument.  A reader who never
    touches the map should still leave knowing what the run found, so the
    findings sit above the fold, and they come from a file rather than from the
    report code so that an analysis can state its own conclusion without the
    viewer having to know anything about it.

    Schema (all fields optional but `headline`)::

        {"findings": [
            {"headline": "...", "detail": "...", "kicker": "...",
             "evidence": [["label", "value"], ...]}
        ]}
    """
    path = Path(outdir) / "analysis" / "findings.json"
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable %s (%s)", path, exc)
        return ""
    items = payload.get("findings") or []
    if not isinstance(items, list) or not items:
        return ""

    cards: list[str] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("headline"):
            continue
        kicker = html.escape(str(item.get("kicker", "") or ""))
        headline = html.escape(str(item["headline"]))
        detail = html.escape(str(item.get("detail", "") or ""))
        rows = ""
        evidence = item.get("evidence") or []
        if isinstance(evidence, list) and evidence:
            cells = "".join(
                f"<tr><th>{html.escape(str(pair[0]))}</th>"
                f"<td>{html.escape(str(pair[1]))}</td></tr>"
                for pair in evidence
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            )
            if cells:
                rows = f"<table class='kd-evidence'>{cells}</table>"
        cards.append(
            "<article class='kd-finding'>"
            + (f"<p class='kd-kicker'>{kicker}</p>" if kicker else "")
            + f"<h3>{headline}</h3>"
            + (f"<p>{detail}</p>" if detail else "")
            + rows
            + "</article>"
        )
    if not cards:
        return ""
    log.info("report: embedded %d finding(s) from %s", len(cards), path)
    return (
        "<section class='kd-findings'><h2>What this run found</h2>"
        + "<div class='kd-finding-grid'>"
        + "".join(cards)
        + "</div></section>"
    )


def _stat_tiles(run: _Run, cfg: Config, stats: dict[str, Any]) -> str:
    tiles = [
        _tile("assemblies", _fmt_int(stats["n_assemblies"]), f"{stats['n_samples']:,} samples"),
        _tile("bins", _fmt_int(stats["n_bins"]), _human_bp(cfg.sketch.bin_size) + " each"),
        _tile(
            "k-mers selected",
            _fmt_int(stats["n_features"]) if stats["n_features"] is not None else "not run",
            f"k = {cfg.sketch.k}, scaled = {cfg.sketch.scaled}",
            empty=stats["n_features"] is None,
        ),
        _tile(
            "components",
            _fmt_int(stats["n_components"]) if stats["n_components"] is not None else "not run",
            stats["variance_note"],
            empty=stats["n_components"] is None,
        ),
        _tile(
            "clusters",
            _fmt_int(stats["n_clusters"]) if stats["n_clusters"] is not None else "not run",
            cfg.cluster.method.upper(),
            empty=stats["n_clusters"] is None,
        ),
        _tile(
            "noise",
            f"{100 * stats['noise_fraction']:.1f}%" if stats["noise_fraction"] is not None
            else "not run",
            "bins left unassigned",
            empty=stats["noise_fraction"] is None,
        ),
        _tile(
            "median sketch",
            _fmt_int(stats["median_sketch"]) if stats["median_sketch"] is not None else "–",
            "hashes per bin",
            empty=stats["median_sketch"] is None,
        ),
        _tile(
            "annotated",
            f"{100 * stats['annotated_fraction']:.0f}%"
            if stats["annotated_fraction"] is not None else "not run",
            "bins with a track hit",
            empty=stats["annotated_fraction"] is None,
        ),
    ]
    return "".join(tiles)


def _notes_html(notes: Sequence[str]) -> str:
    if not notes:
        return ""
    items = "".join(f'<li class="kd-note"><span>{n}</span></li>' for n in notes)
    return f'<ul class="kd-notes">{items}</ul>'


def _methods_html(run: _Run, cfg: Config, notes: Sequence[str]) -> str:
    def code(text: Any) -> str:
        return f"<code>{html.escape(str(text))}</code>"

    chroms = ", ".join(cfg.manifest.chroms) if cfg.manifest.chroms else "all contigs"
    paragraphs = [
        f"Every assembly was tiled into non-overlapping {code(_human_bp(cfg.sketch.bin_size))} "
        f"bins over {html.escape(chroms)}. Each bin is represented by the set of its canonical "
        f"{cfg.sketch.k}-mers whose splitmix64 hash falls below 2⁶⁴/{cfg.sketch.scaled} "
        f"(a FracMinHash sketch, so every assembly independently keeps the same ~1/"
        f"{cfg.sketch.scaled} sample of k-mer space). Bins with fewer than "
        f"{code(cfg.sketch.min_bin_sketch)} retained hashes, or with less than "
        f"{100 * cfg.sketch.min_bin_acgt_frac:.0f}% unambiguous bases, were dropped.",
        f"k-mers were kept when present in at least "
        f"{100 * cfg.select.min_sample_prevalence:.0f}% of <em>samples</em>"
        + (
            ""
            if cfg.select.max_sample_prevalence >= 1.0
            else f" and at most {100 * cfg.select.max_sample_prevalence:.0f}%"
        )
        + f" and in at least {code(cfg.select.min_bins)} bins"
        + (
            f", then deterministically sub-sampled to at most {code(f'{cfg.select.max_features:,}')}"
            " features."
            if cfg.select.max_features
            else "."
        )
        + f" The resulting sparse bin × k-mer matrix was weighted with "
        f"{code(cfg.matrix.weighting)} and {code(cfg.matrix.row_norm)} row normalisation, then "
        f"factored by randomized SVD into {code(cfg.decompose.n_components)} components "
        f"({cfg.decompose.n_iter} power iterations, {cfg.decompose.n_oversamples} oversamples"
        + (f", first {cfg.decompose.drop_first} discarded" if cfg.decompose.drop_first else "")
        + ").",
        f"Components were embedded with UMAP "
        f"({code('n_neighbors=' + str(cfg.embed.n_neighbors))}, "
        f"{code('min_dist=' + str(cfg.embed.min_dist))}, {code(cfg.embed.metric)} metric, "
        f"{cfg.embed.n_components}D) and clustered with {code(cfg.cluster.method)} in the "
        f"{code(cfg.cluster.space)} space "
        f"({code('min_cluster_size=' + str(cfg.cluster.min_cluster_size))}, "
        f"{code('min_samples=' + str(cfg.cluster.min_samples))}, "
        f"{code(cfg.cluster.cluster_selection_method)} selection). A bin counts as carrying a "
        f"feature when the feature covers at least {100 * cfg.enrich.min_frac:.0f}% of it; "
        f"enrichment is the log2 ratio of that rate inside a cluster to the genome-wide rate, "
        f"with a hypergeometric survival-function p-value.",
        f"Every random choice in the pipeline derives from seed {code(cfg.seed)}; the report "
        f"itself sub-samples with seed {code(cfg.report.seed)}. "
        + (
            "plotly.js is embedded in this file, which therefore needs no network access."
            if cfg.report.embed_plotlyjs
            else "plotly.js is loaded from the CDN because <code>report.embed_plotlyjs</code> "
            "is false, so this file needs network access to draw its plots."
        ),
    ]
    if notes:
        paragraphs.append(
            "Caveats for this particular run: "
            + " ".join(n.rstrip(".") + "." for n in notes)
        )
    missing = [k for k, v in sorted(run.stages.items()) if not v]
    if missing:
        paragraphs.append(
            "Stages with no output in this run directory: "
            + ", ".join(code(m) for m in missing)
            + "."
        )
    return "".join(f"<p>{p}</p>" for p in paragraphs)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def _stats(run: _Run, cfg: Config) -> dict[str, Any]:
    bins = run.bins
    n_bins = len(bins)
    n_components: int | None = None
    variance_note = ""
    if run.svd:
        n_components = _int_or_none(run.svd.get("n_components"))
        evr = run.svd.get("explained_variance_ratio")
        if isinstance(evr, (list, tuple)) and evr:
            try:
                total = float(np.nansum(np.asarray(evr, dtype=np.float64)))
                variance_note = f"{100 * total:.1f}% of variance"
            except (TypeError, ValueError):
                variance_note = ""
        if n_components is None:
            sv = run.svd.get("singular_values")
            n_components = len(sv) if isinstance(sv, (list, tuple)) else None
    if n_components is None and run.stages.get("decompose"):
        n_components = int(cfg.decompose.n_components)

    n_clusters: int | None = None
    noise_fraction: float | None = None
    if run.stages.get("cluster") and n_bins:
        labels = bins["cluster"].to_numpy()
        n_clusters = int(np.unique(labels[labels >= 0]).size)
        noise_fraction = float((labels < 0).mean())

    annotated_fraction: float | None = None
    if run.stages.get("annotate") and n_bins:
        # "unannotated" is the sentinel the annotate stage writes when every
        # frac_* column is zero, so it must not count as a track hit.
        dom = bins["dominant_feature"].astype("string").fillna("")
        annotated_fraction = float((~dom.isin(["", "unannotated"])).mean())

    median_sketch: int | None = None
    if n_bins:
        median_sketch = int(np.median(bins["n_sketch"].to_numpy(dtype=np.float64)))

    return {
        "n_bins": n_bins,
        "n_assemblies": int(bins["assembly"].nunique()) if n_bins else 0,
        "n_samples": int(bins["sample"].nunique()) if n_bins else 0,
        "n_features": run.n_features,
        "n_components": n_components,
        "variance_note": variance_note,
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "annotated_fraction": annotated_fraction,
        "median_sketch": median_sketch,
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


#: Last resort when plotly cannot be imported at all.  Pinned rather than
#: floating: a "latest" URL would silently change the rendering engine under a
#: report that was generated for a specific one.
_CDN_FALLBACK_JS = "3.0.1"


def _plotlyjs_version() -> str:
    """Version of the bundled **plotly.js**, which is not plotly.py's version.

    This distinction is load-bearing.  ``plotly.__version__`` is the Python
    package (6.9.0); the JavaScript it ships is a different project on its own
    numbering (3.7.0), and the CDN indexes the *JavaScript* one.  Building a CDN
    URL from the Python version yields ``plotly-6.9.0.min.js``, which does not
    exist -- the CDN answers 403 and the page renders as a blank white
    rectangle, with no console error that names the real problem.
    """
    try:
        from plotly.offline import get_plotlyjs_version

        return str(get_plotlyjs_version())
    except Exception:  # noqa: BLE001 - older plotly.py has no such helper
        pass
    try:
        import re

        from plotly.offline import get_plotlyjs

        match = re.search(r"plotly\.js v([\d.]+)", get_plotlyjs()[:400])
        if match:
            return match.group(1)
    except Exception:  # noqa: BLE001
        pass
    log.warning("could not determine the bundled plotly.js version; pinning %s", _CDN_FALLBACK_JS)
    return _CDN_FALLBACK_JS


def _plotly_block(cfg: Config) -> tuple[str, str]:
    """``(script tag, version)``, embedding the library when asked to."""
    try:
        import plotly  # noqa: F401

        version = _plotlyjs_version()
    except ImportError:  # pragma: no cover - plotly is a hard dependency
        log.warning("plotly is not importable; the report will fall back to the CDN")
        return (
            f'<script src="https://cdn.plot.ly/plotly-{_CDN_FALLBACK_JS}.min.js" '
            'charset="utf-8"></script>',
            "unknown",
        )
    if not cfg.report.embed_plotlyjs:
        return (
            f'<script src="https://cdn.plot.ly/plotly-{version}.min.js" charset="utf-8">'
            "</script>",
            version,
        )
    try:
        from plotly.offline import get_plotlyjs

        return f"<script>{get_plotlyjs()}</script>", version
    except (ImportError, OSError) as exc:  # pragma: no cover
        log.warning("could not inline plotly.js (%s); falling back to the CDN", exc)
        return (
            f'<script src="https://cdn.plot.ly/plotly-{version}.min.js" charset="utf-8">'
            "</script>",
            version,
        )


def _asset(name: str) -> str:
    return (_HERE / name).read_text(encoding="utf-8")


def _json_for_script(payload: dict[str, Any]) -> str:
    """JSON that is safe to paste inside a ``<script>`` element."""
    text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    # ``</script>`` inside a string literal would close the element early.
    return text.replace("<", "\\u003c")


def build_report(cfg: Config, outdir: Path, *, force: bool = False) -> Path:
    """Render ``report/kmer_dust_report.html`` and ``report/summary.json``.

    Returns the path of the HTML file.  Safe to call at any point in a run: any
    stage that has not produced output is reported as missing rather than
    raising.
    """
    root = Path(outdir)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "kmer_dust_report.html"
    summary_path = report_dir / "summary.json"

    if html_path.is_file() and summary_path.is_file() and not force:
        log.info("report already present at %s (use force=True to rebuild)", html_path)
        return html_path

    run = _load_run(cfg, root)
    stats = _stats(run, cfg)
    notes = list(run.notes)
    payload = _build_payload(run, cfg, notes)

    plotly_block, plotly_version = _plotly_block(cfg)
    title = cfg.report.title or cfg.run_name
    subtitle = cfg.report.subtitle or _default_subtitle(run, cfg, stats)
    meta_bits = [
        run.timestamp,
        f"{stats['n_bins']:,} bins",
        f"seed {cfg.seed}",
        f"plotly {plotly_version}",
    ]
    ribbon_subject = run.reference or "the reference"

    document = _asset("template.html")
    replacements = {
        "{{VERSION}}": html.escape(__version__),
        "{{TITLE}}": html.escape(f"{title} · kmer-dust"),
        "{{RUN_NAME}}": html.escape(cfg.run_name),
        "{{HEADLINE}}": html.escape(title),
        "{{SUBTITLE}}": html.escape(subtitle),
        "{{META_LINE}}": html.escape(" · ".join(meta_bits)),
        "{{BIN_SIZE_HUMAN}}": html.escape(_human_bp(cfg.sketch.bin_size)),
        "{{RIBBON_SUBJECT}}": html.escape(ribbon_subject),
        "{{STAT_TILES}}": _stat_tiles(run, cfg, stats),
        "{{FINDINGS}}": _findings_html(outdir),
        "{{NOTES}}": _notes_html(notes),
        "{{METHODS}}": _methods_html(run, cfg, notes),
        "{{FOOTER}}": html.escape(
            f"kmer-dust {__version__} · {cfg.run_name} · generated from {root} · {run.timestamp}"
        ),
        "{{CSS}}": _asset("report.css"),
        "{{APP_JS}}": _asset("report.js"),
        "{{PLOTLY}}": plotly_block,
        "{{DATA_JSON}}": _json_for_script(payload),
    }
    for token, value in replacements.items():
        document = document.replace(token, value)

    tmp = html_path.with_suffix(".html.tmp")
    tmp.write_text(document, encoding="utf-8")
    tmp.replace(html_path)
    size = html_path.stat().st_size
    if size > _SIZE_WARN_BYTES:
        log.warning(
            "report is %.1f MB; lower report.max_points to keep it comfortable to open",
            size / 1024 / 1024,
        )
    log.info("wrote %s (%.1f MB, %d points)", html_path, size / 1024 / 1024, payload["n"])

    summary = _summary(run, cfg, stats, payload, size, plotly_version)
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(summary_path)
    log.info("wrote %s", summary_path)
    return html_path


def _default_subtitle(run: _Run, cfg: Config, stats: dict[str, Any]) -> str:
    chroms = ", ".join(cfg.manifest.chroms) if cfg.manifest.chroms else "all contigs"
    parts = [
        f"{stats['n_assemblies']:,} assemblies from {stats['n_samples']:,} samples",
        chroms,
        f"{_human_bp(cfg.sketch.bin_size)} bins",
        f"{cfg.sketch.k}-mers at scaled={cfg.sketch.scaled}",
    ]
    if run.reference:
        parts.append(f"reference {run.reference}")
    return " · ".join(parts)


def _summary(
    run: _Run,
    cfg: Config,
    stats: dict[str, Any],
    payload: dict[str, Any],
    size: int,
    plotly_version: str,
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    table = payload.get("table")
    if table:
        keys = [c["key"] for c in table["columns"]]
        for row in table["rows"]:
            clusters.append({k: row[i] for i, k in enumerate(keys)})
    return {
        "kmer_dust_version": __version__,
        "run_name": cfg.run_name,
        "generated": run.timestamp,
        "outdir": str(Path(cfg.outdir)),
        "reference": run.reference,
        "stages": {k: bool(v) for k, v in sorted(run.stages.items())},
        "counts": {
            "assemblies": stats["n_assemblies"],
            "samples": stats["n_samples"],
            "bins": stats["n_bins"],
            "kmers_selected": stats["n_features"],
            "components": stats["n_components"],
            "clusters": stats["n_clusters"],
        },
        "noise_fraction": stats["noise_fraction"],
        "annotated_fraction": stats["annotated_fraction"],
        "median_bin_sketch": stats["median_sketch"],
        "report": {
            "html_bytes": int(size),
            "points_plotted": payload["n"],
            "subsampled": bool(payload["n"] < stats["n_bins"]),
            "plotly_version": plotly_version,
            "embed_plotlyjs": bool(cfg.report.embed_plotlyjs),
        },
        "clusters": clusters,
        "config": {
            "k": cfg.sketch.k,
            "bin_size": cfg.sketch.bin_size,
            "scaled": cfg.sketch.scaled,
            "chroms": list(cfg.manifest.chroms),
            "weighting": cfg.matrix.weighting,
            "row_norm": cfg.matrix.row_norm,
            "n_components": cfg.decompose.n_components,
            "n_neighbors": cfg.embed.n_neighbors,
            "min_dist": cfg.embed.min_dist,
            "cluster_method": cfg.cluster.method,
            "min_cluster_size": cfg.cluster.min_cluster_size,
            "seed": cfg.seed,
        },
    }
