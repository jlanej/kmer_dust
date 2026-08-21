"""Where the biology enters: annotation tracks -> per-bin covered fractions.

Everything upstream of this module is deliberately reference-free -- bins are
described only by the k-mers they contain.  This module is the only place that
knows what a centromere *is*, and it exists so that the clusters found without
any alignment can afterwards be *tested* against independent biological
annotation.  Keeping the two apart is the whole point: if the annotation leaked
into the features, the agreement measured in :mod:`kmer_dust.backprop` would be
circular.

Three problems have to be solved carefully.

**Vocabulary.**  The T2T cenSat v2.0 BED and the 461 per-assembly HPRC cenSat
BEDs describe the same satellites with different spellings (``hor_1_1(...)``
versus ``active_hor(...)``, ``HSat3`` versus ``hsat3``, ``cenSat(SST1,SST1v)``
versus ``censat_9_1(SST1_Composite)``).  Reference and assembly bins are only
comparable after both are pushed onto :data:`kmer_dust.schemas.CENSAT_CLASSES`,
so :func:`normalize_censat_name` is written against the *observed* vocabulary of
both files rather than against the published colour key.

**Live HOR arrays.**  The HPRC tracks flag the active alpha-satellite array
explicitly (``active_hor``); the T2T v2.0 track does not -- it writes ``hor``
for everything and encodes "live" in the suffix ``L`` of the single suprachromosomal
family token, e.g. ``hor_1_5(S1C1/5/19H1L)``.  On the real T2T file that rule
(exactly one SF token, ending in ``L``) reproduces the track's own red
``250,0,0`` "live HOR" colour on 31/31 rows with no false positives, so it is
used here to make the reference speak the same dialect as the assemblies.

**Speed.**  The T2T RepeatMasker BED is ~343 MB / 2.7 M rows and is read once per
run, so the parsed, chromosome-subsetted result is cached as parquet keyed by a
hash of (url, contigs).  Overlap computation is a per-contig ``np.searchsorted``
sweep over merged intervals, never a Python loop over bins.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

from kmer_dust import schemas
from kmer_dust.config import Config
from kmer_dust.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "BED_COLUMNS",
    "normalize_censat_name",
    "normalize_repeat_class",
    "read_bed",
    "read_gff_genes",
    "bin_feature_fractions",
    "annotate_bins",
    "load_annotations",
]

#: Contract for the frame returned by :func:`read_bed` and :func:`read_gff_genes`.
BED_COLUMNS: dict[str, str] = {
    "chrom": "string",
    "start": "int64",
    "end": "int64",
    "name": "string",
    "score": "float64",
    "strand": "string",
    "extra": "object",  # list[str] of the columns beyond BED6
}

#: Track kinds understood by :func:`annotate_bins`, and the manifest column that
#: carries the per-assembly file for each.
ASSEMBLY_TRACK_COLUMNS: dict[str, str] = {
    "censat": "censat_bed",
    "repeatmasker": "repeatmasker_bed",
    "segdup": "segdup_bed",
}

_URL_PREFIXES = ("http://", "https://", "ftp://")
_MAX_PARSE_WIDTH = 4096  # refuse to widen a ragged table past this many columns


# --------------------------------------------------------------------------
# vocabulary normalisation
# --------------------------------------------------------------------------

#: Head of a cenSat name (everything before the optional ``(detail)``) mapped
#: onto :data:`kmer_dust.schemas.CENSAT_CLASSES`.  Keys are lower-cased.
_CENSAT_HEADS: dict[str, str] = {
    # alpha satellite, higher-order repeat
    "active_hor": "asat_hor_active",
    "activehor": "asat_hor_active",
    "active-hor": "asat_hor_active",
    "hor_active": "asat_hor_active",
    "livehor": "asat_hor_active",
    "hor": "asat_hor",
    "dhor": "asat_hor",
    "divergent_hor": "asat_hor",
    # alpha satellite, not organised into HORs
    "mon": "asat_mon",
    "monomer": "asat_mon",
    "monomeric": "asat_mon",
    "mixedalpha": "asat_mon",
    "mixed_alpha": "asat_mon",
    # human satellites
    "hsat1a": "hsat1a",
    "hsat1b": "hsat1b",
    "hsat2": "hsat2",
    "hsat3": "hsat3",
    # legacy combined HSat2/3 annotation: unsplittable from the label alone,
    # and dominated by HSat3 in both the T2T and HPRC tracks.
    "hsat2_3": "hsat3",
    "hsat23": "hsat3",
    "hsat2and3": "hsat3",
    "hsat2_and_3": "hsat3",
    "hsat4": "censat_other",
    "hsat5": "censat_other",
    # other satellite families
    "bsat": "bsat",
    "betasat": "bsat",
    "beta": "bsat",
    "gsat": "gsat",
    "gsatii": "gsat",
    "gsatx": "gsat",
    "gammasat": "gsat",
    "tar": "gsat",
    "tar1": "gsat",
    "censat": "censat_other",
    "cen": "censat_other",
    "novel": "censat_other",
    "noveltandem": "censat_other",
    "rdna": "rdna",
    "ct": "ct",
    "transition": "ct",
    "sst1": "subterminal",
    "sst": "subterminal",
    "subterminal": "subterminal",
    "subtelo": "subterminal",
    "subtelomeric": "subterminal",
}

#: Parenthesised details that are *more* specific than a generic ``cenSat(...)``
#: head.  Only consulted when the head itself resolved to ``censat_other``.
_CENSAT_DETAILS: tuple[tuple[str, str], ...] = (
    ("sst1", "subterminal"),
    ("subtelo", "subterminal"),
    ("subterminal", "subterminal"),
    ("tar1", "gsat"),
    ("tar", "gsat"),
    ("gsat", "gsat"),
)

#: Names that are known but deliberately carry no satellite class (assembly gaps).
_CENSAT_IGNORED = frozenset({"gap", "n", "unknown", "na", "."})

#: Trailing ``_<chrom>_<index>`` decorations used by the T2T track
#: (``hor_1_5``, ``mon_X_6``, ``rDNA_21_1``).
_CENSAT_SUFFIX = re.compile(r"(?:_(?:\d+|[XYMxym]))+$")
_CLASS_DETAIL = re.compile(r"^([^(]*)\((.*)\)\s*$", re.DOTALL)

_REPEAT_CLASSES: dict[str, str] = {
    "line": "line",
    "sine": "sine",
    "ltr": "ltr",
    "dna": "dna",
    "satellite": "satellite",
    "sat": "satellite",
    "beta": "satellite",  # RepeatMasker's own beta-satellite class
    "simple_repeat": "simple_repeat",
    "simple": "simple_repeat",
    "low_complexity": "low_complexity",
    "rrna": "rrna",
    "trna": "trna",
    "snrna": "snrna",
    "retroposon": "retroposon",
    "rc": "rc",
    "unknown": "repeat_unknown",
    "unspecified": "repeat_unknown",
    "undefined": "repeat_unknown",
}

#: Reported once each at DEBUG so the vocabularies above can be grown from a real run.
_seen_unknown_censat: set[str] = set()
_seen_unknown_repeat: set[str] = set()


def _note_unknown(store: set[str], kind: str, raw: str) -> None:
    if raw not in store:
        store.add(raw)
        logger.debug("unrecognised %s name %r -- mapped to ''", kind, raw)


def _strip_index_suffix(head: str) -> str:
    return _CENSAT_SUFFIX.sub("", head)


def normalize_censat_name(raw: str) -> str:
    """Map a raw cenSat track name onto :data:`schemas.CENSAT_CLASSES`.

    Case-insensitive, tolerant of the ``class(detail)`` wrapper, of the T2T
    ``_<chrom>_<index>`` decoration and of comma-separated multi-labels such as
    ``GAP,HSat2``.  Returns ``""`` for anything unrecognised -- an annotation we
    cannot place must not silently become a *wrong* class.
    """
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    match = _CLASS_DETAIL.match(text)
    head, detail = (match.group(1).strip(), match.group(2).strip()) if match else (text, "")

    resolved = ""
    head_key = ""
    for token in head.split(","):
        token = token.strip()
        if not token:
            continue
        key = token.lower()
        if key in _CENSAT_IGNORED:
            continue
        resolved = _CENSAT_HEADS.get(key, "")
        if not resolved:
            key = _strip_index_suffix(key)
            resolved = _CENSAT_HEADS.get(key, "")
        if resolved:
            head_key = key
            break

    if not resolved:
        # A few files carry the class only in the parenthesised part.
        for token in detail.split(","):
            key = _strip_index_suffix(token.strip().lower())
            resolved = _CENSAT_HEADS.get(key, "")
            if resolved:
                head_key = key
                break

    if not resolved:
        if head.strip().lower() not in _CENSAT_IGNORED:
            _note_unknown(_seen_unknown_censat, "cenSat", text)
        return ""

    if detail:
        if head_key == "hor" and "," not in detail and detail.upper().endswith("L"):
            # Exactly one suprachromosomal-family token ending in "L" == live array.
            resolved = "asat_hor_active"
        elif resolved == "censat_other":
            lowered = detail.lower()
            for needle, refined in _CENSAT_DETAILS:
                if any(part.strip().startswith(needle) for part in lowered.split(",")):
                    resolved = refined
                    break
    return resolved


def _lookup_repeat(candidate: str | None) -> str:
    if candidate is None:
        return ""
    text = str(candidate).strip()
    if not text or text in {".", "-"}:
        return ""
    text = text.split("/", 1)[0]  # "LINE/L1" and "Satellite/centr" spellings
    key = text.rstrip("?").strip().lower().replace("-", "_").replace(" ", "_")
    return _REPEAT_CLASSES.get(key, "")


def normalize_repeat_class(cls: str, family: str = "") -> str:
    """Map a RepeatMasker class (column 7) onto :data:`schemas.REPEAT_CLASSES`.

    Handles the ``"?"``-suffixed uncertain classes and combined ``class/family``
    spellings.  ``family`` is consulted only when the class is blank or
    unrecognised *and* the family names something specific: the T2T BED writes
    ``undefined`` in the family column for every class it does not subdivide, so
    trusting it unconditionally would quietly turn genuinely unseen classes
    (``srpRNA``, ``scRNA``) into ``repeat_unknown`` and lose the DEBUG record
    that lets the vocabulary be grown.
    """
    primary = _lookup_repeat(cls)
    if primary:
        return primary
    secondary = _lookup_repeat(family)
    if secondary and secondary != "repeat_unknown":
        return secondary
    raw = str(cls or "").strip() or str(family or "").strip()
    if raw and raw not in {".", "-"}:
        _note_unknown(_seen_unknown_repeat, "RepeatMasker class", raw)
    return ""


# --------------------------------------------------------------------------
# reading tracks
# --------------------------------------------------------------------------


def _is_url(path_or_url: str) -> bool:
    return str(path_or_url).startswith(_URL_PREFIXES)


def _cache_name(url: str) -> str:
    """Collision-free but still greppable cache file name for a URL."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    base = url.rstrip("/").rsplit("/", 1)[-1] or "download"
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:80]
    return f"{digest}_{base}"


def _download(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` via ``*.part`` + rename so a partial file never lands."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        import requests

        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
    except ImportError:  # pragma: no cover - requests is a declared dependency
        import shutil
        import urllib.request

        with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as handle:
            shutil.copyfileobj(response, handle, 1 << 20)
    tmp.replace(dest)
    return dest


def _local_copy(path_or_url: str, cache_dir: Path | None) -> Path:
    """Return a local, re-openable path for ``path_or_url`` (downloading if needed)."""
    text = str(path_or_url)
    if not _is_url(text):
        path = Path(text).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"annotation track not found: {path}")
        return path
    root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "kmer_dust"
    dest = root / "tracks" / _cache_name(text)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    # fasta.py owns the shared download/cache helper; fall back so that this
    # module stays usable (and testable) on its own.
    try:
        from kmer_dust.fasta import download as shared_download
    except ImportError:
        return _download(text, dest)
    return Path(shared_download(text, dest))


def _open_text_path(path: Path) -> IO[str]:
    """Open ``path`` as text, transparently decompressing gzip/bgzf by magic bytes."""
    with open(path, "rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _peek_width(path: Path, limit: int = 4096) -> int:
    """Widest tab-delimited row among the first ``limit`` lines."""
    width = 0
    with _open_text_path(path) as handle:
        for i, line in enumerate(handle):
            if i >= limit:
                break
            width = max(width, line.count("\t") + 1)
    return width


_SAW_FIELDS = re.compile(r"saw (\d+)")


def _read_ragged_tsv(path: Path, min_width: int = 3) -> pd.DataFrame:
    """Parse a tab-delimited file whose rows may have *different* widths.

    pandas' C parser needs to know the column count up front and errors out on
    any row that is wider, so start from the observed width and widen on the
    parser's own "expected N fields, saw M" complaint.  ``comment=`` is never
    used: HPRC contig names look like ``HG00408#1#CM085953.1`` and a ``#``
    comment character would truncate them mid-field.
    """
    width = max(_peek_width(path), min_width)
    last: Exception | None = None
    while width <= _MAX_PARSE_WIDTH:
        try:
            with _open_text_path(path) as handle:
                return pd.read_csv(
                    handle,
                    sep="\t",
                    header=None,
                    names=list(range(width)),
                    index_col=False,
                    dtype=str,
                    na_filter=False,
                    quoting=csv.QUOTE_NONE,
                    engine="c",
                )
        except pd.errors.ParserError as exc:
            last = exc
            found = _SAW_FIELDS.search(str(exc))
            seen = int(found.group(1)) if found else 0
            width = max(seen, width * 2)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raise ValueError(f"{path}: rows wider than {_MAX_PARSE_WIDTH} columns") from last


def _empty_bed() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "chrom": pd.Series([], dtype="string"),
            "start": pd.Series([], dtype="int64"),
            "end": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="string"),
            "score": pd.Series([], dtype="float64"),
            "strand": pd.Series([], dtype="string"),
        }
    )
    frame["extra"] = pd.Series([], dtype="object")
    return frame


def read_bed(path_or_url: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Read a BED3..BED9+ file into ``chrom start end name score strand extra``.

    Copes with leading ``track``/``browser`` lines, ``#`` comments and header
    rows (the HPRC segdup BEDs start with ``#chr1<TAB>start1<TAB>...``), plain
    or gzipped input, local paths or ``https://`` URLs, and rows with ragged
    trailing columns.  Non-numeric ``start``/``end`` is the discriminator: any
    row that fails it is metadata, not an interval, and is dropped.
    """
    path = _local_copy(str(path_or_url), cache_dir)
    raw = _read_ragged_tsv(path, min_width=3)
    if raw.empty or raw.shape[1] < 3:
        logger.warning("%s: no BED intervals found", path_or_url)
        return _empty_bed()

    start = pd.to_numeric(raw[1], errors="coerce")
    end = pd.to_numeric(raw[2], errors="coerce")
    keep = start.notna() & end.notna() & raw[0].astype(str).str.len().gt(0)
    dropped = int((~keep).sum())
    if dropped:
        logger.debug("%s: skipped %d non-interval line(s)", path_or_url, dropped)
    raw = raw.loc[keep]
    if raw.empty:
        logger.warning("%s: no BED intervals survived parsing", path_or_url)
        return _empty_bed()

    n = len(raw)
    out = pd.DataFrame(index=pd.RangeIndex(n))
    out["chrom"] = raw[0].to_numpy(dtype=object)
    out["start"] = start.loc[keep].to_numpy(dtype="int64")
    out["end"] = end.loc[keep].to_numpy(dtype="int64")
    out["name"] = raw[3].to_numpy(dtype=object) if raw.shape[1] > 3 else ""
    out["score"] = (
        pd.to_numeric(raw[4], errors="coerce").to_numpy(dtype="float64")
        if raw.shape[1] > 4
        else np.nan
    )
    strand = raw[5].to_numpy(dtype=object) if raw.shape[1] > 5 else None
    out["strand"] = strand if strand is not None else "."
    if raw.shape[1] > 6:
        out["extra"] = pd.Series(raw.iloc[:, 6:].to_numpy(dtype=object).tolist(), dtype="object")
    else:
        out["extra"] = pd.Series([[] for _ in range(n)], dtype="object")

    for col in ("chrom", "name", "strand"):
        out[col] = out[col].fillna("").astype("string")
    out.loc[~out["strand"].isin(["+", "-", "."]), "strand"] = "."
    return out


def read_gff_genes(path_or_url: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Read GFF3 ``gene`` features into the same frame shape as :func:`read_bed`.

    The CAT/Liftoff annotation carries transcripts, exons, introns and CDS in the
    same file; only ``gene`` rows are kept, so a bin's ``frac_gene`` is genic
    footprint rather than exonic density.  GFF3 coordinates are 1-based and
    inclusive, BED is 0-based half-open, hence the ``start - 1``.
    """
    path = _local_copy(str(path_or_url), cache_dir)
    raw = _read_ragged_tsv(path, min_width=9)
    if raw.empty or raw.shape[1] < 9:
        logger.warning("%s: no GFF3 records found", path_or_url)
        return _empty_bed()

    is_gene = raw[2].astype(str).str.lower().eq("gene")
    start = pd.to_numeric(raw[3], errors="coerce")
    end = pd.to_numeric(raw[4], errors="coerce")
    keep = is_gene & start.notna() & end.notna()
    raw = raw.loc[keep]
    if raw.empty:
        logger.warning("%s: no GFF3 'gene' features found", path_or_url)
        return _empty_bed()

    attrs = raw[8].astype(str)
    name = attrs.str.extract(r"(?:^|;)gene_name=([^;]*)", expand=False)
    name = name.fillna(attrs.str.extract(r"(?:^|;)Name=([^;]*)", expand=False))
    name = name.fillna(attrs.str.extract(r"(?:^|;)gene_id=([^;]*)", expand=False))
    name = name.fillna(attrs.str.extract(r"(?:^|;)ID=([^;]*)", expand=False)).fillna("")
    biotype = attrs.str.extract(r"(?:^|;)biotype=([^;]*)", expand=False).fillna("")

    n = len(raw)
    out = pd.DataFrame(index=pd.RangeIndex(n))
    out["chrom"] = raw[0].to_numpy(dtype=object)
    out["start"] = (start.loc[keep].to_numpy(dtype="int64") - 1).clip(min=0)
    out["end"] = end.loc[keep].to_numpy(dtype="int64")
    out["name"] = name.to_numpy(dtype=object)
    out["score"] = pd.to_numeric(raw[5], errors="coerce").to_numpy(dtype="float64")
    out["strand"] = raw[6].to_numpy(dtype=object)
    out["extra"] = pd.Series(
        [[s, b] for s, b in zip(raw[1].astype(str), biotype.astype(str))], dtype="object"
    )
    for col in ("chrom", "name", "strand"):
        out[col] = out[col].fillna("").astype("string")
    out.loc[~out["strand"].isin(["+", "-", "."]), "strand"] = "."
    return out


# --------------------------------------------------------------------------
# interval -> bin coverage
# --------------------------------------------------------------------------


def _contig_values(frame: pd.DataFrame, what: str) -> np.ndarray:
    """Contig key of a frame, accepting either ``contig`` (bins) or ``chrom`` (BED)."""
    for col in ("contig", "chrom"):
        if col in frame.columns:
            return frame[col].astype("string").fillna("").to_numpy(dtype=object)
    raise ValueError(f"{what} frame needs a 'contig' or 'chrom' column")


def _merge_sorted(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge start-sorted intervals so overlaps cannot be counted twice."""
    if starts.size == 0:
        return starts, ends
    cummax = np.maximum.accumulate(ends)
    head = np.empty(starts.size, dtype=bool)
    head[0] = True
    head[1:] = starts[1:] > cummax[:-1]
    idx = np.flatnonzero(head)
    return starts[idx], np.maximum.reduceat(ends, idx)


def _coverage_upto(x: np.ndarray, ms: np.ndarray, me: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """Bases covered by the merged intervals strictly below each position ``x``."""
    j = np.searchsorted(ms, x, side="right") - 1
    cov = np.zeros(x.shape, dtype=np.int64)
    ok = j >= 0
    if not ok.any():
        return cov
    jj = j[ok]
    partial = np.minimum(x[ok], me[jj]) - ms[jj]
    np.maximum(partial, 0, out=partial)
    cov[ok] = pre[jj] + partial
    return cov


def bin_feature_fractions(
    bins: pd.DataFrame, intervals: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    """Covered fraction of every bin by every feature.

    Parameters
    ----------
    bins:
        One row per bin with ``contig`` (or ``chrom``), ``start`` and ``end``.
        Row order is preserved; the rows need not be sorted.
    intervals:
        Annotation intervals with ``chrom`` (or ``contig``), ``start``, ``end``
        and a ``feature`` column naming one of ``features``.
    features:
        Ordered feature vocabulary; the returned columns follow it.

    Returns
    -------
    ``(n_bins, n_features)`` float32 of covered fraction in ``[0, 1]``.

    Overlapping intervals of the same feature are merged first, so a fraction
    can never exceed 1.  The sweep is vectorised: per ``(contig, feature)``
    group the merged intervals become a prefix-sum of covered length, and every
    bin edge is located in it with one :func:`numpy.searchsorted`.
    """
    feature_list = list(features)
    n_bins = int(len(bins))
    n_feat = len(feature_list)
    out = np.zeros((n_bins, n_feat), dtype=np.float32)
    if n_bins == 0 or n_feat == 0 or intervals is None or len(intervals) == 0:
        return out

    bin_contig = _contig_values(bins, "bins")
    bin_start = bins["start"].to_numpy(dtype=np.int64)
    bin_end = bins["end"].to_numpy(dtype=np.int64)
    width = (bin_end - bin_start).astype(np.float64)

    contigs, bin_code = np.unique(bin_contig, return_inverse=True)
    order = np.argsort(bin_code, kind="stable")
    sorted_code = bin_code[order]
    group_lo = np.searchsorted(sorted_code, np.arange(contigs.size), side="left")
    group_hi = np.searchsorted(sorted_code, np.arange(contigs.size), side="right")

    feature_index = {name: i for i, name in enumerate(feature_list)}
    iv_contig = _contig_values(intervals, "intervals")
    iv_feature = intervals["feature"].astype("string").fillna("").to_numpy(dtype=object)
    iv_start = intervals["start"].to_numpy(dtype=np.int64)
    iv_end = intervals["end"].to_numpy(dtype=np.int64)

    contig_pos = {name: i for i, name in enumerate(contigs)}
    ic = np.fromiter(
        (contig_pos.get(c, -1) for c in iv_contig), dtype=np.int64, count=len(iv_contig)
    )
    fc = np.fromiter(
        (feature_index.get(f, -1) for f in iv_feature), dtype=np.int64, count=len(iv_feature)
    )
    lo = np.maximum(iv_start, 0)
    hi = np.maximum(iv_end, lo)
    usable = (ic >= 0) & (fc >= 0) & (hi > lo)
    if not usable.any():
        return out
    ic, fc, lo, hi = ic[usable], fc[usable], lo[usable], hi[usable]

    key = ic * n_feat + fc
    ordering = np.lexsort((lo, key))
    key, lo, hi = key[ordering], lo[ordering], hi[ordering]
    uniq, first = np.unique(key, return_index=True)
    bounds = np.append(first, key.size)

    for gi in range(uniq.size):
        a, b = int(bounds[gi]), int(bounds[gi + 1])
        contig_i, feature_i = divmod(int(uniq[gi]), n_feat)
        rows = order[group_lo[contig_i] : group_hi[contig_i]]
        if rows.size == 0:
            continue
        ms, me = _merge_sorted(lo[a:b], hi[a:b])
        pre = np.concatenate(([0], np.cumsum(me - ms)))
        covered = _coverage_upto(bin_end[rows], ms, me, pre) - _coverage_upto(
            bin_start[rows], ms, me, pre
        )
        w = width[rows]
        frac = np.divide(covered, w, out=np.zeros(rows.size, dtype=np.float64), where=w > 0)
        out[rows, feature_i] = np.clip(frac, 0.0, 1.0).astype(np.float32)
    return out


# --------------------------------------------------------------------------
# track -> (contig, start, end, feature) intervals
# --------------------------------------------------------------------------


def _censat_features(bed: pd.DataFrame) -> pd.Series:
    names = bed["name"].astype(str)
    lookup = {raw: normalize_censat_name(raw) for raw in pd.unique(names)}
    return names.map(lookup)


def _repeat_features(bed: pd.DataFrame) -> pd.Series:
    extra = bed["extra"]
    cls = extra.map(lambda e: e[0] if isinstance(e, (list, tuple)) and len(e) > 0 else "")
    fam = extra.map(lambda e: e[1] if isinstance(e, (list, tuple)) and len(e) > 1 else "")
    pairs = pd.Series(list(zip(cls.astype(str), fam.astype(str))), index=bed.index)
    lookup = {pair: normalize_repeat_class(pair[0], pair[1]) for pair in set(pairs)}
    return pairs.map(lookup)


def _track_intervals(kind: str, url: str, cache_dir: Path | None) -> pd.DataFrame:
    """Parse one track into ``contig start end feature`` (no filtering yet)."""
    if kind == "gene" or re.search(r"\.gff3?(\.gz)?$", url, flags=re.IGNORECASE):
        bed = read_gff_genes(url, cache_dir)
        feature = pd.Series("gene", index=bed.index, dtype=object)
    else:
        bed = read_bed(url, cache_dir)
        if kind == "censat":
            feature = _censat_features(bed)
        elif kind == "repeatmasker":
            feature = _repeat_features(bed)
        elif kind in schemas.EXTRA_FEATURES:
            feature = pd.Series(kind, index=bed.index, dtype=object)
        else:
            raise ValueError(f"unknown annotation track kind: {kind!r}")
    frame = pd.DataFrame(
        {
            "contig": bed["chrom"].astype("string"),
            "start": bed["start"].astype("int64"),
            "end": bed["end"].astype("int64"),
            "feature": pd.Series(feature, dtype="string").fillna(""),
        }
    )
    return frame.loc[frame["feature"].str.len() > 0].reset_index(drop=True)


def _cached_track_intervals(
    kind: str, url: str, contigs: Sequence[str], cache_dir: Path | None, *, force: bool = False
) -> pd.DataFrame:
    """:func:`_track_intervals` restricted to ``contigs``, cached as parquet.

    The T2T RepeatMasker BED is ~343 MB and every assembly in the run would
    otherwise re-parse it; the cache key is a hash of (kind, url, contigs) so a
    changed chromosome selection transparently reparses.
    """
    wanted = sorted({str(c) for c in contigs})
    if cache_dir is None:
        frame = _track_intervals(kind, url, cache_dir)
        return frame.loc[frame["contig"].isin(wanted)].reset_index(drop=True) if wanted else frame

    digest = hashlib.sha1(("\n".join([kind, url, *wanted])).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / "tracks" / f"{kind}-{digest}.parquet"
    if path.exists() and not force:
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            logger.warning("re-parsing %s: unreadable cache %s (%s)", url, path, exc)
    frame = _track_intervals(kind, url, cache_dir)
    if wanted:
        frame = frame.loc[frame["contig"].isin(wanted)].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per writer: the prefetch pool may have two threads racing on the
    # same key if the same track is requested by two assemblies at once.
    tmp = path.with_suffix(f".parquet.{os.getpid()}.{threading.get_ident():x}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)
    return frame


def _prefetch_track_intervals(
    jobs: Sequence[tuple[str, str, tuple[str, ...]]],
    cache_dir: Path | None,
    *,
    force: bool = False,
    workers: int = 8,
) -> None:
    """Warm the parsed-track cache concurrently before the annotation loop.

    Each per-assembly RepeatMasker BED is ~167 MB, so a 24-haplotype run moves
    several gigabytes here -- more than the sketching stage downloads.  Fetching
    them one at a time makes ``annotate`` the second-slowest stage for no good
    reason: the work is almost entirely network wait, and the parse that follows
    releases the GIL inside pandas.  Failures are swallowed on purpose; this is
    a cache warm-up, and the real attempt in :func:`annotate_bins` is where an
    error deserves to be reported against its assembly.
    """
    if cache_dir is None or not jobs:
        return
    unique = sorted({(kind, url, contigs) for kind, url, contigs in jobs})
    if len(unique) < 2:
        return
    workers = max(1, min(int(workers), len(unique)))
    logger.info("annotate: prefetching %d track(s) with %d worker(s)", len(unique), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_cached_track_intervals, kind, url, contigs, cache_dir, force=force): url
            for kind, url, contigs in unique
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - reported for real in annotate_bins
                logger.debug("annotate: prefetch of %s failed (%s)", futures[future], exc)


def _tracks_for_row(row: Mapping[str, Any], cfg: Config) -> dict[str, str]:
    """Which ``kind -> url`` tracks apply to one manifest row (empty URLs dropped)."""
    source = str(row.get("source", "") or "")
    assembly = str(row.get("assembly", "") or "")
    is_reference = source == "t2t" or str(row.get("haplotype", "")) == "ref"
    tracks: dict[str, str] = {}
    if is_reference:
        catalog: dict[str, str] = {}
        try:
            from kmer_dust.catalog import t2t

            catalog = dict(getattr(t2t, "T2T_TRACKS", {}) or {})
        except ImportError as exc:  # pragma: no cover - catalog is a hard dependency at runtime
            logger.warning("no reference track catalog for %s: %s", assembly, exc)
        for kind in cfg.annotate.reference_tracks:
            # A path spelled out in the manifest always wins over the published
            # genome-wide URL.  That is what lets a *sliced* reference -- the
            # test fixture, or any user-cut region -- carry its own rebased
            # tracks; using the genome-wide BED against slice-local coordinates
            # would silently annotate the wrong part of the chromosome.
            column = ASSEMBLY_TRACK_COLUMNS.get(kind)
            local = str(row.get(column, "") or "") if column else ""
            if local and local.lower() != "nan":
                tracks[kind] = local
                continue
            url = str(catalog.get(kind, "") or "")
            if url:
                tracks[kind] = url
            else:
                logger.debug("reference track %r has no URL -- skipped", kind)
        return tracks
    if not cfg.annotate.annotate_assemblies:
        return {}
    for kind in cfg.annotate.assembly_tracks:
        column = ASSEMBLY_TRACK_COLUMNS.get(kind)
        if column is None:
            logger.debug("no per-assembly column for track %r -- skipped", kind)
            continue
        url = str(row.get(column, "") or "")
        if url and url.lower() != "nan":
            tracks[kind] = url
    return tracks


# --------------------------------------------------------------------------
# stage entry points
# --------------------------------------------------------------------------


def _empty_annotations() -> pd.DataFrame:
    frame = schemas.empty_frame(schemas.ANNOTATION_ID_COLUMNS)
    for col in schemas.FEATURE_COLUMNS:
        frame[col] = pd.Series([], dtype="float32")
    return frame


def annotate_bins(
    rows: pd.DataFrame,
    manifest: pd.DataFrame,
    cfg: Config,
    outdir: Path,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Annotate every bin in ``rows`` with the covered fraction of each feature.

    Reference bins are annotated from ``catalog.t2t.T2T_TRACKS``, HPRC bins from
    the per-assembly BEDs named in the manifest.  Output has exactly one row per
    row of ``rows``, in the same order, keyed by ``bin_uid``.  An assembly with
    no usable tracks -- none configured, none in the manifest, or every one of
    them failing to download -- yields ``annotated=False`` and all-zero
    fractions, which lets downstream stages exclude it from denominators
    instead of mistaking "not looked at" for "nothing there".
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "annotations.parquet"
    if path.exists() and not force:
        logger.info("annotate: reusing %s", path)
        return load_annotations(outdir)

    if rows is None or len(rows) == 0:
        logger.warning("annotate: no bins to annotate")
        frame = _empty_annotations()
        _write_annotations(frame, path)
        return frame

    features = list(schemas.FEATURE_VOCAB)
    n_rows = len(rows)
    fracs = np.zeros((n_rows, len(features)), dtype=np.float32)
    annotated = np.zeros(n_rows, dtype=bool)

    if manifest is None or len(manifest) == 0:
        manifest_by_assembly: dict[str, Mapping[str, Any]] = {}
    else:
        manifest_by_assembly = {
            str(rec["assembly"]): rec for rec in manifest.to_dict(orient="records")
        }

    # ``.indices`` gives positional indices, so results can be scattered straight
    # back into ``fracs``; sorting the keys keeps the log (and any warning) stable.
    by_assembly = rows.groupby(rows["assembly"].astype(str), sort=False).indices

    # Resolve every (assembly -> tracks, contigs) pair up front so the downloads
    # can overlap; the loop below then works against a warm cache.
    plan: list[tuple[str, np.ndarray, dict[str, str], list[str]]] = []
    for assembly in sorted(by_assembly):
        take = np.asarray(by_assembly[assembly], dtype=np.int64)
        row = manifest_by_assembly.get(str(assembly))
        if row is None:
            logger.warning("annotate: %s is not in the manifest -- left unannotated", assembly)
            continue
        tracks = _tracks_for_row(row, cfg)
        if not tracks:
            logger.info("annotate: %s has no annotation tracks", assembly)
            continue
        contigs = sorted(set(rows.iloc[take]["contig"].astype(str)))
        plan.append((assembly, take, tracks, contigs))

    _prefetch_track_intervals(
        [(kind, url, tuple(contigs)) for _a, _t, tracks, contigs in plan for kind, url in tracks.items()],
        cache_dir,
        force=force,
        workers=max(1, min(8, int(getattr(cfg, "threads", 4) or 4))),
    )

    for assembly, take, tracks, contigs in plan:
        sub = rows.iloc[take]
        pieces: list[pd.DataFrame] = []
        ok = 0
        for kind, url in tracks.items():
            try:
                piece = _cached_track_intervals(kind, url, contigs, cache_dir, force=force)
            except Exception as exc:  # noqa: BLE001 - one bad track must not kill the run
                logger.warning("annotate: %s %s track failed (%s)", assembly, kind, exc)
                continue
            ok += 1
            if len(piece):
                pieces.append(piece)
            else:
                logger.warning(
                    "annotate: %s %s track has no intervals on %d contig(s) -- name mismatch?",
                    assembly,
                    kind,
                    len(contigs),
                )
        if ok == 0:
            logger.warning("annotate: every track failed for %s", assembly)
            continue
        annotated[take] = True
        if pieces:
            intervals = pd.concat(pieces, ignore_index=True)
            fracs[take, :] = bin_feature_fractions(sub, intervals, features)

    frame = _assemble_annotations(rows, fracs, annotated, features, cfg)
    _write_annotations(frame, path)
    logger.info(
        "annotate: %d bins, %d annotated, %d with a dominant feature",
        len(frame),
        int(frame["annotated"].sum()),
        int((frame["dominant_feature"] != "unannotated").sum()),
    )
    return frame


def _assemble_annotations(
    rows: pd.DataFrame,
    fracs: np.ndarray,
    annotated: np.ndarray,
    features: Sequence[str],
    cfg: Config,
) -> pd.DataFrame:
    """Build the contract frame: ids, dominant feature, then one column per feature."""
    if fracs.size:
        best = np.argmax(fracs, axis=1)  # first max wins -> deterministic given FEATURE_VOCAB
        best_frac = fracs[np.arange(fracs.shape[0]), best]
    else:
        best = np.zeros(len(rows), dtype=np.int64)
        best_frac = np.zeros(len(rows), dtype=np.float32)
    # A bin whose best feature covers less than the threshold is called
    # "unannotated" but keeps its (small) dominant_frac, so the number stays
    # auditable instead of being silently zeroed.
    threshold = max(float(cfg.annotate.min_frac_for_dominant), np.nextafter(0.0, 1.0))
    names = np.asarray(list(features), dtype=object)
    dominant = np.where(best_frac >= threshold, names[best], "unannotated")

    data: dict[str, Any] = {
        "bin_uid": rows["bin_uid"].astype("string").to_numpy(),
        "dominant_feature": dominant,
        "dominant_frac": best_frac.astype(np.float32),
        "annotated": annotated,
    }
    for i, feature in enumerate(features):
        data[schemas.feature_column(feature)] = fracs[:, i]
    frame = pd.DataFrame(data, index=pd.RangeIndex(len(rows)))
    frame = schemas.enforce(frame, schemas.ANNOTATION_ID_COLUMNS, subset=True)
    for col in schemas.FEATURE_COLUMNS:
        frame[col] = frame[col].astype("float32")
    return frame


def _write_annotations(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_annotations(outdir: Path) -> pd.DataFrame:
    """Read back the frame written by :func:`annotate_bins`."""
    path = Path(outdir) / "annotations.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no annotations at {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        return _empty_annotations()
    for col in schemas.FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.float32(0.0)
        frame[col] = frame[col].astype("float32")
    return schemas.enforce(frame, schemas.ANNOTATION_ID_COLUMNS, subset=True)
