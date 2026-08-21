"""Batteries-included test data: a real, tiny slice of the real pangenome.

Developing against synthetic FASTAs is comfortable and misleading.  Real
assemblies have PanSN contig names, soft-masked satellite arrays, cenSat BEDs
with a ``track`` line and ``hor(S1C1H1L)``-style names, and RepeatMasker files
too large to download.  This module pulls a few megabases of *actual* HPRC and
CHM13 sequence -- plus the annotation lines covering it -- into a directory that
the rest of the pipeline can consume exactly like the full dataset.

Two ideas make it cheap:

* **Remote region access.**  Every release-2 assembly publishes ``.fa.gz.fai``
  and ``.fa.gz.gzi`` next to the FASTA, so htslib can fetch one interval over
  HTTPS.  A 6 Mb slice costs a couple of megabytes, not 900.
* **Remote BED slicing.**  The annotation BEDs are up to 425 MB, but they are
  grouped by contig and sorted by start within a contig.  A handful of HTTP
  range requests locate the contig's byte block by bisection, and only the
  interesting part is transferred.  See :func:`slice_remote_bed`.

The sliced FASTA keeps the **original contig name** -- not the
``name:start-end`` that ``samtools faidx`` would emit -- because every
downstream stage joins on the contig name via chromAlias.  The lost provenance
(which slice of which assembly) is written to a ``*.slice.json`` sidecar, and
the annotation coordinates are rewritten into the slice's frame so that a bin
at offset 0 in the local FASTA really is annotated by line 1 of the local BED.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from . import schemas
from .catalog import hprc, t2t
from .catalog import manifest as manifest_mod
from .config import Config
from .log import get_logger, timed
from .preflight import check_remote_access, require_remote_access

__all__ = [
    "fetch_testdata",
    "fetch_reference_tracks",
    "slice_remote_bed",
    "read_chrom_alias",
]

log = get_logger(__name__)

#: Track name -> manifest column, for the three per-assembly BEDs.
_TRACK_COLUMN: dict[str, str] = {
    "censat": "censat_bed",
    "repeatmasker": "repeatmasker_bed",
    "segdup": "segdup_bed",
}

#: Line prefixes that are never BED records.
_BED_HEADERS: tuple[str, ...] = ("#", "track ", "track\t", "browser ")

#: FASTA line width of the written slices.
_FASTA_WIDTH = 60

#: Upper bound on how densely a remote BED is probed when locating a contig.
_MAX_PROBES = 512

#: Per-thread HTTP session holder (see :func:`_session`).
_LOCAL = threading.local()


# --------------------------------------------------------------------------
# chromAlias
# --------------------------------------------------------------------------


def read_chrom_alias(path_or_url: str, cache_dir: Path, *, force: bool = False) -> dict[str, str]:
    """``{contig_name: ucsc_chrom}`` from an HPRC ``*.chromAlias.txt``.

    The file is a TSV whose header line starts with ``# assembly``; column 0 is
    the contig as it appears in the FASTA and column 1 the UCSC name.  Missing
    or unreadable files give an empty mapping rather than an exception -- an
    assembly without a chromAlias is usable, just unplaced.
    """
    if not path_or_url:
        return {}
    try:
        if path_or_url.startswith(("http://", "https://")):
            local = hprc.cache_path(path_or_url, Path(cache_dir))
            hprc.download_file(path_or_url, local, force=force)
        else:
            local = Path(path_or_url)
        text = local.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - a missing alias is not fatal
        log.warning("cannot read chromAlias %s: %s", path_or_url, exc)
        return {}
    alias: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2 or not fields[0] or not fields[1]:
            continue
        alias.setdefault(fields[0].strip(), fields[1].strip())
    return alias


def _contig_for_chrom(alias: dict[str, str], chrom: str) -> str:
    """The (single) contig carrying ``chrom``; ``""`` when absent or ambiguous."""
    hits = sorted(contig for contig, name in alias.items() if name == chrom)
    if not hits:
        return ""
    if len(hits) > 1:
        log.debug("%d contigs claim %s; taking %s", len(hits), chrom, hits[0])
    return hits[0]


# --------------------------------------------------------------------------
# remote BED slicing
# --------------------------------------------------------------------------


def _is_header(line: str) -> bool:
    return not line.strip() or line.startswith(_BED_HEADERS)


def _parse_bed_line(line: str) -> list[str] | None:
    """Split a BED line, or ``None`` when it is a header or malformed."""
    if _is_header(line):
        return None
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 3:
        return None
    try:
        int(fields[1])
        int(fields[2])
    except ValueError:
        return None
    return fields


def _filter_bed(lines: Iterable[str], contig: str, start: int, end: int) -> list[list[str]]:
    """Keep records on ``contig`` overlapping ``[start, end)``, in file order."""
    out: list[list[str]] = []
    for line in lines:
        fields = _parse_bed_line(line)
        if fields is None or fields[0] != contig:
            continue
        lo, hi = int(fields[1]), int(fields[2])
        if hi > start and lo < end:
            out.append(fields)
    return out


def _session() -> Any:
    """A per-thread :class:`requests.Session`.

    Locating a contig block takes a hundred small range requests; without
    connection reuse each one pays a fresh TLS handshake and the slice takes
    half a minute instead of a couple of seconds.  Sessions are not documented
    as thread-safe, hence one per thread.
    """
    import requests

    session = getattr(_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        _LOCAL.session = session
    return session


def _range_get(url: str, first: int, last: int, *, timeout: float = 120.0) -> bytes:
    """HTTP byte range ``[first, last]`` (inclusive, as HTTP counts)."""
    response = _session().get(
        url, headers={"Range": f"bytes={first}-{last}"}, timeout=(10.0, timeout)
    )
    response.raise_for_status()
    return response.content


def _line_at(url: str, offset: int, window: int) -> tuple[int, list[str]] | None:
    """First complete non-header record at or after ``offset``.

    Returns ``(byte_offset_of_that_line, fields)``.  The window grows if the
    file has very long lines (the segdup BEDs have 45 columns).
    """
    for attempt in range(4):
        size = window << attempt
        try:
            buf = _range_get(url, offset, offset + size - 1)
        except Exception as exc:  # noqa: BLE001
            log.debug("range GET failed at %d: %s", offset, exc)
            return None
        if not buf:
            return None
        # A read that does not start at byte 0 almost certainly lands mid-line;
        # discard the partial head.
        cursor = 0 if offset == 0 else buf.find(b"\n") + 1
        if cursor <= 0 and offset != 0:
            continue
        while True:
            nl = buf.find(b"\n", cursor)
            if nl < 0:
                break
            line = buf[cursor:nl].decode("utf-8", "replace")
            fields = _parse_bed_line(line)
            if fields is not None:
                return offset + cursor, fields
            cursor = nl + 1
        # No usable line inside the window -- try a bigger one.
    return None


def _stream_lines(url: str, offset: int, *, timeout: float = 300.0) -> Iterator[str]:
    """Yield whole lines starting at ``offset`` (which must be a line start)."""
    with _session().get(
        url, headers={"Range": f"bytes={offset}-"}, stream=True, timeout=(10.0, timeout)
    ) as response:
        response.raise_for_status()
        for raw in response.iter_lines(chunk_size=1 << 20, decode_unicode=False):
            yield raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def _probe_layout(
    url: str, size: int, probes: int, window: int, seen: dict[int, str]
) -> list[tuple[int, str]]:
    """Sample the file to learn which contig lives at which byte offset.

    ``seen`` memoises probes across refinement rounds so that doubling the
    probe density costs only the new offsets.
    """
    for i in range(probes):
        offset = (size * i) // probes
        if offset in seen:
            continue
        hit = _line_at(url, offset, window)
        seen[offset] = hit[1][0] if hit is not None else ""
    return sorted((offset, name) for offset, name in seen.items() if name)


def _line_before(url: str, offset: int, window: int) -> list[str] | None:
    """Last complete record ending at ``offset``, or ``None`` at the file head.

    Used to confirm that a candidate block start really is the *first* record
    of its contig: if the record just before it has the same contig name, the
    bisection landed inside the block and the answer cannot be trusted.
    """
    if offset <= 0:
        return None
    for attempt in range(4):
        size = window << attempt
        first = max(0, offset - size)
        try:
            buf = _range_get(url, first, offset - 1)
        except Exception as exc:  # noqa: BLE001
            log.debug("backwards range GET failed at %d: %s", offset, exc)
            return None
        if not buf:
            return None
        # ``offset`` is a line start, so the buffer ends with a newline and the
        # last split element is empty; the first is a partial line unless the
        # read began at byte 0.
        candidates = buf.split(b"\n")[:-1]
        if first > 0:
            candidates = candidates[1:]
        for raw in reversed(candidates):
            fields = _parse_bed_line(raw.decode("utf-8", "replace"))
            if fields is not None:
                return fields
        if first == 0:
            return None
    return None


def _bisect_sorted_block(url: str, size: int, contig: str, window: int) -> int | None:
    """Block start for a file sorted lexicographically by contig name.

    Most of these BEDs are ``sort -k1,1 -k2,2n`` output, which makes the whole
    file bisectable in ~25 range requests instead of a hundred probes.  The
    answer is verified against the preceding record before it is trusted.
    """
    lo, hi = 0, size
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        hit = _line_at(url, mid, window)
        if hit is None:
            lo = mid + 1
            continue
        line_start, fields = hit
        if fields[0] >= contig:
            best = line_start
            hi = min(mid, line_start) - 1
        else:
            lo = max(mid, line_start) + 1
    if best is None:
        return None
    hit = _line_at(url, best, window)
    if hit is None or hit[1][0] != contig:
        return None
    previous = _line_before(url, best, window)
    if previous is not None and previous[0] == contig:
        log.debug("sorted-bisect landed inside the %s block; falling back to probing", contig)
        return None
    return best


def _bisect_block_start(url: str, lo: int, hi: int, contig: str, window: int) -> int | None:
    """Smallest offset in ``[lo, hi]`` whose next record is on ``contig``.

    Valid because a contig's records form one contiguous byte block: inside the
    window the predicate "this line belongs to ``contig``" is False then True.
    """
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        hit = _line_at(url, mid, window)
        if hit is None:
            lo = mid + 1
            continue
        line_start, fields = hit
        if fields[0] == contig:
            best = line_start
            hi = min(mid, line_start) - 1
        else:
            lo = max(mid, line_start) + 1
    return best


def _bisect_position(
    url: str, lo: int, hi: int, contig: str, target: int, window: int
) -> int | None:
    """Offset of the first record on ``contig`` with ``start >= target``."""
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        hit = _line_at(url, mid, window)
        if hit is None:
            lo = mid + 1
            continue
        line_start, fields = hit
        if fields[0] != contig or int(fields[1]) >= target:
            best = line_start
            hi = min(mid, line_start) - 1
        else:
            lo = max(mid, line_start) + 1
    return best


def slice_remote_bed(
    url: str,
    contig: str,
    start: int,
    end: int,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    full_download_bytes: int = 8 << 20,
    block_stream_bytes: int = 4 << 20,
    probes: int = 64,
    probe_window: int = 8192,
    back_bytes: int = 1 << 20,
) -> list[list[str]]:
    """Records of ``url`` on ``contig`` overlapping ``[start, end)``.

    Small files (or ones already cached) are simply read.  Large ones are
    sliced with HTTP range requests: probe the file to find the contig's byte
    block, bisect to the block start, and -- if the block is itself large --
    bisect again on the start coordinate before streaming forward.

    The coordinate bisection rewinds ``back_bytes`` before the hit so that
    records *starting* before the region but reaching into it are still seen.
    A record longer than that rewind covers (about a megabyte of BED text, i.e.
    thousands of features) would be missed; the tracks where that could matter
    -- cenSat, telomere -- are small enough to be read in full anyway.

    Returns ``[]`` (with a warning) for anything it cannot slice.  A missing
    annotation must degrade the test data, not abort the download.
    """
    if not url or not contig or end <= start:
        return []
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    local = hprc.cache_path(url, cache_dir) if cache_dir is not None else None
    if local is not None and not force and local.is_file() and local.stat().st_size > 0:
        with open(local, encoding="utf-8", errors="replace") as handle:
            return _filter_bed(handle, contig, start, end)

    size = hprc.remote_size(url)
    if size < 0 or size <= full_download_bytes:
        if local is None:
            log.debug("streaming whole %s (no cache dir)", url)
            try:
                return _filter_bed(_stream_lines(url, 0), contig, start, end)
            except Exception as exc:  # noqa: BLE001
                log.warning("cannot read %s: %s", url, exc)
                return []
        try:
            hprc.download_file(url, local, force=force)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot download %s: %s", url, exc)
            return []
        with open(local, encoding="utf-8", errors="replace") as handle:
            return _filter_bed(handle, contig, start, end)

    try:
        return _slice_by_range(
            url,
            contig,
            start,
            end,
            size=size,
            probes=probes,
            probe_window=probe_window,
            block_stream_bytes=block_stream_bytes,
            back_bytes=back_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - never let a track kill the fetch
        log.warning("range-slicing %s failed (%s); track skipped", url, exc)
        return []


def _locate_block(
    url: str, size: int, contig: str, *, probes: int, window: int
) -> tuple[int, int] | None:
    """Byte range ``[block_start, bound)`` that certainly contains ``contig``.

    Two strategies.  If a coarse probe pass sees contig names in
    non-decreasing order the file is sorted and one bisection finds the block
    directly.  Otherwise -- the HPRC segdup BEDs group by contig but in
    arbitrary order -- the probes are refined until they land inside the block,
    which brackets it for a local bisection.  ``bound`` is only an upper limit:
    the caller stops as soon as the contig changes.
    """
    seen: dict[int, str] = {}
    layout = _probe_layout(url, size, max(8, min(probes, 16)), window, seen)
    if not layout:
        log.warning("could not probe %s; track skipped", url)
        return None
    names = [name for _, name in layout]
    if all(a <= b for a, b in zip(names, names[1:])):
        block_start = _bisect_sorted_block(url, size, contig, window)
        if block_start is not None:
            return block_start, size
        if contig < names[0] or contig > names[-1]:
            log.warning("contig %s is outside %s; track skipped", contig, os.path.basename(url))
            return None

    # Refine the probes until one lands inside the block.  Doubling twice
    # resolves blocks down to ~1/512 of the file, i.e. under a megabyte for a
    # 400 MB BED -- smaller than any real chromosome's annotation.
    hits: list[int] = []
    count = max(8, probes)
    while True:
        layout = _probe_layout(url, size, count, window, seen)
        hits = [i for i, (_, name) in enumerate(layout) if name == contig]
        if hits or count >= _MAX_PROBES:
            break
        count *= 4
        log.debug("contig %s not seen yet; refining to %d probes", contig, count)
    if not hits:
        log.warning(
            "contig %s not found in %d probes of %s; track skipped",
            contig,
            len(layout),
            os.path.basename(url),
        )
        return None

    lower = layout[hits[0] - 1][0] if hits[0] > 0 else 0
    block_start = _bisect_block_start(url, lower, layout[hits[0]][0], contig, window)
    if block_start is None:
        log.warning("could not locate the %s block in %s; track skipped", contig, url)
        return None
    last = hits[-1]
    return block_start, (layout[last + 1][0] if last + 1 < len(layout) else size)


def _slice_by_range(
    url: str,
    contig: str,
    start: int,
    end: int,
    *,
    size: int,
    probes: int,
    probe_window: int,
    block_stream_bytes: int,
    back_bytes: int,
) -> list[list[str]]:
    located = _locate_block(url, size, contig, probes=probes, window=probe_window)
    if located is None:
        return []
    block_start, block_bound = located

    scan_from = block_start
    if block_bound - block_start > block_stream_bytes:
        hit = _bisect_position(url, block_start, block_bound, contig, start, probe_window)
        if hit is not None:
            scan_from = max(block_start, hit - back_bytes)
            if scan_from > block_start:
                snapped = _line_at(url, scan_from, probe_window)
                scan_from = snapped[0] if snapped is not None else block_start

    records: list[list[str]] = []
    entered = False
    for line in _stream_lines(url, scan_from):
        fields = _parse_bed_line(line)
        if fields is None:
            continue
        if fields[0] != contig:
            if entered:
                break
            continue
        entered = True
        lo_pos, hi_pos = int(fields[1]), int(fields[2])
        if lo_pos >= end:
            break
        if hi_pos > start:
            records.append(fields)
    log.debug("sliced %d record(s) of %s from %s", len(records), contig, os.path.basename(url))
    return records


# --------------------------------------------------------------------------
# writing the slice
# --------------------------------------------------------------------------


def _rebase_records(records: Sequence[Sequence[str]], start: int, end: int) -> list[list[str]]:
    """Clip to ``[start, end)`` and shift into the slice's coordinate frame."""
    out: list[list[str]] = []
    for fields in records:
        lo = max(int(fields[1]), start) - start
        hi = min(int(fields[2]), end) - start
        if hi <= lo:
            continue
        row = list(fields)
        row[1] = str(lo)
        row[2] = str(hi)
        # BED9 thickStart/thickEnd must stay inside the feature.
        if len(row) >= 8:
            for i in (6, 7):
                try:
                    value = int(row[i])
                except ValueError:
                    continue
                row[i] = str(min(max(value - start, lo), hi))
        out.append(row)
    return out


def _write_bed(path: Path, records: Sequence[Sequence[str]]) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        for fields in records:
            handle.write("\t".join(fields) + "\n")
    os.replace(tmp, path)
    return path


def _write_bgzip_fasta(path: Path, name: str, sequence: str) -> Path:
    """Write ``sequence`` as a bgzip FASTA named ``name``, with .fai and .gzi."""
    import pysam

    plain = path.with_name(path.name + ".plain.tmp")
    with open(plain, "w", encoding="ascii", newline="\n") as handle:
        handle.write(f">{name}\n")
        for i in range(0, len(sequence), _FASTA_WIDTH):
            handle.write(sequence[i : i + _FASTA_WIDTH] + "\n")
    try:
        pysam.tabix_compress(str(plain), str(path), force=True)
    finally:
        plain.unlink(missing_ok=True)
    for stale in (path.with_name(path.name + ".fai"), path.with_name(path.name + ".gzi")):
        stale.unlink(missing_ok=True)
    pysam.faidx(str(path))
    return path


def _write_chrom_alias(path: Path, contig: str, chrom: str) -> Path:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"# assembly\tucsc\n{contig}\t{chrom}\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(dest: Path, name: str = "checksums.sha256") -> Path:
    path = dest / name
    entries = sorted(
        p for p in dest.rglob("*") if p.is_file() and p.name != name and not p.name.endswith(".tmp")
    )
    lines = [f"{_sha256(p)}  {p.relative_to(dest).as_posix()}\n" for p in entries]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# one assembly
# --------------------------------------------------------------------------


def _fetch_sequence(fasta_url: str, contig: str, start: int, end: int) -> tuple[str, int]:
    """``(sequence, contig_length)`` for ``[start, end)``, clipped to the contig."""
    import pysam

    from .preflight import capture_c_stderr, describe_open_failure

    # htslib writes its real diagnostic to fd 2 from C and pysam raises an
    # OSError naming only the URL, so the reason is lost unless we catch it here.
    with capture_c_stderr() as captured:
        try:
            handle = pysam.FastaFile(fasta_url)
        except Exception as exc:
            raise OSError(describe_open_failure(fasta_url, exc, captured)) from exc
    try:
        try:
            length = int(handle.get_reference_length(contig))
        except (KeyError, ValueError) as exc:
            raise KeyError(f"contig {contig!r} not in {fasta_url}") from exc
        stop = min(end, length)
        if start >= stop:
            raise ValueError(f"contig {contig!r} is only {length} bp; slice {start}-{end} is empty")
        return handle.fetch(contig, start, stop), length
    finally:
        handle.close()


def _slice_assembly(
    row: pd.Series,
    dest: Path,
    cache_dir: Path,
    *,
    chrom: str,
    start: int,
    end: int,
    tracks: Sequence[str],
    force: bool,
    failures: list[str] | None = None,
) -> dict[str, Any] | None:
    """Slice one assembly; returns its sidecar dict, or ``None`` on failure.

    Every ``None`` also appends a one-line reason to ``failures`` when given, so
    the caller can summarise *why* a run produced nothing instead of leaving the
    explanation hundreds of lines up the log.
    """
    assembly = str(row["assembly"])
    sidecar_path = dest / f"{assembly}.slice.json"
    fasta_path = dest / f"{assembly}.fa.gz"

    if not force and sidecar_path.is_file() and fasta_path.is_file():
        try:
            cached = json.loads(sidecar_path.read_text(encoding="utf-8"))
            log.info("%s already sliced; keeping it", assembly)
            return cached
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("unreadable sidecar %s (%s); re-slicing", sidecar_path, exc)

    source = str(row["source"])
    if source == "t2t":
        contig = chrom
    else:
        alias = read_chrom_alias(str(row["chrom_alias"]), cache_dir, force=False)
        contig = _contig_for_chrom(alias, chrom)
        if not contig:
            log.warning("%s: chromAlias has no contig for %s; skipping", assembly, chrom)
            if failures is not None:
                failures.append(f"no-contig-for-{chrom}: {assembly}")
            return None

    try:
        sequence, contig_length = _fetch_sequence(str(row["fasta"]), contig, start, end)
    except Exception as exc:  # noqa: BLE001 - a bad assembly must not stop the fetch
        log.warning("%s: cannot fetch %s:%d-%d (%s); skipping", assembly, contig, start, end, exc)
        if failures is not None:
            # The bucket has to distinguish "could not reach the file" from
            # "reached it, the request was wrong", because only the first one
            # means anything about the environment. _fetch_sequence raises
            # KeyError for a missing contig and ValueError for an out-of-range
            # slice; anything else got as far as htslib opening the URL.
            kind = {
                KeyError: "contig-not-in-fasta",
                ValueError: "slice-out-of-range",
            }.get(type(exc), "remote-open-failed")
            failures.append(f"{kind}: {assembly} ({exc})")
        return None
    stop = start + len(sequence)

    _write_bgzip_fasta(fasta_path, contig, sequence)
    alias_path = _write_chrom_alias(dest / f"{assembly}.chromAlias.txt", contig, chrom)

    written: dict[str, str] = {}
    for track in tracks:
        url = _track_url(row, track)
        if not url:
            continue
        records = slice_remote_bed(url, contig, start, stop, cache_dir=cache_dir)
        out = _write_bed(dest / f"{assembly}.{track}.bed", _rebase_records(records, start, stop))
        written[track] = out.name
        log.info("%s: %d %s record(s)", assembly, len(records), track)

    sidecar = {
        "assembly": assembly,
        "sample": str(row["sample"]),
        "haplotype": str(row["haplotype"]),
        "source": source,
        "population": str(row["population"]),
        "superpopulation": str(row["superpopulation"]),
        "sex": str(row["sex"]),
        "chrom": chrom,
        "contig": contig,
        "contig_length": int(contig_length),
        "offset": int(start),
        "slice_start": int(start),
        "slice_end": int(stop),
        "length": int(len(sequence)),
        "source_fasta": str(row["fasta"]),
        "fasta": fasta_path.name,
        "chrom_alias": alias_path.name,
        "tracks": written,
        "note": (
            "Coordinates in this directory are slice-local: local position 0 is "
            f"{start} on {contig} of the source assembly."
        ),
    }
    tmp = sidecar_path.with_name(sidecar_path.name + ".tmp")
    tmp.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, sidecar_path)
    return sidecar


def _track_url(row: pd.Series, track: str) -> str:
    """Where a track lives for this row: the manifest column, else the T2T default."""
    column = _TRACK_COLUMN.get(track)
    if column is not None:
        url = str(row.get(column, "") or "")
        if url:
            return url
    if str(row.get("source", "")) == "t2t":
        return t2t.T2T_TRACKS.get(track, "")
    return ""


def _manifest_row(sidecar: dict[str, Any], dest: Path) -> dict[str, str]:
    """Manifest row pointing at the local slice."""
    fasta = dest / str(sidecar["fasta"])
    tracks = dict(sidecar.get("tracks") or {})
    return {
        "assembly": str(sidecar["assembly"]),
        "sample": str(sidecar["sample"]),
        "haplotype": str(sidecar["haplotype"]),
        "source": str(sidecar["source"]),
        "fasta": str(fasta),
        "fai": str(fasta.with_name(fasta.name + ".fai")),
        "gzi": str(fasta.with_name(fasta.name + ".gzi")),
        "chrom_alias": str(dest / str(sidecar["chrom_alias"])),
        "censat_bed": str(dest / tracks["censat"]) if "censat" in tracks else "",
        "repeatmasker_bed": str(dest / tracks["repeatmasker"]) if "repeatmasker" in tracks else "",
        "segdup_bed": str(dest / tracks["segdup"]) if "segdup" in tracks else "",
        "population": str(sidecar.get("population", "")),
        "superpopulation": str(sidecar.get("superpopulation", "")),
        "sex": str(sidecar.get("sex", "")),
    }


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------


def fetch_reference_tracks(
    cache_dir: Path, tracks: Sequence[str] | None = None, *, force: bool = False
) -> dict[str, Path]:
    """Download whole T2T annotation tracks into the cache.

    Only worth doing for a full run: RepeatMasker alone is ~340 MB and the
    GENCODE GFF another ~150 MB.  :func:`fetch_testdata` does not call this --
    it slices instead -- but a cluster job that will annotate every reference
    bin wants the files local and warm.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(tracks) if tracks is not None else list(t2t.T2T_TRACKS)
    out: dict[str, Path] = {}
    for name in wanted:
        url = t2t.T2T_TRACKS.get(name, "")
        if not url:
            log.warning("unknown reference track %r; known: %s", name, sorted(t2t.T2T_TRACKS))
            continue
        expected = t2t.T2T_TRACK_BYTES.get(name, 0)
        with timed(log, f"fetching {name} (~{expected / 1e6:.0f} MB)"):
            try:
                out[name] = hprc.download_file(url, hprc.cache_path(url, cache_dir), force=force)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not fetch reference track %r: %s", name, exc)
    return out


def _candidate_manifest(cache_dir: Path, chrom: str, seed: int) -> pd.DataFrame:
    """Every release-2 assembly that could supply ``chrom``, cheaply filtered."""
    cfg = Config.from_dict(
        {
            "seed": seed,
            "manifest": {
                "source": "hprc_release2",
                "chroms": [chrom],
                "require_annotations": ["censat", "chrom_alias"],
                "include_reference": False,
                "seed": seed,
            },
        }
    )
    return manifest_mod.build_manifest(cfg, cache_dir)


def fetch_testdata(
    dest: Path,
    cache_dir: Path,
    *,
    samples: int = 4,
    chrom: str = "chr21",
    span_mb: float = 6.0,
    offset_mb: float = 6.0,
    force: bool = False,
    include_reference: bool = True,
) -> Path:
    """Build a self-contained test dataset under ``dest``; returns its manifest path.

    ``samples`` HPRC *samples* are chosen deterministically (both haplotypes of
    each), plus CHM13v2.0 when ``include_reference``.  For every assembly the
    ``[offset_mb, offset_mb + span_mb)`` window of ``chrom`` is written as a
    bgzip FASTA with ``.fai``/``.gzi``, together with the cenSat, RepeatMasker
    and segdup lines covering it, a one-line chromAlias, and a ``*.slice.json``
    sidecar recording the offset.  Finally a manifest TSV and a SHA-256
    checksum file are written.

    Defaults download roughly 40-60 MB.  Assemblies whose ``chrom`` is missing
    or too short are skipped and replaced by the next candidate, so the result
    has ``samples`` samples whenever the catalog can supply them.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if samples < 0:
        raise ValueError("samples must be >= 0")
    if span_mb <= 0:
        raise ValueError("span_mb must be > 0")
    start = int(round(offset_mb * 1e6))
    end = start + int(round(span_mb * 1e6))
    if start < 0:
        raise ValueError("offset_mb must be >= 0")
    # Fails instantly and says exactly why, instead of letting every remote open
    # die one at a time with a message that names only the URL.
    require_remote_access()

    failures: list[str] = []
    log.info("test data: %d sample(s) x %s:%d-%d -> %s", samples, chrom, start, end, dest)

    sidecars: list[dict[str, Any]] = []
    pool_size = 0
    if samples > 0:
        pool = _candidate_manifest(cache_dir, chrom, seed=7)
        pool_size = int(len(pool))
        order = manifest_mod.ordered_samples(pool, 7) if len(pool) else []
        by_sample = {
            str(name): rows.sort_values(["haplotype", "assembly"], kind="stable")
            for name, rows in pool.groupby("sample", sort=False)
        }
        accepted = 0
        # Walk far more candidates than requested.  Whole-chromosome contigs are
        # rarer than one would hope -- only ~14% of release-2 haplotypes carry a
        # single ``chr21`` contig, the acrocentrics being the hardest to
        # assemble -- so a skipped sample must be replaced, not lost.  Rejecting
        # a sample costs two cached 8 KB chromAlias reads.
        for sample in order[: samples * 10 + 20]:
            if accepted >= samples:
                break
            rows = by_sample.get(sample)
            if rows is None:
                continue
            produced: list[dict[str, Any]] = []
            with timed(log, f"slicing {sample}"):
                for _, row in rows.iterrows():
                    sidecar = _slice_assembly(
                        row,
                        dest,
                        cache_dir,
                        chrom=chrom,
                        start=start,
                        end=end,
                        tracks=("censat", "repeatmasker", "segdup"),
                        force=force,
                        failures=failures,
                    )
                    if sidecar is not None:
                        produced.append(sidecar)
            if not produced:
                log.info("%s: no usable haplotype for %s; trying the next sample", sample, chrom)
                continue
            sidecars.extend(produced)
            accepted += 1
        if accepted < samples:
            log.warning("only %d of %d requested samples could be sliced", accepted, samples)

    if include_reference:
        reference = pd.Series(t2t.reference_manifest_row())
        with timed(log, "slicing chm13v2.0"):
            sidecar = _slice_assembly(
                reference,
                dest,
                cache_dir,
                chrom=chrom,
                start=start,
                end=end,
                tracks=("censat", "repeatmasker", "segdup", "telomere"),
                force=force,
                failures=failures,
            )
        if sidecar is None:
            log.warning("could not slice the CHM13 reference; test data has no reference row")
        else:
            sidecars.insert(0, sidecar)

    rows = [_manifest_row(sidecar, dest) for sidecar in sidecars]
    frame = (
        schemas.enforce(pd.DataFrame(rows), schemas.MANIFEST_COLUMNS)
        if rows
        else schemas.empty_frame(schemas.MANIFEST_COLUMNS)
    )
    path = manifest_mod.write_manifest(frame, dest / "manifest.tsv")
    _write_checksums(dest)
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    log.info("test data ready: %d assemblies, %.1f MB in %s", len(frame), total / 1e6, dest)
    if len(frame) == 0:
        _explain_empty_fetch(failures, pool_size, chrom)
    return path


def _explain_empty_fetch(failures: Sequence[str], pool_size: int, chrom: str) -> None:
    """Say, in the last lines of the log, why a fetch produced nothing.

    Without this the transcript ends on three lines that between them report a
    duration, a symptom and a count -- and the causal records are a few hundred
    lines further up, one per rejected haplotype, all nearly identical.
    """
    log.error("fetch produced NO assemblies. Funnel:")
    log.error("  candidate assemblies in the catalog : %d", pool_size)
    log.error("  slice attempts that failed          : %d", len(failures))
    if not failures:
        log.error(
            "  no assembly was even attempted -- the candidate pool for %s was empty, so "
            "the HPRC catalog itself came back with nothing usable (check the warnings "
            "above from kmer_dust.catalog).",
            chrom,
        )
    else:
        buckets: dict[str, list[str]] = {}
        for reason in failures:
            kind, _, detail = reason.partition(": ")
            buckets.setdefault(kind, []).append(detail)
        for kind, details in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            log.error("  %-22s x%-4d e.g. %s", kind, len(details), details[0])
    ok, explanation = check_remote_access()
    log.error("  htslib: %s", explanation)
    # Only draw the network conclusion when the failures actually *are* remote
    # opens. A wrong diagnosis in an error message costs more than none at all:
    # "check your egress" sends the reader down the wrong path when the real
    # answer was an out-of-range slice or a missing chromosome.
    remote_failures = sum(1 for f in failures if f.startswith("remote-open-failed"))
    if ok and remote_failures:
        log.error(
            "  htslib CAN do https, so the %d failed remote open(s) point at the network "
            "or the endpoint rather than a broken install: check egress to "
            "s3-us-west-2.amazonaws.com from this host.",
            remote_failures,
        )
