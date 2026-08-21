"""One uniform way to read sequence, wherever the FASTA actually lives.

The manifest mixes four kinds of input and the rest of the pipeline must not
care which it got:

===========================  =========================================
plain local FASTA            ``pysam`` builds/loads a ``.fai`` next to it
bgzip local FASTA            ``pysam`` with ``.fai`` + ``.gzi``
``https://`` plain FASTA     ``pysam``/htslib range-reads with a remote ``.fai``
``https://`` bgzip FASTA     ``pysam``/htslib with remote ``.fai`` + ``.gzi``
===========================  =========================================

The indexed path matters enormously for HPRC release 2: the assemblies live in
S3 as ~1 GB bgzipped FASTAs and htslib can pull a single 10 Mb chromosome block
out of one in a couple of seconds -- but *only* because ``.fa.gz.fai`` and
``.fa.gz.gzi`` are published alongside.  Without an index htslib does not fail,
it silently tries to *build* one, which means downloading the entire file.  That
is why :class:`FastaSource` probes for the index with a cheap ``HEAD`` before it
ever hands a URL to htslib, and drops to a streaming parser when the probe says
"no index" rather than letting htslib decide.

The streaming fallback is deliberately dumb (sequential, no random access, no
contig lengths known up front) because it exists only for local scratch FASTAs
and for remote files somebody forgot to index.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import IO

from .log import get_logger

logger = get_logger(__name__)

__all__ = [
    "FastaSource",
    "load_chrom_alias",
    "normalize_chrom",
    "open_text",
    "download",
    "DEFAULT_BLOCK",
]

#: Default streaming block size.  8 Mb of sequence is small enough to keep a
#: 250 Mb chromosome out of memory and large enough that per-block NumPy
#: overhead and per-request htslib latency both disappear into the noise.
DEFAULT_BLOCK: int = 8_000_000

#: (connect, read) timeouts for every HTTP call we make.
HTTP_TIMEOUT: tuple[float, float] = (15.0, 120.0)
#: How many times a transient HTTP failure is retried before giving up.
HTTP_RETRIES: int = 3
#: Status codes worth retrying; anything else is a real answer.
HTTP_RETRY_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

_GZIP_SUFFIXES = (".gz", ".bgz", ".bgzf")


# --------------------------------------------------------------------------
# tiny URL / path helpers
# --------------------------------------------------------------------------


def _is_url(path_or_url: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", str(path_or_url)))


def _is_http(path_or_url: str) -> bool:
    return str(path_or_url).lower().startswith(("http://", "https://"))


def _looks_gzipped(path_or_url: str) -> bool:
    """Extension test; local files additionally get a magic-number check."""
    name = str(path_or_url).split("?", 1)[0].lower()
    if name.endswith(_GZIP_SUFFIXES):
        return True
    if not _is_url(path_or_url):
        try:
            with open(path_or_url, "rb") as handle:
                return handle.read(2) == b"\x1f\x8b"
        except OSError:
            return False
    return False


def _cache_name(url: str) -> str:
    """Deterministic, collision-resistant cache file name for a URL.

    The digest prefix keeps two different ``chm13v2.0.fa.fai`` URLs apart; the
    readable suffix keeps the cache directory greppable by a human.
    """
    base = url.rsplit("/", 1)[-1].split("?", 1)[0] or "download"
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:96]
    return f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]}_{base}"


def _requests():
    """Import ``requests`` lazily so ``import kmer_dust`` stays cheap."""
    import requests

    return requests


# --------------------------------------------------------------------------
# download / open_text
# --------------------------------------------------------------------------


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Fetch ``url`` to ``dest``, atomically, and return ``dest``.

    Writes to a per-process ``.part`` file and ``os.replace``s it into place, so
    a killed job can never leave a truncated file that a later run would happily
    treat as a complete cache entry.  When the server advertises a
    ``Content-Length`` (and is not transfer-encoding the body) the byte count is
    checked, because a silently truncated ``.fai`` is far worse than a crash.

    Non-URL ``url`` values are copied from the local filesystem, which lets the
    caller treat manifest columns uniformly without sniffing them first.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        logger.debug("cached %s -> %s", url, dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")

    if not _is_url(url):
        src = Path(url)
        if not src.exists():
            raise FileNotFoundError(f"no such file: {url}")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return dest
    if not _is_http(url):
        raise ValueError(f"download() only speaks http(s) or local paths, got {url!r}")

    requests = _requests()
    last_error: Exception | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as resp:
                if resp.status_code in HTTP_RETRY_STATUS:
                    raise OSError(f"HTTP {resp.status_code} for {url}")
                resp.raise_for_status()
                # Content-Length describes the *encoded* body; iter_content
                # transparently un-gzips a Content-Encoding, so only compare
                # when there is no such encoding in play.
                encoded = "content-encoding" in {k.lower() for k in resp.headers}
                declared = resp.headers.get("Content-Length")
                written = 0
                with open(tmp, "wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
            if declared is not None and not encoded:
                try:
                    expected = int(declared)
                except ValueError:
                    expected = -1
                if expected >= 0 and written != expected:
                    raise OSError(
                        f"short read for {url}: got {written} bytes, expected {expected}"
                    )
            os.replace(tmp, dest)
            logger.debug("downloaded %s (%d bytes) -> %s", url, written, dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised at the end
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt == HTTP_RETRIES or not _retryable(exc):
                break
            logger.warning("download attempt %d/%d failed for %s: %s",
                           attempt, HTTP_RETRIES, url, exc)
    raise OSError(f"failed to download {url}: {last_error}") from last_error


def _retryable(exc: BaseException) -> bool:
    """A 404 will still be a 404 in two seconds; a dropped socket may not be."""
    requests = _requests()
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None and 400 <= status < 500 and status not in HTTP_RETRY_STATUS:
            return False
    return True


def _remote_exists(url: str) -> bool | None:
    """``True``/``False`` if we could tell, ``None`` if the probe itself failed.

    The tri-state matters: "the server says 404" and "the network blipped" call
    for different fallbacks, and conflating them either loses a usable index or
    triggers an accidental whole-genome download.
    """
    if not _is_http(url):
        return None
    requests = _requests()
    for attempt in range(1, 3):
        try:
            resp = requests.head(url, allow_redirects=True, timeout=HTTP_TIMEOUT)
            if resp.status_code == 405:  # HEAD not allowed; ask for one byte instead
                resp = requests.get(
                    url, headers={"Range": "bytes=0-0"}, stream=True, timeout=HTTP_TIMEOUT
                )
                resp.close()
            return resp.status_code < 400
        except Exception as exc:  # noqa: BLE001 - probe failure is information, not an error
            logger.debug("index probe %d/2 failed for %s: %s", attempt, url, exc)
    return None


def _open_binary(path_or_url: str, cache_dir: Path | None = None) -> IO[bytes]:
    """Binary stream over a local or remote file, transparently un-gzipped."""
    if _is_url(path_or_url):
        if not _is_http(path_or_url):
            raise ValueError(f"cannot stream {path_or_url!r}: only http(s) is supported")
        if cache_dir is not None:
            local = download(str(path_or_url), Path(cache_dir) / _cache_name(str(path_or_url)))
            return _open_binary(str(local))
        requests = _requests()
        resp = requests.get(path_or_url, stream=True, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        raw = resp.raw
        raw.decode_content = True
        stream: IO[bytes] = gzip.GzipFile(fileobj=raw) if _looks_gzipped(path_or_url) else raw
        # Keep the Response alive for as long as the stream is: dropping it can
        # return the connection to the pool underneath us mid-read.
        stream._kmer_dust_response = resp  # type: ignore[attr-defined]
        return stream
    if _looks_gzipped(path_or_url):
        return gzip.open(path_or_url, "rb")
    return open(path_or_url, "rb")


def open_text(path_or_url: str, cache_dir: Path | None = None) -> IO[str]:
    """Text handle over a local/remote, plain/gzipped file.

    ``cache_dir`` turns a remote read into a cached local read, which is what
    every catalog and BED consumer wants: those files are small, read repeatedly
    and worth keeping.  Callers that genuinely want a one-shot stream pass
    ``None``.
    """
    return io.TextIOWrapper(
        _open_binary(str(path_or_url), cache_dir), encoding="utf-8", errors="replace"
    )


# --------------------------------------------------------------------------
# chromosome naming
# --------------------------------------------------------------------------

_MITO_NAMES = frozenset({"m", "mt", "mito", "mtdna", "chrmt"})


def normalize_chrom(name: str) -> str:
    """Map a contig name onto ``chr1``..``chr22``/``chrX``/``chrY``/``chrM``.

    Returns ``""`` for everything else -- unplaced contigs, scaffolds and bare
    GenBank accessions.  A PanSN name is unwrapped first, but only its *final*
    field is ever consulted: ``CHM13#0#chrX`` (minigraph-cactus, CHM13-in-a-graph)
    resolves to ``chrX``, while ``HG00408#1#CM085953.1`` still resolves to ``""``.
    That asymmetry is deliberate -- an accession does denote a chromosome, but
    only the assembly's own chromAlias file knows which one, and guessing from
    the accession number would be wrong for every sample but the one it came
    from.
    """
    if not name:
        return ""
    core = str(name).strip()
    if not core:
        return ""
    if "#" in core:
        # PanSN: <sample>#<haplotype>#<contig>.  Only the contig field can carry
        # a chromosome name; if it does not, we fall through to "" as usual.
        core = core.rsplit("#", 1)[-1].strip()
        if not core:
            return ""
    if core.lower().startswith("chr"):
        core = core[3:]
    lowered = core.lower()
    if lowered in _MITO_NAMES:
        return "chrM"
    if lowered in {"x", "y"}:
        return f"chr{lowered.upper()}"
    if core.isdigit():
        number = int(core)
        if 1 <= number <= 22:
            return f"chr{number}"
    return ""


_UCSC_RANDOM = re.compile(r"^(chr(?:\d{1,2}|[XYM]|MT))_[^\s]*_(?:random|alt|fix)$", re.IGNORECASE)
_UCSC_UNKNOWN = re.compile(r"^chrUn[_.]", re.IGNORECASE)


def parse_ucsc_placement(name: str) -> tuple[str, bool]:
    """``(chromosome, is_localised)`` for a UCSC-style sequence name.

    HPRC chromAlias files use three shapes, and collapsing them loses the thing
    this pipeline most wants to look at::

        chr13                              -> ("chr13", True)   placed
        chr13_JBHIKM010000006.1_random     -> ("chr13", False)  chromosome known, position not
        chrUn_JBHIKM010000019.1            -> ("",      False)  chromosome unknown

    The middle case is 34 % of the assembled sequence in a typical release-2
    haplotype -- ~1 Gb per assembly -- and it is not junk.  It is precisely the
    unlocalised, repeat-rich material (satellite arrays, rDNA, the acrocentric
    short arms) that resists placement *because* it is repetitive, which is the
    same reason it is interesting here.  Treating it as unplaced throws away the
    chromosome assignment that the assembler was confident enough to record.

    The second element is what keeps that honest downstream: a ``_random``
    contig's coordinates are contig-local, not chromosome-local, so any analysis
    that reasons about genomic position must restrict itself to localised bins.
    """
    text = str(name or "").strip()
    if not text:
        return "", False
    if _UCSC_UNKNOWN.match(text):
        return "", False
    match = _UCSC_RANDOM.match(text)
    if match:
        return normalize_chrom(match.group(1)), False
    chrom = normalize_chrom(text)
    return chrom, bool(chrom)


def load_chrom_alias(path_or_url: str, cache_dir: Path | None = None) -> dict[str, str]:
    """Read an HPRC ``.chromAlias.txt`` into ``{any alias -> UCSC name}``.

    The file is a TSV whose header line is commented::

        # assembly	ucsc	genbank
        HG00408#1#CM085953.1	chr2	CM085953.1

    Every non-empty cell in a row is registered as a key for that row's ``ucsc``
    value, so the map works whether the FASTA uses PanSN names, bare accessions
    or UCSC names.  Rows with an empty ``ucsc`` cell (unplaced scaffolds) are
    skipped: an empty value would be indistinguishable from "not in the map",
    and both mean "unplaced" downstream anyway.

    Values are returned verbatim rather than passed through
    :func:`normalize_chrom`, so callers can still see e.g. ``chr1_random``.  A
    missing file is an empty map, not an error -- ``chrom_alias`` is optional in
    the manifest and a run without it simply has more unplaced bins.
    """
    alias: dict[str, str] = {}
    if not path_or_url:
        return alias
    try:
        handle = open_text(str(path_or_url), cache_dir)
    except (OSError, ValueError) as exc:
        logger.warning("chrom alias %s unreadable (%s); treating every contig as unplaced",
                       path_or_url, exc)
        return alias

    ucsc_col = 1
    with handle:
        for lineno, raw in enumerate(handle):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if line.startswith("#"):
                fields = [f.strip().lower() for f in line.lstrip("#").strip().split("\t")]
                if "ucsc" in fields:
                    ucsc_col = fields.index("ucsc")
                continue
            fields = line.split("\t")
            if len(fields) <= ucsc_col:
                logger.debug("chrom alias %s line %d has %d fields; skipped",
                             path_or_url, lineno + 1, len(fields))
                continue
            ucsc = fields[ucsc_col].strip()
            if not ucsc:
                continue
            for field in fields:
                key = field.strip()
                if key:
                    alias.setdefault(key, ucsc)
    logger.debug("chrom alias %s: %d keys", path_or_url, len(alias))
    return alias


# --------------------------------------------------------------------------
# streaming FASTA parsing
# --------------------------------------------------------------------------


def _iter_raw_lines(handle: IO[bytes], chunk: int = 1 << 22) -> Iterator[bytes]:
    """Newline-split a binary stream in big chunks.

    ``for line in handle`` is correct but spends all its time in Python on a
    60-column FASTA; chunked splitting is several times faster and the extra
    bookkeeping is four lines.
    """
    tail = b""
    while True:
        data = handle.read(chunk)
        if not data:
            break
        parts = (tail + data).split(b"\n")
        tail = parts.pop()
        yield from parts
    if tail:
        yield tail


def _stream_fasta_blocks(
    handle: IO[bytes], block: int, wanted: frozenset[str] | None
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(contig, block)`` for a sequential FASTA stream.

    Blocks are contiguous and non-overlapping; the last block of a contig is
    whatever is left over.  A contig whose sequence is empty yields nothing at
    all, exactly as the indexed path does for a zero-length reference.
    """
    name = ""
    keep = False
    buf = bytearray()
    remaining = set(wanted) if wanted is not None else None
    for line in _iter_raw_lines(handle):
        if line[:1] == b">":
            if keep and buf:
                yield name, bytes(buf)
            buf.clear()
            header = line[1:].strip()
            name = header.split()[0].decode("utf-8", "replace") if header else ""
            keep = wanted is None or name in wanted
            if remaining is not None:
                remaining.discard(name)
                if keep is False and not remaining:
                    break  # every requested contig has been seen; stop reading
            continue
        if not keep:
            continue
        buf += line.strip()
        while len(buf) >= block:
            yield name, bytes(buf[:block])
            del buf[:block]
    if keep and buf:
        yield name, bytes(buf)


# --------------------------------------------------------------------------
# FastaSource
# --------------------------------------------------------------------------


class FastaSource:
    """Random-access (or, failing that, sequential) reader for one assembly.

    Construction never reads sequence: it only works out *how* the file can be
    read.  The expensive decision -- index or no index -- is made once here so
    that a mistake shows up as a log line at startup instead of a 1 GB download
    in the middle of a Slurm array job.
    """

    def __init__(
        self,
        fasta: str,
        fai: str = "",
        gzi: str = "",
        cache_dir: Path | None = None,
    ):
        self.fasta = str(fasta)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._remote = _is_url(self.fasta)
        self._gzipped = _looks_gzipped(self.fasta)
        self._requested_fai = str(fai or "")
        self._requested_gzi = str(gzi or "")
        self._handle = None  # pysam.FastaFile, opened lazily
        self._lengths: dict[str, int] | None = None
        self._stream: IO[bytes] | None = None
        self._closed = False

        if not self._remote and not Path(self.fasta).exists():
            raise FileNotFoundError(f"FASTA not found: {self.fasta}")

        self._fai, self._gzi = self._resolve_index()
        self.indexed = self._fai is not None

    # -- index resolution ---------------------------------------------------

    def _resolve_index(self) -> tuple[str | None, str | None]:
        """Decide which index files (if any) htslib should be handed.

        Local indexes are settled by ``exists()``.  Remote ones are settled by a
        ``HEAD``, *always* -- including for explicitly supplied URLs -- because
        htslib's response to a missing remote index is to build one by streaming
        the whole FASTA, and no manifest is trustworthy enough to risk that.
        """
        fai = self._requested_fai or f"{self.fasta}.fai"
        gzi = self._requested_gzi or f"{self.fasta}.gzi"
        if self._gzipped and not self._requested_gzi and self.fasta.lower().endswith(".gz"):
            gzi = f"{self.fasta}.gzi"

        def present(path: str) -> bool:
            if not path:
                return False
            if _is_url(path):
                probe = _remote_exists(path)
                if probe is None:
                    # Neither answer is safe to assume: "present" risks htslib
                    # downloading a gigabyte to build an index, "absent" risks
                    # streaming the same gigabyte.  Fail loudly and let the
                    # caller retry -- transient network trouble is retryable,
                    # a wasted whole-genome transfer is not.
                    raise OSError(
                        f"could not determine whether {path} exists; refusing to guess"
                    )
                return probe
            return Path(path).exists()

        have_fai = present(fai)
        if not have_fai:
            if self._remote:
                logger.warning(
                    "no usable .fai for %s; falling back to a full sequential stream",
                    self.fasta,
                )
                return None, None
            # Locally, pysam can build the index itself -- cheap and worth it.
            built = self._try_build_local_fai()
            if not built:
                return None, None
            fai = built

        gzi_path: str | None = None
        if self._gzipped:
            if present(gzi):
                gzi_path = gzi
            elif self._remote:
                logger.warning(
                    "%s is bgzipped and has a .fai but no .gzi; falling back to streaming",
                    self.fasta,
                )
                return None, None
        return fai, gzi_path

    def _try_build_local_fai(self) -> str | None:
        try:
            import pysam
        except ImportError:  # pragma: no cover - pysam is a hard dependency
            logger.warning("pysam unavailable; streaming %s", self.fasta)
            return None
        try:
            pysam.faidx(self.fasta)
        except Exception as exc:  # noqa: BLE001 - any failure just means "stream it"
            logger.info("could not index %s (%s); streaming instead", self.fasta, exc)
            return None
        candidate = Path(f"{self.fasta}.fai")
        return str(candidate) if candidate.exists() else None

    def _local_index_copy(self, url_or_path: str | None) -> str | None:
        """Index files must be local for htslib when their name is non-default."""
        if not url_or_path or not _is_url(url_or_path):
            return url_or_path
        target = self.cache_dir or Path(tempfile.gettempdir()) / "kmer_dust_index_cache"
        return str(download(url_or_path, Path(target) / _cache_name(url_or_path)))

    # -- lazy pysam handle --------------------------------------------------

    def _pysam(self):
        if self._handle is not None:
            return self._handle
        if not self.indexed:
            raise RuntimeError(f"{self.fasta} has no index; use iter_contigs() instead")
        import pysam

        kwargs: dict[str, str] = {}
        fai, gzi = self._fai, self._gzi
        # htslib finds default-named remote indexes on its own; anything else
        # (a renamed or relocated .fai) has to be materialised locally first.
        if fai and fai != f"{self.fasta}.fai":
            kwargs["filepath_index"] = self._local_index_copy(fai) or fai
        if gzi and gzi != f"{self.fasta}.gzi":
            kwargs["filepath_index_compressed"] = self._local_index_copy(gzi) or gzi
        self._handle = pysam.FastaFile(self.fasta, **kwargs)
        return self._handle

    # -- public interface ---------------------------------------------------

    @property
    def contigs(self) -> list[str]:
        """Contig names in file order.

        Cheap when indexed.  When streaming this has to read the whole file to
        find the headers, so callers on the streaming path should prefer
        ``iter_contigs(contigs=...)``, which filters as it goes.
        """
        return list(self.contig_lengths())

    def contig_lengths(self) -> dict[str, int]:
        """``{contig: length}``.  Streaming sources pay a full pass for this."""
        if self._lengths is not None:
            return dict(self._lengths)
        if self.indexed:
            handle = self._pysam()
            self._lengths = {
                str(name): int(length)
                for name, length in zip(handle.references, handle.lengths)
            }
        else:
            logger.info("scanning %s sequentially to enumerate contigs", self.fasta)
            lengths: dict[str, int] = {}
            with self._open_stream() as stream:
                for name, chunk in _stream_fasta_blocks(stream, DEFAULT_BLOCK, None):
                    lengths[name] = lengths.get(name, 0) + len(chunk)
            self._lengths = lengths
        return dict(self._lengths)

    def fetch(self, contig: str, start: int = 0, end: int | None = None) -> bytes:
        """Sequence of ``contig[start:end]`` as ASCII bytes (case preserved).

        Soft-masking is *not* stripped: lower-case is repeat annotation, and
        :func:`kmer_dust.hashing.encode_bases` treats both cases identically.
        """
        start = max(0, int(start))
        if self.indexed:
            handle = self._pysam()
            lengths = self.contig_lengths()
            if contig not in lengths:
                raise KeyError(f"{contig!r} is not in {self.fasta}")
            stop = lengths[contig] if end is None else min(int(end), lengths[contig])
            if stop <= start:
                return b""
            return handle.fetch(reference=contig, start=start, end=stop).encode("ascii")
        pieces: list[bytes] = []
        with self._open_stream() as stream:
            for name, chunk in _stream_fasta_blocks(stream, DEFAULT_BLOCK, frozenset({contig})):
                if name == contig:
                    pieces.append(chunk)
        seq = b"".join(pieces)
        if not pieces:
            raise KeyError(f"{contig!r} is not in {self.fasta}")
        return seq[start:] if end is None else seq[start : int(end)]

    def iter_contigs(
        self,
        contigs: Sequence[str] | None = None,
        block: int = DEFAULT_BLOCK,
    ) -> Iterator[tuple[str, int, bytes]]:
        """Stream sequence as ``(contig, contig_length, block)`` triples.

        Blocks are **contiguous and non-overlapping**: for one contig the
        concatenation of the blocks is exactly its sequence, in order.  That
        makes the caller responsible for stitching k-mers across a boundary by
        carrying the last ``k-1`` bases of one block onto the front of the next
        -- :mod:`kmer_dust.sketch` does exactly that, and gets a byte-identical
        sketch for any block size.  Blocks are never padded and never repeated,
        so a caller that forgets the carry silently loses ``k-1`` k-mers per
        boundary rather than crashing.

        ``contig_length`` is the full length of the contig, known in advance on
        the indexed path.  On the *streaming* path it is ``-1``: a sequential
        parser cannot know a contig's length until it has passed the end of it.
        Callers must therefore accumulate the length from the blocks they
        receive and treat a positive value as a cross-check, not a promise.

        ``contigs=None`` means "all of them", in file order.
        """
        if self._closed:
            raise ValueError("FastaSource is closed")
        block = max(1, int(block))
        if self.indexed:
            lengths = self.contig_lengths()
            names: Iterable[str] = list(lengths) if contigs is None else list(contigs)
            for name in names:
                if name not in lengths:
                    raise KeyError(f"{name!r} is not in {self.fasta}")
                length = lengths[name]
                for start in range(0, length, block):
                    stop = min(start + block, length)
                    yield name, length, self.fetch(name, start, stop)
            return
        wanted = None if contigs is None else frozenset(contigs)
        with self._open_stream() as stream:
            for name, chunk in _stream_fasta_blocks(stream, block, wanted):
                yield name, -1, chunk

    def _open_stream(self) -> IO[bytes]:
        # Deliberately not cached: every streaming consumer wants the file from
        # the top, and a half-consumed shared handle would be a silent bug.
        return _open_binary(self.fasta, None)

    def close(self) -> None:
        self._closed = True
        if self._handle is not None:
            try:
                self._handle.close()
            except (OSError, ValueError) as exc:  # pragma: no cover - defensive
                logger.debug("closing %s raised %s", self.fasta, exc)
            self._handle = None

    def __enter__(self) -> FastaSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        mode = "indexed" if self.indexed else "streaming"
        return f"FastaSource({self.fasta!r}, {mode})"
