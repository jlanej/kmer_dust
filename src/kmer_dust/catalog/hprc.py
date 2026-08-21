"""HPRC release-2 data tables: cached downloads and typed accessors.

The HPRC publishes its release-2 inventory as a handful of CSV "data tables" in
a GitHub repository, each of which points at S3 objects.  Nothing else in
kmer-dust is allowed to know those URLs: this module is the single place where
the real world leaks in, so that a change upstream is a one-file fix and so the
rest of the pipeline can be exercised with local files.

Three properties matter more than elegance here:

* **Caching.**  A table is downloaded once into ``cache_dir`` and re-read from
  disk forever after.  A run on a cluster node with no outbound network still
  works as long as the cache was warmed.
* **Determinism.**  Accessors sort their output, so two runs against the same
  cached CSVs produce byte-identical manifests.
* **Defensiveness.**  The upstream CSVs are hand-maintained.  They have ragged
  rows, trailing commas that invent empty columns, ``N/A`` sentinels in half a
  dozen spellings, and free-text fields containing quoted commas.  Everything
  here parses with the C parser, warns instead of raising, and normalises to
  ``""`` rather than ``NaN`` so downstream ``string`` columns stay clean.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pandas as pd

from ..log import get_logger

__all__ = [
    "HPRC_DATA_TABLES_BASE",
    "S3_HTTPS_ENDPOINT",
    "ANNOTATION_TABLES",
    "POPULATION_TO_SUPERPOPULATION",
    "SUPERPOPULATIONS",
    "s3_to_https",
    "cache_path",
    "download_file",
    "remote_size",
    "fetch_table",
    "release2_index",
    "annotation_index",
    "sample_metadata",
    "t2t_sequence_table",
    "complete_chromosomes",
]

log = get_logger(__name__)

# --------------------------------------------------------------------------
# where the data lives
# --------------------------------------------------------------------------

#: Raw-GitHub prefix holding every release-2 data table (verified 2026-08).
HPRC_DATA_TABLES_BASE: Final[str] = (
    "https://raw.githubusercontent.com/human-pangenomics/"
    "hprc_intermediate_assembly/main/data_tables/"
)

#: The public HTTPS endpoint for the ``human-pangenomics`` bucket.  The data
#: tables cite ``s3://`` URIs, which pysam/htslib cannot open without AWS
#: credentials; the bucket is world-readable over plain HTTPS.
S3_HTTPS_ENDPOINT: Final[str] = "https://s3-us-west-2.amazonaws.com"

#: Path (relative to :data:`HPRC_DATA_TABLES_BASE`) of the assembly index.
RELEASE2_INDEX_TABLE: Final[str] = "assemblies_release2_v1.0.index.csv"

#: Path of the sample metadata table.
SAMPLE_METADATA_TABLE: Final[str] = "sample/hprc_release2_sample_metadata.csv"

#: Per-assembly annotation indexes, keyed by the name kmer-dust uses internally.
#: Every one of these CSVs has columns ``sample_id,haplotype,assembly_name,location``.
ANNOTATION_TABLES: Final[dict[str, str]] = {
    "censat": "annotation/censat/censat_hprc_r2_v1.0.index.csv",
    "repeatmasker": "annotation/repeat_masker/repeat_masker_bed_hprc_r2_v1.0.index.csv",
    "segdup": "annotation/segdups/segdups_hprc_r2_v1.1.index.csv",
    "chrom_alias": "annotation/chrom_assignment/chrom_alias_hprc_r2_v1.0.index.csv",
    "t2t_sequences": "annotation/chrom_assignment/t2t_sequences_hprc_r2_v1.0.index.csv",
}

#: Tolerated spellings of the annotation kinds.
_KIND_ALIASES: Final[dict[str, str]] = {
    "censat": "censat",
    "censat_bed": "censat",
    "repeat_masker": "repeatmasker",
    "repeatmasker": "repeatmasker",
    "repeatmasker_bed": "repeatmasker",
    "rm": "repeatmasker",
    "segdup": "segdup",
    "segdups": "segdup",
    "segdup_bed": "segdup",
    "chrom_alias": "chrom_alias",
    "chromalias": "chrom_alias",
    "alias": "chrom_alias",
    "t2t_sequences": "t2t_sequences",
    "t2t_chromosomes": "t2t_sequences",
}

#: Population code -> 1000G-style superpopulation.  Covers the 26 1000G
#: populations plus the two extra codes that actually occur in the release-2
#: metadata: ``ASL`` (African Americans, St. Louis) and ``MKK`` (Maasai,
#: Kinyawa).  Anything unknown maps to ``""`` and is warned about once.
POPULATION_TO_SUPERPOPULATION: Final[dict[str, str]] = {
    # African
    "ACB": "AFR",
    "ASW": "AFR",
    "ESN": "AFR",
    "GWD": "AFR",
    "LWK": "AFR",
    "MSL": "AFR",
    "YRI": "AFR",
    "ASL": "AFR",
    "MKK": "AFR",
    # Admixed American
    "CLM": "AMR",
    "MXL": "AMR",
    "PEL": "AMR",
    "PUR": "AMR",
    # East Asian
    "CDX": "EAS",
    "CHB": "EAS",
    "CHS": "EAS",
    "JPT": "EAS",
    "KHV": "EAS",
    # European
    "CEU": "EUR",
    "FIN": "EUR",
    "GBR": "EUR",
    "IBS": "EUR",
    "TSI": "EUR",
    # South Asian
    "BEB": "SAS",
    "GIH": "SAS",
    "ITU": "SAS",
    "PJL": "SAS",
    "STU": "SAS",
}

SUPERPOPULATIONS: Final[tuple[str, ...]] = ("AFR", "AMR", "EAS", "EUR", "SAS")

#: Values that upstream uses to mean "missing".
_NULLISH: Final[frozenset[str]] = frozenset({"", "na", "n/a", "nan", "none", "null", "-", "."})

#: Haplotype tokens that may appear in an ``assembly_name``.
_HAPLOTYPE_TOKENS: Final[frozenset[str]] = frozenset({"pat", "mat", "hap1", "hap2"})

#: ``warnings.catch_warnings`` mutates process-global state, and manifest
#: building parses tables from a thread pool -- serialise the parse step.
_PARSE_LOCK = threading.Lock()
_WARNED_POPULATIONS: set[str] = set()


# --------------------------------------------------------------------------
# URL / cache plumbing
# --------------------------------------------------------------------------


def s3_to_https(uri: str) -> str:
    """Rewrite an ``s3://bucket/key`` URI as a public HTTPS URL.

    Anything that is not an ``s3://`` URI is returned unchanged, so this is safe
    to map over a column that already mixes URIs and URLs.  Missing values
    become ``""``.
    """
    if uri is None or not isinstance(uri, str):
        return ""
    text = uri.strip()
    if not text or text.lower() in _NULLISH:
        return ""
    if not text.startswith("s3://"):
        return text
    bucket, _, key = text[len("s3://") :].partition("/")
    if not bucket or not key:
        log.warning("cannot convert malformed S3 URI %r", uri)
        return ""
    return f"{S3_HTTPS_ENDPOINT}/{bucket}/{key}"


def cache_path(url: str, cache_dir: Path) -> Path:
    """Deterministic on-disk location for ``url`` under ``cache_dir``.

    The URL digest is a *directory* rather than a filename prefix so that the
    cached copy keeps its original extension -- pysam and gzip both sniff on the
    name, and a debugging human wants to see ``chm13v2.0_censat_v2.0.bed``.
    """
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    name = os.path.basename(url.split("?", 1)[0]) or "download"
    return Path(cache_dir) / digest / name


def _is_usable(path: Path) -> bool:
    """True when a cached file exists and is not obviously truncated."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.name.endswith(".gz"):
            with open(path, "rb") as handle:
                if handle.read(2) != b"\x1f\x8b":
                    log.warning("cached file %s is not gzip; re-downloading", path)
                    return False
    except OSError as exc:  # pragma: no cover - filesystem weirdness
        log.warning("cannot stat cached file %s: %s", path, exc)
        return False
    return True


def download_file(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    timeout: tuple[float, float] = (10.0, 120.0),
    retries: int = 3,
) -> Path:
    """Download ``url`` to ``dest``, atomically and idempotently.

    Named ``download_file`` rather than ``download`` so it cannot be confused
    with :func:`kmer_dust.fasta.download`, which serves the FASTA layer.

    The write goes to a process/thread-unique temporary file and is then
    ``os.replace``d into position, so a concurrent reader never sees a partial
    file and an interrupted run leaves the cache consistent.
    """
    import requests  # local import: the catalog is importable without network deps

    dest = Path(dest)
    if not force and _is_usable(dest):
        log.debug("cache hit %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident():x}.tmp")
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            if tmp.stat().st_size == 0:
                raise OSError(f"empty response body for {url}")
            os.replace(tmp, dest)
            log.debug("downloaded %s -> %s (%d bytes)", url, dest, dest.stat().st_size)
            return dest
        except Exception as exc:  # noqa: BLE001 - re-raised below after retries
            last_error = exc
            if isinstance(exc, KeyboardInterrupt):  # pragma: no cover
                raise
            log.warning("download attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            time.sleep(min(2.0**attempt, 10.0))
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:  # pragma: no cover
                    pass
    raise RuntimeError(f"failed to download {url}: {last_error}")


def remote_size(url: str, *, timeout: float = 30.0) -> int:
    """Content-Length of ``url`` in bytes, or ``-1`` when the server won't say.

    Used to decide whether a track can be slurped whole or has to be sliced.
    """
    import requests

    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        return int(length) if length is not None else -1
    except Exception as exc:  # noqa: BLE001 - a missing size is not fatal
        log.debug("HEAD failed for %s: %s", url, exc)
        return -1


def url_exists(url: str, *, timeout: float = 30.0) -> bool:
    """True when ``url`` answers a HEAD request with a 2xx status."""
    import requests

    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return bool(response.ok)
    except Exception as exc:  # noqa: BLE001
        log.debug("HEAD failed for %s: %s", url, exc)
        return False


# --------------------------------------------------------------------------
# table parsing
# --------------------------------------------------------------------------


def _blank_nullish(series: pd.Series) -> pd.Series:
    """Map every flavour of "missing" to the empty string."""
    text = series.astype("string").fillna("")
    text = text.str.strip()
    return text.mask(text.str.lower().isin(_NULLISH), "").astype("string")


def _read_csv(path: Path, sep: str) -> pd.DataFrame:
    """Parse a hand-maintained CSV without ever raising on a ragged row.

    ``on_bad_lines="warn"`` keeps the good rows and reports the ragged ones; the
    warnings are captured and re-emitted through the logger so a run's stderr
    stays in one format.
    """
    with _PARSE_LOCK, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = pd.read_csv(
            path,
            sep=sep,
            engine="c",
            on_bad_lines="warn",
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
    for entry in caught:
        log.warning("while parsing %s: %s", path.name, entry.message)
    # Trailing commas in the header invent all-empty "Unnamed: n" columns.
    drop = [
        col
        for col in frame.columns
        if str(col).startswith("Unnamed:") and not frame[col].str.strip().any()
    ]
    if drop:
        frame = frame.drop(columns=drop)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def fetch_table(url: str, cache_dir: Path, *, force: bool = False, sep: str = ",") -> pd.DataFrame:
    """Download (once) and parse a data table.

    Every column comes back as pandas ``string`` with ``""`` for missing, which
    is what the manifest contract wants and which sidesteps the usual
    "``NaN`` sneaks into a string column" bug.
    """
    dest = cache_path(url, Path(cache_dir))
    download_file(url, dest, force=force)
    try:
        frame = _read_csv(dest, sep)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        if force:
            raise RuntimeError(f"cannot parse {url} (cached at {dest}): {exc}") from exc
        log.warning("cached copy of %s looks corrupt (%s); re-downloading", url, exc)
        download_file(url, dest, force=True)
        frame = _read_csv(dest, sep)
    for col in frame.columns:
        frame[col] = _blank_nullish(frame[col])
    return frame


def _table_url(relative: str) -> str:
    return HPRC_DATA_TABLES_BASE + relative.lstrip("/")


# --------------------------------------------------------------------------
# typed accessors
# --------------------------------------------------------------------------


def _haplotype_label(assembly_name: str, hap_number: str) -> str:
    """Haplotype token as the HPRC itself spells it.

    ``assembly_name`` looks like ``HG00408_pat_hprc_r2_v1.0.1`` or
    ``NA18906_hap2_hprc_r2_v1.0.1``; the second underscore-delimited field is
    the authoritative label.  Reconstructing it from the numeric ``haplotype``
    column would silently rename half the trio-phased assemblies.

    A handful of release-2 entries do not follow that shape at all -- the HG002
    Q100 assemblies are named ``hg002v1.1.pat`` and ``hg002v1.1.mat_MT`` -- so
    after the positional field fails we scan every ``_``/``.``-delimited token
    for a haplotype word before falling back to the numeric column.  Sample ids
    are ``HG#####``/``NA#####``, so a token scan cannot collide with one.
    """
    parts = assembly_name.split("_")
    if len(parts) > 1 and parts[1].lower() in _HAPLOTYPE_TOKENS:
        return parts[1].lower()
    for token in re.split(r"[._]", assembly_name):
        if token.lower() in _HAPLOTYPE_TOKENS:
            return token.lower()
    if hap_number in {"1", "2"}:
        return f"hap{hap_number}"
    return ""


def release2_index(cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """One row per release-2 *haplotype* assembly, with HTTPS URLs.

    Columns: ``sample, haplotype, assembly, fasta, fai, gzi``.

    The upstream CSV also carries two ``haplotype == 0`` rows for GRCh38 and
    CHM13.  Those are references, not haplotypes -- kmer-dust adds its own
    reference row from :mod:`kmer_dust.catalog.t2t` -- so they are dropped here.
    """
    raw = fetch_table(_table_url(RELEASE2_INDEX_TABLE), cache_dir, force=force)
    required = {"sample_id", "haplotype", "assembly_name", "assembly"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"release-2 index is missing column(s) {sorted(missing)}")
    if raw.empty:
        return _empty_index()

    frame = pd.DataFrame(
        {
            "sample": raw["sample_id"],
            "assembly": raw["assembly_name"],
            "fasta": raw["assembly"].map(s3_to_https),
            "fai": raw.get("assembly_fai", pd.Series("", index=raw.index)).map(s3_to_https),
            "gzi": raw.get("assembly_gzi", pd.Series("", index=raw.index)).map(s3_to_https),
        }
    )
    frame["haplotype"] = [
        _haplotype_label(name, hap)
        for name, hap in zip(raw["assembly_name"], raw["haplotype"], strict=True)
    ]

    reference_rows = ~raw["haplotype"].isin(["1", "2"])
    if reference_rows.any():
        log.debug(
            "dropping %d non-haplotype row(s) from the release-2 index: %s",
            int(reference_rows.sum()),
            ", ".join(sorted(frame.loc[reference_rows, "assembly"])),
        )
        frame = frame.loc[~reference_rows]

    usable = (frame["assembly"] != "") & (frame["sample"] != "") & (frame["fasta"] != "")
    if not usable.all():
        log.warning("dropping %d release-2 row(s) with no usable FASTA URL", int((~usable).sum()))
        frame = frame.loc[usable]

    frame = frame.drop_duplicates(subset="assembly", keep="first")
    frame = frame[["sample", "haplotype", "assembly", "fasta", "fai", "gzi"]]
    frame = frame.sort_values(["sample", "haplotype", "assembly"], kind="stable")
    return frame.reset_index(drop=True).astype("string")


def _empty_index() -> pd.DataFrame:
    cols = ["sample", "haplotype", "assembly", "fasta", "fai", "gzi"]
    return pd.DataFrame({c: pd.Series([], dtype="string") for c in cols})


def annotation_index(kind: str, cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Per-assembly annotation locations for one track ``kind``.

    ``kind`` is one of ``censat``, ``repeatmasker``, ``segdup``, ``chrom_alias``
    or ``t2t_sequences`` (a few obvious spellings are accepted).  Columns:
    ``sample, assembly, url``.
    """
    canonical = _KIND_ALIASES.get(str(kind).strip().lower())
    if canonical is None:
        raise ValueError(
            f"unknown annotation kind {kind!r}; expected one of {sorted(ANNOTATION_TABLES)}"
        )
    raw = fetch_table(_table_url(ANNOTATION_TABLES[canonical]), cache_dir, force=force)
    empty = pd.DataFrame({c: pd.Series([], dtype="string") for c in ("sample", "assembly", "url")})
    if raw.empty:
        return empty
    if not {"assembly_name", "location"} <= set(raw.columns):
        log.warning("annotation index %r lacks assembly_name/location columns", canonical)
        return empty
    frame = pd.DataFrame(
        {
            "sample": raw.get("sample_id", pd.Series("", index=raw.index)),
            "assembly": raw["assembly_name"],
            "url": raw["location"].map(s3_to_https),
        }
    )
    frame = frame.loc[(frame["assembly"] != "") & (frame["url"] != "")]
    frame = frame.drop_duplicates(subset="assembly", keep="first")
    frame = frame.sort_values("assembly", kind="stable").reset_index(drop=True)
    return frame.astype("string")


def sample_metadata(cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Sample-level metadata, reduced to what the manifest actually needs.

    Columns: ``sample, population, superpopulation, sex, trio_available``.

    The upstream CSV has fifteen columns, one of which (``population_descriptor``)
    is free text containing quoted commas, and a handful of rows are ragged.
    Rather than trust the whole table we keep four fields and normalise them.
    """
    raw = fetch_table(_table_url(SAMPLE_METADATA_TABLE), cache_dir, force=force)
    empty = pd.DataFrame(
        {
            "sample": pd.Series([], dtype="string"),
            "population": pd.Series([], dtype="string"),
            "superpopulation": pd.Series([], dtype="string"),
            "sex": pd.Series([], dtype="string"),
            "trio_available": pd.Series([], dtype="bool"),
        }
    )
    if raw.empty or "sample_id" not in raw.columns:
        log.warning("sample metadata table is empty or malformed; proceeding without it")
        return empty

    population = raw.get("population_abbreviation", pd.Series("", index=raw.index)).str.upper()
    sex = raw.get("sex", pd.Series("", index=raw.index)).str.lower()
    sex = sex.where(sex.isin(["male", "female"]), "")
    trio = raw.get("trio_available", pd.Series("", index=raw.index)).str.upper() == "TRUE"

    super_pop = population.map(lambda code: POPULATION_TO_SUPERPOPULATION.get(code, ""))
    unmapped = sorted(set(population[(population != "") & (super_pop == "")]))
    new = [code for code in unmapped if code not in _WARNED_POPULATIONS]
    if new:
        _WARNED_POPULATIONS.update(new)
        log.warning(
            "population code(s) with no superpopulation mapping: %s "
            "(rows keep population but get superpopulation='')",
            ", ".join(new),
        )

    frame = pd.DataFrame(
        {
            "sample": raw["sample_id"],
            "population": population,
            "superpopulation": super_pop,
            "sex": sex,
            "trio_available": trio,
        }
    )
    frame = frame.loc[frame["sample"] != ""]
    frame = frame.drop_duplicates(subset="sample", keep="first")
    frame = frame.sort_values("sample", kind="stable").reset_index(drop=True)
    for col in ("sample", "population", "superpopulation", "sex"):
        frame[col] = frame[col].astype("string")
    frame["trio_available"] = frame["trio_available"].astype(bool)
    return frame


def t2t_sequence_table(cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Index of per-assembly ``*.t2t_chromosomes.tsv`` files.

    Columns: ``sample, assembly, url``.  The pointed-at TSVs are what
    :func:`complete_chromosomes` reads; there is one per haplotype assembly, so
    they are only fetched when ``ManifestConfig.require_t2t_chrom`` is set.
    """
    return annotation_index("t2t_sequences", cache_dir, force=force)


def complete_chromosomes(url: str, cache_dir: Path, *, force: bool = False) -> frozenset[str]:
    """Chromosomes assembled telomere-to-telomere in one haplotype.

    A chromosome counts as T2T-complete when its sequence is a single ungapped
    ``contig`` -- ``level == "contig"`` and ``num_gaps == 0``.  A scaffold with
    zero gaps would be a contradiction, so both conditions are checked.

    Returns an empty set (with a warning) when the TSV is missing or unreadable:
    a broken annotation should shrink the manifest, never abort the run.
    """
    if not url:
        return frozenset()
    try:
        table = fetch_table(url, cache_dir, force=force, sep="\t")
    except Exception as exc:  # noqa: BLE001 - one bad haplotype must not kill the run
        log.warning("cannot read t2t_chromosomes table %s: %s", url, exc)
        return frozenset()
    if table.empty or not {"chr_name", "level"} <= set(table.columns):
        log.warning("t2t_chromosomes table %s has unexpected columns", url)
        return frozenset()
    gaps = pd.to_numeric(table.get("num_gaps", "0"), errors="coerce").fillna(1)
    keep = (table["level"].str.lower() == "contig") & (gaps == 0)
    return frozenset(name for name in table.loc[keep, "chr_name"] if name)


def superpopulation_of(population: str) -> str:
    """Superpopulation for a population code, ``""`` when unknown."""
    return POPULATION_TO_SUPERPOPULATION.get(str(population).strip().upper(), "")


def warm_cache(cache_dir: Path, kinds: Iterable[str] | None = None, *, force: bool = False) -> None:
    """Download every data table needed to build a manifest offline later."""
    cache_dir = Path(cache_dir)
    release2_index(cache_dir, force=force)
    sample_metadata(cache_dir, force=force)
    for kind in kinds if kinds is not None else ANNOTATION_TABLES:
        annotation_index(kind, cache_dir, force=force)
