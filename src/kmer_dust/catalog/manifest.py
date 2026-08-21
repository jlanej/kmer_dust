"""Build the assembly manifest: which haplotypes take part in a run.

The manifest is the contract between "the outside world" and every other stage.
It is a plain TSV (``schemas.MANIFEST_COLUMNS``) so a user can hand-edit one,
check it into a repository, or diff two runs; and it is the *only* place where
assemblies get chosen, so a run is reproducible from it alone.

Three sources are supported:

``hprc_release2``
    Join the release-2 assembly index with the annotation indexes and the
    sample metadata, then apply every filter in :class:`~kmer_dust.config.ManifestConfig`.
``file``
    Read a TSV the user wrote.  Filters that can be evaluated locally still
    apply, so ``--max-samples 4`` behaves the same way whatever the source.
``local_dir``
    Glob a directory of FASTAs, one assembly per file.

Sub-selection is the delicate part.  When a cap forces a choice, the aim is a
*useful* subset rather than an arbitrary prefix: whole samples (both
haplotypes) before half-samples, and a spread across superpopulations before a
pile of Europeans.  The tie-break is a BLAKE2b digest of ``f"{seed}:{sample}"``
-- not Python's ``hash()``, which is salted per process and would make runs
irreproducible.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from .. import schemas
from ..config import Config
from ..log import get_logger
from . import hprc, t2t

__all__ = ["build_manifest", "write_manifest", "read_manifest", "ordered_samples"]

log = get_logger(__name__)

#: ``require_annotations`` name -> manifest column holding its URL.
_ANNOTATION_COLUMN: dict[str, str] = {
    "censat": "censat_bed",
    "repeatmasker": "repeatmasker_bed",
    "segdup": "segdup_bed",
    "segdups": "segdup_bed",
    "chrom_alias": "chrom_alias",
    "chromalias": "chrom_alias",
}

#: FASTA extensions recognised by ``source = local_dir``.
_FASTA_GLOBS: tuple[str, ...] = ("*.fa", "*.fa.gz", "*.fasta", "*.fasta.gz")

#: Haplotype tokens recognised in a local FASTA filename.
_HAPLOTYPE_TOKENS: frozenset[str] = frozenset({"pat", "mat", "hap1", "hap2"})


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def build_manifest(cfg: Config, cache_dir: Path, *, force: bool = False) -> pd.DataFrame:
    """Resolve ``cfg.manifest`` into a concrete list of assemblies.

    Returns exactly ``schemas.MANIFEST_COLUMNS``, the reference row first when
    ``cfg.manifest.include_reference``, in a deterministic order.  An input that
    is empty but valid (no matching samples, an empty directory) yields an empty
    frame with the right dtypes rather than an exception -- the caller decides
    whether zero assemblies is an error.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = cfg.manifest.source

    if source == "hprc_release2":
        frame = _from_hprc(cfg, cache_dir, force=force)
    elif source == "file":
        frame = _from_file(Path(cfg.manifest.path))
    elif source == "local_dir":
        frame = _from_local_dir(Path(cfg.manifest.path))
    else:  # config.validate() already rejects this, but be explicit
        raise ValueError(f"unknown manifest source {source!r}")

    frame = _apply_filters(frame, cfg, cache_dir, force=force)
    frame = _apply_caps(frame, cfg)
    frame = _finalize(frame, cfg)
    log.info(
        "manifest: %d assemblies from %d samples (source=%s)",
        len(frame),
        frame["sample"].nunique() if len(frame) else 0,
        source,
    )
    return frame


def write_manifest(df: pd.DataFrame, path: Path) -> Path:
    """Write a manifest TSV atomically; returns ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = schemas.enforce(df, schemas.MANIFEST_COLUMNS)
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, sep="\t", index=False, na_rep="", lineterminator="\n")
    os.replace(tmp, path)
    log.debug("wrote manifest %s (%d rows)", path, len(out))
    return path


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest TSV, tolerating missing optional columns.

    Rows lacking a ``schemas.MANIFEST_REQUIRED`` field are dropped with a
    warning: a half-written row is far more likely to be a typo than an
    intention, and letting it through would fail much later inside pysam.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        comment=None,
    )
    frame.columns = [str(c).strip() for c in frame.columns]
    for column in schemas.MANIFEST_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[list(schemas.MANIFEST_COLUMNS)]
    for column in frame.columns:
        frame[column] = frame[column].astype("string").fillna("").str.strip()
    if len(frame):
        bad = pd.Series(False, index=frame.index)
        for column in schemas.MANIFEST_REQUIRED:
            bad |= frame[column] == ""
        if bad.any():
            log.warning(
                "dropping %d manifest row(s) missing one of %s",
                int(bad.sum()),
                ", ".join(schemas.MANIFEST_REQUIRED),
            )
            frame = frame.loc[~bad]
    return schemas.enforce(frame.reset_index(drop=True), schemas.MANIFEST_COLUMNS)


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def _empty_manifest() -> pd.DataFrame:
    return schemas.empty_frame(schemas.MANIFEST_COLUMNS)


def _from_hprc(cfg: Config, cache_dir: Path, *, force: bool) -> pd.DataFrame:
    """Join the release-2 index with annotations and sample metadata."""
    index = hprc.release2_index(cache_dir, force=force)
    if index.empty:
        log.warning("release-2 index is empty; manifest will contain no HPRC assemblies")
        return _empty_manifest()

    frame = pd.DataFrame(
        {
            "assembly": index["assembly"],
            "sample": index["sample"],
            "haplotype": index["haplotype"],
            "source": "hprc",
            "fasta": index["fasta"],
            "fai": index["fai"],
            "gzi": index["gzi"],
        }
    )

    for kind, column in (
        ("chrom_alias", "chrom_alias"),
        ("censat", "censat_bed"),
        ("repeatmasker", "repeatmasker_bed"),
        ("segdup", "segdup_bed"),
    ):
        try:
            table = hprc.annotation_index(kind, cache_dir, force=force)
        except Exception as exc:  # noqa: BLE001 - a missing index is not fatal
            log.warning("annotation index %r unavailable (%s); column left empty", kind, exc)
            frame[column] = ""
            continue
        mapping = dict(zip(table["assembly"], table["url"], strict=True))
        frame[column] = frame["assembly"].map(mapping).astype("string").fillna("")
        n_missing = int((frame[column] == "").sum())
        if n_missing:
            log.debug("%d/%d assemblies have no %s annotation", n_missing, len(frame), kind)

    try:
        meta = hprc.sample_metadata(cache_dir, force=force)
    except Exception as exc:  # noqa: BLE001
        log.warning("sample metadata unavailable (%s); population columns left empty", exc)
        meta = pd.DataFrame(columns=["sample", "population", "superpopulation", "sex"])
    for column in ("population", "superpopulation", "sex"):
        mapping = dict(zip(meta["sample"], meta[column], strict=True)) if len(meta) else {}
        frame[column] = frame["sample"].map(mapping).astype("string").fillna("")

    return schemas.enforce(frame, schemas.MANIFEST_COLUMNS)


def _from_file(path: Path) -> pd.DataFrame:
    """Read a user-written manifest TSV."""
    frame = read_manifest(path)
    log.info("read %d assemblies from %s", len(frame), path)
    return frame


def _split_local_stem(stem: str) -> tuple[str, str]:
    """``HG002_pat`` -> ``("HG002", "pat")``; anything else -> ``(stem, "hap1")``.

    A directory of assemblies almost always encodes the haplotype in the file
    name, and losing it would collapse both haplotypes of a sample into one
    ``sample``/``haplotype`` pair -- which the prevalence maths cares about.
    """
    parts = stem.split("_")
    for cut in range(len(parts) - 1, 0, -1):
        if parts[cut].lower() in _HAPLOTYPE_TOKENS:
            return "_".join(parts[:cut]), parts[cut].lower()
    return stem, "hap1"


def _from_local_dir(path: Path) -> pd.DataFrame:
    """One assembly per FASTA file in ``path``."""
    if not path.is_dir():
        raise NotADirectoryError(f"manifest.path is not a directory: {path}")
    seen: dict[Path, None] = {}
    for pattern in _FASTA_GLOBS:
        for hit in sorted(path.glob(pattern)):
            seen.setdefault(hit.resolve(), None)
    files = sorted(seen)
    if not files:
        log.warning("no %s files under %s", "/".join(_FASTA_GLOBS), path)
        return _empty_manifest()

    rows: list[dict[str, str]] = []
    for fasta in files:
        stem = fasta.name
        for suffix in (".fa.gz", ".fasta.gz", ".fa", ".fasta"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        sample, haplotype = _split_local_stem(stem)
        fai = fasta.with_name(fasta.name + ".fai")
        gzi = fasta.with_name(fasta.name + ".gzi")
        rows.append(
            {
                "assembly": stem,
                "sample": sample,
                "haplotype": haplotype,
                "source": "local",
                "fasta": str(fasta),
                "fai": str(fai) if fai.is_file() else "",
                "gzi": str(gzi) if gzi.is_file() else "",
                "chrom_alias": "",
                "censat_bed": "",
                "repeatmasker_bed": "",
                "segdup_bed": "",
                "population": "",
                "superpopulation": "",
                "sex": "",
            }
        )
    log.info("found %d FASTA file(s) under %s", len(rows), path)
    return schemas.enforce(pd.DataFrame(rows), schemas.MANIFEST_COLUMNS)


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------


def _normalised(values: Iterable[str]) -> list[str]:
    return [str(v).strip() for v in values if str(v).strip()]


def _apply_filters(
    frame: pd.DataFrame, cfg: Config, cache_dir: Path, *, force: bool
) -> pd.DataFrame:
    """Apply every ``ManifestConfig`` predicate, cheapest first."""
    if frame.empty:
        return frame
    mcfg = cfg.manifest

    allow = _normalised(mcfg.samples)
    if allow:
        wanted = set(allow)
        unknown = wanted - set(frame["sample"])
        if unknown:
            log.warning("requested sample(s) not in the catalog: %s", ", ".join(sorted(unknown)))
        frame = frame.loc[frame["sample"].isin(wanted)]
        log.debug("samples allow-list keeps %d assemblies", len(frame))

    exclude = set(_normalised(mcfg.exclude_samples))
    if exclude:
        frame = frame.loc[~frame["sample"].isin(exclude)]
        log.debug("exclude_samples keeps %d assemblies", len(frame))

    populations = {p.upper() for p in _normalised(mcfg.populations)}
    if populations:
        keep = frame["population"].str.upper().isin(populations) | frame[
            "superpopulation"
        ].str.upper().isin(populations)
        dropped = int((~keep).sum())
        frame = frame.loc[keep]
        log.debug("population filter %s drops %d assemblies", sorted(populations), dropped)

    # Annotation requirements are a statement about the HPRC catalog: a local
    # FASTA directory has no annotation index at all, and dropping every row
    # because of the default ``require_annotations=["censat"]`` would be a
    # baffling way to return an empty manifest.
    annotatable = frame["source"] == "hprc"
    for name in _normalised(mcfg.require_annotations):
        column = _ANNOTATION_COLUMN.get(name.lower())
        if column is None:
            log.warning("require_annotations: unknown track %r, ignored", name)
            continue
        keep = ~annotatable | (frame[column].fillna("") != "")
        dropped = int((~keep).sum())
        frame = frame.loc[keep]
        annotatable = annotatable.loc[keep]
        if dropped:
            log.info("require_annotations=%s drops %d assemblies", name, dropped)

    if mcfg.require_t2t_chrom:
        frame = _filter_t2t_complete(frame, cfg, cache_dir, force=force)

    return frame.reset_index(drop=True)


def _filter_t2t_complete(
    frame: pd.DataFrame, cfg: Config, cache_dir: Path, *, force: bool
) -> pd.DataFrame:
    """Keep haplotypes where every requested chromosome is a gapless contig.

    This needs one small TSV *per assembly* -- several hundred HTTP GETs -- so it
    runs last (after the cheap filters have shrunk the candidate list), in a
    thread pool, and against the on-disk cache.  A haplotype whose TSV cannot be
    read is dropped rather than assumed complete: the flag exists precisely to
    guarantee completeness.
    """
    from concurrent.futures import ThreadPoolExecutor

    chroms = _normalised(cfg.manifest.chroms)
    if not chroms:
        log.warning("require_t2t_chrom is set but manifest.chroms is empty; filter skipped")
        return frame
    hprc_rows = frame["source"] == "hprc"
    candidates = sorted(set(frame.loc[hprc_rows, "assembly"]))
    if not candidates:
        return frame

    try:
        index = hprc.t2t_sequence_table(cache_dir, force=force)
    except Exception as exc:  # noqa: BLE001
        log.warning("t2t_sequences index unavailable (%s); require_t2t_chrom skipped", exc)
        return frame
    urls = dict(zip(index["assembly"], index["url"], strict=True))

    required = set(chroms)
    workers = max(1, min(16, int(cfg.threads) * 4))
    log.info(
        "checking T2T completeness of %s for %d assemblies (%d threads)",
        ",".join(chroms),
        len(candidates),
        workers,
    )

    def _check(assembly: str) -> bool:
        url = urls.get(assembly, "")
        if not url:
            return False
        return required <= hprc.complete_chromosomes(url, cache_dir, force=force)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verdicts = dict(zip(candidates, pool.map(_check, candidates), strict=True))

    keep = ~hprc_rows | frame["assembly"].map(verdicts).fillna(False).astype(bool)
    dropped = int((~keep).sum())
    if dropped:
        log.info("require_t2t_chrom drops %d assemblies", dropped)
    return frame.loc[keep]


# --------------------------------------------------------------------------
# deterministic sub-selection
# --------------------------------------------------------------------------


def _sample_key(sample: str, seed: int) -> int:
    """Stable pseudo-random ordering key.

    ``hash()`` is salted per process, so it cannot be used: two runs of the same
    config would pick different samples.  BLAKE2b of ``seed:sample`` is stable
    across processes, machines and Python versions.
    """
    digest = hashlib.blake2b(f"{seed}:{sample}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def ordered_samples(frame: pd.DataFrame, seed: int) -> list[str]:
    """Every sample, ordered "best first" and balanced across superpopulations.

    Within a superpopulation, samples with both haplotypes come first (a
    half-sample weakens the per-sample prevalence statistics that drive k-mer
    selection); ties break on :func:`_sample_key`.  Groups are then visited
    round-robin so that a cap of *n* samples spreads across ancestries instead
    of taking whichever group sorts first.

    Samples with no superpopulation (GIAB and other extras) are appended only
    after every known group is exhausted -- they cannot contribute to the
    spread, so they should not consume one of its slots.
    """
    groups: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for sample, rows in frame.groupby("sample", sort=True):
        superpop = str(rows["superpopulation"].iloc[0] or "")
        complete = rows["haplotype"].nunique() >= 2
        groups[superpop].append((0 if complete else 1, _sample_key(str(sample), seed), str(sample)))

    known = sorted(name for name in groups if name)
    queues = {name: sorted(groups[name]) for name in known}
    picked: list[str] = []
    while any(queues[name] for name in known):
        for name in known:
            if queues[name]:
                picked.append(queues[name].pop(0)[2])
    picked.extend(entry[2] for entry in sorted(groups.get("", [])))
    return picked


def _apply_caps(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Honour ``max_samples`` / ``max_assemblies`` deterministically.

    Both caps count HPRC/local assemblies only; the reference row is added
    afterwards and is never what a user means by "four samples".
    """
    if frame.empty:
        return frame
    mcfg = cfg.manifest
    max_samples = int(mcfg.max_samples or 0)
    max_assemblies = int(mcfg.max_assemblies or 0)
    if max_samples <= 0 and max_assemblies <= 0:
        return frame

    order = ordered_samples(frame, int(mcfg.seed))
    if max_samples > 0 and len(order) > max_samples:
        log.info("max_samples=%d selects from %d candidate samples", max_samples, len(order))
        order = order[:max_samples]
    chosen = set(order)
    frame = frame.loc[frame["sample"].isin(chosen)]

    if max_assemblies > 0 and len(frame) > max_assemblies:
        by_sample = {
            str(sample): rows.sort_values(["haplotype", "assembly"], kind="stable")
            for sample, rows in frame.groupby("sample", sort=False)
        }
        keep: list[str] = []
        for sample in order:
            rows = by_sample.get(sample)
            if rows is None:
                continue
            room = max_assemblies - len(keep)
            if room <= 0:
                break
            keep.extend(rows["assembly"].tolist()[:room])
        log.info(
            "max_assemblies=%d keeps %d of %d assemblies", max_assemblies, len(keep), len(frame)
        )
        frame = frame.loc[frame["assembly"].isin(set(keep))]
    return frame.reset_index(drop=True)


def _finalize(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sort, prepend the reference row, and enforce the column contract."""
    if len(frame):
        frame = frame.sort_values(["sample", "haplotype", "assembly"], kind="stable")
        frame = frame.reset_index(drop=True)
    if cfg.manifest.include_reference:
        row = t2t.reference_manifest_row()
        if frame.empty or row["assembly"] not in set(frame["assembly"]):
            reference = schemas.enforce(pd.DataFrame([row]), schemas.MANIFEST_COLUMNS)
            frame = pd.concat([reference, frame], ignore_index=True) if len(frame) else reference
    if frame.empty:
        return _empty_manifest()
    return schemas.enforce(frame.reset_index(drop=True), schemas.MANIFEST_COLUMNS)


def manifest_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Small dict of headline numbers, handy for logs and the report."""
    if frame.empty:
        return {"assemblies": 0, "samples": 0, "superpopulations": {}, "sources": {}}
    return {
        "assemblies": int(len(frame)),
        "samples": int(frame["sample"].nunique()),
        "superpopulations": frame["superpopulation"].value_counts().to_dict(),
        "sources": frame["source"].value_counts().to_dict(),
    }


def annotation_columns(tracks: Sequence[str]) -> list[str]:
    """Manifest columns backing a list of track names (unknown names ignored)."""
    out: list[str] = []
    for name in tracks:
        column = _ANNOTATION_COLUMN.get(str(name).strip().lower())
        if column and column not in out:
            out.append(column)
    return out
