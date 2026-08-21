"""T2T-CHM13v2.0: the reference assembly and its annotation tracks.

CHM13v2.0 is the anchor of the whole analysis.  It is the one assembly whose
centromeres, satellite arrays and segmental duplications are annotated to
completion, so a cluster that lands on a CHM13 bin inherits a *name*; every
HPRC bin in the same cluster then inherits that name by association.  Getting
these five URLs right is therefore load-bearing.

Every URL below was verified with an HTTP HEAD against the public
``human-pangenomics`` bucket before being hard-coded (2026-08).  Two choices
deserve a note:

* The **uncompressed** ``chm13v2.0.fa`` is used, not the bgzip analysis-set
  copy.  The bgzip file ships a ``.gzi`` but no ``.fa.gz.fai``, so htslib
  cannot do remote region access on it; the plain FASTA has a published
  ``.fai`` and random access works over HTTPS.
* The segmental-duplication track is ``chm13v2.0_SD.bed`` -- a BED9 of SD
  intervals -- found by listing the annotation prefix.  Its sibling
  ``chm13v2.0_SD.full.bed`` is the same intervals with 35 extra analysis
  columns and a ``#``-commented header, which buys nothing here.
"""

from __future__ import annotations

from typing import Final

from ..log import get_logger
from .hprc import S3_HTTPS_ENDPOINT, url_exists

__all__ = [
    "T2T_ASSEMBLY",
    "T2T_SAMPLE",
    "T2T_FASTA",
    "T2T_FAI",
    "T2T_GZI",
    "T2T_TRACKS",
    "T2T_CHROMS",
    "reference_manifest_row",
    "verify_track_urls",
]

log = get_logger(__name__)

_BASE: Final[str] = f"{S3_HTTPS_ENDPOINT}/human-pangenomics/T2T/CHM13/assemblies"
_ANNOTATION: Final[str] = f"{_BASE}/annotation"

#: Assembly id used everywhere in the pipeline for the reference.
T2T_ASSEMBLY: Final[str] = "chm13v2.0"

#: Sample id for the reference (CHM13 is a complete hydatidiform mole; the v2.0
#: assembly grafts on HG002's chrY, which is why ``sex`` stays blank below).
T2T_SAMPLE: Final[str] = "CHM13"

T2T_FASTA: Final[str] = f"{_BASE}/chm13v2.0.fa"
T2T_FAI: Final[str] = f"{_BASE}/chm13v2.0.fa.fai"
#: The reference FASTA is not block-compressed, so there is no index to name.
T2T_GZI: Final[str] = ""

#: Annotation tracks, keyed by the names used in ``AnnotateConfig``.
T2T_TRACKS: Final[dict[str, str]] = {
    "censat": f"{_ANNOTATION}/chm13v2.0_censat_v2.0.bed",
    "repeatmasker": f"{_ANNOTATION}/chm13v2.0_RepeatMasker_4.1.2p1.2022Apr14.bed",
    "segdup": f"{_ANNOTATION}/chm13v2.0_SD.bed",
    "telomere": f"{_ANNOTATION}/chm13v2.0_telomere.bed",
    "gene": f"{_ANNOTATION}/chm13v2.0_GENCODEv35_CAT_Liftoff.vep.gff3.gz",
}

#: Approximate on-disk size of each track, for deciding whether a slice is
#: worth the trouble.  Measured, not guessed; only used for log messages and
#: for the "is this cheap enough to download whole" heuristic in ``fetch``.
T2T_TRACK_BYTES: Final[dict[str, int]] = {
    "censat": 194_081,
    "repeatmasker": 342_862_439,
    "segdup": 6_685_777,
    "telomere": 941,
    "gene": 158_318_946,
}

#: Contig names in ``chm13v2.0.fa``.  Already UCSC-style, so a reference row
#: needs no chromAlias file.
T2T_CHROMS: Final[tuple[str, ...]] = tuple(
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
)


def reference_manifest_row() -> dict[str, str]:
    """The manifest row for CHM13v2.0, as ``schemas.MANIFEST_COLUMNS``.

    ``chrom_alias`` is empty because the FASTA already uses ``chrN`` names, and
    ``sex`` is empty rather than "female": CHM13 itself is 46,XX but v2.0
    carries HG002's chrY, so neither label is honest.
    """
    return {
        "assembly": T2T_ASSEMBLY,
        "sample": T2T_SAMPLE,
        "haplotype": "ref",
        "source": "t2t",
        "fasta": T2T_FASTA,
        "fai": T2T_FAI,
        "gzi": T2T_GZI,
        "chrom_alias": "",
        "censat_bed": T2T_TRACKS["censat"],
        "repeatmasker_bed": T2T_TRACKS["repeatmasker"],
        "segdup_bed": T2T_TRACKS["segdup"],
        "population": "",
        "superpopulation": "",
        "sex": "",
    }


def verify_track_urls(*, timeout: float = 30.0) -> dict[str, bool]:
    """HEAD every reference URL; ``{name: reachable}``.

    Not called by the pipeline -- the URLs are verified at authoring time -- but
    it is the first thing to run when a run starts failing to fetch tracks.
    """
    targets = {"fasta": T2T_FASTA, "fai": T2T_FAI, **T2T_TRACKS}
    result = {name: url_exists(url, timeout=timeout) for name, url in targets.items()}
    for name, ok in result.items():
        if not ok:
            log.warning("T2T track %r is unreachable: %s", name, targets[name])
    return result
