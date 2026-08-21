"""Catalogs of real assemblies: HPRC release 2 and T2T-CHM13v2.0.

This subpackage is the pipeline's only contact with the outside world.  Every
URL, every quirk of a hand-maintained CSV and every "the segdup track lives
under a different name than you would guess" fact lives here, so the rest of
kmer-dust deals in manifests and local paths.

Typical use::

    from kmer_dust.catalog import build_manifest, write_manifest
    manifest = build_manifest(cfg, cache_dir)
    write_manifest(manifest, cfg.path("manifest.tsv"))
"""

from __future__ import annotations

from .hprc import (
    HPRC_DATA_TABLES_BASE,
    POPULATION_TO_SUPERPOPULATION,
    annotation_index,
    fetch_table,
    release2_index,
    s3_to_https,
    sample_metadata,
    t2t_sequence_table,
)
from .manifest import build_manifest, read_manifest, write_manifest
from .t2t import T2T_ASSEMBLY, T2T_FAI, T2T_FASTA, T2T_TRACKS, reference_manifest_row

__all__ = [
    "HPRC_DATA_TABLES_BASE",
    "POPULATION_TO_SUPERPOPULATION",
    "T2T_ASSEMBLY",
    "T2T_FAI",
    "T2T_FASTA",
    "T2T_TRACKS",
    "annotation_index",
    "build_manifest",
    "fetch_table",
    "read_manifest",
    "reference_manifest_row",
    "release2_index",
    "s3_to_https",
    "sample_metadata",
    "t2t_sequence_table",
    "write_manifest",
]
