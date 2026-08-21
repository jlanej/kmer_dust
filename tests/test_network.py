"""Live checks against the real HPRC release-2 and T2T-CHM13 files.

Opt-in (``KMER_DUST_TEST_NETWORK=1`` or ``-m network``); CI never runs these.
They exist because the whole remote-access design rests on one fact that is
cheap to verify and catastrophic to get wrong: the release-2 bgzip assemblies
publish ``.fa.gz.fai`` *and* ``.fa.gz.gzi`` next to them, so a 200 kb slice is
an HTTP range request, not a 3 GB download.  A regression that silently falls
back to streaming would still pass every offline test in this suite.

Note that the analysis-set bgzip CHM13 has a ``.gzi`` but no ``.fa.gz.fai``,
which is why the reference here is the *uncompressed* ``chm13v2.0.fa``.
"""

from __future__ import annotations

import numpy as np
import pytest

from kmer_dust.catalog import t2t
from kmer_dust.fasta import FastaSource, download, load_chrom_alias
from kmer_dust.hashing import encode_bases, max_hash_for_scaled, sketch_contig

pytestmark = pytest.mark.network

S3 = "https://s3-us-west-2.amazonaws.com/human-pangenomics"
HPRC_BASE = f"{S3}/working/HPRC/HG00408/assemblies/release2/HG00408_pat_hprc_r2_v1.0.1.fa.gz"
HPRC_CONTIG = "HG00408#1#CM085953.1"  # chr2 of this haplotype
SLICE = 200_000
K = 31


def sketch_stats(seq: bytes, scaled: int = 200):
    codes = encode_bases(seq)
    _, hashes = sketch_contig(
        codes, k=K, bin_size=10_000, max_hash=max_hash_for_scaled(scaled)
    )
    acgt = int((codes <= 3).sum())
    return acgt, hashes


def test_hprc_release2_remote_slice_sketches():
    src = FastaSource(HPRC_BASE, fai=HPRC_BASE + ".fai", gzi=HPRC_BASE + ".gzi")
    try:
        lengths = src.contig_lengths()
        assert HPRC_CONTIG in lengths
        assert lengths[HPRC_CONTIG] > 200_000_000  # chr2 is ~240 Mb
        seq = src.fetch(HPRC_CONTIG, 20_000_000, 20_000_000 + SLICE)
    finally:
        src.close()
    assert len(seq) == SLICE
    acgt, hashes = sketch_stats(seq)
    assert acgt > 0.95 * SLICE, "a random chr2 slice should be almost all ACGT"
    expected = (SLICE - K + 1) / 200
    assert 0.5 * expected < hashes.size < 2.0 * expected
    assert np.unique(hashes).size > 0.5 * hashes.size


def test_t2t_chm13_remote_slice_sketches():
    src = FastaSource(t2t.T2T_FASTA, fai=t2t.T2T_FAI)
    try:
        lengths = src.contig_lengths()
        assert "chr21" in lengths and "chrM" in lengths
        assert lengths["chrM"] == 16_569
        assert 45_000_000 < lengths["chr21"] < 47_000_000
        seq = src.fetch("chr21", 20_000_000, 20_000_000 + SLICE)
    finally:
        src.close()
    assert len(seq) == SLICE
    acgt, hashes = sketch_stats(seq)
    assert acgt > 0.95 * SLICE
    expected = (SLICE - K + 1) / 200
    assert 0.5 * expected < hashes.size < 2.0 * expected


def test_remote_slices_are_reproducible():
    """Two fetches of the same interval must give the same bytes."""
    src = FastaSource(t2t.T2T_FASTA, fai=t2t.T2T_FAI)
    try:
        a = src.fetch("chr21", 6_000_000, 6_010_000)
        b = src.fetch("chr21", 6_000_000, 6_010_000)
    finally:
        src.close()
    assert a == b
    assert len(a) == 10_000


def test_t2t_track_urls_resolve(tmp_path):
    for name, url in t2t.T2T_TRACKS.items():
        if not url or name in {"repeatmasker", "gene"}:
            continue  # those two are hundreds of MB; not worth a CI-less test
        dest = tmp_path / f"{name}{''.join(url[-8:])}"
        path = download(url, dest)
        assert path.exists()
        assert path.stat().st_size > 0


def test_hprc_chrom_alias_maps_contigs_to_chromosomes(cache_dir):
    from kmer_dust.catalog import hprc

    index = hprc.annotation_index("chrom_alias", cache_dir)
    row = index[index["assembly"] == "HG00408_pat_hprc_r2_v1.0.1"]
    assert len(row) == 1
    alias = load_chrom_alias(row.iloc[0]["url"], cache_dir)
    assert alias.get(HPRC_CONTIG) == "chr2"
    assert sum(1 for v in alias.values() if v.startswith("chr")) >= 20
