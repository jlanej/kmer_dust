"""FASTA access, block iteration and chromosome-name normalisation.

The block-boundary test is the important one here.  ``sketch`` streams a contig
in ``block``-sized pieces and stitches k-mers across the seams by carrying the
last ``k-1`` bases forward; if ``iter_contigs`` ever overlaps, gaps or reorders
its output, the sketch of a contig silently depends on the block size and no two
runs are comparable.  So we sketch one contig at seven different block sizes and
demand a byte-identical hash stream.
"""

from __future__ import annotations

import gzip
import shutil

import numpy as np
import pytest
from conftest import random_sequence, write_fasta

from kmer_dust.fasta import (
    FastaSource,
    download,
    load_chrom_alias,
    normalize_chrom,
    open_text,
)
from kmer_dust.hashing import encode_bases, max_hash_for_scaled, sketch_contig

TOY_LENGTHS = {
    "HG00408#1#CM085953.1": 1200,
    "chr21": 1200,
    "chr22": 600,
    "chrN_all": 400,
    "tiny_contig": 20,
    "scaffold_00007": 500,
    "chrX": 345,
}


# --------------------------------------------------------------------------
# FastaSource -- plain local FASTA
# --------------------------------------------------------------------------


def test_contigs_are_reported_in_file_order(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        assert src.contigs == list(TOY_LENGTHS)
    finally:
        src.close()


def test_contig_lengths(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        assert src.contig_lengths() == TOY_LENGTHS
    finally:
        src.close()


def test_fetch_whole_and_slice(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        whole = src.fetch("chr21")
        assert isinstance(whole, bytes)
        assert len(whole) == 1200
        assert set(whole.upper()) <= set(b"ACGTN")
        assert src.fetch("chr21", 100, 200) == whole[100:200]
        assert src.fetch("chr21", 1100) == whole[1100:]
        assert src.fetch("chr21", 0, 1200) == whole
    finally:
        src.close()


def test_fetch_preserves_soft_masking(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        seq = src.fetch("chr22")
        assert any(chr(b).islower() for b in seq), "lower-case bases must survive the read"
    finally:
        src.close()


def test_fetch_of_an_all_n_contig(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        assert src.fetch("chrN_all").upper() == b"N" * 400
    finally:
        src.close()


def test_fetch_unknown_contig_raises(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        with pytest.raises((KeyError, ValueError)):
            src.fetch("no_such_contig")
    finally:
        src.close()


def test_empty_fasta_has_no_contigs(tmp_path):
    path = tmp_path / "empty.fa"
    path.write_bytes(b"")
    src = FastaSource(str(path))
    try:
        assert src.contigs == []
        assert src.contig_lengths() == {}
        assert list(src.iter_contigs()) == []
    finally:
        src.close()


# --------------------------------------------------------------------------
# iter_contigs
# --------------------------------------------------------------------------


def test_iter_contigs_yields_all_contigs_by_default(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        seen: dict[str, list[bytes]] = {}
        lengths: dict[str, int] = {}
        for name, length, block in src.iter_contigs(block=1_000_000):
            seen.setdefault(name, []).append(block)
            lengths[name] = length
        assert set(seen) == set(TOY_LENGTHS)
        assert lengths == TOY_LENGTHS
        for name, blocks in seen.items():
            assert b"".join(blocks) == src.fetch(name)
    finally:
        src.close()


def test_iter_contigs_restricts_to_the_requested_contigs(toy_fasta):
    src = FastaSource(str(toy_fasta))
    try:
        names = {name for name, _, _ in src.iter_contigs(["chr21", "chrX"])}
        assert names == {"chr21", "chrX"}
        assert list(src.iter_contigs([])) == []
    finally:
        src.close()


@pytest.mark.parametrize("block", [1, 7, 31, 97, 512, 1199, 1200, 5000])
def test_iter_contigs_blocks_are_exact_contiguous_and_non_overlapping(toy_fasta, block):
    src = FastaSource(str(toy_fasta))
    try:
        whole = src.fetch("chr21")
        blocks = [b for _, _, b in src.iter_contigs(["chr21"], block=block)]
        assert b"".join(blocks) == whole
        assert all(len(b) == block for b in blocks[:-1])
        assert 0 < len(blocks[-1]) <= block
        assert sum(len(b) for b in blocks) == len(whole)
    finally:
        src.close()


def _streaming_sketch(src, contig, *, k, bin_size, max_hash, block):
    """Sketch a contig from ``iter_contigs``, stitching across seams.

    This is exactly the algorithm ``sketch.py`` must implement: carry the last
    ``k-1`` bases into the next chunk, and attribute each k-mer to the bin
    holding its absolute first base.
    """
    tail = b""
    offset = 0
    starts: list[np.ndarray] = []
    hashes: list[np.ndarray] = []
    for _, _, blk in src.iter_contigs([contig], block=block):
        chunk = tail + blk
        chunk_origin = offset - len(tail)
        local, h = sketch_contig(encode_bases(chunk), k=k, bin_size=1, max_hash=max_hash)
        if local.size:
            starts.append(local.astype(np.int64) + chunk_origin)
            hashes.append(h)
        offset += len(blk)
        # Carrying the tail of *chunk* (not of blk) keeps this correct even when
        # a block is shorter than k-1 bases.
        tail = chunk[-(k - 1) :] if k > 1 else b""
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)
    return np.concatenate(starts) // bin_size, np.concatenate(hashes)


@pytest.mark.parametrize("block", [37, 101, 256, 999, 1200, 100_000])
def test_block_boundary_equivalence(toy_fasta, block):
    """One block or seven: the stitched hash stream must be identical."""
    k, bin_size = 31, 500
    max_hash = max_hash_for_scaled(4)
    src = FastaSource(str(toy_fasta))
    try:
        ref_bins, ref_hashes = sketch_contig(
            encode_bases(src.fetch("chr21")), k=k, bin_size=bin_size, max_hash=max_hash
        )
        bins, hashes = _streaming_sketch(
            src, "chr21", k=k, bin_size=bin_size, max_hash=max_hash, block=block
        )
    finally:
        src.close()
    assert hashes.size == ref_hashes.size
    assert np.array_equal(hashes, ref_hashes)
    assert np.array_equal(bins.astype(np.int32), ref_bins)


def test_block_boundary_equivalence_across_an_n_run(toy_fasta):
    """The seam must not resurrect k-mers that span an N run."""
    k, bin_size = 31, 100
    max_hash = max_hash_for_scaled(1)
    src = FastaSource(str(toy_fasta))
    try:
        whole = src.fetch("chrX")
        ref_bins, ref_hashes = sketch_contig(
            encode_bases(whole), k=k, bin_size=bin_size, max_hash=max_hash
        )
        for block in (40, 119, 121, 128):
            bins, hashes = _streaming_sketch(
                src, "chrX", k=k, bin_size=bin_size, max_hash=max_hash, block=block
            )
            assert np.array_equal(hashes, ref_hashes), block
            assert np.array_equal(bins.astype(np.int32), ref_bins), block
    finally:
        src.close()


# --------------------------------------------------------------------------
# indexed / compressed inputs
# --------------------------------------------------------------------------


def test_bgzip_fasta_with_index(tmp_path, toy_fasta):
    pysam = pytest.importorskip("pysam")
    bgz = tmp_path / "toy.bgz.fa.gz"
    pysam.tabix_compress(str(toy_fasta), str(bgz), force=True)
    pysam.faidx(str(bgz))
    src = FastaSource(str(bgz))
    try:
        assert src.contig_lengths() == TOY_LENGTHS
        plain = FastaSource(str(toy_fasta))
        try:
            assert src.fetch("chr21") == plain.fetch("chr21")
            assert src.fetch("chrX", 50, 150) == plain.fetch("chrX", 50, 150)
        finally:
            plain.close()
    finally:
        src.close()


def test_explicit_fai_argument_is_honoured(tmp_path, toy_fasta):
    pysam = pytest.importorskip("pysam")
    pysam.faidx(str(toy_fasta))
    fai = toy_fasta.with_suffix(".fa.fai")
    assert fai.exists()
    moved = tmp_path / "elsewhere.fai"
    shutil.copyfile(fai, moved)
    src = FastaSource(str(toy_fasta), fai=str(moved))
    try:
        assert src.contig_lengths() == TOY_LENGTHS
    finally:
        src.close()


# --------------------------------------------------------------------------
# open_text / download
# --------------------------------------------------------------------------


def test_open_text_reads_a_plain_file(tmp_path):
    path = tmp_path / "plain.txt"
    path.write_text("a\nb\n")
    with open_text(str(path)) as handle:
        assert handle.read() == "a\nb\n"


def test_open_text_transparently_decompresses_gzip(tmp_path):
    path = tmp_path / "gz.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("x\ty\n")
    with open_text(str(path)) as handle:
        assert handle.read() == "x\ty\n"


def test_open_text_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        open_text(str(tmp_path / "nope.txt"))


def test_download_reuses_an_existing_destination(tmp_path):
    """No network: an existing destination must short-circuit the fetch."""
    dest = tmp_path / "already.txt"
    dest.write_text("cached\n")
    out = download("https://example.invalid/already.txt", dest)
    assert out == dest
    assert dest.read_text() == "cached\n"


# --------------------------------------------------------------------------
# chrom alias / normalisation
# --------------------------------------------------------------------------


def test_load_chrom_alias_skips_the_comment_header(toy_chrom_alias):
    alias = load_chrom_alias(str(toy_chrom_alias))
    assert alias["HG00408#1#CM085953.1"] == "chr2"
    assert alias["chr21"] == "chr21"
    assert alias["chrX"] == "chrX"
    assert not any(key.startswith("#") for key in alias)
    # a contig with no UCSC name must not claim one
    assert alias.get("scaffold_00007", "") == ""


def test_load_chrom_alias_missing_file_is_an_empty_mapping(tmp_path):
    """A manifest row may name an alias file that was never downloaded."""
    assert load_chrom_alias(str(tmp_path / "absent.txt")) == {}
    assert load_chrom_alias("") == {}


NORMALIZE_CASES = [
    ("chr1", "chr1"),
    ("1", "chr1"),
    ("chr21", "chr21"),
    ("21", "chr21"),
    ("chr22", "chr22"),
    ("chrX", "chrX"),
    ("X", "chrX"),
    ("chrY", "chrY"),
    ("Y", "chrY"),
    ("chrM", "chrM"),
    ("M", "chrM"),
    ("MT", "chrM"),
    ("chrMT", "chrM"),
    # unplaced / unresolvable -> ""
    ("", ""),
    ("scaffold_00007", ""),
    ("JAGYYT010000042.1", ""),
    ("chrUn_JTFH01000001v1", ""),
    ("chr1_KI270706v1_random", ""),
    ("chrEBV", ""),
    ("chr0", ""),
    ("chr23", ""),
]


@pytest.mark.parametrize("raw,expected", NORMALIZE_CASES, ids=[c[0] or "empty" for c in NORMALIZE_CASES])
def test_normalize_chrom(raw, expected):
    assert normalize_chrom(raw) == expected


@pytest.mark.parametrize("raw,expected", [("HG00408#1#chr21", "chr21"), ("CHM13#0#chrX", "chrX")])
def test_normalize_chrom_strips_a_pansn_prefix(raw, expected):
    # EXTENSION beyond docs/API.md, deliberately left failing as a request:
    # HPRC release-2 contigs are PanSN over *accessions* (HG00408#1#CM085953.1),
    # which genuinely cannot be resolved without the chromAlias -- but PanSN
    # over chromosome names is common in pangenome FASTAs (minigraph-cactus,
    # CHM13 in a graph) and costs one rsplit("#", 1) to support.  Without it,
    # such an assembly needs a chromAlias file to contribute any bins at all.
    assert normalize_chrom(raw) == expected


def test_normalize_chrom_never_guesses_a_wrong_chromosome():
    """The safety property that must hold whatever is decided above."""
    for raw in ("HG00408#1#CM085953.1", "CM085953.1", "JAGYYT010000042.1", "scaffold_7"):
        assert normalize_chrom(raw) in {"", "chr2"}


def test_normalize_chrom_is_idempotent():
    for name in ("chr1", "chrX", "chrM", ""):
        assert normalize_chrom(normalize_chrom(name)) == normalize_chrom(name)


def test_normalize_chrom_of_a_bare_genbank_accession():
    # CM085953.1 is chr2 of HG00408 hap1, but only a per-assembly chromAlias can
    # say so.  Either behaviour is defensible; what must never happen is a wrong
    # chromosome.  (Contract not pinned by docs/API.md -- confirm with the
    # fasta.py author.)
    assert normalize_chrom("CM085953.1") in {"", "chr2"}


def test_normalize_chrom_beats_a_synthetic_contig_name(tmp_path):
    """A contig literally named chr21 needs no alias table to be recognised."""
    seq = random_sequence(200, np.random.default_rng(0))
    path = write_fasta(tmp_path / "syn.fa", [("chr21", seq)])
    src = FastaSource(str(path))
    try:
        assert [normalize_chrom(c) for c in src.contigs] == ["chr21"]
    finally:
        src.close()


# --------------------------------------------------------------------------
# UCSC placement: the distinction that decides whether ~1 Gb per haplotype is
# analysed or silently discarded.
# --------------------------------------------------------------------------

PLACEMENT_CASES = [
    # localised chromosomes: real coordinates
    ("chr13", ("chr13", True)),
    ("chr1", ("chr1", True)),
    ("chrX", ("chrX", True)),
    ("chrM", ("chrM", True)),
    # chromosome known, position within it not -- coordinates are contig-local
    ("chr13_JBHIKM010000006.1_random", ("chr13", False)),
    ("chr21_JBHIKM010000017.1_random", ("chr21", False)),
    ("chr1_KI270706v1_random", ("chr1", False)),
    ("chrX_KI270880v1_alt", ("chrX", False)),
    # chromosome genuinely unknown
    ("chrUn_JBHIKM010000019.1", ("", False)),
    ("chrUn_GL000195v1", ("", False)),
    # not a chromosome name at all
    ("scaffold_00007", ("", False)),
    ("JAGYYT010000042.1", ("", False)),
    ("", ("", False)),
    ("chrEBV", ("", False)),
]


@pytest.mark.parametrize("raw,expected", PLACEMENT_CASES, ids=[c[0] or "empty" for c in PLACEMENT_CASES])
def test_parse_ucsc_placement(raw, expected):
    from kmer_dust.fasta import parse_ucsc_placement

    assert parse_ucsc_placement(raw) == expected


def test_a_random_contig_keeps_its_chromosome_but_not_its_coordinates():
    """The whole point of the second element.

    `chr13_..._random` really is chr13 -- the assembler said so -- so it belongs
    in a chr13 run. But its `start` is an offset into that contig, not into
    chr13, so anything reasoning about genomic position has to be able to tell
    the two apart.
    """
    from kmer_dust.fasta import parse_ucsc_placement

    chrom, placed = parse_ucsc_placement("chr13_JBHIKM010000006.1_random")
    assert chrom == "chr13", "the chromosome assignment must survive"
    assert placed is False, "but it must not claim chromosome coordinates"


def test_placement_never_invents_a_chromosome():
    """The safety property: a bare accession still resolves to nothing."""
    from kmer_dust.fasta import parse_ucsc_placement

    for raw in ("CM085953.1", "HG00408#1#CM085953.1", "JAGYYT010000042.1", "scaffold_7"):
        assert parse_ucsc_placement(raw) == ("", False)


def test_placement_agrees_with_normalize_chrom_on_clean_names():
    from kmer_dust.fasta import normalize_chrom, parse_ucsc_placement

    for raw in ("chr1", "chr22", "chrX", "chrY", "chrM", "1", "X", "MT"):
        chrom, placed = parse_ucsc_placement(raw)
        assert chrom == normalize_chrom(raw)
        assert placed is bool(chrom)
