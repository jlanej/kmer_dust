"""Shared fixtures for the kmer-dust test suite.

Two kinds of test input live here.

*Tiny committed fixtures* (``tests/data/``) are hand-written files whose exact
content is part of a test's meaning -- a cenSat BED whose names exercise every
branch of the normalisation table, a chromAlias with a real PanSN contig name.
They are a few kB and are read-only; anything that would let pysam or a stage
write next to them (``.fai`` sidecars, ``.done`` markers) copies them into
``tmp_path`` first.

*Synthetic assemblies* are generated at test time from a fixed seed rather than
committed, because they need to be hundreds of kB and because the interesting
property is the structure we plant in them, not the bytes.  The builder makes a
small pangenome: one shared backbone mutated per haplotype, plus two "satellite"
regions tiled from a shared pool of monomer variants.  Backbone bins are similar
only to the *same* bin in other assemblies; satellite bins are similar to *every
other satellite bin anywhere*.  That is precisely the signal the whole pipeline
is supposed to find, so it is what the smoke test asserts on.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kmer_dust import schemas
from kmer_dust.config import Config

DATA_DIR = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# marker plumbing
# --------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Network tests are opt-in: CI must stay offline and deterministic."""
    if os.environ.get("KMER_DUST_TEST_NETWORK", "").lower() in {"1", "true", "yes"}:
        return
    if "network" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="set KMER_DUST_TEST_NETWORK=1 (or -m network) to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------
# sequence construction helpers
# --------------------------------------------------------------------------

_ALPHABET = np.frombuffer(b"ACGT", dtype=np.uint8)
_COMPLEMENT = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")

#: 171 bp is the alpha-satellite monomer length; using the real number keeps the
#: synthetic arrays honest about how many distinct k-mers a satellite bin holds.
MONOMER_LEN = 171
#: Distinct monomer variants in the shared pool.  Every satellite bin in every
#: assembly draws from this pool, which is what makes them mutually similar.
N_MONOMERS = 48
#: Half-open satellite intervals planted in every synthetic contig.
SAT_REGIONS: tuple[tuple[int, int], ...] = ((100_000, 140_000), (220_000, 250_000))
SYNTH_CONTIG_LEN = 300_000
SYNTH_CONTIG = "chr21"


def random_sequence(n: int, rng: np.random.Generator) -> bytes:
    return _ALPHABET[rng.integers(0, 4, n)].tobytes()


def revcomp(seq: str | bytes) -> str | bytes:
    if isinstance(seq, str):
        return seq.translate(str.maketrans("ACGTacgtNn", "TGCAtgcaNn"))[::-1]
    return seq.translate(_COMPLEMENT)[::-1]


def mutate(seq: bytes, rate: float, rng: np.random.Generator) -> bytes:
    """Substitute a fraction ``rate`` of positions; length is preserved."""
    arr = np.frombuffer(seq, dtype=np.uint8).copy()
    if rate <= 0.0 or arr.size == 0:
        return arr.tobytes()
    n_mut = rng.binomial(arr.size, rate)
    if n_mut == 0:
        return arr.tobytes()
    pos = rng.choice(arr.size, size=n_mut, replace=False)
    arr[pos] = _ALPHABET[rng.integers(0, 4, n_mut)]
    return arr.tobytes()


def monomer_pool(rng: np.random.Generator, n: int = N_MONOMERS) -> list[bytes]:
    """``n`` variants of one consensus monomer, ~4 % divergent from it."""
    consensus = random_sequence(MONOMER_LEN, rng)
    return [mutate(consensus, 0.04, rng) for _ in range(n)]


def satellite_array(pool: Sequence[bytes], length: int, rng: np.random.Generator) -> bytes:
    """Tile ``pool`` monomers in random order until ``length`` bases are covered."""
    n_units = length // MONOMER_LEN + 1
    order = rng.integers(0, len(pool), n_units)
    return b"".join(pool[i] for i in order)[:length]


def write_fasta(path: Path, records: Sequence[tuple[str, bytes]], width: int = 60) -> Path:
    with open(path, "wb") as handle:
        for name, seq in records:
            handle.write(b">" + name.encode() + b"\n")
            for i in range(0, len(seq), width):
                handle.write(seq[i : i + width] + b"\n")
    return path


# --------------------------------------------------------------------------
# synthetic pangenome
# --------------------------------------------------------------------------


def satellite_bin_indices(bin_size: int, contig_len: int = SYNTH_CONTIG_LEN) -> set[int]:
    """Bins *fully* inside a planted satellite region, in shard-local indices."""
    out: set[int] = set()
    for start, end in SAT_REGIONS:
        if end > contig_len:
            continue
        first = -(-start // bin_size)  # ceil: only bins entirely inside count
        last = end // bin_size  # exclusive
        out.update(range(first, last))
    return out


def build_synthetic_assemblies(
    outdir: Path,
    *,
    n_assemblies: int = 6,
    contig_len: int = SYNTH_CONTIG_LEN,
    seed: int = 17,
    snv_rate: float = 0.004,
    write_annotations: bool = True,
) -> pd.DataFrame:
    """Build a tiny multi-assembly "pangenome" and return its manifest.

    Every assembly is the same backbone with independent point mutations, so
    equivalent bins stay recognisably related; the two satellite regions are
    re-tiled independently from a *shared* monomer pool, so satellite bins are
    related to each other regardless of assembly or position.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = np.random.default_rng(seed)
    backbone = bytearray(random_sequence(contig_len, root))
    pool = monomer_pool(root)

    # A shorter contig simply has fewer planted regions; never let a slice
    # assignment past the end silently *extend* the sequence.
    regions = [(s, e) for s, e in SAT_REGIONS if e <= contig_len]

    rows: list[dict[str, str]] = []
    for i in range(n_assemblies):
        rng = np.random.default_rng([seed, i])
        seq = bytearray(mutate(bytes(backbone), snv_rate, rng))
        for start, end in regions:
            seq[start:end] = satellite_array(pool, end - start, rng)
        assert len(seq) == contig_len
        sample = f"SYN{i // 2:03d}"
        haplotype = "pat" if i % 2 == 0 else "mat"
        assembly = f"{sample}_{haplotype}_syn_v1"
        fasta = outdir / f"{assembly}.fa"
        write_fasta(fasta, [(SYNTH_CONTIG, bytes(seq))])

        alias = outdir / f"{assembly}.chromAlias.txt"
        alias.write_text(f"# assembly\tucsc\tgenbank\n{SYNTH_CONTIG}\t{SYNTH_CONTIG}\tsyn.1\n")

        censat = ""
        if write_annotations:
            bed = outdir / f"{assembly}.censat.bed"
            lines = ['track name="cenSat" description="synthetic" itemRgb="On"']
            for j, (start, end) in enumerate(regions, start=1):
                lines.append(
                    f"{SYNTH_CONTIG}\t{start}\t{end}\thor_1_{j}(S1C1H1L)"
                    f"\t100\t.\t{start}\t{end}\t255,146,0"
                )
            bed.write_text("\n".join(lines) + "\n")
            censat = str(bed)

        rows.append(
            {
                "assembly": assembly,
                "sample": sample,
                "haplotype": haplotype,
                "source": "local",
                "fasta": str(fasta),
                "fai": "",
                "gzi": "",
                "chrom_alias": str(alias),
                "censat_bed": censat,
                "repeatmasker_bed": "",
                "segdup_bed": "",
                "population": "SYN",
                "superpopulation": "EUR",
                "sex": "female",
            }
        )
    return schemas.enforce(pd.DataFrame(rows), schemas.MANIFEST_COLUMNS)


# --------------------------------------------------------------------------
# fake sketch shards -- for stages downstream of sketch that must be tested
# without running sketch itself
# --------------------------------------------------------------------------


def write_sketch_shard(
    outdir: Path,
    assembly: str,
    hashes_per_bin: Sequence[Sequence[int]],
    *,
    sample: str | None = None,
    haplotype: str = "pat",
    source: str = "local",
    contig: str = "chr21",
    chrom: str = "chr21",
    bin_size: int = 10_000,
) -> dict[str, Path]:
    """Write a hand-specified ``<assembly>.{bins,sketch}.parquet`` + ``.done``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample = sample if sample is not None else assembly.split("_")[0]
    bins = []
    pairs: list[tuple[int, int]] = []
    for bin_idx, hashes in enumerate(hashes_per_bin):
        start = bin_idx * bin_size
        bins.append(
            {
                "bin_idx": bin_idx,
                "bin_uid": schemas.bin_uid(assembly, contig, start),
                "assembly": assembly,
                "sample": sample,
                "haplotype": haplotype,
                "source": source,
                "contig": contig,
                "chrom": chrom,
                "placed": True,
                "start": start,
                "end": start + bin_size,
                "n_acgt": bin_size,
                "n_kmers": bin_size - 30,
                "n_sketch": len(hashes),
                "gc": 0.5,
                "nfrac": 0.0,
            }
        )
        pairs.extend((bin_idx, int(h)) for h in hashes)
    if bins:
        bins_df = schemas.enforce(pd.DataFrame(bins), schemas.BIN_COLUMNS)
    else:
        bins_df = schemas.empty_frame(schemas.BIN_COLUMNS)
    sketch_df = pd.DataFrame(pairs, columns=["bin_idx", "hash"])
    if sketch_df.empty:
        sketch_df = schemas.empty_frame(schemas.SKETCH_COLUMNS)
    else:
        sketch_df = schemas.enforce(sketch_df, schemas.SKETCH_COLUMNS)
        sketch_df = sketch_df.sort_values(["bin_idx", "hash"], kind="stable").reset_index(drop=True)
    paths = {
        "bins": outdir / f"{assembly}.bins.parquet",
        "sketch": outdir / f"{assembly}.sketch.parquet",
        "done": outdir / f"{assembly}.done",
    }
    bins_df.to_parquet(paths["bins"], index=False)
    sketch_df.to_parquet(paths["sketch"], index=False)
    paths["done"].write_text("ok\n")
    return paths


def manifest_from_shards(specs: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Minimal manifest for ``(assembly, sample)`` pairs written by the helper."""
    if not specs:
        return schemas.empty_frame(schemas.MANIFEST_COLUMNS)
    rows = []
    for i, (assembly, sample) in enumerate(specs):
        rows.append(
            {
                "assembly": assembly,
                "sample": sample,
                "haplotype": "pat" if i % 2 == 0 else "mat",
                "source": "local",
                "fasta": "",
                "fai": "",
                "gzi": "",
                "chrom_alias": "",
                "censat_bed": "",
                "repeatmasker_bed": "",
                "segdup_bed": "",
                "population": "",
                "superpopulation": "",
                "sex": "",
            }
        )
    return schemas.enforce(pd.DataFrame(rows), schemas.MANIFEST_COLUMNS)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def toy_fasta(tmp_path: Path) -> Path:
    """A writable copy -- pysam drops a ``.fai`` next to whatever it opens."""
    dest = tmp_path / "toy.fa"
    shutil.copyfile(DATA_DIR / "toy.fa", dest)
    return dest


@pytest.fixture
def toy_chrom_alias(tmp_path: Path) -> Path:
    dest = tmp_path / "toy.chromAlias.txt"
    shutil.copyfile(DATA_DIR / "toy.chromAlias.txt", dest)
    return dest


@pytest.fixture
def toy_censat_bed() -> Path:
    return DATA_DIR / "toy.censat.bed"


@pytest.fixture
def toy_repeatmasker_bed() -> Path:
    return DATA_DIR / "toy.repeatmasker.bed"


@pytest.fixture
def toy_segdup_bed() -> Path:
    return DATA_DIR / "toy.segdup.bed"


@pytest.fixture
def make_config(tmp_path: Path):
    """Factory for a small, fully-offline :class:`Config`.

    Defaults are chosen so every stage has enough rows to be non-degenerate on a
    six-assembly / 300 kb toy pangenome while still finishing in seconds.
    """

    def _make(**overrides) -> Config:
        data = {
            "run_name": "test",
            "outdir": str(tmp_path / "out"),
            "datadir": str(tmp_path / "data"),
            "threads": 1,
            "seed": 1234,
            "manifest": {
                "source": "file",
                "path": str(tmp_path / "manifest.tsv"),
                "chroms": [SYNTH_CONTIG],
                "include_reference": False,
                "require_annotations": [],
                "require_t2t_chrom": False,
            },
            "sketch": {
                "k": 31,
                "bin_size": 10_000,
                "scaled": 50,
                "min_bin_sketch": 5,
                "include_unplaced": True,
                "threads": 1,
            },
            "select": {
                # Six samples: a k-mer in all six is prevalence 1.0, and the
                # planted satellite k-mers are exactly those, so the default
                # 0.95 ceiling would throw away the signal we are testing.
                "min_sample_prevalence": 0.0,
                "max_sample_prevalence": 1.0,
                "min_bins": 2,
                "max_features": 0,
                "n_buckets": 4,
            },
            "matrix": {"weighting": "idf", "row_norm": "l2"},
            "decompose": {"n_components": 8, "n_iter": 4},
            "embed": {"n_neighbors": 5, "min_dist": 0.05, "n_components": 2},
            "cluster": {"min_cluster_size": 5, "min_samples": 2},
            "annotate": {"reference_tracks": [], "assembly_tracks": ["censat"]},
            "enrich": {"min_cluster_size": 3, "min_frac": 0.25, "top_features": 3},
            "report": {"embed_plotlyjs": False},
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key] = {**data[key], **value}
            else:
                data[key] = value
        return Config.from_dict(data)

    return _make


@pytest.fixture
def tiny_cfg(make_config) -> Config:
    return make_config()


@pytest.fixture
def synthetic_assemblies(tmp_path: Path):
    """Factory building the synthetic pangenome under ``tmp_path/assemblies``."""

    def _build(**kwargs) -> pd.DataFrame:
        return build_synthetic_assemblies(tmp_path / "assemblies", **kwargs)

    return _build


@pytest.fixture
def shard_writer(tmp_path: Path):
    """Factory writing hand-specified sketch shards into ``tmp_path/sketch``."""

    def _write(assembly: str, hashes_per_bin, **kwargs) -> dict[str, Path]:
        return write_sketch_shard(tmp_path / "sketch", assembly, hashes_per_bin, **kwargs)

    return _write


@pytest.fixture
def sketch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "sketch"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260820)


def block_network(mp: pytest.MonkeyPatch) -> None:
    """Make any HTTP call raise, so an "offline" test is provably offline."""
    import urllib.request

    def _forbidden(*args, **kwargs):  # pragma: no cover - only fires on a bug
        raise AssertionError("this test must not touch the network")

    mp.setattr(urllib.request, "urlopen", _forbidden)
    try:
        import requests

        mp.setattr(requests.Session, "request", _forbidden)
        mp.setattr(requests, "get", _forbidden)
    except ImportError:  # pragma: no cover
        pass


def smoke_config(base: Path, manifest_path: Path) -> Config:
    """The config the offline end-to-end test runs; see test_pipeline_smoke."""
    return Config.from_dict(
        {
            "run_name": "smoke",
            "outdir": str(base / "out"),
            "datadir": str(base / "data"),
            "threads": 1,
            "seed": 1234,
            "manifest": {
                "source": "file",
                "path": str(manifest_path),
                "chroms": [SYNTH_CONTIG],
                "include_reference": False,
                "require_annotations": [],
                "require_t2t_chrom": False,
            },
            "sketch": {
                "k": 31,
                "bin_size": 10_000,
                "scaled": 50,
                "min_bin_sketch": 5,
                "include_unplaced": True,
                "threads": 1,
            },
            # Six samples: the planted satellite k-mers are in all of them, so
            # the default 0.95 prevalence ceiling would delete the signal.
            "select": {
                "min_sample_prevalence": 0.0,
                "max_sample_prevalence": 1.0,
                "min_bins": 2,
                "max_features": 0,
                "n_buckets": 4,
            },
            "matrix": {"weighting": "idf", "row_norm": "l2"},
            "decompose": {"n_components": 8, "n_iter": 7},
            "embed": {"n_neighbors": 5, "min_dist": 0.05, "n_components": 2},
            "cluster": {"method": "hdbscan", "min_cluster_size": 5, "min_samples": 2},
            "annotate": {"reference_tracks": [], "assembly_tracks": ["censat"]},
            "enrich": {"min_cluster_size": 3, "min_frac": 0.25, "top_features": 3},
            # True on purpose: the report claims to work offline, and the
            # only way to test that claim is to build the inlined version.
            "report": {"embed_plotlyjs": True, "title": "smoke"},
        }
    )


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory: pytest.TempPathFactory):
    """Run the whole pipeline once, offline, on the synthetic pangenome.

    Session-scoped because it is the most expensive thing in the suite and both
    the end-to-end assertions and the report tests read the same run directory.
    """
    base = tmp_path_factory.mktemp("smoke")
    manifest = build_synthetic_assemblies(base / "assemblies", n_assemblies=6, seed=17)
    manifest_path = base / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    cfg = smoke_config(base, manifest_path)

    from kmer_dust import pipeline  # imported late: other tests must still collect

    with pytest.MonkeyPatch.context() as mp:
        block_network(mp)
        result = pipeline.run_all(cfg)
    return cfg, manifest, result


@pytest.fixture(autouse=True)
def _deterministic_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the thread counts that silently reorder floating-point reductions.

    The environment variables alone are not enough for *this* process: BLAS
    reads them when numpy is imported, which already happened before any
    fixture ran. They still matter, because the sketch stage's worker processes
    inherit them. To pin the parent as well we go through threadpoolctl, which
    sklearn already depends on and which changes the pools at runtime.

    Without this the end-to-end smoke test is genuinely load-sensitive: a
    different BLAS thread count reorders the reductions inside the randomized
    SVD, the components shift in the last bits, and a planted cluster sitting
    near the HDBSCAN threshold can land on either side of it. Observed once,
    while an unrelated 8-thread job was saturating the machine.
    """
    # NUMBA_NUM_THREADS is deliberately absent: numba refuses to change it
    # once a kernel has been compiled, which happens on the first sketch.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - sklearn pulls it in
        yield
        return
    with threadpool_limits(limits=1):
        yield
