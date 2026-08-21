"""``kmer-dust`` command line interface.

Every pipeline stage is its own subcommand taking the same ``--config`` YAML, so
the identical invocation works from a shell, from a Slurm array job and from a
Snakemake rule.  ``kmer-dust run`` chains them all in one process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import typer

from . import __version__
from .config import Config
from .log import get_logger, setup_logging
from .pipeline import STAGES, run_all, run_stage

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Without this, click refuses `kmer-dust --version` with "Missing command"
    # before the root callback ever runs.  `no_args_is_help` still makes a bare
    # invocation print help, so the only behaviour this changes is that root
    # options work on their own -- which is what `docker run <image> --version`
    # needs.
    invoke_without_command=True,
    help=(
        "Alignment-free clustering of genomic bins across HPRC assemblies and T2T-CHM13.\n\n"
        "Tile -> FracMinHash sketch -> bin x k-mer matrix -> randomized SVD -> UMAP -> HDBSCAN "
        "-> annotate -> back-propagate."
    ),
)

logger = get_logger("kmer_dust.cli")

_CONFIG_OPT = typer.Option(..., "--config", "-c", help="Run configuration YAML.", exists=True)
_FORCE_OPT = typer.Option(False, "--force", "-f", help="Recompute even if outputs exist.")


def _load(config: Path, overrides: list[str] | None = None) -> Config:
    cfg = Config.load(config)
    for override in overrides or []:
        if "=" not in override:
            raise typer.BadParameter(f"--set expects key=value, got {override!r}")
        key, _, raw = override.partition("=")
        _apply_override(cfg, key.strip(), raw.strip())
    cfg.validate()
    return cfg


def _apply_override(cfg: Config, dotted: str, raw: str) -> None:
    """``--set cluster.min_cluster_size=200`` without a config edit."""
    import yaml

    parts = dotted.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise typer.BadParameter(f"unknown config section {part!r} in {dotted!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise typer.BadParameter(f"unknown config key {dotted!r}")
    current = getattr(target, leaf)
    value = yaml.safe_load(raw)
    if isinstance(current, bool):
        value = bool(value)
    elif isinstance(current, int) and not isinstance(current, bool):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    elif isinstance(current, list) and not isinstance(value, list):
        value = [v.strip() for v in str(raw).split(",") if v.strip()]
    setattr(target, leaf, value)


@app.callback()
def _root(
    ctx: typer.Context,
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Repeat for DEBUG."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only."),
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    setup_logging(verbose, quiet=quiet)
    if version:
        typer.echo(f"kmer-dust {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        # invoke_without_command=True means we get here for a bare `kmer-dust`;
        # show the help that no_args_is_help would have shown.
        typer.echo(ctx.get_help())
        raise typer.Exit()


# --------------------------------------------------------------------------
# per-stage commands, generated so they can never drift from pipeline.STAGES
# --------------------------------------------------------------------------

_STAGE_HELP = {
    "manifest": "Resolve which assemblies take part and write manifest.tsv.",
    "sketch": "Tile every assembly into bins and FracMinHash their k-mers.",
    "select": "Choose the k-mer columns by cross-sample prevalence.",
    "matrix": "Assemble the sparse bin x k-mer matrix.",
    "decompose": "Randomized SVD of the matrix.",
    "embed": "UMAP embedding of the components.",
    "cluster": "HDBSCAN/DBSCAN over the embedding.",
    "annotate": "Overlay cenSat / RepeatMasker / segdup tracks onto every bin.",
    "enrich": "Cluster x feature enrichment and cluster naming.",
    "backprop": "Write per-assembly cluster BEDs and the label-transfer report.",
    "report": "Build the interactive HTML report.",
}


def _make_stage_command(stage: str):
    def _cmd(
        config: Path = _CONFIG_OPT,
        force: bool = _FORCE_OPT,
        set_: list[str] | None = typer.Option(
            None, "--set", "-s", help="Override a config value, e.g. -s embed.n_neighbors=15."
        ),
    ) -> None:
        cfg = _load(config, set_)
        result = run_stage(cfg, stage, force=force)
        typer.echo(json.dumps({"stage": result.name, "seconds": round(result.seconds, 2), **result.detail}))

    _cmd.__name__ = f"{stage}_cmd"
    _cmd.__doc__ = _STAGE_HELP[stage]
    return _cmd


for _stage in STAGES:
    if _stage == "sketch":
        continue  # sketch takes an extra flag; see below
    app.command(name=_stage, help=_STAGE_HELP[_stage])(_make_stage_command(_stage))


@app.command(name="sketch", help=_STAGE_HELP["sketch"])
def sketch_cmd(
    config: Path = _CONFIG_OPT,
    force: bool = _FORCE_OPT,
    assembly: list[str] | None = typer.Option(
        None,
        "--assembly",
        "-a",
        help=(
            "Sketch only this assembly (repeatable). This is the Slurm-array entry point: "
            "one task per assembly, each touching only its own shard."
        ),
    ),
    assembly_file: Path | None = typer.Option(
        None,
        "--assembly-file",
        help="File of assembly ids, one per line (blank lines and '#' comments ignored).",
    ),
    set_: list[str] | None = typer.Option(None, "--set", "-s", help="Override a config value."),
) -> None:
    cfg = _load(config, set_)
    wanted = list(assembly or [])
    if assembly_file:
        for line in assembly_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                wanted.append(line)
    result = run_stage(cfg, "sketch", force=force, assemblies=wanted or None)
    typer.echo(
        json.dumps({"stage": result.name, "seconds": round(result.seconds, 2), **result.detail})
    )


# --------------------------------------------------------------------------
# whole-run and utility commands
# --------------------------------------------------------------------------


@app.command()
def run(
    config: Path = _CONFIG_OPT,
    force: bool = _FORCE_OPT,
    force_from: str | None = typer.Option(
        None, "--force-from", help=f"Re-run this stage and everything after it ({'|'.join(STAGES)})."
    ),
    only: str | None = typer.Option(
        None, "--only", help="Comma-separated subset of stages, in pipeline order."
    ),
    set_: list[str] | None = typer.Option(None, "--set", "-s", help="Override a config value."),
) -> None:
    """Run every stage in one process."""
    cfg = _load(config, set_)
    stages = STAGES
    if only:
        wanted = [s.strip() for s in only.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in STAGES]
        if unknown:
            raise typer.BadParameter(f"unknown stage(s): {unknown}")
        stages = tuple(s for s in STAGES if s in wanted)
    if force_from and force_from not in STAGES:
        raise typer.BadParameter(f"unknown stage {force_from!r}")
    results = run_all(cfg, stages=stages, force=force, force_from=force_from)
    total = sum(r.seconds for r in results)
    typer.echo(
        json.dumps(
            {
                "run": cfg.run_name,
                "outdir": str(cfg.outdir),
                "seconds": round(total, 2),
                "stages": [{"name": r.name, "seconds": round(r.seconds, 2), **r.detail} for r in results],
            },
            indent=2,
        )
    )


@app.command()
def fetch(
    dest: Path = typer.Option(Path("data/testdata"), "--dest", help="Where to write the slices."),
    datadir: Path = typer.Option(Path("data"), "--datadir", help="Root for the catalog cache."),
    samples: int = typer.Option(4, "--samples", help="Number of HPRC samples to slice."),
    chrom: str = typer.Option("chr21", "--chrom", help="Chromosome to slice from each assembly."),
    span_mb: float = typer.Option(6.0, "--span-mb", help="Megabases per assembly."),
    offset_mb: float = typer.Option(6.0, "--offset-mb", help="Offset into the chromosome."),
    no_reference: bool = typer.Option(False, "--no-reference", help="Skip T2T-CHM13."),
    tracks: bool = typer.Option(False, "--tracks", help="Also pre-warm the big T2T annotation BEDs."),
    force: bool = _FORCE_OPT,
) -> None:
    """Download small slices of *real* HPRC and T2T assemblies for tests and demos."""
    from .fetch import fetch_reference_tracks, fetch_testdata

    cache_dir = datadir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fetch_testdata(
        dest,
        cache_dir,
        samples=samples,
        chrom=chrom,
        span_mb=span_mb,
        offset_mb=offset_mb,
        force=force,
        include_reference=not no_reference,
    )
    if tracks:
        fetch_reference_tracks(cache_dir, force=force)
    typer.echo(json.dumps({"manifest": str(manifest_path)}))


@app.command("init-config")
def init_config(
    output: Path = typer.Argument(..., help="Where to write the YAML."),
    preset: str = typer.Option("smoke", "--preset", help="smoke | chr21 | full"),
    force: bool = _FORCE_OPT,
) -> None:
    """Write a starter configuration file."""
    if output.exists() and not force:
        raise typer.BadParameter(f"{output} exists; pass --force to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    src = _find_preset(preset)
    if src is not None:
        output.write_text(src.read_text())
    else:
        # Installed without the repo or the image's preset directory alongside.
        # The dataclass defaults are a valid config, just an unopinionated one.
        logger.warning(
            "preset %r not found (looked in %s); writing the built-in defaults instead",
            preset,
            ", ".join(str(d) for d in _preset_dirs()),
        )
        Config().dump(output)
    typer.echo(str(output))


def _preset_dirs() -> list[Path]:
    """Where to look for the shipped preset YAMLs, most specific first.

    The presets live in ``workflow/config/`` in the repository and are copied to
    ``/opt/kmer-dust/presets`` in the container image; keeping one source and
    two lookup paths beats duplicating the files into the package.
    """
    dirs: list[Path] = []
    env = os.environ.get("KMER_DUST_PRESETS")
    if env:
        dirs.append(Path(env))
    dirs.append(Path(__file__).resolve().parents[2] / "workflow" / "config")
    dirs.append(Path.cwd() / "workflow" / "config")
    return dirs


def _find_preset(name: str) -> Path | None:
    for directory in _preset_dirs():
        candidate = directory / f"{name}.yaml"
        if candidate.is_file():
            return candidate
    return None


@app.command()
def info() -> None:
    """Print versions and the state of the optional accelerators."""
    payload: dict[str, object] = {"kmer_dust": __version__, "python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "pandas", "pyarrow", "pysam", "sklearn", "umap", "numba", "plotly"):
        try:
            payload[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001 - reporting, not control flow
            payload[mod] = f"unavailable ({type(exc).__name__})"
    try:
        from .hashing import NUMBA_AVAILABLE

        payload["numba_kernel"] = NUMBA_AVAILABLE
    except Exception:  # noqa: BLE001
        payload["numba_kernel"] = False
    typer.echo(json.dumps(payload, indent=2))


@app.command("validate-config")
def validate_config(config: Path = _CONFIG_OPT) -> None:
    """Load, validate and echo the fully-resolved configuration."""
    cfg = Config.load(config)
    typer.echo(json.dumps(cfg.to_dict(), indent=2, default=str))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
