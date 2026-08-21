"""Stage graph and the local (non-Slurm) runner.

The pipeline is a linear chain of restartable stages.  Each stage knows how to
load the outputs of the ones before it, so exactly the same code runs whether
the chain is executed in one process (``kmer-dust run``), as separate Slurm jobs
(``kmer-dust <stage>`` from ``hpc/run_stage.sbatch``), or under Snakemake.

Stage modules are imported lazily so that a partially installed checkout can
still import ``kmer_dust`` -- and so that ``kmer-dust --help`` does not pay for
numba, umap and plotly.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config
from .log import get_logger, timed

logger = get_logger(__name__)

#: Stage names in dependency order.
STAGES: tuple[str, ...] = (
    "manifest",
    "sketch",
    "select",
    "matrix",
    "decompose",
    "embed",
    "cluster",
    "annotate",
    "enrich",
    "backprop",
    "report",
)


@dataclass
class StageResult:
    name: str
    seconds: float
    detail: dict[str, Any]


class RunContext:
    """Lazily-loaded, memoised handles on every stage output of a run.

    A stage asks the context for what it needs; the context either hands back an
    object produced earlier in this process or reads it from disk.  That is what
    makes ``kmer-dust cluster --config run.yaml`` work as a standalone Slurm job.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cache: dict[str, Any] = {}
        self.out = Path(cfg.outdir)
        self.cache_dir = Path(cfg.datadir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

    # -- generic memoisation -------------------------------------------------

    def _get(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def set(self, key: str, value: Any) -> Any:
        self._cache[key] = value
        return value

    # -- typed accessors -----------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.out / "manifest.tsv"

    @property
    def manifest(self):
        from .catalog import manifest as manifest_mod

        return self._get("manifest", lambda: manifest_mod.read_manifest(self.manifest_path))

    @property
    def sketch_dir(self) -> Path:
        return self.cfg.stage_dir("sketch")

    @property
    def kmers(self):
        from . import select as select_mod

        return self._get("kmers", lambda: select_mod.load_kmers(self.cfg.stage_dir("kmers")))

    @property
    def matrix(self):
        from . import matrix as matrix_mod

        return self._get("matrix", lambda: matrix_mod.load_matrix(self.cfg.stage_dir("matrix")))[0]

    @property
    def rows(self):
        from . import matrix as matrix_mod

        return self._get("matrix", lambda: matrix_mod.load_matrix(self.cfg.stage_dir("matrix")))[1]

    @property
    def pcs(self):
        from . import decompose as decompose_mod

        return self._get("pcs", lambda: decompose_mod.load_pcs(self.cfg.stage_dir("decompose")))

    @property
    def embedding(self):
        from . import embed as embed_mod

        return self._get("embed", lambda: embed_mod.load_embedding(self.cfg.stage_dir("embed")))

    @property
    def clusters(self):
        from . import cluster as cluster_mod

        return self._get(
            "clusters", lambda: cluster_mod.load_clusters(self.cfg.stage_dir("cluster"))
        )

    @property
    def annotations(self):
        from . import annotate as annotate_mod

        return self._get(
            "annotations", lambda: annotate_mod.load_annotations(self.cfg.stage_dir("annotate"))
        )

    @property
    def enrichment(self):
        from . import enrich as enrich_mod

        return self._get("enrich", lambda: enrich_mod.load_enrichment(self.cfg.stage_dir("enrich")))


# --------------------------------------------------------------------------
# individual stages
# --------------------------------------------------------------------------


def stage_manifest(ctx: RunContext, *, force: bool = False) -> StageResult:
    from .catalog import manifest as manifest_mod

    cfg = ctx.cfg
    if ctx.manifest_path.exists() and not force:
        df = manifest_mod.read_manifest(ctx.manifest_path)
        logger.info("manifest already present with %d assemblies", len(df))
    else:
        df = manifest_mod.build_manifest(cfg, ctx.cache_dir, force=force)
        manifest_mod.write_manifest(df, ctx.manifest_path)
    ctx.set("manifest", df)
    n_samples = df["sample"].nunique()
    return StageResult(
        "manifest", 0.0, {"assemblies": int(len(df)), "samples": int(n_samples)}
    )


def stage_sketch(
    ctx: RunContext, *, force: bool = False, assemblies: Sequence[str] | None = None
) -> StageResult:
    """Sketch every assembly in the manifest, or just the ones named.

    ``assemblies`` is what makes a Slurm array possible: one task per assembly,
    each walking only its own shard.  Without it N parallel ``kmer-dust sketch``
    jobs would each iterate the whole manifest and race on the same outputs.
    Each task writes its own summary file so the shards cannot clobber one
    another; the aggregate is rebuilt from whatever summaries exist.
    """
    from . import sketch as sketch_mod

    manifest = ctx.manifest
    suffix = ""
    if assemblies:
        wanted = list(dict.fromkeys(assemblies))
        known = set(manifest["assembly"].astype(str))
        unknown = [a for a in wanted if a not in known]
        if unknown:
            raise ValueError(
                f"assembly/assemblies not in {ctx.manifest_path}: {', '.join(unknown)}"
            )
        manifest = manifest[manifest["assembly"].astype(str).isin(wanted)]
        logger.info("sketching %d of %d assemblies", len(manifest), len(ctx.manifest))
        suffix = "." + _shard_tag(wanted)

    summary = sketch_mod.sketch_manifest(
        manifest,
        ctx.cfg,
        threads=ctx.cfg.sketch.threads or ctx.cfg.threads,
        force=force,
        cache_dir=ctx.cache_dir,
    )
    summary.to_csv(ctx.sketch_dir / f"sketch_summary{suffix}.tsv", sep="\t", index=False)
    failed = summary[summary["status"] != "ok"] if "status" in summary else summary.iloc[:0]
    if len(failed):
        logger.error("%d assemblies failed to sketch:\n%s", len(failed), failed.to_string())
    return StageResult(
        "sketch",
        0.0,
        {
            "shards": int(len(summary)),
            "failed": int(len(failed)),
            "bins": int(summary.get("n_bins", 0).sum()) if len(summary) else 0,
            "hashes": int(summary.get("n_hashes", 0).sum()) if len(summary) else 0,
        },
    )


def stage_select(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import select as select_mod

    kmers = select_mod.select_kmers(
        ctx.sketch_dir, ctx.manifest, ctx.cfg, ctx.cfg.stage_dir("kmers"), force=force
    )
    ctx.set("kmers", kmers)
    return StageResult("select", 0.0, {"features": int(len(kmers))})


def stage_matrix(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import matrix as matrix_mod

    mat, rows = matrix_mod.build_matrix(
        ctx.sketch_dir, ctx.kmers, ctx.manifest, ctx.cfg, ctx.cfg.stage_dir("matrix"), force=force
    )
    ctx.set("matrix", (mat, rows))
    return StageResult(
        "matrix",
        0.0,
        {"rows": int(mat.shape[0]), "cols": int(mat.shape[1]), "nnz": int(mat.nnz)},
    )


def stage_decompose(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import decompose as decompose_mod

    pcs = decompose_mod.decompose(
        ctx.matrix, ctx.cfg, ctx.cfg.stage_dir("decompose"), force=force
    )
    ctx.set("pcs", pcs)
    return StageResult("decompose", 0.0, {"components": int(pcs.shape[1])})


def stage_embed(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import embed as embed_mod

    emb = embed_mod.embed(ctx.pcs, ctx.cfg, ctx.cfg.stage_dir("embed"), force=force)
    ctx.set("embed", emb)
    return StageResult("embed", 0.0, {"dims": int(emb.shape[1])})


def stage_cluster(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import cluster as cluster_mod

    coords = ctx.embedding if ctx.cfg.cluster.space == "embedding" else ctx.pcs
    clusters = cluster_mod.cluster(
        coords, ctx.rows, ctx.cfg, ctx.cfg.stage_dir("cluster"), force=force
    )
    ctx.set("clusters", clusters)
    labels = clusters["cluster"]
    n_clusters = int((labels.unique() >= 0).sum())
    noise = float((labels < 0).mean()) if len(labels) else 0.0
    return StageResult("cluster", 0.0, {"clusters": n_clusters, "noise_frac": round(noise, 4)})


def stage_annotate(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import annotate as annotate_mod

    ann = annotate_mod.annotate_bins(
        ctx.rows,
        ctx.manifest,
        ctx.cfg,
        ctx.cfg.stage_dir("annotate"),
        cache_dir=ctx.cache_dir,
        force=force,
    )
    ctx.set("annotations", ann)
    annotated = int(ann["annotated"].sum()) if "annotated" in ann else 0
    return StageResult("annotate", 0.0, {"rows": int(len(ann)), "annotated": annotated})


def stage_enrich(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import enrich as enrich_mod

    enrichment, names = enrich_mod.enrich_clusters(
        ctx.rows,
        ctx.clusters,
        ctx.annotations,
        ctx.cfg,
        ctx.cfg.stage_dir("enrich"),
        force=force,
    )
    ctx.set("enrich", (enrichment, names))
    return StageResult("enrich", 0.0, {"tests": int(len(enrichment)), "named": int(len(names))})


def stage_backprop(ctx: RunContext, *, force: bool = False) -> StageResult:
    from . import backprop as backprop_mod

    _enrichment, names = ctx.enrichment
    outdir = ctx.cfg.stage_dir("backprop")
    paths = backprop_mod.write_cluster_beds(
        ctx.rows, ctx.clusters, names, ctx.cfg, outdir, force=force
    )
    transfer = backprop_mod.cluster_transfer_report(
        ctx.rows, ctx.clusters, ctx.annotations, names, ctx.cfg, outdir
    )
    return StageResult(
        "backprop", 0.0, {"beds": len(paths), "clusters_compared": int(len(transfer))}
    )


def stage_report(ctx: RunContext, *, force: bool = False) -> StageResult:
    from .viz import report as report_mod

    path = report_mod.build_report(ctx.cfg, ctx.out, force=force)
    return StageResult("report", 0.0, {"path": str(path)})


STAGE_FUNCS: dict[str, Callable[..., StageResult]] = {
    "manifest": stage_manifest,
    "sketch": stage_sketch,
    "select": stage_select,
    "matrix": stage_matrix,
    "decompose": stage_decompose,
    "embed": stage_embed,
    "cluster": stage_cluster,
    "annotate": stage_annotate,
    "enrich": stage_enrich,
    "backprop": stage_backprop,
    "report": stage_report,
}


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------


def _shard_tag(assemblies: Sequence[str]) -> str:
    """Short, stable filename tag for a subset of assemblies."""
    if len(assemblies) == 1:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(assemblies[0]))
        return safe[:80]
    digest = hashlib.sha1("\n".join(sorted(assemblies)).encode()).hexdigest()[:10]
    return f"{len(assemblies)}shards-{digest}"


def run_stage(
    cfg: Config,
    stage: str,
    *,
    force: bool = False,
    ctx: RunContext | None = None,
    **kwargs: Any,
):
    if stage not in STAGE_FUNCS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    ctx = ctx or RunContext(cfg)
    start = time.perf_counter()
    with timed(logger, f"stage {stage}"):
        result = STAGE_FUNCS[stage](ctx, force=force, **kwargs)
    result.seconds = time.perf_counter() - start
    return result


def run_all(
    cfg: Config,
    *,
    stages: tuple[str, ...] = STAGES,
    force: bool = False,
    force_from: str | None = None,
) -> list[StageResult]:
    """Run the chain in one process, writing ``run_summary.json`` as it goes.

    ``force_from`` re-runs that stage and everything after it, which is the
    knob you actually want when re-tuning clustering without re-sketching 3 Gb
    of sequence.
    """
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.dump(cfg.path("config.resolved.yaml"))
    ctx = RunContext(cfg)
    forcing = force
    results: list[StageResult] = []
    for stage in stages:
        if force_from and stage == force_from:
            forcing = True
        result = run_stage(cfg, stage, force=forcing, ctx=ctx)
        results.append(result)
        logger.info("[%s] %.1fs %s", result.name, result.seconds, json.dumps(result.detail))
        _write_run_summary(cfg, results)
    return results


def _write_run_summary(cfg: Config, results: list[StageResult]) -> None:
    payload = {
        "run_name": cfg.run_name,
        "outdir": str(cfg.outdir),
        "stages": [
            {"name": r.name, "seconds": round(r.seconds, 3), **r.detail} for r in results
        ],
    }
    path = cfg.path("run_summary.json")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
