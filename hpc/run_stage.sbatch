#!/usr/bin/env bash
#
# Run exactly one kmer-dust stage as a Slurm job, without Snakemake.
#
#   sbatch hpc/run_stage.sbatch manifest workflow/config/chr21.yaml
#   sbatch --array=0-59%20 hpc/run_stage.sbatch sketch workflow/config/chr21.yaml
#   sbatch --export=ALL,SIF=/scratch/$USER/kmer-dust.sif hpc/run_stage.sbatch select ...
#
# For people who would rather drive the pipeline by hand.  Stages are ordered:
#
#   manifest -> sketch -> select -> matrix -> decompose -> embed -> cluster
#            -> annotate -> enrich -> backprop -> report
#
# `annotate` needs only `matrix`, so it can run alongside decompose/embed/
# cluster; everything else is strictly sequential.  Every stage is restartable
# and skips work whose output already exists, so re-submitting a stage after a
# walltime kill resumes rather than restarts.
#
# ---------------------------------------------------------------------------
# EDIT BEFORE FIRST USE
# ---------------------------------------------------------------------------
#   * --account / --partition below (or delete if your site has defaults).
#   * --cpus-per-task / --mem / --time: the defaults here are sized for a
#     single-chromosome run (chr21.yaml). For full.yaml, override per stage on
#     the command line, e.g.
#       sbatch --mem=256G --time=24:00:00 hpc/run_stage.sbatch decompose ...
#     Snakemake does this scaling for you; that is the argument for using it.
#   * The site-setup block: module loads / venv activation.
# ---------------------------------------------------------------------------

#SBATCH --job-name=kmer-dust-stage
#SBATCH --output=slurm-logs/%x-%A_%a.out
#SBATCH --error=slurm-logs/%x-%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
##SBATCH --account=CHANGE_ME
##SBATCH --partition=CHANGE_ME

set -euo pipefail

STAGES="manifest sketch select matrix decompose embed cluster annotate enrich backprop report run"

STAGE="${1:-${STAGE:-}}"
CONFIG="${2:-${CONFIG:-workflow/config/chr21.yaml}}"
SIF="${SIF:-}"
ASSEMBLY="${ASSEMBLY:-}"
EXTRA_ARGS=("${@:3}")

die() {
    printf 'run_stage: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Run one kmer-dust stage as a Slurm job.

Usage: sbatch hpc/run_stage.sbatch STAGE [CONFIG] [extra kmer-dust args...]

  STAGE    manifest|sketch|select|matrix|decompose|embed|cluster|
           annotate|enrich|backprop|report|run
  CONFIG   run config (default: $CONFIG, else workflow/config/chr21.yaml)

Environment (pass with --export=ALL,NAME=value):
  SIF              path to a .sif; the stage then runs via `apptainer exec`
  APPTAINER_BIND   extra bind mounts, e.g. "/scratch,/projects"
  ASSEMBLY         sketch only: one assembly id instead of all of them
  KMER_DUST_ENV    a file to `source` to put kmer-dust on PATH

Array jobs: with --array set and STAGE=sketch, task N sketches the Nth
assembly of <outdir>/manifest.tsv, so `manifest` must have run first. Size the
array from the manifest:  --array=0-$(( $(wc -l < results/chr21/manifest.tsv) - 2 ))

See hpc/README.md for a full walkthrough.
USAGE
}

case "$STAGE" in
    -h|--help|"") usage; [ -z "$STAGE" ] && exit 1; exit 0 ;;
esac

# shellcheck disable=SC2076  # a literal substring test is exactly what is wanted
case " $STAGES " in
    *" $STAGE "*) ;;
    *) die "unknown stage '$STAGE'; expected one of: $STAGES" ;;
esac

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm-logs

# ---------------------------------------------------------------------------
# Site setup -- EDIT THIS BLOCK (see hpc/submit.sbatch for the rationale)
# ---------------------------------------------------------------------------
if [ -n "${KMER_DUST_ENV:-}" ]; then
    # shellcheck disable=SC1090  # a site-supplied path, unknowable here
    source "$KMER_DUST_ENV"
fi
# ---------------------------------------------------------------------------

[ -f "$CONFIG" ] || die "no such config: $CONFIG (run from the repo root)"

# Give the numerical libraries exactly the cpus Slurm allocated. Without this
# OpenBLAS starts one thread per physical core on the node and a 8-cpu job
# thrashes a 128-core machine.
THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
# $HOME is often read-only or absent on compute nodes; keep every cache local.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${TMPDIR:-/tmp}/numba-$USER}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-$USER}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

# Build the command: either the container or whatever is on PATH.
if [ -n "$SIF" ]; then
    [ -f "$SIF" ] || die "SIF=$SIF does not exist (build it with hpc/build_sif.sh)"
    command -v apptainer >/dev/null 2>&1 && RUNTIME=apptainer || RUNTIME=singularity
    command -v "$RUNTIME" >/dev/null 2>&1 || die "neither apptainer nor singularity on PATH"
    RUN=("$RUNTIME" exec)
    # $PWD, $HOME and /tmp are bound automatically; a scratch filesystem is not.
    [ -n "${APPTAINER_BIND:-}" ] && RUN+=(--bind "$APPTAINER_BIND")
    RUN+=("$SIF" kmer-dust)
elif [ -n "${KMER_DUST_BIN:-}" ]; then
    read -r -a RUN <<< "$KMER_DUST_BIN"
elif command -v kmer-dust >/dev/null 2>&1; then
    RUN=(kmer-dust)
else
    die "kmer-dust is not on PATH and SIF is unset; edit the site-setup block in $0"
fi

STAGE_ARGS=("$STAGE" --config "$CONFIG")

# ---------------------------------------------------------------------------
# Array-job scatter over the sketch stage
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "$STAGE" = "sketch" ] && [ -z "$ASSEMBLY" ]; then
    OUTDIR="$(sed -n 's/^outdir:[[:space:]]*//p' "$CONFIG" | head -1 | tr -d '"'"'")"
    [ -n "$OUTDIR" ] || die "could not read outdir from $CONFIG"
    MANIFEST="$OUTDIR/manifest.tsv"
    [ -s "$MANIFEST" ] || die "$MANIFEST is missing -- run the manifest stage first"
    # +2 skips the header and converts the 0-based task id to a 1-based line.
    ASSEMBLY="$(awk -F'\t' -v n="$SLURM_ARRAY_TASK_ID" 'NR == n + 2 {print $1}' "$MANIFEST")"
    [ -n "$ASSEMBLY" ] || die "array task $SLURM_ARRAY_TASK_ID is past the end of $MANIFEST"
    printf 'run_stage: array task %s -> %s\n' "$SLURM_ARRAY_TASK_ID" "$ASSEMBLY" >&2
fi

if [ -n "$ASSEMBLY" ]; then
    [ "$STAGE" = "sketch" ] || die "ASSEMBLY only applies to the sketch stage"
    # Guard rather than fail 400 array tasks in: older builds of the CLI sketch
    # the whole manifest and have no way to select one haplotype.
    if ! "${RUN[@]}" sketch --help 2>&1 | grep -q -- '--assembly'; then
        die "this build of kmer-dust has no 'sketch --assembly'; drop the --array
  and submit a single sketch job, or use the Snakemake workflow."
    fi
    STAGE_ARGS+=(--assembly "$ASSEMBLY")
fi

STAGE_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

printf 'run_stage: %s %s\n' "${RUN[*]}" "${STAGE_ARGS[*]}" >&2
printf 'run_stage: threads=%s host=%s\n' "$THREADS" "$(hostname)" >&2

exec "${RUN[@]}" "${STAGE_ARGS[@]}"
