#!/usr/bin/env bash
#
# Snakemake controller job for kmer-dust.
#
#   sbatch hpc/submit.sbatch workflow/config/chr21.yaml
#   sbatch --export=ALL,SIF=/scratch/$USER/kmer-dust.sif hpc/submit.sbatch workflow/config/full.yaml
#
# This job does almost no work itself: it builds the DAG and then submits one
# Slurm job per rule via the snakemake-executor-plugin-slurm, waiting on them.
# That is why it asks for 2 cpus and a long walltime -- it must outlive every
# job it spawns, including the sketch scatter.
#
# ---------------------------------------------------------------------------
# EDIT BEFORE FIRST USE
# ---------------------------------------------------------------------------
#   * --account / --partition below (or delete them if your site has defaults).
#     The same values belong in workflow/profiles/slurm/config.yaml, which is
#     what the *spawned* jobs use; this header only covers the controller.
#   * --time: must exceed the whole run, not one stage. full.yaml is days.
#   * The "site setup" block further down: module loads / venv activation.
#   * SNAKEMAKE_BIND: filesystems the container must see beyond $PWD and /tmp.
#
# Not every cluster lets a *compute* node call sbatch. If yours does not, run
# the same snakemake command from a login node inside tmux/screen instead --
# everything below the site-setup block works unchanged there.
# ---------------------------------------------------------------------------

#SBATCH --job-name=kmer-dust
#SBATCH --output=slurm-logs/kmer-dust-%j.out
#SBATCH --error=slurm-logs/kmer-dust-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
##SBATCH --account=CHANGE_ME
##SBATCH --partition=CHANGE_ME

set -euo pipefail

CONFIG="${1:-${CONFIG:-workflow/config/chr21.yaml}}"
PROFILE="${PROFILE:-workflow/profiles/slurm}"
SIF="${SIF:-}"
JOBS="${JOBS:-}"
EXTRA_ARGS=("${@:2}")

die() {
    printf 'submit: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Snakemake controller job for kmer-dust.

Usage: sbatch hpc/submit.sbatch [CONFIG] [extra snakemake args...]

  CONFIG   run config (default: $CONFIG, else workflow/config/chr21.yaml)

Environment (pass with --export=ALL,NAME=value):
  SIF              path to a .sif; enables --use-apptainer
  SNAKEMAKE_BIND   extra bind mounts, e.g. "/scratch,/projects"
  PROFILE          snakemake profile dir  (default: workflow/profiles/slurm)
  JOBS             override the profile's concurrent-job limit
  KMER_DUST_ENV    a file to `source` to put snakemake on PATH

Edit the #SBATCH header (account, partition, walltime) and the site-setup
block before first use. See hpc/README.md for a full walkthrough.
USAGE
}

case "${CONFIG}" in
    -h|--help) usage; exit 0 ;;
esac

# Slurm starts the job in the submitting directory, but only if it still exists
# and is readable from the compute node; be explicit about it either way.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm-logs

# ---------------------------------------------------------------------------
# Site setup -- EDIT THIS BLOCK
# ---------------------------------------------------------------------------
# Whatever it takes to put `snakemake` (and, if you are not using a container,
# `kmer-dust`) on PATH on a compute node. Typical shapes:
#
#   module purge && module load python/3.11 apptainer
#   source /scratch/$USER/venvs/kmer-dust/bin/activate
#
# Left as a no-op by default so the script works unmodified anywhere PATH is
# already correct.
if [ -n "${KMER_DUST_ENV:-}" ]; then
    # shellcheck disable=SC1090  # a site-supplied path, unknowable here
    source "$KMER_DUST_ENV"
fi
# ---------------------------------------------------------------------------

[ -f "$CONFIG" ] || die "no such config: $CONFIG (run from the repo root)"
[ -d "$PROFILE" ] || die "no such profile directory: $PROFILE"
command -v snakemake >/dev/null 2>&1 \
    || die "snakemake is not on PATH; edit the site-setup block in $0"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    printf 'submit: not running under Slurm -- driving the workflow directly.\n' >&2
    printf 'submit: this is fine from tmux on a login node; ^C will abandon submitted jobs.\n' >&2
fi

SNAKE_ARGS=(--snakefile workflow/Snakefile
            --configfile "$CONFIG"
            --profile "$PROFILE")

# The controller itself must not hog the node it landed on.
SNAKE_ARGS+=(--local-cores "${SLURM_CPUS_PER_TASK:-2}")
[ -n "$JOBS" ] && SNAKE_ARGS+=(--jobs "$JOBS")

if [ -n "$SIF" ]; then
    [ -f "$SIF" ] || die "SIF=$SIF does not exist (build it with hpc/build_sif.sh)"
    # An absolute path is required: the compute node's CWD is not guaranteed to
    # match the controller's.
    case "$SIF" in
        /*) ;;
        *) SIF="$PWD/$SIF" ;;
    esac
    SNAKE_ARGS+=(--use-apptainer --config "container=$SIF")
    # Apptainer binds $PWD, $HOME and /tmp by default. Anything else -- the
    # scratch filesystem holding `outdir`, a project directory holding the
    # assemblies -- has to be named. Set SNAKEMAKE_BIND="/scratch,/projects".
    if [ -n "${SNAKEMAKE_BIND:-}" ]; then
        SNAKE_ARGS+=(--apptainer-args "--cleanenv --bind ${SNAKEMAKE_BIND}")
    fi
fi

SNAKE_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

printf 'submit: repo=%s\n' "$REPO_ROOT" >&2
printf 'submit: snakemake %s\n' "${SNAKE_ARGS[*]}" >&2

# `unlock` clears a stale lock left behind by a controller the scheduler killed
# (OOM, walltime). It is safe here because Slurm guarantees only one controller
# per job, and refusing to start is a worse failure than clearing a dead lock.
snakemake "${SNAKE_ARGS[@]}" --unlock >/dev/null 2>&1 || true

snakemake "${SNAKE_ARGS[@]}"
