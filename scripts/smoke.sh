#!/usr/bin/env bash
#
# End-to-end smoke test: fetch the tiny fixture (once), run every pipeline
# stage over it, and assert that each stage's contracted output landed on disk.
#
# This is not a unit test and does not replace one.  What it catches is the
# class of failure unit tests structurally cannot: a stage writing a file the
# next stage does not read, a CLI flag that no longer exists, a schema change
# that only shows up once real sequence has flowed through all eleven stages.
# It runs on real (if small) HPRC and CHM13 sequence for the same reason.
#
# Exit status is 0 only if every expected output exists and is non-empty.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="workflow/config/smoke.yaml"
USE_SNAKEMAKE=0
CORES=2
SKIP_FETCH=0
FORCE=0
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<'USAGE'
Run the kmer-dust pipeline end to end over the small real-data fixture.

Usage: scripts/smoke.sh [options]

  -c, --config FILE   run config       (default: workflow/config/smoke.yaml)
      --snakemake     drive the run through workflow/Snakefile instead of
                      `kmer-dust run`, which also exercises the DAG
      --cores N       cores for --snakemake                    (default: 2)
      --skip-fetch    assume the fixture is already present
  -f, --force         wipe the output directory and recompute everything
  -h, --help          this text

Environment:
  KMER_DUST_BIN  path to the kmer-dust executable (default: found on PATH,
                 falling back to `$PYTHON -m kmer_dust`)
  PYTHON         interpreter for the fallback and helpers  (default: python3)
USAGE
}

die() {
    printf 'smoke: %s\n' "$*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        -c|--config)  [ $# -ge 2 ] || die "$1 needs a value"; CONFIG="$2"; shift 2 ;;
        --snakemake)  USE_SNAKEMAKE=1; shift ;;
        --cores)      [ $# -ge 2 ] || die "$1 needs a value"; CORES="$2"; shift 2 ;;
        --skip-fetch) SKIP_FETCH=1; shift ;;
        -f|--force)   FORCE=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            die "unknown argument: $1 (try --help)" ;;
    esac
done

case "$CORES" in ''|*[!0-9]*) die "--cores must be a positive integer" ;; esac
[ "$CORES" -ge 1 ] || die "--cores must be at least 1"

# Every path in the config is relative to the repository root, so the run has
# to start there no matter where the user invoked the script from.
cd "$REPO_ROOT"
[ -f "$CONFIG" ] || die "no such config: $CONFIG"

if [ -n "${KMER_DUST_BIN:-}" ]; then
    # A command, not just a path: "python -m kmer_dust" is a legitimate value,
    # so split it on whitespace rather than treating it as one argv[0].
    read -r -a KMER_DUST <<< "$KMER_DUST_BIN"
elif command -v kmer-dust >/dev/null 2>&1; then
    KMER_DUST=(kmer-dust)
elif "$PYTHON" -c 'import kmer_dust' >/dev/null 2>&1; then
    KMER_DUST=("$PYTHON" -m kmer_dust)
else
    die "kmer-dust is not installed; run: pip install -e '.[dev]'"
fi

# The config is the single source of truth for where things go, so read outdir
# and the fixture manifest out of it rather than duplicating them here.
PATHS="$(KD_CONFIG="$CONFIG" "$PYTHON" -c '
import os, sys, yaml
with open(os.environ["KD_CONFIG"]) as handle:
    cfg = yaml.safe_load(handle) or {}
outdir = str(cfg.get("outdir") or "").strip()
if not outdir:
    sys.exit("config has no outdir")
manifest = dict(cfg.get("manifest") or {})
# A manifest that is not source=file has no fixture to fetch.
fixture = str(manifest.get("path") or "").strip() if manifest.get("source") == "file" else ""
print(outdir)
print(fixture)
')" || die "could not read $CONFIG"
OUTDIR="$(printf '%s\n' "$PATHS" | sed -n 1p)"
FIXTURE="$(printf '%s\n' "$PATHS" | sed -n 2p)"
[ -n "$OUTDIR" ] || die "could not read outdir from $CONFIG"

if [ -n "$FIXTURE" ]; then
    if [ "$SKIP_FETCH" -eq 0 ]; then
        # Not passed --force: the fixture is expensive to rebuild and --force
        # here is about recomputing *outputs*, not re-downloading inputs.
        "$REPO_ROOT/scripts/fetch_test_data.sh" --dest "$(dirname -- "$FIXTURE")"
    fi
    [ -s "$FIXTURE" ] || die "fixture manifest $FIXTURE is missing; run scripts/fetch_test_data.sh"
fi

if [ "$FORCE" -eq 1 ] && [ -d "$OUTDIR" ]; then
    printf 'smoke: clearing %s\n' "$OUTDIR" >&2
    rm -rf -- "$OUTDIR"
fi

START=$SECONDS
if [ "$USE_SNAKEMAKE" -eq 1 ]; then
    command -v snakemake >/dev/null 2>&1 \
        || die "snakemake not installed; run: pip install -e '.[workflow]'"
    SNAKE_ARGS=(--snakefile workflow/Snakefile --configfile "$CONFIG"
                --cores "$CORES" --rerun-triggers mtime)
    # Local cores, no container: this checks the DAG, not the deployment image.
    [ "$FORCE" -eq 1 ] && SNAKE_ARGS+=(--forceall)
    snakemake "${SNAKE_ARGS[@]}"
else
    "${KMER_DUST[@]}" run --config "$CONFIG"
fi
ELAPSED=$(( SECONDS - START ))

# One representative output per stage.  sketch/*.done is checked by pattern
# because how many shards there are depends on how many haplotypes the fixture
# could find with an intact chromosome.
EXPECTED=(
    "$OUTDIR/manifest.tsv"
    "$OUTDIR/kmers/kmers.parquet"
    "$OUTDIR/kmers/prevalence.parquet"
    "$OUTDIR/matrix/matrix.npz"
    "$OUTDIR/matrix/rows.parquet"
    "$OUTDIR/decompose/pcs.npy"
    "$OUTDIR/decompose/svd.json"
    "$OUTDIR/embed/umap.npy"
    "$OUTDIR/cluster/clusters.parquet"
    "$OUTDIR/annotate/annotations.parquet"
    "$OUTDIR/enrich/enrichment.parquet"
    "$OUTDIR/enrich/cluster_names.parquet"
    "$OUTDIR/backprop/clusters.all.bed.gz"
    "$OUTDIR/report/kmer_dust_report.html"
    "$OUTDIR/report/summary.json"
)

FAILED=0
for path in "${EXPECTED[@]}"; do
    if [ -s "$path" ]; then
        printf '  ok      %s\n' "$path"
    else
        printf '  MISSING %s\n' "$path" >&2
        FAILED=1
    fi
done

# `! -name '_*'` skips the workflow's own sketch_all marker; the stage writes a
# real <assembly>.done per haplotype either way.
SHARDS="$(find "$OUTDIR/sketch" -name '*.done' ! -name '_*' 2>/dev/null | wc -l | tr -d ' ')"
if [ "${SHARDS:-0}" -ge 1 ]; then
    printf '  ok      %s sketch shard(s)\n' "$SHARDS"
else
    printf '  MISSING %s/*.done -- no assembly was sketched\n' "$OUTDIR/sketch" >&2
    FAILED=1
fi

# summary.json is where the run says what it actually found; echoing the scalar
# fields turns a green tick into something a human can sanity-check.
if [ -s "$OUTDIR/report/summary.json" ]; then
    KD_SUMMARY="$OUTDIR/report/summary.json" "$PYTHON" -c '
import json, os
with open(os.environ["KD_SUMMARY"]) as handle:
    summary = json.load(handle)
if isinstance(summary, dict):
    for key, value in summary.items():
        if not isinstance(value, (dict, list)):
            print(f"  {key}: {value}")
' || printf 'smoke: could not parse summary.json\n' >&2
fi

if [ "$FAILED" -ne 0 ]; then
    die "FAILED after ${ELAPSED}s"
fi
printf 'smoke: PASSED in %ss -- report at %s/report/kmer_dust_report.html\n' \
    "$ELAPSED" "$OUTDIR"
