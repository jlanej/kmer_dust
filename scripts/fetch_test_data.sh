#!/usr/bin/env bash
#
# Materialise the smoke-test fixture: small slices of *real* HPRC release-2
# haplotypes and T2T-CHM13v2.0 under data/testdata/.
#
# The slicing itself lives in `kmer-dust fetch` (kmer_dust/fetch.py), not here.
# That is deliberate: the S3 layout, the chromAlias lookup and the BED rebasing
# are things the package already has to know, and a second copy in bash would
# drift.  What this script adds is the operational shell around it -- argument
# validation, idempotence, a parameter stamp so a changed window forces a
# refetch, and a verification pass that every file the manifest points at
# actually exists.
#
# Idempotence is the point: re-running without --force is a no-op, which is
# what makes the actions/cache step in .github/workflows/ci.yml worth having.
#
#   IMPORTANT: coordinates in the fixture are slice-relative -- local position
#   0 is `offset` on the source contig.  The cenSat/RepeatMasker/segdup BEDs
#   written alongside are rebased to match, so annotation is self-consistent,
#   but the genome-wide T2T tracks must NOT be used against these bins.  That
#   is why workflow/config/smoke.yaml sets `annotate.reference_tracks: []`.
#
# Defaults download ~20 MB and cover the chr21 alpha-satellite HOR array in
# every haplotype sampled so far, which is what makes the fixture interesting:
# a slice of unique euchromatin would leave every bin looking the same.

set -euo pipefail

DEST="data/testdata"
DATADIR="data"
CHROM="chr21"
SAMPLES=2
SPAN_MB=4
OFFSET_MB=7
INCLUDE_REFERENCE=1
FORCE=0
KMER_DUST_BIN="${KMER_DUST_BIN:-}"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<'USAGE'
Fetch the smoke-test fixture (small real slices of HPRC + CHM13 assemblies).

Usage: scripts/fetch_test_data.sh [options]

  -d, --dest DIR       fixture directory            (default: data/testdata)
      --datadir DIR    catalog cache root           (default: data)
  -c, --chrom NAME     chromosome to slice          (default: chr21)
  -n, --samples N      HPRC samples (both haplotypes each)   (default: 2)
      --span-mb N      Mb per assembly              (default: 4)
      --offset-mb N    Mb into the chromosome       (default: 7)
      --no-reference   skip the T2T-CHM13v2.0 slice
  -f, --force          refetch even if the fixture is already present
  -h, --help           this text

Environment:
  KMER_DUST_BIN  path to the kmer-dust executable (default: found on PATH,
                 falling back to `$PYTHON -m kmer_dust`)
  PYTHON         interpreter for the fallback      (default: python3)

The default window covers the chr21 HOR array; see the header of this script
for why that matters and why the coordinates are slice-relative.
USAGE
}

die() {
    printf 'fetch_test_data: %s\n' "$*" >&2
    exit 1
}

is_number() {
    case "$1" in
        ''|*[!0-9.]*|*.*.*) return 1 ;;
        *) return 0 ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        -d|--dest)      [ $# -ge 2 ] || die "$1 needs a value"; DEST="$2"; shift 2 ;;
        --datadir)      [ $# -ge 2 ] || die "$1 needs a value"; DATADIR="$2"; shift 2 ;;
        -c|--chrom)     [ $# -ge 2 ] || die "$1 needs a value"; CHROM="$2"; shift 2 ;;
        -n|--samples)   [ $# -ge 2 ] || die "$1 needs a value"; SAMPLES="$2"; shift 2 ;;
        --span-mb)      [ $# -ge 2 ] || die "$1 needs a value"; SPAN_MB="$2"; shift 2 ;;
        --offset-mb)    [ $# -ge 2 ] || die "$1 needs a value"; OFFSET_MB="$2"; shift 2 ;;
        --no-reference) INCLUDE_REFERENCE=0; shift ;;
        -f|--force)     FORCE=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              die "unknown argument: $1 (try --help)" ;;
    esac
done

case "$SAMPLES" in ''|*[!0-9]*) die "--samples must be a non-negative integer" ;; esac
is_number "$SPAN_MB"   || die "--span-mb must be a number"
is_number "$OFFSET_MB" || die "--offset-mb must be a number"
[ -n "$CHROM" ] || die "--chrom must not be empty"
[ "$SAMPLES" -gt 0 ] || [ "$INCLUDE_REFERENCE" -eq 1 ] \
    || die "--samples 0 together with --no-reference would fetch nothing"

# Resolve the CLI once.  `python -m kmer_dust` is the fallback because an
# editable install without its console script on PATH is a common CI state.
if [ -z "$KMER_DUST_BIN" ]; then
    if command -v kmer-dust >/dev/null 2>&1; then
        KMER_DUST=(kmer-dust)
    elif "$PYTHON" -c 'import kmer_dust' >/dev/null 2>&1; then
        KMER_DUST=("$PYTHON" -m kmer_dust)
    else
        die "kmer-dust is not installed; run: pip install -e '.[dev]'"
    fi
else
    # A command, not just a path: "python -m kmer_dust" is a legitimate value.
    read -r -a KMER_DUST <<< "$KMER_DUST_BIN"
fi

MANIFEST="$DEST/manifest.tsv"
STAMP="$DEST/.fetch_params"
# Anything that changes which sequence ends up on disk belongs in the stamp.
WANT="chrom=$CHROM samples=$SAMPLES span_mb=$SPAN_MB offset_mb=$OFFSET_MB reference=$INCLUDE_REFERENCE"

if [ "$FORCE" -eq 0 ] && [ -s "$MANIFEST" ] && [ -f "$STAMP" ]; then
    if [ "$(cat "$STAMP")" = "$WANT" ]; then
        printf 'fetch_test_data: %s is up to date (%s assemblies); --force to refetch\n' \
            "$MANIFEST" "$(( $(wc -l < "$MANIFEST") - 1 ))"
        exit 0
    fi
    # `kmer-dust fetch` keeps an existing slice whenever its sidecar is present,
    # regardless of the window asked for, so a changed window has to be forced
    # or the fixture would silently stay at the old coordinates.
    printf 'fetch_test_data: parameters changed, forcing a refetch\n  was: %s\n  now: %s\n' \
        "$(cat "$STAMP")" "$WANT" >&2
    FORCE=1
fi

mkdir -p "$DEST" "$DATADIR"
rm -f "$STAMP"

FETCH_ARGS=(fetch --dest "$DEST" --datadir "$DATADIR" --samples "$SAMPLES"
            --chrom "$CHROM" --span-mb "$SPAN_MB" --offset-mb "$OFFSET_MB")
[ "$INCLUDE_REFERENCE" -eq 0 ] && FETCH_ARGS+=(--no-reference)
[ "$FORCE" -eq 1 ] && FETCH_ARGS+=(--force)

printf 'fetch_test_data: %s %s\n' "${KMER_DUST[*]}" "${FETCH_ARGS[*]}" >&2
"${KMER_DUST[@]}" "${FETCH_ARGS[@]}"

# Verify rather than trust.  A fixture that is half-downloaded fails later, in
# the middle of a pipeline stage, where the error is much harder to read.
KD_MANIFEST="$MANIFEST" "$PYTHON" - <<'PYTHON_VERIFY'
"""Check the fetched manifest against the on-disk contract."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

REQUIRED = ("assembly", "sample", "haplotype", "source", "fasta")
PATH_COLUMNS = ("fasta", "fai", "gzi", "chrom_alias", "censat_bed",
                "repeatmasker_bed", "segdup_bed")

manifest = Path(os.environ["KD_MANIFEST"])
if not manifest.is_file():
    sys.exit(f"fetch_test_data: {manifest} was not written")

with open(manifest, newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if not rows:
    sys.exit(f"fetch_test_data: {manifest} has no assemblies")

problems: list[str] = []
try:
    from kmer_dust.schemas import MANIFEST_COLUMNS

    missing = [c for c in MANIFEST_COLUMNS if c not in rows[0]]
    if missing:
        problems.append(f"manifest is missing contract columns: {missing}")
except ImportError:  # verification is best-effort if the package moved
    pass

for row in rows:
    assembly = row.get("assembly") or "<unnamed>"
    for column in REQUIRED:
        if not (row.get(column) or "").strip():
            problems.append(f"{assembly}: empty required column {column!r}")
    for column in PATH_COLUMNS:
        value = (row.get(column) or "").strip()
        if value and not Path(value).is_file():
            problems.append(f"{assembly}: {column} points at a missing file {value}")

if problems:
    for problem in problems:
        print(f"fetch_test_data: {problem}", file=sys.stderr)
    sys.exit(1)

samples = {row["sample"] for row in rows}
print(f"fetch_test_data: verified {len(rows)} assemblies from {len(samples)} sample(s)",
      file=sys.stderr)
PYTHON_VERIFY

printf '%s' "$WANT" > "$STAMP"
printf 'fetch_test_data: fixture ready in %s\n' "$DEST"
