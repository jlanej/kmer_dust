#!/usr/bin/env bash
#
# Convert the published kmer-dust container into a local Apptainer image.
#
# Why bother, when Snakemake can pull `docker://...` itself: on a real cluster
# every compute node pulling the same image from GHCR is slow, is often blocked
# outright (compute nodes usually have no route to the internet), and counts
# against GHCR's anonymous rate limit.  Building the .sif once onto a shared
# filesystem and pointing every job at that file avoids all three.
#
# SITE-SPECIFIC BITS YOU MAY NEED TO EDIT OR OVERRIDE:
#   * --tmpdir  Apptainer unpacks the whole image here before squashing it.
#               $TMPDIR on a login node is often a few hundred MB of tmpfs,
#               which is not enough.  Point it at scratch.
#   * --output  Put the .sif somewhere every compute node can read.  $HOME is
#               frequently mounted noexec or not mounted at all on compute
#               nodes; /scratch or a project filesystem is the usual answer.
#
# --fakeroot is NOT needed here.  Building a SIF *from an OCI image* is an
# unprivileged operation in Apptainer >= 1.0; --fakeroot is only required when
# building from a definition file that runs %post as root.  If your site has
# disabled unprivileged user namespaces entirely, ask an admin to build the
# image, or use --docker-daemon on a workstation and copy the .sif across.

set -euo pipefail

IMAGE="ghcr.io/jlanej/kmer-dust:latest"
OUTPUT=""
TMPDIR_OVERRIDE=""
CACHE_OVERRIDE=""
USE_DOCKER_DAEMON=0
FORCE=0

usage() {
    cat <<'USAGE'
Build a kmer-dust .sif from the published container image.

Usage: hpc/build_sif.sh [options]

  -i, --image REF      source image  (default: ghcr.io/jlanej/kmer-dust:latest)
  -o, --output FILE    .sif to write (default: derived from the image tag)
  -t, --tmpdir DIR     scratch for the build   (default: $APPTAINER_TMPDIR,
                       else $TMPDIR, else ./.apptainer/tmp)
      --cache DIR      layer cache             (default: ./.apptainer/cache)
      --docker-daemon  pull from the local Docker daemon instead of a registry
                       (for an image you just built, or an air-gapped host)
  -f, --force          overwrite an existing .sif
  -h, --help           this text

Examples:
  hpc/build_sif.sh -o /scratch/$USER/kmer-dust_v0.1.0.sif \
                   -i ghcr.io/jlanej/kmer-dust:v0.1.0 \
                   -t /scratch/$USER/apptainer-tmp

  docker build -f docker/Dockerfile -t kmer-dust:dev .
  hpc/build_sif.sh --docker-daemon -i kmer-dust:dev -o kmer-dust_dev.sif
USAGE
}

die() {
    printf 'build_sif: %s\n' "$*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--image)      [ $# -ge 2 ] || die "$1 needs a value"; IMAGE="$2"; shift 2 ;;
        -o|--output)     [ $# -ge 2 ] || die "$1 needs a value"; OUTPUT="$2"; shift 2 ;;
        -t|--tmpdir)     [ $# -ge 2 ] || die "$1 needs a value"; TMPDIR_OVERRIDE="$2"; shift 2 ;;
        --cache)         [ $# -ge 2 ] || die "$1 needs a value"; CACHE_OVERRIDE="$2"; shift 2 ;;
        --docker-daemon) USE_DOCKER_DAEMON=1; shift ;;
        -f|--force)      FORCE=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "unknown argument: $1 (try --help)" ;;
    esac
done

[ -n "$IMAGE" ] || die "--image must not be empty"

# apptainer is the current name; singularity is the same tool on older sites.
if command -v apptainer >/dev/null 2>&1; then
    RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
    RUNTIME=singularity
    printf 'build_sif: apptainer not found, falling back to singularity\n' >&2
else
    die "neither apptainer nor singularity is on PATH (try: module load apptainer)"
fi

if [ -z "$OUTPUT" ]; then
    # ghcr.io/owner/kmer-dust:v1.2.3 -> kmer-dust_v1.2.3.sif
    base="${IMAGE##*/}"
    name="${base%%:*}"
    tag="${base##*:}"
    [ "$tag" = "$base" ] && tag="latest"
    OUTPUT="${name}_${tag//[^A-Za-z0-9._-]/_}.sif"
fi

if [ -e "$OUTPUT" ] && [ "$FORCE" -eq 0 ]; then
    die "$OUTPUT already exists (pass --force to rebuild)"
fi

# Apptainer needs room for the unpacked image *and* the squashed result. The
# kmer-dust image is ~1 GB unpacked, so a 100 MB tmpfs $TMPDIR will fail with a
# confusing "no space left on device" halfway through.
BUILD_TMPDIR="${TMPDIR_OVERRIDE:-${APPTAINER_TMPDIR:-${SINGULARITY_TMPDIR:-${TMPDIR:-$PWD/.apptainer/tmp}}}}"
BUILD_CACHE="${CACHE_OVERRIDE:-${APPTAINER_CACHEDIR:-${SINGULARITY_CACHEDIR:-$PWD/.apptainer/cache}}}"
mkdir -p "$BUILD_TMPDIR" "$BUILD_CACHE" "$(dirname -- "$OUTPUT")"

# Export under both names so the fallback runtime picks them up too.
export APPTAINER_TMPDIR="$BUILD_TMPDIR" SINGULARITY_TMPDIR="$BUILD_TMPDIR"
export APPTAINER_CACHEDIR="$BUILD_CACHE" SINGULARITY_CACHEDIR="$BUILD_CACHE"

AVAILABLE_KB="$(df -Pk "$BUILD_TMPDIR" | awk 'NR==2 {print $4}')"
if [ -n "${AVAILABLE_KB:-}" ] && [ "$AVAILABLE_KB" -lt 3000000 ]; then
    printf 'build_sif: warning: only %s MB free on %s; builds want ~3 GB\n' \
        "$((AVAILABLE_KB / 1024))" "$BUILD_TMPDIR" >&2
fi

BUILD_ARGS=()
[ "$FORCE" -eq 1 ] && BUILD_ARGS+=(--force)

printf 'build_sif: %s -> %s\n' "$IMAGE" "$OUTPUT" >&2
printf 'build_sif:   tmpdir=%s cache=%s\n' "$BUILD_TMPDIR" "$BUILD_CACHE" >&2

build_from() {
    "$RUNTIME" build "${BUILD_ARGS[@]}" "$OUTPUT" "$1"
}

if [ "$USE_DOCKER_DAEMON" -eq 1 ]; then
    command -v docker >/dev/null 2>&1 || die "--docker-daemon needs docker on PATH"
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || die "docker daemon has no image $IMAGE (build it first)"
    build_from "docker-daemon://${IMAGE}"
elif ! build_from "docker://${IMAGE}"; then
    # Registry pull failed. On a login node with no egress that is expected;
    # a locally built image is the usual way out.
    printf 'build_sif: registry pull failed\n' >&2
    if command -v docker >/dev/null 2>&1 && docker image inspect "$IMAGE" >/dev/null 2>&1; then
        printf 'build_sif: falling back to the local docker daemon\n' >&2
        BUILD_ARGS+=(--force)  # a partial .sif may be sitting there
        build_from "docker-daemon://${IMAGE}"
    else
        die "could not pull $IMAGE and it is not in a local docker daemon.
  If this is a login node with no internet, build on a machine that has it and
  copy the .sif across, or run this on a compute node with egress."
    fi
fi

[ -s "$OUTPUT" ] || die "$OUTPUT was not written"

# A .sif that exists but cannot run its entrypoint is the failure this catches
# -- usually a missing kernel feature or a noexec mount on the target path.
printf 'build_sif: verifying\n' >&2
"$RUNTIME" exec "$OUTPUT" kmer-dust --version
"$RUNTIME" exec "$OUTPUT" kmer-dust info

SIZE_MB="$(du -m "$OUTPUT" | awk '{print $1}')"
cat >&2 <<EOF
build_sif: wrote $OUTPUT (${SIZE_MB} MB)

Point the workflow at it:
  snakemake --snakefile workflow/Snakefile --configfile workflow/config/chr21.yaml \\
            --profile workflow/profiles/slurm --use-apptainer \\
            --config container=$(cd "$(dirname -- "$OUTPUT")" && pwd)/$(basename -- "$OUTPUT")
EOF
