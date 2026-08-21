# kmer-dust batch image.
#
# Two constraints shaped this file:
#
#  1. It is a *batch* image.  There is no server, so there is no HEALTHCHECK and
#     no exposed port; the entrypoint is the CLI itself.
#  2. The real deployment target is Apptainer on a shared HPC cluster, not
#     Docker.  Apptainer runs the image as the *invoking host user*, with the
#     image filesystem read-only, so anything that assumes a writable $HOME, a
#     writable site-packages, or uid 0 breaks there even though it works under
#     `docker run`.  Everything below that looks paranoid follows from that:
#     caches point at /tmp (always writable, and bind-mounted from the host by
#     Apptainer), and OpenMP/BLAS thread counts are pinned to 1 so a Slurm cpu
#     allocation is not silently oversubscribed.
#
# Build from the repository root, so pyproject.toml and src/ are in context:
#     docker build -f docker/Dockerfile -t kmer-dust:dev .

# Pinned to the 3.11 line on Debian bookworm.  Patch releases are deliberately
# not pinned: a floating patch keeps CVE fixes flowing, and the wheels we rely
# on (numpy, scipy, pysam, numba) are ABI-stable across 3.11.x.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm

# --------------------------------------------------------------------------
# Stage 1 -- build a wheel from the repo and install it into a throwaway venv.
#
# A venv (rather than --user or --prefix) is what makes stage 2 a clean copy:
# console-script shebangs point at /opt/venv/bin/python, whose symlink target
# (/usr/local/bin/python3.11) exists in the runtime stage because both stages
# share the same base image.
# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# At the time of writing every dependency publishes a manylinux wheel, so none
# of this is strictly needed -- but without it a single upstream project
# dropping wheels turns into an opaque build failure.  It stays in this stage
# and never reaches the runtime image.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      build-essential \
      zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src

# pyproject.toml is the single source of truth for dependencies; there is
# deliberately no requirements.txt here to drift out of sync with it.
COPY pyproject.toml README.md LICENSE ./
RUN pip install --upgrade pip setuptools wheel build

COPY src/ ./src/

# --no-isolation reuses the setuptools/wheel installed above instead of pip
# building a second, network-fetched build environment.
RUN python -m build --wheel --no-isolation --outdir /tmp/wheels . \
 && pip install /tmp/wheels/*.whl \
 && python -c "import kmer_dust; print('built kmer-dust', kmer_dust.__version__)"

# --------------------------------------------------------------------------
# Stage 2 -- runtime.  No compilers, no headers, no build cache.
# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# Supplied by CI (docker/metadata-action); harmless defaults for local builds.
ARG VERSION=0.1.0
ARG REVISION=unknown
ARG CREATED=unknown
ARG SOURCE_URL=https://github.com/jlanej/kmer_dust

LABEL org.opencontainers.image.title="kmer-dust" \
      org.opencontainers.image.description="Alignment-free clustering of genomic bins across HPRC assemblies and T2T-CHM13 via FracMinHash k-mer sketches, randomized SVD, UMAP and HDBSCAN." \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.url="${SOURCE_URL}" \
      org.opencontainers.image.documentation="${SOURCE_URL}/blob/main/hpc/README.md" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}" \
      org.opencontainers.image.base.name="docker.io/library/python:3.11-slim-bookworm"

# ca-certificates: both pysam/htslib and requests fetch https S3 URLs.
# libcurl4 / libgomp1 are normally vendored inside the wheels; the system
# copies cost ~2 MB and remove a class of failure if a wheel is ever replaced
# by a source build.
RUN apt-get update \
 && apt-get install --no-install-recommends -y ca-certificates libcurl4 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

# The three preset configs are the only repo files the CLI reads at run time
# (`kmer-dust init-config --preset chr21`).  They are copied rather than made
# package data so that `workflow/config/` stays the single source of truth for
# both the container and a Snakemake `--configfile`.
COPY workflow/config/ /opt/kmer-dust/presets/
ENV KMER_DUST_PRESETS=/opt/kmer-dust/presets

# A named non-root user for `docker run`.  Under Apptainer this uid is ignored
# (the container runs as the host user), which is precisely why nothing below
# may depend on being uid 1000 or on /home/kmerdust existing.
RUN useradd --create-home --uid 1000 --shell /bin/bash kmerdust

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Every cache the scientific-Python stack writes is redirected under /tmp,
# which Apptainer bind-mounts from the host and is therefore writable by any
# uid even though the image is read-only.  NUMBA_CACHE_DIR is the load-bearing
# one: kmer_dust.hashing compiles with cache=True and numba's default target is
# the package's own __pycache__ inside site-packages.  HOME is set for the
# Docker case only -- Apptainer overrides it with the host $HOME, which is why
# the XDG/MPL/NUMBA variables are set explicitly rather than relying on $HOME.
ENV HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    XDG_CONFIG_HOME=/tmp/.config \
    MPLCONFIGDIR=/tmp/.cache/matplotlib \
    NUMBA_CACHE_DIR=/tmp/.cache/numba

# Respect the Slurm cpu allocation.  Left alone, each BLAS/OpenMP runtime
# spawns one thread per *physical* core, so a 2-cpu job on a 128-core node
# starts 128 threads and thrashes.  kmer-dust does its own parallelism from
# cfg.threads; override at run time (e.g. `-e OPENBLAS_NUM_THREADS=8`) if you
# deliberately want threaded BLAS.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

# Pre-create the cache tree world-writable so that `docker run --user 65534`
# (or an Apptainer run where /tmp is *not* bound from the host) still works.
# /work is the conventional bind point for the run directory; it is
# world-writable because we cannot know the uid Apptainer will hand us.
RUN mkdir -p /tmp/.cache/numba /tmp/.cache/matplotlib /tmp/.config /work \
 && chmod -R a+rwX /tmp/.cache /tmp/.config \
 && chmod 1777 /work

WORKDIR /work
USER 1000:1000

# Batch image: no HEALTHCHECK on purpose -- there is nothing long-lived to probe.
ENTRYPOINT ["kmer-dust"]
CMD ["--help"]
