# Running kmer-dust on a Slurm cluster

A walkthrough of one real submission, from an empty scratch directory to an
HTML report. It assumes a fairly ordinary cluster: Slurm, Apptainer available
as a module, a shared `/scratch` filesystem, and no internet access from the
compute nodes.

Files in this directory:

| file | what it is |
| --- | --- |
| `build_sif.sh` | turn the published container into a local `.sif` |
| `submit.sbatch` | the Snakemake controller job -- the normal way to run |
| `run_stage.sbatch` | one stage, standalone, for people who refuse Snakemake |

Three things need editing before the first run, and they are flagged in each
file under `EDIT BEFORE FIRST USE`: your Slurm **account** and **partition**,
your **scratch paths**, and the **bind mounts** the container needs.

---

## 0. What the run actually costs

Read this before asking for an allocation.

| config | haplotypes | bins | wall time | peak memory | where it hurts |
| --- | --- | --- | --- | --- | --- |
| `smoke.yaml` | 4 slices | ~1.6k | seconds | < 1 GB | nothing; it is a test |
| `chr21.yaml` | ~60 | ~270k | hours | ~64 GB | `sketch` (network), `embed` |
| `full.yaml` | 464 + CHM13 | ~1.4e8 | days | ~256 GB | `decompose`, `embed` |

`sketch` is I/O bound: each job streams an assembly over https from the HPRC
S3 bucket. It is where nearly all of a full run's wall time goes, and it is the
one stage whose parallelism matters. Everything after `matrix` is a single
large-memory job.

Two measured surprises from a real 24-haplotype chr21 run, both worth planning
around:

* **`embed` is the wall-clock bottleneck, not `decompose`.** Setting UMAP's
  `random_state` -- which `embed.deterministic: true` does, and which you want
  for anything reproducible -- pins UMAP and pynndescent to a single thread.
  105k bins took **18 minutes**; the randomized SVD over the same matrix took
  7 seconds. Either budget for it, or set `embed.deterministic: false` and give
  it `embed.n_jobs` cores while you are still tuning.
* **`annotate` downloads more than `sketch` does.** The per-haplotype
  RepeatMasker BEDs average 167 MB each, so 24 haplotypes move ~6 GB and all
  464 would move ~70 GB. The stage prefetches them through a thread pool
  (`threads`, capped at 8) and caches the *parsed* intervals under
  `datadir/cache/tracks/`, so you pay once -- but on a cluster whose compute
  nodes have no egress you must warm that cache from the login node
  (`kmer-dust run --config … --only manifest,annotate`) before submitting, or
  drop `repeatmasker` from `annotate.assembly_tracks` and keep it for the
  reference only.

---

## 1. Get the code and a place to put results

```bash
git clone https://github.com/jlanej/kmer_dust.git
cd kmer_dust

# Results, intermediates and the container all belong on scratch, not in $HOME.
# A full run writes terabytes of parquet.
export KD_SCRATCH=/scratch/$USER/kmer-dust
mkdir -p "$KD_SCRATCH"
```

## 2. Build the container image

```bash
module load apptainer          # or: module load singularity

hpc/build_sif.sh \
    --image ghcr.io/jlanej/kmer-dust:latest \
    --output "$KD_SCRATCH/kmer-dust.sif" \
    --tmpdir "$KD_SCRATCH/apptainer-tmp"
```

`--tmpdir` matters: Apptainer unpacks the whole image before squashing it, and
the default `$TMPDIR` on a login node is often a small tmpfs. The script warns
if there is less than 3 GB free.

If the login node has no route to ghcr.io, build on a machine that does and
copy the `.sif` across:

```bash
# on a workstation with docker
docker build -f docker/Dockerfile -t kmer-dust:dev .
hpc/build_sif.sh --docker-daemon -i kmer-dust:dev -o kmer-dust.sif
scp kmer-dust.sif cluster:/scratch/$USER/kmer-dust/
```

`--fakeroot` is not needed. Building a SIF from an OCI image is unprivileged in
Apptainer >= 1.0; `--fakeroot` is only for definition-file builds that run
`%post` as root.

The script finishes by running `kmer-dust --version` and `kmer-dust info`
inside the image. If that fails, the image is unusable on this cluster (usually
a `noexec` mount or disabled user namespaces) and nothing below will work.

## 3. Choose and edit a config

```bash
cp workflow/config/chr21.yaml "$KD_SCRATCH/chr21.yaml"
$EDITOR "$KD_SCRATCH/chr21.yaml"
```

At minimum change:

```yaml
outdir: /scratch/YOURUSER/kmer-dust/results/chr21   # not $HOME
datadir: /scratch/YOURUSER/kmer-dust/data           # catalog + track cache
```

Every knob is commented in place. The ones worth a second look on a first run
are `manifest.max_samples` (start small), `sketch.scaled` (the memory/signal
trade-off) and `cluster.min_cluster_size`.

Validate it before burning an allocation:

```bash
apptainer exec "$KD_SCRATCH/kmer-dust.sif" \
    kmer-dust validate-config --config "$KD_SCRATCH/chr21.yaml"
```

## 4. Point the Slurm profile at your cluster

Edit `workflow/profiles/slurm/config.yaml`:

```yaml
default-resources:
  - slurm_partition=general        # uncomment and set
  - slurm_account=my_allocation    # uncomment and set
jobs: 200                          # your queue's per-user limit
```

The per-rule `mem_mb` / `runtime` / `cpus_per_task` are computed by the
Snakefile from the science config, so you do not normally set them. If a stage
is consistently killed, either raise `mem_scale` (below) or pin that one rule
under `set-resources`.

## 5. Bind mounts — the thing that actually goes wrong

Apptainer bind-mounts `$PWD`, `$HOME` and `/tmp` automatically. **Nothing
else.** If `outdir` is on `/scratch` and you launch from `/home`, the container
will not see `/scratch` and the job dies with a confusing "no such file or
directory" naming a path that plainly exists.

Name every other filesystem you touch:

```bash
export SNAKEMAKE_BIND="/scratch,/projects"
```

`submit.sbatch` turns that into `--apptainer-args "--cleanenv --bind ..."`.
`--cleanenv` drops your *shell's* environment (a stray `PYTHONPATH` pointing at
a host venv is a classic way to break a container run); the image's own
environment — `PATH`, `NUMBA_CACHE_DIR`, the thread limits — is unaffected.

## 6. Submit

```bash
sbatch --export=ALL,SIF="$KD_SCRATCH/kmer-dust.sif",SNAKEMAKE_BIND="/scratch" \
       hpc/submit.sbatch "$KD_SCRATCH/chr21.yaml"
```

That single job is the *controller*: it builds the DAG and submits one Slurm
job per rule, then waits. It needs a walltime longer than the entire run —
check the `--time` in the `#SBATCH` header, which defaults to 48 h.

If your cluster forbids `sbatch` from a compute node, run the same thing from a
login node inside `tmux`:

```bash
tmux new -s kmer-dust
SIF="$KD_SCRATCH/kmer-dust.sif" SNAKEMAKE_BIND=/scratch \
    bash hpc/submit.sbatch "$KD_SCRATCH/chr21.yaml"
```

## 7. Watch it

```bash
squeue -u "$USER"                                   # the spawned jobs
tail -f slurm-logs/kmer-dust-<jobid>.out            # the controller
tail -f "$KD_SCRATCH/results/chr21/logs/sketch/"*.log   # a stage
```

Each rule writes `<outdir>/logs/<stage>.log`; the sketch scatter writes
`<outdir>/logs/sketch/<assembly>.log`. Snakemake's own log is under
`.snakemake/log/`.

## 8. When something fails

Re-submitting the same command resumes: every stage skips work whose output
already exists, and Snakemake only re-runs rules whose outputs are missing.

| symptom | fix |
| --- | --- |
| `Directory cannot be locked` | a previous controller was killed. `submit.sbatch` clears the lock on start; to do it by hand, `snakemake --snakefile workflow/Snakefile --configfile <cfg> --unlock` |
| a job is OOM-killed | the profile retries twice with 1.7x the memory. If it still dies, `--config mem_scale=2` on the whole run, or pin that rule under `set-resources` |
| every `sketch` job fails at once | the compute nodes cannot reach `s3-us-west-2.amazonaws.com`. Pre-stage the assemblies and use a `source: file` manifest |
| `no such file` for a path that exists | a missing bind mount; see step 5 |
| `MISSING sketch/*.done` after a green run | the manifest filtered everything out. Loosen `require_t2t_chrom` / `require_annotations` |
| numba recompiles in every job | `NUMBA_CACHE_DIR` is not writable. Inside the container it is `/tmp/.cache/numba`; make sure `/tmp` is bound |

## 9. Without Snakemake

`run_stage.sbatch` runs one stage. The order is
`manifest -> sketch -> select -> matrix -> decompose -> embed -> cluster ->
annotate -> enrich -> backprop -> report`, with `annotate` free to run
alongside `decompose`/`embed`/`cluster` since it only needs `matrix`.

```bash
export CFG="$KD_SCRATCH/chr21.yaml" SIF="$KD_SCRATCH/kmer-dust.sif"

m=$(sbatch --parsable --export=ALL,SIF hpc/run_stage.sbatch manifest "$CFG")

# One array task per assembly. The manifest exists by now, so size the array
# from it (minus the header, minus one for the 0-based index).
n=$(( $(wc -l < "$KD_SCRATCH/results/chr21/manifest.tsv") - 2 ))
s=$(sbatch --parsable --dependency=afterok:$m --array=0-$n%50 \
           --export=ALL,SIF hpc/run_stage.sbatch sketch "$CFG")

k=$(sbatch --parsable --dependency=afterok:$s --mem=64G \
           --export=ALL,SIF hpc/run_stage.sbatch select "$CFG")
# ...and so on, or simply:
sbatch --dependency=afterok:$s --mem=128G --time=24:00:00 \
       --export=ALL,SIF hpc/run_stage.sbatch run "$CFG"
```

The array scatter needs a `kmer-dust` build whose `sketch` subcommand accepts
`--assembly`; `run_stage.sbatch` checks for it and refuses with a clear message
otherwise, in which case submit a single (much slower) `sketch` job instead.

Note what you give up: the per-stage memory and walltime that the Snakefile
computes from your config, the automatic retry with escalating memory, and the
dependency bookkeeping. That is most of the reason the Snakemake path exists.

## 10. Getting the result off the cluster

```bash
scp -r cluster:"$KD_SCRATCH/results/chr21/report" .
open report/kmer_dust_report.html
```

`report/kmer_dust_report.html` is self-contained (plotly.js is inlined), so it
opens without a network connection. `backprop/clusters.all.bed.gz` is the
per-assembly cluster painting, ready for a genome browser;
`enrich/cluster_names.parquet` is what each cluster turned out to be.
