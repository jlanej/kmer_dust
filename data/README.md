# `data/`

This directory is **gitignored**. It is where `kmer-dust fetch` puts downloaded
inputs:

```
data/
  cache/       # catalog CSVs pulled from the HPRC data-table repo
  testdata/    # small real slices of real assemblies (see tests/testdata.manifest.tsv)
  refs/        # T2T-CHM13v2.0 annotation tracks
  assemblies/  # full assemblies, only on HPC-scale runs
```

Nothing here is required to *build* the package -- run `kmer-dust fetch smoke`
to populate it.
