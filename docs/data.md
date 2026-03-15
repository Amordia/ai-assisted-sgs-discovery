# Data

Raw DNS cutouts are not tracked in Git. They are regenerated with the download
scripts listed in `repro/manifest.json` and require a valid JHTDB token.

## Required local inputs

The reproduction manifest documents all expected HDF5 inputs under
`required_inputs`. To regenerate them:

```bash
conda run --no-capture-output -n fluid python repro/run_pipeline.py --groups downloads
```

Set the token first:

```bash
export JHTDB_AUTH_TOKEN=...
```

## Tracked artifacts

The repository tracks compact CSV summaries and derived figure/table artifacts,
not raw velocity fields or processed tensor caches. This keeps the public
repository small and ensures that the data license and access rules remain
controlled by JHTDB.

## Local cache files

Files matching `*.h5`, including processed oracle caches such as
`*_processed_*.h5`, are intentionally ignored. They can be regenerated from the
raw cutouts and current code.
