# Reproducibility

The canonical reproduction entry points are in `repro/`.

## Environment

```bash
conda env create -f environment.yml
conda activate fluid
bash repro/bootstrap_env.sh
```

## Verify tracked artifacts

```bash
conda run --no-capture-output -n fluid python repro/verify_artifacts.py --groups checks,benchmarks,solver,figures
```

## Re-run computations

The runner can execute groups declared in `repro/manifest.json`:

```bash
conda run --no-capture-output -n fluid python repro/run_pipeline.py --groups checks,benchmarks,solver,figures --skip-existing
```

Use `--list` to inspect available steps:

```bash
conda run --no-capture-output -n fluid python repro/run_pipeline.py --list
```

## Optional proposal channel

The deterministic mutation library is the canonical search path for reproduced
results. The optional LLM proposal channel can be enabled with `GEMINI_API_KEY`
for proposal-channel experiments.
