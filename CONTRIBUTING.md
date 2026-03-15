# Contributing

This repository is primarily a research artifact for reproducing the SGS
closure discovery experiments. Contributions should preserve reproducibility.

## Development workflow

1. Create a branch from `main`.
2. Keep code changes separate from regenerated result artifacts when practical.
3. Run the targeted reproduction checks before opening a pull request.
4. Do not commit raw JHTDB HDF5 files, local `.env` files, or machine-specific
   cache products.

## Validation

Use the manifest runner for reproducible checks:

```bash
conda run --no-capture-output -n fluid python repro/verify_artifacts.py --groups checks,benchmarks,solver
```

For code changes that affect tensor features, run:

```bash
conda run --no-capture-output -n fluid python tests/test_nonuniform_gradients.py
```

## Data and credentials

JHTDB credentials must be supplied through environment variables or a local
`.env` file. Never commit credentials or downloaded `.h5` files.
