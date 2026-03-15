# AI-assisted SGS Discovery

This repository contains the final code, benchmark outputs, and reproducibility metadata for the geometry-consistent neuro-symbolic discovery workflow. The goal is reproducibility: a new user should be able to recreate the tracked CSV artifacts and figures from the repository state without depending on local machine paths or unpublished scratch code.

## Scope

The repository keeps the final versions of:

- the corrected SGS oracle and grid-metric handling;
- the symbolic library, optimizer, and MCTS search code;
- matched `a priori`, transfer, and solver-coupled benchmark scripts;
- the canonical result CSVs used to build the paper tables and figures;
- compact derived tables and figures for result inspection.

Raw JHTDB HDF5 files are intentionally not tracked in Git. They are regenerated through the download scripts and require a valid JHTDB token.

## Environment

The canonical environment is the `fluid` conda environment described in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate fluid
bash repro/bootstrap_env.sh
```

The bootstrap step installs the vendored `givernylocal` and `giverny` packages from the repository into the active environment.

## Configuration

Copy `.env.example` to `.env`, or export the same variables in your shell.

Required:

- `JHTDB_AUTH_TOKEN`: token used by all download scripts.

Optional:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`

The tracked benchmark results do not depend on Gemini access. The canonical reproducible path uses the deterministic fallback proposal mechanism in the search code together with fixed seeds.

## Repository layout

- `src/sgs_discovery/`: importable core package for oracles, grid metrics, symbolic tensors, search, optimization, and shared solver utilities.
- `scripts/download/`: JHTDB cutout download entry points.
- `scripts/benchmarks/`: `a priori`, transfer, search, ablation, and methodology-check scripts.
- `scripts/solver/`: solver-coupled `a posteriori` rollout screens.
- `scripts/figures/`: derived table and figure builders.
- `repro/`: canonical manifest, runner, environment bootstrap, and artifact verifier.
- `results/`: tracked compact CSV/PDF/PNG result artifacts.
- `paper/`: publication-facing placeholder; journal templates and manuscript build products are not part of the public code release.
- `docs/`: data-access and reproducibility notes.
- `tests/`: regression tests for numerical assumptions.

## Canonical reproduction flow

List the configured steps:

```bash
conda run --no-capture-output -n fluid python repro/run_pipeline.py --list
```

Run the synthetic regression test plus all paper-facing computations, skipping work whose outputs already exist:

```bash
conda run --no-capture-output -n fluid python repro/run_pipeline.py \
  --groups tests,benchmarks,solver \
  --skip-existing
```

Run the methodology-check bundle that addresses common reviewer concerns about
the final code path:

```bash
conda run --no-capture-output -n fluid python repro/run_pipeline.py \
  --groups checks \
  --skip-existing
```

If you need to regenerate raw datasets from JHTDB first:

```bash
conda run --no-capture-output -n fluid python repro/run_pipeline.py --groups downloads
```

Validate that the tracked artifacts exist and still match the expected schema:

```bash
conda run --no-capture-output -n fluid python repro/verify_artifacts.py \
  --groups tests,benchmarks,solver
```

To also validate the methodology-check artifacts:

```bash
conda run --no-capture-output -n fluid python repro/verify_artifacts.py \
  --groups checks
```

To also require the raw HDF5 inputs:

```bash
conda run --no-capture-output -n fluid python repro/verify_artifacts.py \
  --groups downloads,tests,benchmarks,solver \
  --check-inputs
```

## Canonical tracked outputs

The core result directories kept in Git are:

- `results/baselines`
- `results/holdout`
- `results/external_cutout`
- `results/temporal_transfer`
- `results/filter_transfer`
- `results/isotropic_ood`
- `results/channel5200_ood`
- `results/modern_baselines`
- `results/corrected_research_large`
- `results/corrected_scan`
- `results/corrected_search`
- `results/cross_flow`
- `results/objective_ablation`
- `results/aposteriori_isotropic`
- `results/aposteriori_rollout`
- `results/aposteriori_channel`
- `results/commutation`
- `results/llm_ablation`
- `results/subsample_sensitivity`
- `results/methodology_checks`
- `results/comparison_post_dimensional_fix`

The comparison pack of derived tables and figures is regenerated with:

- `scripts/figures/build_postfix_comparison_pack.py`

The extended reproduction flow also keeps:

- the curated corrected candidate scan (`scripts/benchmarks/corrected_candidate_scan.py`);
- the lightweight corrected search probe (`scripts/benchmarks/corrected_search_probe.py`);
- the coefficient robustness and cross-flow transfer study (`scripts/benchmarks/cross_flow_seed_study.py`);
- the optimizer-objective ablation (`scripts/benchmarks/objective_ablation.py`);
- the short-horizon isotropic solver screen (`scripts/solver/aposteriori_isotropic_les.py`);
- the commutation-error quantification (`scripts/benchmarks/commutation_error_study.py`);
- the deterministic-vs-Gemini proposal-channel ablation (`scripts/benchmarks/proposal_channel_ablation.py`);
- the frozen-subsample sensitivity sweep (`scripts/benchmarks/subsample_sensitivity.py`).

## Notes on determinism

- The corrected benchmark and search scripts use fixed random seeds where the paper depends on exact ranked outputs.
- Search scripts can optionally talk to Gemini, but the reproducible paper path does not require that service.
- The repository uses a package-plus-scripts layout so that core algorithms are importable while experiment entry points remain easy to audit.
