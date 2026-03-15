# Methodology Checks After the Dimensional-Consistency Fix

This directory records follow-up experiments that address three recurring
methodological questions about the final code path without modifying the paper
manuscript directly.

## 1. LLM proposal-channel ablation

Files:

- `results/llm_ablation/search_rows.csv`
- `results/llm_ablation/search_summary.csv`
- `proposal_channel_ablation.py`
- `mcts_agent.py`

Setup:

- Same corrected roots (`Leonard`, `L_WALE`), same seeds (`11`, `29`), same
  dual-oracle objective, same MCTS depth/iteration budget.
- Two search channels are compared:
  - `deterministic`: hard-coded mutation library only.
  - `gemini`: Gemini proposal channel enabled through the repository `.env`.
- The final parser namespace was improved so Gemini outputs using
  `sqrt(...)`, `Abs(...)`, `exp(...)`, and `log(...)` are accepted instead of
  being discarded as avoidable parse errors.

Key result:

- Deterministic search reaches a lower mean best loss:
  - deterministic: `1.8545 +- 0.0130`
  - gemini: `2.2225 +- 0.1074`
- Gemini runs are faster only because they evaluate fewer nodes:
  - deterministic: mean tree size `20.5`
  - gemini: mean tree size `6.5`
- After the parser fix, Gemini proposals are fully parsed in the tracked
  ablation runs (`llm_success_fraction = 1.0`).

Interpretation:

- In the present final implementation, the LLM channel is not required to
  recover the strongest reported search outcomes.
- Its practical role is proposal diversification rather than a necessary source
  of the paper's main discovery.
- The deterministic mutation library remains the canonical reproducible path.

## 2. Commutation-error quantification on the current filtering/discretization path

Files:

- `results/commutation/commutation_summary.csv`
- `results/commutation/channel_commutation_profiles.csv`
- `results/commutation/commutation_profiles.pdf`
- `commutation_error_study.py`

Setup:

- Compare `G(du/dx)` against `d(Gu)/dx` on the same datasets used by the SGS
  oracle.
- Use the current repository filter path:
  - isotropic: Gaussian filtering with `mode='wrap'`
  - channel: Gaussian filtering with `mode='nearest'` and explicit non-uniform
    wall-normal coordinates for differentiation

Key result:

- Channel data:
  - strain relative commutation RMS: `0.0690`
  - Laplacian-strain relative commutation RMS: `0.4143`
  - near-wall/profile peak for Laplacian-strain relative commutation: `18.07`
- Isotropic data:
  - strain relative commutation RMS: `0.6117`
  - Laplacian-strain relative commutation RMS: `0.5481`

Interpretation:

- Commutation error is not negligible on the current benchmark path.
- It is moderate for first-order wall-bounded derived features, but large for
  the high-order Laplacian-strain feature, especially near the wall.
- This should be treated as a real limitation of the current `a priori`
  benchmark, not as a negligible implementation detail.

## 3. Frozen-subsample sensitivity and extreme-event coverage

Files:

- `results/subsample_sensitivity/ratio_sweep_rows.csv`
- `results/subsample_sensitivity/ratio_sweep_summary.csv`
- `results/subsample_sensitivity/subsample_stability.pdf`
- `subsample_sensitivity.py`
- `optimizer.py`

Setup:

- Refit two representative symbolic models:
  - `Jaumann_hybrid`
  - `Champion`
- Sweep the frozen optimization subsample ratio over
  `1%, 2%, 5%, 10%, 20%, 50%`
- Repeat each ratio with four independent seeds.
- Measure both final full-field metrics and the fraction of the strongest true
  backscatter events captured by the sampled indices.
- The optimizer was improved so the frozen subsample ratio is now an explicit
  public parameter rather than a hard-coded internal constant.

Key result:

- `Jaumann_hybrid` channel stress correlation remains in the narrow range
  `0.1077` to `0.1132` across the entire ratio sweep.
- `Champion` channel stress correlation remains in the narrow range
  `0.2962` to `0.2977` across the entire ratio sweep.
- Coverage of the strongest backscatter events scales roughly linearly with the
  chosen sample ratio:
  - at `1%`, top-1% channel-event coverage is about `1.0%`
  - at `10%`, top-1% channel-event coverage is about `10.4%`
  - at `50%`, top-1% channel-event coverage is about `50.5%`

Interpretation:

- This is not a formal proof of unbiasedness.
- It is, however, a direct empirical sensitivity check showing that the current
  frozen-subsample optimizer is much more stable than the criticism would imply
  on the final code path.
- Within the tested range, the main fitted metrics are nearly invariant to the
  subsample ratio.
