# Post-Fix Comparison Pack

This directory records post-dimensional-consistency versions of the experimental
tables and figures that appeared in the manuscript. The original manuscript
files were left untouched on purpose. These files are a side-by-side lookup pack
for later comparison against the older paper snapshot.

## Model-name mapping

- `Champion` = current dimensionally consistent Laplacian-strain branch.
- `Jaumann_hybrid` = current dimensionally consistent Jaumann hybrid with a `Delta^2 S_j` term.
- `Wstretch_hybrid` = current dimensionally consistent vortex-stretching hybrid.
- `WALE_canonical_physical` = the modern-baseline table row corresponding to canonical WALE.

## Table mapping

- Old `tab:baseline` -> `tables/tab_baseline_updated.*`
- Old `tab:objective_ablation` -> `tables/tab_objective_ablation_updated.*`
- Old `tab:holdout` -> `tables/tab_holdout_updated.*`
- Old `tab:transfer_combined` -> `tables/tab_transfer_combined_updated.*`
- Old `tab:external` -> `tables/tab_external_updated.*`
- Old `tab:modern_baselines` -> `tables/tab_modern_baselines_updated.*`
- Old `tab:aposteriori_periodic` -> `tables/tab_aposteriori_periodic_updated.*`
- Old `tab:aposteriori_channel` -> `tables/tab_aposteriori_channel_updated.*`
- Old expanded-search discussion -> `tables/extra_search_campaign_updated.*`

## Figure mapping

- Old `fig:channel_physical` -> `figures/fig_channel_physical_updated.pdf`
- Old `fig:validation_map` -> `figures/fig_validation_map_updated.pdf`
- Old `fig:solver_screen` -> `figures/fig_solver_screen_updated.pdf`

## Source result files

- `results/baselines/summary_metrics.csv`
- `results/holdout/blocked_holdout_summary.csv`
- `results/external_cutout/shifted_cutout_summary.csv`
- `results/temporal_transfer/temporal_transfer_summary.csv`
- `results/filter_transfer/filter_width_transfer_summary.csv`
- `results/isotropic_ood/isotropic_ood_summary.csv`
- `results/channel5200_ood/channel5200_transfer_summary.csv`
- `results/modern_baselines/apriori_summary.csv`
- `results/objective_ablation/objective_ablation_summary.csv`
- `results/corrected_research_large/apriori_screen.csv`
- `results/aposteriori_isotropic/isotropic_short_horizon_summary.csv`
- `results/aposteriori_rollout/isotropic_rollout_summary.csv`
- `results/aposteriori_channel/channel_rollout_pchip_summary_4frames.csv`

## Notes

- These artifacts reflect the rerun, physically consistent benchmark chain.
- Some old narrative claims no longer hold under the updated results; this pack
  is intentionally descriptive rather than editorial.
- The objective-ablation table now corresponds to the current `Champion`
  definition used in code, not the older pre-fix naked-dimensional form.
