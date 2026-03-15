# Methodology Check Summary

| Aspect | What was improved | Primary output | Key finding |
| --- | --- | --- | --- |
| LLM channel | Added explicit deterministic-vs-Gemini ablation and improved the Gemini parser namespace in `mcts_agent.py` | `results/llm_ablation/search_summary.csv` | Deterministic search remains stronger (`1.8545` vs `2.2225` mean best loss); Gemini is optional, not required |
| Commutation error | Added direct quantification of filter-derivative commutation error on isotropic and channel benchmarks | `results/commutation/commutation_summary.csv` | Channel strain commutation is moderate (`0.069` RMS relative), but Laplacian-strain commutation is large (`0.414` RMS relative; near-wall peak `18.07`) |
| Subsample sensitivity | Made optimizer subsample ratio configurable and swept `1%`-`50%` with repeated refits | `results/subsample_sensitivity/ratio_sweep_summary.csv` | Main fitted metrics are nearly invariant to subsample ratio; extreme-event coverage scales roughly with the sample ratio |
