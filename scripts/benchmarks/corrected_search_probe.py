#!/usr/bin/env python3
"""
corrected_search_probe.py
=========================
Run a small corrected-oracle MCTS probe from a Leonard+canonical-WALE seed.

This is not the final large search. Its purpose is narrower:
  * verify that the corrected search no longer collapses onto the archived
    Laplacian-strain hybrid,
  * show which compact closure families are promoted once proposal-order bias
    is removed and corrected-aware fallback proposals are enabled.

Outputs
-------
results/corrected_search/probe_top_models.csv
"""

from __future__ import annotations

import csv
import os

import numpy as np

from sgs_discovery.mcts_agent import NeuroSymbolicMCTS
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import corrected_root_library


def main() -> None:
    out_dir = "results/corrected_search"
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(0)
    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")
    engine = TensorSymbolicEngine()
    optimizer = LeafNodeOptimizer(
        engine,
        [oracle_iso, oracle_chan],
        lambda_pi=1.0,
        lambda_diss=1.0,
        lambda_l1=1e-5,
        max_iter=120,
    )

    root_expr = dict(corrected_root_library())["L_WALE"]

    mcts = NeuroSymbolicMCTS(
        engine=engine,
        oracles=[oracle_iso, oracle_chan],
        optimizer=optimizer,
        exploration_weight=1.414,
        max_depth=3,
        gemini_api_key="",
    )

    mcts.search(root_expr=root_expr, n_iterations=3, log_interval=1)
    top_nodes = mcts.get_top_k(10)

    out_path = os.path.join(out_dir, "probe_top_models.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "loss", "constants", "expr"],
        )
        writer.writeheader()
        for rank, node in enumerate(top_nodes, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "loss": float(node.mse),
                    "constants": str(node.best_constants),
                    "expr": str(node.expr),
                }
            )

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
