#!/usr/bin/env python3
"""
main.py — Neuro-Symbolic-SGS: End-to-End Discovery Loop
=========================================================
Assembles all modules and runs an MCTS search to discover
SGS closure expressions for Large Eddy Simulation.

After search: generates paper-ready CSV files for a priori analysis.

Usage
-----
    python3 main.py
"""

from __future__ import annotations

import os
import json
import time

import numpy as np
from scipy import stats

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.grid_metrics import reshape_tensor_field
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.mcts_agent import NeuroSymbolicMCTS, Node
from sgs_discovery.symbolic_closures import corrected_root_library


# ══════════════════════════════════════════════════════════════════
# Post-processing utilities
# ══════════════════════════════════════════════════════════════════

def predict_tau(engine, expr, oracle, best_constants):
    """Evaluate the best expression on FULL data (no sub-sampling)."""
    return engine.lambdify_tensor_expr(
        expr,
        oracle.S, oracle.Omega, oracle.L,
        oracle.S_d, oracle.S_jaumann, oracle.Lap_S, oracle.Delta,
        oracle.omega_vec, oracle.W_vec, oracle.h_scalar,
        best_constants,
    )


def save_dissipation_scatter(engine, expr, best_constants, oracle, name, out_dir):
    """Module 5.1: Pi = -tau_ij * S_ij scatter CSV (20k sampled points)."""
    tau_pred = predict_tau(engine, expr, oracle, best_constants)
    tau_true = oracle.tau

    pi_pred = -np.einsum('nij,nij->n', tau_pred, oracle.S)
    pi_true = -np.einsum('nij,nij->n', tau_true, oracle.S)

    # Sample 20k points
    n = len(pi_pred)
    n_sample = min(20000, n)
    idx = np.random.choice(n, size=n_sample, replace=False)

    import csv
    path = os.path.join(out_dir, f"dissipation_scatter_{name}.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["pi_true", "pi_pred"])
        for i in idx:
            writer.writerow([float(pi_true[i]), float(pi_pred[i])])
    print(f"  📊 Dissipation scatter ({name}): {path}")


def save_pearson_correlation(engine, expr, best_constants, oracle, name, out_dir):
    """Module 5.2: Full-tensor Pearson correlation (6 independent components)."""
    tau_pred = predict_tau(engine, expr, oracle, best_constants)
    tau_true = oracle.tau

    components = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    labels = ["11", "22", "33", "12", "13", "23"]

    import csv
    path = os.path.join(out_dir, f"pearson_correlations_{name}.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["component", "pearson_r", "p_value"])
        for (i, j), label in zip(components, labels):
            r, p = stats.pearsonr(tau_true[:, i, j], tau_pred[:, i, j])
            writer.writerow([label, float(r), float(p)])
            print(f"    τ_{label}: r = {r:.6f}")
    print(f"  📊 Pearson correlations ({name}): {path}")


def save_wall_normal_profile(engine, expr, best_constants, oracle, out_dir):
    """Module 5.3: <tau_12> vs y for channel flow."""
    if oracle.y_coords is None:
        print("  ⚠ No y_coords available — skipping wall-normal profile")
        return

    tau_pred = predict_tau(engine, expr, oracle, best_constants)
    tau_true = oracle.tau

    tau_true_3d = reshape_tensor_field(oracle, tau_true)
    tau_pred_3d = reshape_tensor_field(oracle, tau_pred)
    tau12_true_mean = np.mean(tau_true_3d[..., 0, 1], axis=(0, 2))
    tau12_pred_mean = np.mean(tau_pred_3d[..., 0, 1], axis=(0, 2))

    import csv
    path = os.path.join(out_dir, "wall_normal_tau12_profile.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["y", "tau12_true_mean", "tau12_pred_mean"])
        for y, tau_true_mean, tau_pred_mean in zip(oracle.y_coords, tau12_true_mean, tau12_pred_mean):
            writer.writerow([float(y), float(tau_true_mean), float(tau_pred_mean)])
    print(f"  📊 Wall-normal τ_12 profile: {path}")


def export_mcts_tree(root: Node, out_dir: str):
    """Module 5.4: Export MCTS tree as JSON."""
    def node_to_dict(node: Node) -> dict:
        return {
            "expr": str(node.expr),
            "N": node.N,
            "Q": round(node.Q, 6),
            "mse": node.mse if np.isfinite(node.mse) else None,
            "value": round(node.value, 6),
            "best_constants": node.best_constants,
            "children": [node_to_dict(c) for c in node.children],
        }

    tree = node_to_dict(root)
    path = os.path.join(out_dir, "mcts_tree.json")
    with open(path, 'w') as f:
        json.dump(tree, f, indent=2, default=str)
    print(f"  🌳 MCTS tree exported: {path}")


# ══════════════════════════════════════════════════════════════════
# Main
def main() -> None:
    print("=" * 72)
    print("  Neuro-Symbolic-SGS  —  Automated SGS Closure Discovery")
    print("  ═══════════════════════════════════════════════════════")
    print("  Features: S, Omega, L, S_d, S_j, Lap_S, omega, W, Delta")
    print("  Constants: c1..c8 (sparse L1-regularized limit)")
    print("  Datasets: Isotropic (wrap) + Channel (nearest) w/ Local NMSE")
    print("=" * 72)
    print()

    # ── 1. Data Oracles ──
    oracle_iso = JHTDBOracle(
        h5_path="jhtdb_u_tensor_64.h5",
        filter_width=1.0,
        boundary_mode='wrap',
    )
    oracle_chan = JHTDBOracle(
        h5_path="channel_u_tensor_64.h5",
        filter_width=1.0,
        boundary_mode='nearest',
    )
    print(f"[DATA ISO]  {oracle_iso} | Samples: {len(oracle_iso)}")
    print(f"[DATA CHAN] {oracle_chan} | Samples: {len(oracle_chan)}")
    print()

    # ── 2. Engine & Optimizer ──
    engine = TensorSymbolicEngine()
    optimizer = LeafNodeOptimizer(engine, [oracle_iso, oracle_chan], lambda_pi=1.0, max_iter=400)
    print(f"[PHYS]  {engine}")
    print(f"[OPT]   {optimizer}")
    print(f"[RULES] Dual-Oracle NMSE + 10% Sub-sampling + Bounded Physics Backscatter")
    print()

    # ── 3. Seed Expression ──
    # Seeding with a combination of structural anchor (L) and wall-damping (S_d)
    initial_expr = dict(corrected_root_library())["L_WALE"]
    print(f"[SEED]  Initial closure:  τ_ij = {initial_expr}")
    print()

    # ── 4. MCTS Search ──
    mcts = NeuroSymbolicMCTS(
    
        engine=engine,
        oracles=[oracle_iso, oracle_chan],
        optimizer=optimizer,
        exploration_weight=1.414,
        max_depth=10,
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        http_proxy=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
    )
    print(f"[MCTS]  {mcts}")
    print("-" * 72)

    t0 = time.perf_counter()
    root = mcts.search(
        root_expr=initial_expr,
        n_iterations=500,
        log_interval=10,
    )
    elapsed = time.perf_counter() - t0

    # ── 5. Report Top-3 ──
    print()
    print("=" * 72)
    print(f"  SEARCH COMPLETE — Elapsed: {elapsed:.1f}s")
    print("=" * 72)
    print()

    top_k = mcts.get_top_k(k=3)
    if not top_k:
        print("  ⚠  No valid closures found.")
        return

    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║            Top-3 SGS Closure Expressions Found             ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

    for rank, node in enumerate(top_k, start=1):
        print(f"  ┌─── Rank #{rank} ───────────────────────────────────────────┐")
        print(f"  │  Loss:       {node.mse:.6e}")
        print(f"  │  Value:      {node.value:.6f}")
        print(f"  │  Constants:  {node.best_constants}")
        print(f"  │  Expression:")
        print(f"  │    τ_ij = {node.expr}")
        print(f"  └────────────────────────────────────────────────────────────┘")
        print()

    # ══════════════════════════════════════════════════════════════════
    # 6. PAPER-READY POST-PROCESSING (100% full data, no sub-sampling)
    # ══════════════════════════════════════════════════════════════════
    print("=" * 72)
    print("  POST-PROCESSING: A Priori Analysis")
    print("=" * 72)
    print()

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    best_node = top_k[0]
    best_expr = best_node.expr
    best_consts = best_node.best_constants
    print(f"  Best model: τ_ij = {best_expr}")
    print(f"  Constants:  {best_consts}")
    print()

    # 6.1 SGS Dissipation Scatter
    print("  [1/4] Computing SGS dissipation rate (Pi) scatter...")
    save_dissipation_scatter(engine, best_expr, best_consts, oracle_iso, "ISO", out_dir)
    save_dissipation_scatter(engine, best_expr, best_consts, oracle_chan, "CHAN", out_dir)

    # 6.2 Pearson Correlation Matrix
    print("\n  [2/4] Computing full-tensor Pearson correlations...")
    save_pearson_correlation(engine, best_expr, best_consts, oracle_iso, "ISO", out_dir)
    save_pearson_correlation(engine, best_expr, best_consts, oracle_chan, "CHAN", out_dir)

    # 6.3 Channel Wall-Normal Profiles
    print("\n  [3/4] Computing channel wall-normal τ_12 profiles...")
    save_wall_normal_profile(engine, best_expr, best_consts, oracle_chan, out_dir)

    # 6.4 MCTS Tree Export
    print("\n  [4/4] Exporting MCTS tree structure...")
    export_mcts_tree(root, out_dir)

    print(f"\n🎉 All post-processing complete! Results saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
