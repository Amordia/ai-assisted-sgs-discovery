#!/usr/bin/env python3
"""
benchmark_models.py
===================
Evaluate repository-native SGS baselines and the archived champion model
on the same dual-oracle setup used by the neuro-symbolic search.

Outputs:
  - results/baselines/summary_metrics.csv
  - results/baselines/component_metrics.csv
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import stats

from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.grid_metrics import wall_profile_corr, wall_profile_rmse
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs


@dataclass
class ModelSpec:
    name: str
    expr: sp.MatrixExpr
    optimize: bool = True
    fixed_constants: dict[str, float] | None = None


def mean_component_correlation(tau_true: np.ndarray, tau_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    labels = ["11", "22", "33", "12", "13", "23"]
    by_component: dict[str, float] = {}
    values = []
    for (i, j), label in zip(components, labels):
        pred = tau_pred[:, i, j]
        true = tau_true[:, i, j]
        if np.std(pred) < 1e-15 or np.std(true) < 1e-15:
            r = np.nan
        else:
            r, _ = stats.pearsonr(true, pred)
        by_component[label] = float(r)
        values.append(float(r))
    return float(np.nanmean(values)), by_component


def dissipation_metrics(oracle: JHTDBOracle, tau_pred: np.ndarray) -> dict[str, float]:
    pi_true = -np.einsum("nij,nij->n", oracle.tau, oracle.S)
    pi_pred = -np.einsum("nij,nij->n", tau_pred, oracle.S)
    if np.std(pi_true) < 1e-15 or np.std(pi_pred) < 1e-15:
        pi_corr = np.nan
    else:
        pi_corr = float(np.corrcoef(pi_true, pi_pred)[0, 1])
    return {
        "pi_corr": pi_corr,
        "pi_rmse": float(np.sqrt(np.mean((pi_pred - pi_true) ** 2))),
        "pi_bias": float(np.mean(pi_pred - pi_true)),
        "pi_true_neg_frac": float(np.mean(pi_true < 0.0)),
        "pi_pred_neg_frac": float(np.mean(pi_pred < 0.0)),
    }

def wall_profile_metrics(oracle: JHTDBOracle, tau_pred: np.ndarray) -> dict[str, float]:
    if oracle.y_coords is None:
        return {}
    return {
        "wall_tau12_corr": wall_profile_corr(oracle, tau_pred),
        "wall_tau12_rmse": wall_profile_rmse(oracle, tau_pred),
    }


def main() -> None:
    np.random.seed(0)
    out_dir = "results/baselines"
    os.makedirs(out_dir, exist_ok=True)

    oracle_iso = JHTDBOracle(
        h5_path="jhtdb_u_tensor_64.h5",
        filter_width=1.0,
        boundary_mode="wrap",
    )
    oracle_chan = JHTDBOracle(
        h5_path="channel_u_tensor_64.h5",
        filter_width=1.0,
        boundary_mode="nearest",
    )

    engine = TensorSymbolicEngine()
    optimizer = LeafNodeOptimizer(
        engine,
        [oracle_iso, oracle_chan],
        lambda_pi=1.0,
        lambda_diss=1.0,
        lambda_l1=1e-5,
        max_iter=400,
    )

    exprs = native_model_exprs()

    models = [
        ModelSpec("Zero", sp.zeros(3), optimize=False, fixed_constants={}),
        ModelSpec("Leonard", exprs["Leonard"]),
        ModelSpec("WALE_like", exprs["WALE_like"]),
        ModelSpec("Smagorinsky_like", exprs["Smagorinsky_like"]),
        ModelSpec("WALE_canonical", exprs["WALE_canonical"]),
        ModelSpec("Mixed_L_WALE_like", exprs["Mixed_L_WALE_like"]),
        ModelSpec("Jaumann_hybrid", exprs["Jaumann_hybrid"]),
        ModelSpec("Champion", exprs["Champion"]),
        ModelSpec("Wstretch_hybrid", exprs["Wstretch_hybrid"]),
    ]

    summary_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []

    for model in models:
        print("=" * 72)
        print(f"[MODEL] {model.name}")
        print(f"expr = {model.expr}")

        if model.optimize:
            best_constants, opt_loss = optimizer.optimize(model.expr)
        else:
            best_constants = model.fixed_constants or {}
            if best_constants:
                tau_iso = engine.lambdify_tensor_expr(
                    model.expr,
                    oracle_iso.S,
                    oracle_iso.Omega,
                    oracle_iso.L,
                    oracle_iso.S_d,
                    oracle_iso.S_jaumann,
                    oracle_iso.Lap_S,
                    oracle_iso.Delta,
                    oracle_iso.omega_vec,
                    oracle_iso.W_vec,
                    oracle_iso.h_scalar,
                    best_constants,
                )
                tau_chan = engine.lambdify_tensor_expr(
                    model.expr,
                    oracle_chan.S,
                    oracle_chan.Omega,
                    oracle_chan.L,
                    oracle_chan.S_d,
                    oracle_chan.S_jaumann,
                    oracle_chan.Lap_S,
                    oracle_chan.Delta,
                    oracle_chan.omega_vec,
                    oracle_chan.W_vec,
                    oracle_chan.h_scalar,
                    best_constants,
                )
                opt_loss = oracle_iso.evaluate_mse(tau_iso) + oracle_chan.evaluate_mse(tau_chan)
            else:
                opt_loss = oracle_iso.evaluate_mse(np.zeros_like(oracle_iso.tau)) + oracle_chan.evaluate_mse(np.zeros_like(oracle_chan.tau))

        print(f"constants = {best_constants}")
        print(f"loss = {opt_loss:.6e}")

        tau_iso = engine.lambdify_tensor_expr(
            model.expr,
            oracle_iso.S,
            oracle_iso.Omega,
            oracle_iso.L,
            oracle_iso.S_d,
            oracle_iso.S_jaumann,
            oracle_iso.Lap_S,
            oracle_iso.Delta,
            oracle_iso.omega_vec,
            oracle_iso.W_vec,
            oracle_iso.h_scalar,
            best_constants,
        )
        tau_chan = engine.lambdify_tensor_expr(
            model.expr,
            oracle_chan.S,
            oracle_chan.Omega,
            oracle_chan.L,
            oracle_chan.S_d,
            oracle_chan.S_jaumann,
            oracle_chan.Lap_S,
            oracle_chan.Delta,
            oracle_chan.omega_vec,
            oracle_chan.W_vec,
            oracle_chan.h_scalar,
            best_constants,
        )

        nmse_iso = oracle_iso.evaluate_mse(tau_iso)
        nmse_chan = oracle_chan.evaluate_mse(tau_chan)
        mean_r_iso, comp_iso = mean_component_correlation(oracle_iso.tau, tau_iso)
        mean_r_chan, comp_chan = mean_component_correlation(oracle_chan.tau, tau_chan)
        pi_iso = dissipation_metrics(oracle_iso, tau_iso)
        pi_chan = dissipation_metrics(oracle_chan, tau_chan)
        wall_chan = wall_profile_metrics(oracle_chan, tau_chan)

        summary_rows.append({
            "model": model.name,
            "loss": float(opt_loss),
            "constants": str(best_constants),
            "nmse_iso": float(nmse_iso),
            "nmse_chan": float(nmse_chan),
            "mean_r_iso": mean_r_iso,
            "mean_r_chan": mean_r_chan,
            "tau12_r_chan": comp_chan["12"],
            "pi_r_iso": pi_iso["pi_corr"],
            "pi_r_chan": pi_chan["pi_corr"],
            "wall_tau12_corr_chan": wall_chan.get("wall_tau12_corr"),
        })

        for dataset_name, comps in [("ISO", comp_iso), ("CHAN", comp_chan)]:
            row = {"model": model.name, "dataset": dataset_name}
            row.update({f"r_{k}": v for k, v in comps.items()})
            component_rows.append(row)

    summary_path = os.path.join(out_dir, "summary_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    component_path = os.path.join(out_dir, "component_metrics.csv")
    with open(component_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(component_rows[0].keys()))
        writer.writeheader()
        writer.writerows(component_rows)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {component_path}")


if __name__ == "__main__":
    main()
