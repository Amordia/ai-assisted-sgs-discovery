#!/usr/bin/env python3
"""
objective_ablation.py
=====================
Quantify how the optimizer's physics terms affect the champion model.

The same symbolic champion expression is re-fitted under multiple objective
choices and evaluated on the archived full-field cutouts. Repeating the fit
over several random frozen sub-samples distinguishes genuine objective effects
from optimizer-seed noise.

Outputs:
  - results/objective_ablation/objective_ablation_rows.csv
  - results/objective_ablation/objective_ablation_summary.csv
"""

from __future__ import annotations

import csv
import os

import numpy as np
import sympy as sp
from scipy import stats

from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.grid_metrics import wall_profile_corr
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs


SEEDS = tuple(range(12))


def mean_component_correlation(
    tau_true: np.ndarray, tau_pred: np.ndarray
) -> tuple[float, dict[str, float]]:
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    labels = ["11", "22", "33", "12", "13", "23"]
    detail: dict[str, float] = {}
    values = []
    for (i, j), label in zip(components, labels):
        true = tau_true[:, i, j]
        pred = tau_pred[:, i, j]
        if np.std(true) < 1e-15 or np.std(pred) < 1e-15:
            r = np.nan
        else:
            r, _ = stats.pearsonr(true, pred)
        detail[label] = float(r)
        values.append(float(r))
    return float(np.nanmean(values)), detail


def dissipation_stats(oracle: JHTDBOracle, tau_pred: np.ndarray) -> dict[str, float]:
    pi_true = -np.einsum("nij,nij->n", oracle.tau, oracle.S)
    pi_pred = -np.einsum("nij,nij->n", tau_pred, oracle.S)
    if np.std(pi_true) < 1e-15 or np.std(pi_pred) < 1e-15:
        pi_corr = float("nan")
    else:
        pi_corr = float(np.corrcoef(pi_true, pi_pred)[0, 1])
    return {
        "pi_corr": pi_corr,
        "pi_bias": float(np.mean(pi_pred - pi_true)),
        "pi_rmse": float(np.sqrt(np.mean((pi_pred - pi_true) ** 2))),
        "pi_pred_neg_frac": float(np.mean(pi_pred < 0.0)),
    }

def main() -> None:
    np.random.seed(0)
    out_dir = "results/objective_ablation"
    os.makedirs(out_dir, exist_ok=True)

    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")

    engine = TensorSymbolicEngine()
    champion_expr = native_model_exprs()["Champion"]

    objectives = [
        {"name": "NMSE_only", "lambda_pi": 0.0, "lambda_diss": 0.0},
        {"name": "NMSE_plus_backscatter", "lambda_pi": 1.0, "lambda_diss": 0.0},
        {"name": "NMSE_plus_dissipation", "lambda_pi": 0.0, "lambda_diss": 1.0},
        {"name": "Full_objective", "lambda_pi": 1.0, "lambda_diss": 1.0},
    ]

    rows: list[dict[str, object]] = []
    for obj_idx, objective in enumerate(objectives):
        optimizer = LeafNodeOptimizer(
            engine,
            [oracle_iso, oracle_chan],
            lambda_pi=objective["lambda_pi"],
            lambda_diss=objective["lambda_diss"],
            lambda_l1=1e-5,
            max_iter=400,
        )
        for seed in SEEDS:
            fit_seed = 30_000 + 100 * obj_idx + seed
            np.random.seed(fit_seed)
            constants, train_loss = optimizer.optimize(champion_expr)

            tau_iso = engine.lambdify_tensor_expr(
                champion_expr,
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
                constants,
            )
            tau_chan = engine.lambdify_tensor_expr(
                champion_expr,
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
                constants,
            )

            mean_r_iso, _ = mean_component_correlation(oracle_iso.tau, tau_iso)
            mean_r_chan, comps_chan = mean_component_correlation(oracle_chan.tau, tau_chan)
            pi_iso = dissipation_stats(oracle_iso, tau_iso)
            pi_chan = dissipation_stats(oracle_chan, tau_chan)

            row = {
                "objective": objective["name"],
                "seed": seed,
                "fit_seed": fit_seed,
                "lambda_pi": objective["lambda_pi"],
                "lambda_diss": objective["lambda_diss"],
                "train_loss": float(train_loss),
                "constants": str(constants),
                "nmse_iso": float(oracle_iso.evaluate_mse(tau_iso)),
                "nmse_chan": float(oracle_chan.evaluate_mse(tau_chan)),
                "mean_r_iso": mean_r_iso,
                "mean_r_chan": mean_r_chan,
                "tau12_r_chan": comps_chan["12"],
                "pi_r_iso": pi_iso["pi_corr"],
                "pi_r_chan": pi_chan["pi_corr"],
                "pi_bias_iso": pi_iso["pi_bias"],
                "pi_bias_chan": pi_chan["pi_bias"],
                "pi_rmse_iso": pi_iso["pi_rmse"],
                "pi_rmse_chan": pi_chan["pi_rmse"],
                "neg_frac_iso": pi_iso["pi_pred_neg_frac"],
                "neg_frac_chan": pi_chan["pi_pred_neg_frac"],
                "wall_tau12_corr_chan": wall_profile_corr(oracle_chan, tau_chan),
            }
            rows.append(row)

            print(
                f"objective={objective['name']:>22s} seed={seed:02d} "
                f"mean_r_chan={mean_r_chan:.3f} pi_r_chan={pi_chan['pi_corr']:.3f}"
            )

    rows_path = os.path.join(out_dir, "objective_ablation_rows.csv")
    with open(rows_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        "train_loss",
        "nmse_iso",
        "nmse_chan",
        "mean_r_iso",
        "mean_r_chan",
        "tau12_r_chan",
        "pi_r_iso",
        "pi_r_chan",
        "pi_bias_iso",
        "pi_bias_chan",
        "pi_rmse_iso",
        "pi_rmse_chan",
        "neg_frac_iso",
        "neg_frac_chan",
        "wall_tau12_corr_chan",
    ]
    summary_rows: list[dict[str, object]] = []
    for objective in objectives:
        subset = [row for row in rows if row["objective"] == objective["name"]]
        summary = {
            "objective": objective["name"],
            "lambda_pi": objective["lambda_pi"],
            "lambda_diss": objective["lambda_diss"],
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in subset], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values))
        summary_rows.append(summary)

    summary_path = os.path.join(out_dir, "objective_ablation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {rows_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
