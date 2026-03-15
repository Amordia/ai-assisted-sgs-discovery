#!/usr/bin/env python3
"""
cross_flow_seed_study.py
========================
Assess two linked robustness questions for repository-native SGS models:

1. Are coefficient fits stable across different frozen 10% optimizer
   sub-samples?
2. Do coefficients fitted on one flow family transfer to the other?

Outputs:
  - results/cross_flow/cross_flow_seed_rows.csv
  - results/cross_flow/cross_flow_seed_summary.csv
  - results/cross_flow/champion_coeff_rows.csv
  - results/cross_flow/champion_coeff_summary.csv
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import sympy as sp
from scipy import stats

from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.grid_metrics import wall_profile_corr
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs


SEEDS = tuple(range(12))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    expr: sp.MatrixExpr


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


def dissipation_correlation(oracle: JHTDBOracle, tau_pred: np.ndarray) -> float:
    pi_true = -np.einsum("nij,nij->n", oracle.tau, oracle.S)
    pi_pred = -np.einsum("nij,nij->n", tau_pred, oracle.S)
    if np.std(pi_true) < 1e-15 or np.std(pi_pred) < 1e-15:
        return float("nan")
    return float(np.corrcoef(pi_true, pi_pred)[0, 1])

def predict(
    engine: TensorSymbolicEngine,
    expr: sp.MatrixExpr,
    oracle: JHTDBOracle,
    constants: dict[str, float],
) -> np.ndarray:
    return engine.lambdify_tensor_expr(
        expr,
        oracle.S,
        oracle.Omega,
        oracle.L,
        oracle.S_d,
        oracle.S_jaumann,
        oracle.Lap_S,
        oracle.Delta,
        oracle.omega_vec,
        oracle.W_vec,
        oracle.h_scalar,
        constants,
    )


def summarize(values: Iterable[float]) -> tuple[float, float]:
    array = np.array(list(values), dtype=float)
    return float(np.nanmean(array)), float(np.nanstd(array))


def main() -> None:
    np.random.seed(0)
    out_dir = "results/cross_flow"
    os.makedirs(out_dir, exist_ok=True)

    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")

    engine = TensorSymbolicEngine()
    exprs = native_model_exprs()

    models = [
        ModelSpec("Leonard", exprs["Leonard"]),
        ModelSpec("WALE_like", exprs["WALE_like"]),
        ModelSpec("Smagorinsky_like", exprs["Smagorinsky_like"]),
        ModelSpec("WALE_canonical", exprs["WALE_canonical"]),
        ModelSpec("Champion", exprs["Champion"]),
    ]

    regimes = [
        ("ISO_only", [oracle_iso]),
        ("CHAN_only", [oracle_chan]),
        ("Joint", [oracle_iso, oracle_chan]),
    ]

    seed_rows: list[dict[str, object]] = []
    champion_coeff_rows: list[dict[str, object]] = []

    for regime_idx, (regime_name, train_oracles) in enumerate(regimes):
        for model_idx, model in enumerate(models):
            optimizer = LeafNodeOptimizer(
                engine,
                train_oracles,
                lambda_pi=1.0,
                lambda_diss=1.0,
                lambda_l1=1e-5,
                max_iter=400,
            )
            for seed in SEEDS:
                fit_seed = 10_000 + 1000 * regime_idx + 100 * model_idx + seed
                np.random.seed(fit_seed)
                constants, train_loss = optimizer.optimize(model.expr)

                tau_iso = predict(engine, model.expr, oracle_iso, constants)
                tau_chan = predict(engine, model.expr, oracle_chan, constants)
                mean_r_iso, comp_iso = mean_component_correlation(oracle_iso.tau, tau_iso)
                mean_r_chan, comp_chan = mean_component_correlation(oracle_chan.tau, tau_chan)

                row = {
                    "regime": regime_name,
                    "model": model.name,
                    "seed": seed,
                    "fit_seed": fit_seed,
                    "train_loss": float(train_loss),
                    "constants": str(constants),
                    "nmse_iso": float(oracle_iso.evaluate_mse(tau_iso)),
                    "nmse_chan": float(oracle_chan.evaluate_mse(tau_chan)),
                    "mean_r_iso": mean_r_iso,
                    "mean_r_chan": mean_r_chan,
                    "r11_iso": comp_iso["11"],
                    "r12_iso": comp_iso["12"],
                    "r11_chan": comp_chan["11"],
                    "r12_chan": comp_chan["12"],
                    "pi_r_iso": dissipation_correlation(oracle_iso, tau_iso),
                    "pi_r_chan": dissipation_correlation(oracle_chan, tau_chan),
                    "wall_tau12_corr_chan": wall_profile_corr(oracle_chan, tau_chan),
                }
                seed_rows.append(row)

                if model.name == "Champion":
                    champion_coeff_rows.append({
                        "regime": regime_name,
                        "seed": seed,
                        "fit_seed": fit_seed,
                        "train_loss": float(train_loss),
                        "c1": float(constants.get("c1", np.nan)),
                        "c2": float(constants.get("c2", np.nan)),
                        "c3": float(constants.get("c3", np.nan)),
                    })

                print(
                    f"regime={regime_name:>9s} model={model.name:>16s} seed={seed:02d} "
                    f"mean_r_iso={mean_r_iso:.3f} mean_r_chan={mean_r_chan:.3f}"
                )

    seed_path = os.path.join(out_dir, "cross_flow_seed_rows.csv")
    with open(seed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_metrics = [
        "train_loss",
        "nmse_iso",
        "nmse_chan",
        "mean_r_iso",
        "mean_r_chan",
        "r11_iso",
        "r12_iso",
        "r11_chan",
        "r12_chan",
        "pi_r_iso",
        "pi_r_chan",
        "wall_tau12_corr_chan",
    ]
    summary_rows: list[dict[str, object]] = []
    for regime_name, _ in regimes:
        for model in models:
            rows = [r for r in seed_rows if r["regime"] == regime_name and r["model"] == model.name]
            summary_row: dict[str, object] = {"regime": regime_name, "model": model.name}
            for metric in summary_metrics:
                mean, std = summarize(float(r[metric]) for r in rows)
                summary_row[f"{metric}_mean"] = mean
                summary_row[f"{metric}_std"] = std
            summary_rows.append(summary_row)

    summary_path = os.path.join(out_dir, "cross_flow_seed_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    coeff_path = os.path.join(out_dir, "champion_coeff_rows.csv")
    with open(coeff_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(champion_coeff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(champion_coeff_rows)

    coeff_summary_rows: list[dict[str, object]] = []
    for regime_name, _ in regimes:
        rows = [r for r in champion_coeff_rows if r["regime"] == regime_name]
        coeff_summary = {"regime": regime_name}
        for coeff in ["c1", "c2", "c3", "train_loss"]:
            values = np.array([float(r[coeff]) for r in rows], dtype=float)
            coeff_summary[f"{coeff}_mean"] = float(np.nanmean(values))
            coeff_summary[f"{coeff}_std"] = float(np.nanstd(values))
            coeff_summary[f"{coeff}_median_abs"] = float(np.nanmedian(np.abs(values)))
        coeff_summary["c1_positive_frac"] = float(np.mean([float(r["c1"]) > 0.0 for r in rows]))
        coeff_summary["c2_positive_frac"] = float(np.mean([float(r["c2"]) > 0.0 for r in rows]))
        coeff_summary["c3_negative_frac"] = float(np.mean([float(r["c3"]) < 0.0 for r in rows]))
        coeff_summary["ppn_sign_pattern_frac"] = float(
            np.mean(
                [
                    float(r["c1"]) > 0.0 and float(r["c2"]) > 0.0 and float(r["c3"]) < 0.0
                    for r in rows
                ]
            )
        )
        coeff_summary_rows.append(coeff_summary)

    coeff_summary_path = os.path.join(out_dir, "champion_coeff_summary.csv")
    with open(coeff_summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(coeff_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(coeff_summary_rows)

    print(f"Saved {seed_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {coeff_path}")
    print(f"Saved {coeff_summary_path}")


if __name__ == "__main__":
    main()
