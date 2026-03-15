#!/usr/bin/env python3
"""
isotropic_ood_benchmark.py
==========================
Fit SGS models on the original dual-oracle setup and evaluate them on external
isotropic JHTDB datasets that were not used during search or fitting.

Outputs:
  - results/isotropic_ood/isotropic_ood_summary.csv
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
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs


@dataclass
class ModelSpec:
    name: str
    expr: sp.MatrixExpr


def mean_component_correlation(
    tau_true: np.ndarray, tau_pred: np.ndarray
) -> tuple[float, dict[str, float]]:
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    labels = ["11", "22", "33", "12", "13", "23"]
    values = []
    detail: dict[str, float] = {}
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


def dissipation_corr(oracle: JHTDBOracle, tau_pred: np.ndarray) -> float:
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


def main() -> None:
    np.random.seed(0)
    out_dir = "results/isotropic_ood"
    os.makedirs(out_dir, exist_ok=True)

    train_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    train_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")

    test_cases = [
        (
            "isotropic1024coarse",
            JHTDBOracle(
                "jhtdb_u_tensor_64_coarse_ood.h5",
                filter_width=1.0,
                dx=2.0 * np.pi / 1024.0,
                boundary_mode="wrap",
            ),
        ),
        (
            "isotropic4096_matched_filter",
            JHTDBOracle(
                "jhtdb_u_tensor_64_4096_ood.h5",
                filter_width=4.0,
                dx=2.0 * np.pi / 4096.0,
                boundary_mode="wrap",
            ),
        ),
    ]

    engine = TensorSymbolicEngine()
    optimizer = LeafNodeOptimizer(
        engine,
        [train_iso, train_chan],
        lambda_pi=1.0,
        lambda_diss=1.0,
        lambda_l1=1e-5,
        max_iter=400,
    )

    exprs = native_model_exprs()

    models = [
        ModelSpec("Leonard", exprs["Leonard"]),
        ModelSpec("WALE_like", exprs["WALE_like"]),
        ModelSpec("Smagorinsky_like", exprs["Smagorinsky_like"]),
        ModelSpec("WALE_canonical", exprs["WALE_canonical"]),
        ModelSpec("Jaumann_hybrid", exprs["Jaumann_hybrid"]),
        ModelSpec("Champion", exprs["Champion"]),
        ModelSpec("Wstretch_hybrid", exprs["Wstretch_hybrid"]),
    ]

    fitted_constants: dict[str, dict[str, float]] = {}
    fitted_losses: dict[str, float] = {}
    rows: list[dict[str, object]] = []

    for idx, model in enumerate(models):
        np.random.seed(7000 + idx)
        best_constants, train_loss = optimizer.optimize(model.expr)
        fitted_constants[model.name] = best_constants
        fitted_losses[model.name] = float(train_loss)
        print(f"trained {model.name}: loss={train_loss:.6e}, constants={best_constants}")

    for case_name, test_iso in test_cases:
        for model in models:
            tau_iso = predict(engine, model.expr, test_iso, fitted_constants[model.name])
            mean_r_iso, comp_iso = mean_component_correlation(test_iso.tau, tau_iso)
            row = {
                "case": case_name,
                "model": model.name,
                "train_loss": fitted_losses[model.name],
                "constants": str(fitted_constants[model.name]),
                "nmse_iso_transfer": float(test_iso.evaluate_mse(tau_iso)),
                "mean_r_iso_transfer": mean_r_iso,
                "r11_iso_transfer": comp_iso["11"],
                "r12_iso_transfer": comp_iso["12"],
                "pi_r_iso_transfer": dissipation_corr(test_iso, tau_iso),
            }
            rows.append(row)
            print(row)

    out_path = os.path.join(out_dir, "isotropic_ood_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
