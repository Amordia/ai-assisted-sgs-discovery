#!/usr/bin/env python3
"""
holdout_benchmark.py
====================
Blocked hold-out evaluation for candidate SGS models.

The flow fields are split along the z direction into contiguous slabs.
Each fold trains on 3/4 of the slabs and evaluates on the held-out quarter.
This is not an independent-flow test, but it is stricter than evaluating on
the same full field used during fitting.
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
from sgs_discovery.grid_metrics import infer_grid_shape, wall_profile_corr
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs


@dataclass
class OracleView:
    S: np.ndarray
    Omega: np.ndarray
    tau: np.ndarray
    L: np.ndarray
    S_d: np.ndarray
    Delta: np.ndarray
    omega_vec: np.ndarray
    W_vec: np.ndarray
    h_scalar: np.ndarray
    S_jaumann: np.ndarray
    Lap_S: np.ndarray
    var_tau: np.ndarray
    var_pi: float
    mean_pi: float
    y_coords: np.ndarray | None
    n_samples: int
    grid_shape: tuple[int, int, int]

    def evaluate_mse(self, predicted_tau: np.ndarray, true_tau: np.ndarray | None = None) -> float:
        if true_tau is None:
            true_tau = self.tau
        residual = predicted_tau - true_tau
        safe_var = np.where(self.var_tau > 1e-12, self.var_tau, 1e-12)
        local_mse = (residual ** 2) / safe_var[:, None, None]
        return float(np.mean(local_mse))


@dataclass
class ModelSpec:
    name: str
    expr: sp.MatrixExpr

def make_view(oracle: JHTDBOracle, indices: np.ndarray, grid_shape: tuple[int, int, int]) -> OracleView:
    S = oracle.S[indices]
    tau = oracle.tau[indices]
    pi_true = -np.einsum("nij,nij->n", tau, S)
    return OracleView(
        S=S,
        Omega=oracle.Omega[indices],
        tau=tau,
        L=oracle.L[indices],
        S_d=oracle.S_d[indices],
        Delta=oracle.Delta[indices],
        omega_vec=oracle.omega_vec[indices],
        W_vec=oracle.W_vec[indices],
        h_scalar=oracle.h_scalar[indices],
        S_jaumann=oracle.S_jaumann[indices],
        Lap_S=oracle.Lap_S[indices],
        var_tau=oracle.var_tau[indices],
        var_pi=float(np.var(pi_true)),
        mean_pi=float(np.mean(pi_true)),
        y_coords=oracle.y_coords.copy() if oracle.y_coords is not None else None,
        n_samples=len(indices),
        grid_shape=grid_shape,
    )


def mean_component_correlation(tau_true: np.ndarray, tau_pred: np.ndarray) -> tuple[float, dict[str, float]]:
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


def dissipation_correlation(oracle: OracleView, tau_pred: np.ndarray) -> float:
    pi_true = -np.einsum("nij,nij->n", oracle.tau, oracle.S)
    pi_pred = -np.einsum("nij,nij->n", tau_pred, oracle.S)
    if np.std(pi_true) < 1e-15 or np.std(pi_pred) < 1e-15:
        return float("nan")
    return float(np.corrcoef(pi_true, pi_pred)[0, 1])

def predict(engine: TensorSymbolicEngine, expr: sp.MatrixExpr, oracle: OracleView, constants: dict[str, float]) -> np.ndarray:
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
    out_dir = "results/holdout"
    os.makedirs(out_dir, exist_ok=True)

    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")

    z_iso, y_iso, x_iso = infer_grid_shape(oracle_iso)
    z_chan, y_chan, x_chan = infer_grid_shape(oracle_chan)

    idx_iso_grid = np.arange(oracle_iso.n_samples).reshape(z_iso, y_iso, x_iso)
    idx_chan_grid = np.arange(oracle_chan.n_samples).reshape(z_chan, y_chan, x_chan)

    engine = TensorSymbolicEngine()

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

    fold_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    n_folds = 4
    fold_size_iso = z_iso // n_folds
    fold_size_chan = z_chan // n_folds

    for fold in range(n_folds):
        z0_iso, z1_iso = fold * fold_size_iso, (fold + 1) * fold_size_iso
        z0_chan, z1_chan = fold * fold_size_chan, (fold + 1) * fold_size_chan

        test_iso = idx_iso_grid[z0_iso:z1_iso, :, :].reshape(-1)
        train_iso = np.setdiff1d(idx_iso_grid.reshape(-1), test_iso, assume_unique=True)
        test_chan = idx_chan_grid[z0_chan:z1_chan, :, :].reshape(-1)
        train_chan = np.setdiff1d(idx_chan_grid.reshape(-1), test_chan, assume_unique=True)

        train_iso_view = make_view(oracle_iso, train_iso, (z_iso - fold_size_iso, y_iso, x_iso))
        test_iso_view = make_view(oracle_iso, test_iso, (fold_size_iso, y_iso, x_iso))
        train_chan_view = make_view(oracle_chan, train_chan, (z_chan - fold_size_chan, y_chan, x_chan))
        test_chan_view = make_view(oracle_chan, test_chan, (fold_size_chan, y_chan, x_chan))

        optimizer = LeafNodeOptimizer(
            engine,
            [train_iso_view, train_chan_view],
            lambda_pi=1.0,
            lambda_diss=1.0,
            lambda_l1=1e-5,
            max_iter=400,
        )

        for model_idx, model in enumerate(models):
            np.random.seed(1000 + 17 * fold + model_idx)
            best_constants, train_loss = optimizer.optimize(model.expr)

            tau_iso = predict(engine, model.expr, test_iso_view, best_constants)
            tau_chan = predict(engine, model.expr, test_chan_view, best_constants)

            mean_r_iso, _ = mean_component_correlation(test_iso_view.tau, tau_iso)
            mean_r_chan, comps_chan = mean_component_correlation(test_chan_view.tau, tau_chan)

            row = {
                "fold": fold,
                "model": model.name,
                "train_loss": float(train_loss),
                "constants": str(best_constants),
                "nmse_iso_test": float(test_iso_view.evaluate_mse(tau_iso)),
                "nmse_chan_test": float(test_chan_view.evaluate_mse(tau_chan)),
                "mean_r_iso_test": mean_r_iso,
                "mean_r_chan_test": mean_r_chan,
                "tau12_r_chan_test": comps_chan["12"],
                "pi_r_iso_test": dissipation_correlation(test_iso_view, tau_iso),
                "pi_r_chan_test": dissipation_correlation(test_chan_view, tau_chan),
                "wall_tau12_corr_chan_test": wall_profile_corr(test_chan_view, tau_chan),
            }
            fold_rows.append(row)
            print(f"fold={fold} model={model.name} mean_r_chan={mean_r_chan:.3f} pi_r_chan={row['pi_r_chan_test']:.3f}")

    metrics = [
        "nmse_iso_test",
        "nmse_chan_test",
        "mean_r_iso_test",
        "mean_r_chan_test",
        "tau12_r_chan_test",
        "pi_r_iso_test",
        "pi_r_chan_test",
        "wall_tau12_corr_chan_test",
    ]

    for model in models:
        rows = [r for r in fold_rows if r["model"] == model.name]
        agg: dict[str, object] = {"model": model.name}
        for metric in metrics:
            values = np.array([float(r[metric]) for r in rows], dtype=float)
            agg[f"{metric}_mean"] = float(np.nanmean(values))
            agg[f"{metric}_std"] = float(np.nanstd(values))
        aggregate_rows.append(agg)

    with open(os.path.join(out_dir, "blocked_holdout_folds.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    with open(os.path.join(out_dir, "blocked_holdout_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Saved {out_dir}/blocked_holdout_folds.csv")
    print(f"Saved {out_dir}/blocked_holdout_summary.csv")


if __name__ == "__main__":
    main()
