#!/usr/bin/env python3
"""
corrected_candidate_scan.py
===========================
Systematically screen a curated family of corrected-oracle SGS closures.

The corrected wall-normal metric handling substantially reordered model
performance, so this script focuses on the most credible recovery path:
mixed Leonard/WALE/topological closures that can balance
  * isotropic structural fidelity,
  * wall-bounded structural fidelity,
  * channel shear-stress agreement,
  * channel SGS dissipation,
  * wall-normal profile quality.

Outputs
-------
results/corrected_scan/candidate_scan_summary.csv
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import stats

from sgs_discovery.grid_metrics import wall_profile_corr
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.physics_env import TensorSymbolicEngine, c1, c2, c3, c4
from sgs_discovery.symbolic_closures import corrected_term_dictionary, native_model_exprs


@dataclass
class ModelSpec:
    name: str
    expr: sp.MatrixExpr


def mean_component_correlation(
    tau_true: np.ndarray, tau_pred: np.ndarray,
) -> tuple[float, dict[str, float]]:
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    labels = ["11", "22", "33", "12", "13", "23"]
    values: list[float] = []
    by_component: dict[str, float] = {}
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


def dissipation_corr(oracle: JHTDBOracle, tau_pred: np.ndarray) -> float:
    pi_true = -np.einsum("nij,nij->n", oracle.tau, oracle.S)
    pi_pred = -np.einsum("nij,nij->n", tau_pred, oracle.S)
    if np.std(pi_true) < 1e-15 or np.std(pi_pred) < 1e-15:
        return float("nan")
    return float(np.corrcoef(pi_true, pi_pred)[0, 1])


def clipped(value: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def main() -> None:
    out_dir = "results/corrected_scan"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading corrected oracles...")
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
        max_iter=300,
    )

    exprs = native_model_exprs()
    terms = corrected_term_dictionary()

    models = [
        ModelSpec("Leonard", exprs["Leonard"]),
        ModelSpec("WALE_canonical", exprs["WALE_canonical"]),
        ModelSpec("WALE_like", exprs["WALE_like"]),
        ModelSpec("Smagorinsky_like", exprs["Smagorinsky_like"]),
        ModelSpec("Mixed_L_WALE_like", exprs["Mixed_L_WALE_like"]),
        ModelSpec("Champion", exprs["Champion"]),
        ModelSpec("L_plus_WALE_canonical", c1 * terms["L"] + c2 * terms["WALE_canonical"]),
        ModelSpec("L_plus_WALE_both", c1 * terms["L"] + c2 * terms["WALE_canonical"] + c3 * terms["WALE_like"]),
        ModelSpec("L_plus_WALEcanon_Sj", exprs["Jaumann_hybrid"]),
        ModelSpec("L_plus_WALEcanon_omega", c1 * terms["L"] + c2 * terms["WALE_canonical"] + c3 * terms["omega_outer"]),
        ModelSpec("L_plus_WALEcanon_W", c1 * terms["L"] + c2 * terms["WALE_canonical"] + c3 * terms["Wstretch"]),
        ModelSpec("L_plus_WALEcanon_omegaW", c1 * terms["L"] + c2 * terms["WALE_canonical"] + c3 * terms["omega_W"]),
        ModelSpec(
            "L_plus_WALEcanon_WALElike_Sj",
            c1 * terms["L"] + c2 * terms["WALE_canonical"] + c3 * terms["WALE_like"] + c4 * terms["Jaumann"],
        ),
        ModelSpec(
            "L_plus_WALEcanon_WALElike_omega",
            c1 * terms["L"] + c2 * terms["WALE_canonical"] + c3 * terms["WALE_like"] + c4 * terms["omega_outer"],
        ),
        ModelSpec(
            "L_plus_WALEcanon_WALElike_W",
            exprs["Wstretch_hybrid"],
        ),
        ModelSpec(
            "L_plus_WALEcanon_WALElike_omegaW",
            c1 * terms["L"] + c2 * terms["WALE_canonical"] + c3 * terms["WALE_like"] + c4 * terms["omega_W"],
        ),
    ]

    rows: list[dict[str, object]] = []
    for model in models:
        print("=" * 72)
        print(f"[MODEL] {model.name}")
        if not engine.is_physically_valid(model.expr):
            print("  skipped: failed physical validity check")
            continue

        # Keep the frozen 10% sub-samples aligned across models.
        np.random.seed(0)
        constants, loss = optimizer.optimize(model.expr)
        print(f"  constants = {constants}")
        print(f"  loss = {loss:.6e}")

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
            constants,
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
            constants,
        )

        mean_r_iso, comp_iso = mean_component_correlation(oracle_iso.tau, tau_iso)
        mean_r_chan, comp_chan = mean_component_correlation(oracle_chan.tau, tau_chan)
        pi_r_iso = dissipation_corr(oracle_iso, tau_iso)
        pi_r_chan = dissipation_corr(oracle_chan, tau_chan)
        wall_corr = wall_profile_corr(oracle_chan, tau_chan)
        nmse_iso = float(oracle_iso.evaluate_mse(tau_iso))
        nmse_chan = float(oracle_chan.evaluate_mse(tau_chan))

        # Heuristic ranking for corrected candidate screening.
        balanced_score = (
            0.30 * clipped(mean_r_iso)
            + 0.30 * clipped(mean_r_chan)
            + 0.15 * max(clipped(comp_chan["12"]), 0.0)
            + 0.15 * max(clipped(pi_r_chan), 0.0)
            + 0.10 * max(clipped(wall_corr), 0.0)
        )
        channel_score = (
            0.35 * clipped(mean_r_chan)
            + 0.20 * max(clipped(comp_chan["12"]), 0.0)
            + 0.20 * max(clipped(pi_r_chan), 0.0)
            + 0.25 * max(clipped(wall_corr), 0.0)
        )

        row = {
            "model": model.name,
            "expr": str(model.expr),
            "constants": str(constants),
            "loss": float(loss),
            "nmse_iso": nmse_iso,
            "nmse_chan": nmse_chan,
            "mean_r_iso": mean_r_iso,
            "mean_r_chan": mean_r_chan,
            "tau12_r_chan": comp_chan["12"],
            "pi_r_iso": pi_r_iso,
            "pi_r_chan": pi_r_chan,
            "wall_tau12_corr_chan": wall_corr,
            "balanced_score": balanced_score,
            "channel_score": channel_score,
        }
        rows.append(row)

        print(
            "  metrics:",
            f"mean_r_iso={mean_r_iso:.3f}",
            f"mean_r_chan={mean_r_chan:.3f}",
            f"tau12={comp_chan['12']:.3f}",
            f"pi_chan={pi_r_chan:.3f}",
            f"wall={wall_corr:.3f}",
            f"balanced={balanced_score:.3f}",
        )

    rows.sort(key=lambda row: row["balanced_score"], reverse=True)

    out_path = os.path.join(out_dir, "candidate_scan_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nTop corrected candidates by balanced_score:")
    for idx, row in enumerate(rows[:5], start=1):
        print(
            f"{idx:>2d}. {row['model']}: "
            f"balanced={row['balanced_score']:.3f}, "
            f"mean_r_iso={row['mean_r_iso']:.3f}, "
            f"mean_r_chan={row['mean_r_chan']:.3f}, "
            f"tau12={row['tau12_r_chan']:.3f}, "
            f"pi_chan={row['pi_r_chan']:.3f}, "
            f"wall={row['wall_tau12_corr_chan']:.3f}"
        )

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
