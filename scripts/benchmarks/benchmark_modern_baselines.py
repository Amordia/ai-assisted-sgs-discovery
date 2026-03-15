#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy import stats
from scipy.optimize import minimize_scalar

from sgs_discovery.grid_metrics import wall_profile_corr
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs
from sgs_discovery.structured_sgs import (
    feature_bundle,
    flatten_tensor_field,
    load_filtered_volume,
    tau_amd,
    tau_dynamic_smagorinsky,
    tau_smagorinsky,
    tau_vreman,
    tau_wale,
)


@dataclass
class SymbolicModel:
    name: str
    expr: sp.MatrixExpr


def mean_component_correlation(tau_true: np.ndarray, tau_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    labels = ["11", "22", "33", "12", "13", "23"]
    values: list[float] = []
    by_component: dict[str, float] = {}
    for (i, j), label in zip(components, labels):
        pred = tau_pred[:, i, j]
        true = tau_true[:, i, j]
        if np.std(pred) < 1.0e-15 or np.std(true) < 1.0e-15:
            r = np.nan
        else:
            r, _ = stats.pearsonr(true, pred)
        by_component[label] = float(r)
        values.append(float(r))
    return float(np.nanmean(values)), by_component


def dissipation_corr(oracle: JHTDBOracle, tau_pred: np.ndarray) -> float:
    pi_true = -np.einsum("nij,nij->n", oracle.tau, oracle.S)
    pi_pred = -np.einsum("nij,nij->n", tau_pred, oracle.S)
    if np.std(pi_true) < 1.0e-15 or np.std(pi_pred) < 1.0e-15:
        return float("nan")
    return float(np.corrcoef(pi_true, pi_pred)[0, 1])


def summarize_model(model: str, family: str, constants: str, oracle_iso: JHTDBOracle, oracle_chan: JHTDBOracle, tau_iso: np.ndarray, tau_chan: np.ndarray) -> dict[str, object]:
    mean_r_iso, comp_iso = mean_component_correlation(oracle_iso.tau, tau_iso)
    mean_r_chan, comp_chan = mean_component_correlation(oracle_chan.tau, tau_chan)
    return {
        "model": model,
        "family": family,
        "constants": constants,
        "nmse_iso": float(oracle_iso.evaluate_mse(tau_iso)),
        "nmse_chan": float(oracle_chan.evaluate_mse(tau_chan)),
        "mean_r_iso": mean_r_iso,
        "mean_r_chan": mean_r_chan,
        "tau12_r_chan": comp_chan["12"],
        "pi_r_iso": dissipation_corr(oracle_iso, tau_iso),
        "pi_r_chan": dissipation_corr(oracle_chan, tau_chan),
        "wall_tau12_corr_chan": wall_profile_corr(oracle_chan, tau_chan),
    }


def dual_oracle_scalar_loss(scale: float, oracle_iso: JHTDBOracle, oracle_chan: JHTDBOracle, tau_iso_base: np.ndarray, tau_chan_base: np.ndarray) -> float:
    total = 0.0
    for oracle, tau_base in ((oracle_iso, tau_iso_base), (oracle_chan, tau_chan_base)):
        tau = scale * tau_base
        total += oracle.evaluate_mse(tau)

        dissipation = -np.einsum("nij,nij->n", tau, oracle.S)
        backscatter = np.maximum(0.0, -dissipation)
        total += -min(float(np.mean(backscatter)) / oracle.var_pi, 0.5)

        rel_err = (float(np.mean(dissipation)) - oracle.mean_pi) / (abs(oracle.mean_pi) + 1.0e-12)
        total += np.log1p(abs(rel_err)) ** 2
    return float(total)


def fit_scalar_scale(
    oracle_iso: JHTDBOracle,
    oracle_chan: JHTDBOracle,
    tau_iso_base: np.ndarray,
    tau_chan_base: np.ndarray,
    lower: float = -50.0,
    upper: float = 50.0,
) -> float:
    result = minimize_scalar(
        lambda s: dual_oracle_scalar_loss(s, oracle_iso, oracle_chan, tau_iso_base, tau_chan_base),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    return float(result.x)


def evaluate_symbolic(
    engine: TensorSymbolicEngine,
    model: SymbolicModel,
    constants: dict[str, float],
    oracle: JHTDBOracle,
) -> np.ndarray:
    return engine.lambdify_tensor_expr(
        model.expr,
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
    out_dir = "results/modern_baselines"
    os.makedirs(out_dir, exist_ok=True)

    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")
    vol_iso = load_filtered_volume("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    vol_chan = load_filtered_volume("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")

    bundle_iso = feature_bundle(
        vol_iso.u_bar_t2,
        axis_coords=vol_iso.axis_coords,
        mode=vol_iso.boundary_mode,
        prev_u=vol_iso.u_bar_t1,
        sigma_extra=1.0,
    )
    bundle_chan = feature_bundle(
        vol_chan.u_bar_t2,
        axis_coords=vol_chan.axis_coords,
        mode=vol_chan.boundary_mode,
        prev_u=vol_chan.u_bar_t1,
        sigma_extra=1.0,
    )

    rows: list[dict[str, object]] = []

    rows.append(
        summarize_model(
            model="Bardina_Leonard",
            family="modern_canonical",
            constants="{}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=oracle_iso.L,
            tau_chan=oracle_chan.L,
        )
    )

    tau_iso = flatten_tensor_field(tau_smagorinsky(bundle_iso["S"], vol_iso.delta_eff))
    tau_chan = flatten_tensor_field(tau_smagorinsky(bundle_chan["S"], vol_chan.delta_eff))
    rows.append(
        summarize_model(
            model="Smagorinsky_canonical",
            family="modern_canonical",
            constants="{'Cs': 0.17}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=tau_iso,
            tau_chan=tau_chan,
        )
    )
    scale = fit_scalar_scale(oracle_iso, oracle_chan, tau_iso, tau_chan, lower=0.0, upper=50.0)
    rows.append(
        summarize_model(
            model="Smagorinsky_tuned",
            family="modern_tuned",
            constants=f"{{'scale': {scale:.6e}, 'Cs_eff': {0.17 * np.sqrt(max(scale, 0.0)):.6e}}}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=scale * tau_iso,
            tau_chan=scale * tau_chan,
        )
    )

    tau_iso, coeff_iso = tau_dynamic_smagorinsky(
        vol_iso.u_bar_t2,
        bundle_iso["S"],
        vol_iso.delta_eff,
        axis_coords=vol_iso.axis_coords,
        mode=vol_iso.boundary_mode,
        plane_average=False,
    )
    tau_chan, coeff_chan = tau_dynamic_smagorinsky(
        vol_chan.u_bar_t2,
        bundle_chan["S"],
        vol_chan.delta_eff,
        axis_coords=vol_chan.axis_coords,
        mode=vol_chan.boundary_mode,
        plane_average=True,
    )
    rows.append(
        summarize_model(
            model="Dynamic_Smagorinsky",
            family="modern_dynamic",
            constants=f"{{'C_iso': {float(np.mean(coeff_iso)):.6e}, 'C_chan_mean': {float(np.mean(coeff_chan)):.6e}}}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=flatten_tensor_field(tau_iso),
            tau_chan=flatten_tensor_field(tau_chan),
        )
    )

    tau_iso = flatten_tensor_field(tau_wale(bundle_iso["S"], bundle_iso["S_d"], vol_iso.delta_eff))
    tau_chan = flatten_tensor_field(tau_wale(bundle_chan["S"], bundle_chan["S_d"], vol_chan.delta_eff))
    rows.append(
        summarize_model(
            model="WALE_canonical_physical",
            family="modern_canonical",
            constants="{'Cw': 0.325}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=tau_iso,
            tau_chan=tau_chan,
        )
    )
    scale = fit_scalar_scale(oracle_iso, oracle_chan, tau_iso, tau_chan, lower=0.0, upper=50.0)
    rows.append(
        summarize_model(
            model="WALE_tuned",
            family="modern_tuned",
            constants=f"{{'scale': {scale:.6e}, 'Cw_eff': {0.325 * np.sqrt(max(scale, 0.0)):.6e}}}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=scale * tau_iso,
            tau_chan=scale * tau_chan,
        )
    )

    tau_iso = flatten_tensor_field(tau_vreman(bundle_iso["grad_u"], bundle_iso["S"], vol_iso.axis_widths))
    tau_chan = flatten_tensor_field(tau_vreman(bundle_chan["grad_u"], bundle_chan["S"], vol_chan.axis_widths))
    rows.append(
        summarize_model(
            model="Vreman_canonical",
            family="modern_canonical",
            constants=f"{{'Cv': {2.5 * 0.17 ** 2:.6e}}}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=tau_iso,
            tau_chan=tau_chan,
        )
    )
    scale = fit_scalar_scale(oracle_iso, oracle_chan, tau_iso, tau_chan, lower=0.0, upper=50.0)
    rows.append(
        summarize_model(
            model="Vreman_tuned",
            family="modern_tuned",
            constants=f"{{'scale': {scale:.6e}, 'Cv_eff': {scale * (2.5 * 0.17 ** 2):.6e}}}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=scale * tau_iso,
            tau_chan=scale * tau_chan,
        )
    )

    tau_iso = flatten_tensor_field(tau_amd(bundle_iso["grad_u"], bundle_iso["S"], vol_iso.axis_widths))
    tau_chan = flatten_tensor_field(tau_amd(bundle_chan["grad_u"], bundle_chan["S"], vol_chan.axis_widths))
    rows.append(
        summarize_model(
            model="AMD_canonical",
            family="modern_canonical",
            constants="{'Ca': 0.3}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=tau_iso,
            tau_chan=tau_chan,
        )
    )
    scale = fit_scalar_scale(oracle_iso, oracle_chan, tau_iso, tau_chan, lower=0.0, upper=50.0)
    rows.append(
        summarize_model(
            model="AMD_tuned",
            family="modern_tuned",
            constants=f"{{'scale': {scale:.6e}, 'Ca_eff': {scale * 0.3:.6e}}}",
            oracle_iso=oracle_iso,
            oracle_chan=oracle_chan,
            tau_iso=scale * tau_iso,
            tau_chan=scale * tau_chan,
        )
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

    symbolic_models = [
        SymbolicModel("Leonard_fitted", exprs["Leonard"]),
        SymbolicModel("Champion", exprs["Champion"]),
        SymbolicModel("Jaumann_hybrid", exprs["Jaumann_hybrid"]),
        SymbolicModel("Wstretch_hybrid", exprs["Wstretch_hybrid"]),
    ]

    for idx, model in enumerate(symbolic_models):
        np.random.seed(7000 + idx)
        constants, _loss = optimizer.optimize(model.expr)
        tau_iso = evaluate_symbolic(engine, model, constants, oracle_iso)
        tau_chan = evaluate_symbolic(engine, model, constants, oracle_chan)
        rows.append(
            summarize_model(
                model=model.name,
                family="discovered_or_fitted",
                constants=str(constants),
                oracle_iso=oracle_iso,
                oracle_chan=oracle_chan,
                tau_iso=tau_iso,
                tau_chan=tau_chan,
            )
        )

    rows.sort(key=lambda row: (float(row["nmse_iso"]) + float(row["nmse_chan"])))

    out_path = os.path.join(out_dir, "apriori_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
