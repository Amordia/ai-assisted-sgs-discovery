#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.structured_sgs import (
    _DEFAULT_DT,
    box_lengths_from_volume,
    divergence_tensor,
    feature_bundle,
    flatten_scalar_field,
    flatten_tensor_field,
    flatten_vector_field,
    isotropic_shell_spectrum,
    load_filtered_volume,
    spectrum_corr,
    tau_amd,
    tau_dynamic_smagorinsky,
    tau_wale,
    volume_mean_kinetic_energy,
)
from sgs_discovery.symbolic_closures import native_model_exprs

SIM_DT = 0.0002


@dataclass
class SymbolicModel:
    name: str
    expr: sp.MatrixExpr


def make_kgrid(shape: tuple[int, int, int], box_lengths: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z, Y, X = shape
    kz = 2.0 * np.pi * np.fft.fftfreq(Z, d=box_lengths[0] / Z)
    ky = 2.0 * np.pi * np.fft.fftfreq(Y, d=box_lengths[1] / Y)
    kx = 2.0 * np.pi * np.fft.fftfreq(X, d=box_lengths[2] / X)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    return KX, KY, KZ, K2


def project_div_free(field: np.ndarray, kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    KX, KY, KZ, K2 = kgrid
    fhat = np.fft.fftn(field, axes=(0, 1, 2))
    dot = KX * fhat[..., 0] + KY * fhat[..., 1] + KZ * fhat[..., 2]
    mask = K2 > 0.0
    out = fhat.copy()
    out[..., 0][mask] -= KX[mask] * dot[mask] / K2[mask]
    out[..., 1][mask] -= KY[mask] * dot[mask] / K2[mask]
    out[..., 2][mask] -= KZ[mask] * dot[mask] / K2[mask]
    out[..., 0][~mask] = 0.0
    out[..., 1][~mask] = 0.0
    out[..., 2][~mask] = 0.0
    return np.fft.ifftn(out, axes=(0, 1, 2)).real


def spectral_gradient_vector(u: np.ndarray, kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    KX, KY, KZ, _ = kgrid
    uhat = np.fft.fftn(u, axes=(0, 1, 2))
    dudx = np.fft.ifftn(1j * KX[..., None] * uhat, axes=(0, 1, 2)).real
    dudy = np.fft.ifftn(1j * KY[..., None] * uhat, axes=(0, 1, 2)).real
    dudz = np.fft.ifftn(1j * KZ[..., None] * uhat, axes=(0, 1, 2)).real
    return dudx, dudy, dudz


def laplacian_vector(u: np.ndarray, kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    _, _, _, K2 = kgrid
    uhat = np.fft.fftn(u, axes=(0, 1, 2))
    return np.fft.ifftn((-K2[..., None]) * uhat, axes=(0, 1, 2)).real


def convective_term(u: np.ndarray, kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    dudx, dudy, dudz = spectral_gradient_vector(u, kgrid)
    return u[..., 0:1] * dudx + u[..., 1:2] * dudy + u[..., 2:3] * dudz


def divergence_rms(u: np.ndarray, kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
    KX, KY, KZ, _ = kgrid
    uhat = np.fft.fftn(u, axes=(0, 1, 2))
    div = np.fft.ifftn(1j * (KX[..., None] * uhat[..., 0:1] + KY[..., None] * uhat[..., 1:2] + KZ[..., None] * uhat[..., 2:3]), axes=(0, 1, 2)).real
    return float(np.sqrt(np.mean(div ** 2)))


def fit_symbolic_constants(oracle_iso: JHTDBOracle, oracle_chan: JHTDBOracle) -> tuple[TensorSymbolicEngine, dict[str, dict[str, float]], list[SymbolicModel]]:
    engine = TensorSymbolicEngine()
    optimizer = LeafNodeOptimizer(
        engine,
        [oracle_iso, oracle_chan],
        lambda_pi=1.0,
        lambda_diss=1.0,
        lambda_l1=1.0e-5,
        max_iter=400,
    )

    exprs = native_model_exprs()
    models = [
        SymbolicModel("Champion", exprs["Champion"]),
        SymbolicModel("Jaumann_hybrid", exprs["Jaumann_hybrid"]),
        SymbolicModel("Wstretch_hybrid", exprs["Wstretch_hybrid"]),
    ]
    constants_by_model: dict[str, dict[str, float]] = {}
    for idx, model in enumerate(models):
        np.random.seed(7000 + idx)
        constants, _ = optimizer.optimize(model.expr)
        constants_by_model[model.name] = constants
    return engine, constants_by_model, models


def symbolic_tau(
    engine: TensorSymbolicEngine,
    expr: sp.MatrixExpr,
    constants: dict[str, float],
    bundle: dict[str, np.ndarray],
    delta_eff: np.ndarray,
) -> np.ndarray:
    tau_flat = engine.lambdify_tensor_expr(
        expr,
        flatten_tensor_field(bundle["S"]),
        flatten_tensor_field(bundle["Omega"]),
        flatten_tensor_field(bundle["L"]),
        flatten_tensor_field(bundle["S_d"]),
        flatten_tensor_field(bundle["S_j"]),
        flatten_tensor_field(bundle["Lap_S"]),
        flatten_scalar_field(delta_eff),
        flatten_vector_field(bundle["omega_vec"]),
        flatten_vector_field(bundle["W_vec"]),
        bundle["h_scalar"].reshape(-1, 1),
        constants,
    )
    return tau_flat.reshape(bundle["S"].shape)


def calibrate_viscosity(
    u0: np.ndarray,
    u1: np.ndarray,
    tau_true: np.ndarray,
    axis_coords: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dt: float,
) -> float:
    du_dt = (u1 - u0) / dt
    conv = convective_term(u0, kgrid)
    div_tau = divergence_tensor(tau_true, axis_coords)
    forcing_free_rhs = du_dt + project_div_free(conv + div_tau, kgrid)
    lap = laplacian_vector(u0, kgrid)
    numerator = float(np.sum(lap * forcing_free_rhs))
    denominator = float(np.sum(lap * lap) + 1.0e-30)
    return max(numerator / denominator, 0.0)


def build_tau_predictor(
    name: str,
    engine: TensorSymbolicEngine,
    symbolic_models: dict[str, SymbolicModel],
    constants_by_model: dict[str, dict[str, float]],
    axis_coords: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    delta_eff: np.ndarray,
    axis_widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    feature_dt: float,
):
    def predict(u: np.ndarray, prev_u: np.ndarray) -> np.ndarray:
        bundle = feature_bundle(
            u,
            axis_coords=axis_coords,
            mode="wrap",
            prev_u=prev_u,
            dt=feature_dt,
            sigma_extra=1.0,
        )
        if name == "No_model":
            return np.zeros(bundle["S"].shape, dtype=np.float64)
        if name == "Bardina_Leonard":
            return bundle["L"]
        if name == "Dynamic_Smagorinsky":
            tau, _ = tau_dynamic_smagorinsky(
                u,
                bundle["S"],
                delta_eff,
                axis_coords=axis_coords,
                mode="wrap",
                plane_average=False,
            )
            return tau
        if name == "WALE_canonical":
            return tau_wale(bundle["S"], bundle["S_d"], delta_eff)
        if name == "AMD_canonical":
            return tau_amd(bundle["grad_u"], bundle["S"], axis_widths)
        model = symbolic_models[name]
        return symbolic_tau(engine, model.expr, constants_by_model[name], bundle, delta_eff)

    return predict


def rhs(
    u: np.ndarray,
    prev_u: np.ndarray,
    tau_predictor,
    viscosity: float,
    axis_coords: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    tau = tau_predictor(u, prev_u)
    conv = convective_term(u, kgrid)
    div_tau = divergence_tensor(tau, axis_coords)
    projected = project_div_free(conv + div_tau, kgrid)
    return -projected + viscosity * laplacian_vector(u, kgrid)


def integrate_rk4(
    u0: np.ndarray,
    prev_u: np.ndarray,
    tau_predictor,
    viscosity: float,
    axis_coords: tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray],
    kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dt: float,
    n_steps: int,
) -> np.ndarray:
    u = project_div_free(u0, kgrid)
    prev = project_div_free(prev_u, kgrid)
    sub_dt = dt / n_steps
    for _ in range(n_steps):
        k1 = rhs(u, prev, tau_predictor, viscosity, axis_coords, kgrid)
        u2 = project_div_free(u + 0.5 * sub_dt * k1, kgrid)
        k2 = rhs(u2, prev, tau_predictor, viscosity, axis_coords, kgrid)
        u3 = project_div_free(u + 0.5 * sub_dt * k2, kgrid)
        k3 = rhs(u3, prev, tau_predictor, viscosity, axis_coords, kgrid)
        u4 = project_div_free(u + sub_dt * k3, kgrid)
        k4 = rhs(u4, prev, tau_predictor, viscosity, axis_coords, kgrid)
        u_next = u + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        prev = u
        u = project_div_free(u_next, kgrid)
    return u


def main() -> None:
    out_dir = "results/aposteriori_isotropic"
    os.makedirs(out_dir, exist_ok=True)

    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")
    oracle_ap = JHTDBOracle("jhtdb_u_tensor_64_periodic.h5", filter_width=1.0, boundary_mode="wrap")
    volume = load_filtered_volume("jhtdb_u_tensor_64_periodic.h5", filter_width=1.0, boundary_mode="wrap")
    box_lengths = box_lengths_from_volume(volume)
    kgrid = make_kgrid(volume.grid_shape, box_lengths)

    u_prev = volume.u_bar_t1
    u0 = volume.u_bar_t2
    u_target = volume.u_bar_t3
    if u_prev is None or u_target is None:
        raise RuntimeError("isotropic a posteriori test requires Velocity_t1 and Velocity_t3")

    tau_true = oracle_ap.tau.reshape(volume.grid_shape + (3, 3))
    div_tau_true = divergence_tensor(tau_true, volume.axis_coords)
    viscosity = calibrate_viscosity(
        u0=u0,
        u1=u_target,
        tau_true=tau_true,
        axis_coords=volume.axis_coords,
        kgrid=kgrid,
        dt=SIM_DT,
    )

    engine, constants_by_model, symbolic_model_list = fit_symbolic_constants(oracle_iso, oracle_chan)
    symbolic_models = {model.name: model for model in symbolic_model_list}

    target_k, target_spec = isotropic_shell_spectrum(u_target, box_lengths)
    rhs_true = (u_target - u0) / SIM_DT

    models = [
        "No_model",
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "WALE_canonical",
        "AMD_canonical",
        "Champion",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
    ]

    rows: list[dict[str, object]] = []
    for model_name in models:
        tau_predictor = build_tau_predictor(
            name=model_name,
            engine=engine,
            symbolic_models=symbolic_models,
            constants_by_model=constants_by_model,
            axis_coords=volume.axis_coords,
            delta_eff=volume.delta_eff,
            axis_widths=volume.axis_widths,
            feature_dt=_DEFAULT_DT,
        )
        t0 = perf_counter()
        try:
            tau0 = tau_predictor(u0, u_prev)
            rhs0 = rhs(u0, u_prev, tau_predictor, viscosity, volume.axis_coords, kgrid)
            u_pred = integrate_rk4(
                u0=u0,
                prev_u=u_prev,
                tau_predictor=tau_predictor,
                viscosity=viscosity,
                axis_coords=volume.axis_coords,
                kgrid=kgrid,
                dt=SIM_DT,
                n_steps=4,
            )
            pred_k, pred_spec = isotropic_shell_spectrum(u_pred, box_lengths)
            row = {
                "model": model_name,
                "viscosity": viscosity,
                "velocity_rmse": float(np.sqrt(np.mean((u_pred - u_target) ** 2))),
                "rhs_rmse": float(np.sqrt(np.mean((rhs0 - rhs_true) ** 2))),
                "closure_div_rmse": float(np.sqrt(np.mean((divergence_tensor(tau0, volume.axis_coords) - div_tau_true) ** 2))),
                "energy_pred": volume_mean_kinetic_energy(u_pred),
                "energy_true": volume_mean_kinetic_energy(u_target),
                "energy_rel_err": float((volume_mean_kinetic_energy(u_pred) - volume_mean_kinetic_energy(u_target)) / (volume_mean_kinetic_energy(u_target) + 1.0e-30)),
                "spectrum_corr": spectrum_corr(target_spec, pred_spec),
                "div_rms": divergence_rms(u_pred, kgrid),
                "runtime_s": float(perf_counter() - t0),
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "model": model_name,
                "viscosity": viscosity,
                "velocity_rmse": np.nan,
                "rhs_rmse": np.nan,
                "closure_div_rmse": np.nan,
                "energy_pred": np.nan,
                "energy_true": volume_mean_kinetic_energy(u_target),
                "energy_rel_err": np.nan,
                "spectrum_corr": np.nan,
                "div_rms": np.nan,
                "runtime_s": float(perf_counter() - t0),
                "status": f"failed: {type(exc).__name__}",
            }
        rows.append(row)
        print(row)

    out_path = os.path.join(out_dir, "isotropic_short_horizon_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
