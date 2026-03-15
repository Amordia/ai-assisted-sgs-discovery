#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from aposteriori_isotropic_les import SymbolicModel, symbolic_tau
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.structured_sgs import (
    feature_bundle,
    flatten_scalar_field,
    flatten_tensor_field,
    flatten_vector_field,
    load_filtered_sequence,
    tau_amd,
    tau_dynamic_smagorinsky,
    tau_vreman,
    tau_wale,
    volume_mean_kinetic_energy,
)
from sgs_discovery.symbolic_closures import native_model_exprs


EPS = 1.0e-30
DEFAULT_INPUT = "channel_fullheight_u_tensor_64_rollout.h5"
DEFAULT_OUTPUT = "results/aposteriori_channel/channel_rollout_summary.csv"


@dataclass
class ChannelOps:
    y_coords: np.ndarray
    D1: np.ndarray
    D2: np.ndarray
    kx: np.ndarray
    kz: np.ndarray
    KX: np.ndarray
    KZ: np.ndarray
    K2: np.ndarray
    poisson_cache: dict[tuple[int, int], tuple[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal wall-bounded a posteriori channel rollout benchmark.")
    parser.add_argument("--h5", default=DEFAULT_INPUT, help="Full-height channel rollout HDF5.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="CSV summary output.")
    parser.add_argument("--max-frames", type=int, default=7, help="Maximum number of frames to use from the rollout file.")
    parser.add_argument("--substeps", type=int, default=2, help="Integrator substeps per DNS interval.")
    parser.add_argument("--calib-steps", type=int, default=2, help="Teacher-forced transitions used to fit effective viscosity.")
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model list. Empty means all defaults.",
    )
    return parser.parse_args()


def finite_diff_weights(nodes: np.ndarray, x0: float, deriv_order: int) -> np.ndarray:
    n = len(nodes)
    shifted = nodes - x0
    vand = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    for power in range(n):
        vand[power, :] = shifted ** power
    rhs[deriv_order] = float(math.factorial(deriv_order))
    return np.linalg.solve(vand, rhs)


def derivative_matrix(coords: np.ndarray, deriv_order: int, stencil_size: int = 5) -> np.ndarray:
    n = len(coords)
    if n < stencil_size:
        raise ValueError(f"need at least {stencil_size} points, got {n}")
    if stencil_size <= deriv_order:
        raise ValueError("stencil_size must exceed deriv_order")
    half = stencil_size // 2
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i < half:
            start = 0
        elif i >= n - half:
            start = n - stencil_size
        else:
            start = i - half
        stop = start + stencil_size
        stencil_idx = np.arange(start, stop)
        weights = finite_diff_weights(coords[stencil_idx], coords[i], deriv_order)
        mat[i, stencil_idx] = weights
    return mat


def apply_y_matrix(field: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if field.ndim == 3:
        return np.einsum("ij,zjx->zix", matrix, field, optimize=True)
    if field.ndim == 4:
        return np.einsum("ij,zjxc->zixc", matrix, field, optimize=True)
    raise ValueError(f"unsupported field rank for y-matrix application: {field.ndim}")


def build_channel_ops(sequence) -> ChannelOps:
    z_coords = np.asarray(sequence.z_coords, dtype=np.float64)
    x_coords = np.asarray(sequence.x_coords, dtype=np.float64)
    y_coords = np.asarray(sequence.y_coords, dtype=np.float64)
    z_len = float(z_coords[-1] - z_coords[0] + np.abs(np.gradient(z_coords)).mean())
    x_len = float(x_coords[-1] - x_coords[0] + np.abs(np.gradient(x_coords)).mean())
    kz = 2.0 * np.pi * np.fft.fftfreq(len(z_coords), d=z_len / len(z_coords))
    kx = 2.0 * np.pi * np.fft.fftfreq(len(x_coords), d=x_len / len(x_coords))
    KZ, KX = np.meshgrid(kz, kx, indexing="ij")
    K2 = KX * KX + KZ * KZ

    D1 = derivative_matrix(y_coords, deriv_order=1, stencil_size=5)
    D2 = derivative_matrix(y_coords, deriv_order=2, stencil_size=5)
    eye = np.eye(len(y_coords), dtype=np.complex128)
    zero_mode_matrix = D2.astype(np.complex128).copy()
    zero_mode_matrix[0, :] = D1[0, :]
    zero_mode_matrix[-1, :] = D1[-1, :]

    poisson_cache: dict[tuple[int, int], tuple[str, object]] = {}
    for iz, kz_val in enumerate(kz[: len(kz) // 2 + 1]):
        for ix, kx_val in enumerate(kx[: len(kx) // 2 + 1]):
            key = (iz, ix)
            k2 = float(kx_val * kx_val + kz_val * kz_val)
            matrix = (D2.astype(np.complex128) - k2 * eye).copy()
            matrix[0, :] = D1[0, :]
            matrix[-1, :] = D1[-1, :]
            if iz == 0 and ix == 0:
                poisson_cache[key] = ("pinv", np.linalg.pinv(zero_mode_matrix))
            else:
                poisson_cache[key] = ("lu", lu_factor(matrix))

    return ChannelOps(
        y_coords=y_coords,
        D1=D1,
        D2=D2,
        kx=kx,
        kz=kz,
        KX=KX,
        KZ=KZ,
        K2=K2,
        poisson_cache=poisson_cache,
    )


def folded_mode_index(size: int, idx: int) -> int:
    return idx if idx <= size // 2 else size - idx


def solve_poisson(rhs: np.ndarray, ops: ChannelOps, iz: int, ix: int) -> np.ndarray:
    key = (folded_mode_index(len(ops.kz), iz), folded_mode_index(len(ops.kx), ix))
    rhs_mode = np.asarray(rhs, dtype=np.complex128).copy()
    kind, solver = ops.poisson_cache[key]
    if key == (0, 0):
        rhs_mode[1:-1] -= np.mean(rhs_mode[1:-1])
        return solver @ rhs_mode
    return lu_solve(solver, rhs_mode)


def enforce_no_slip(u: np.ndarray) -> np.ndarray:
    out = np.asarray(u, dtype=np.float64).copy()
    out[:, 0, :, :] = 0.0
    out[:, -1, :, :] = 0.0
    return out


def project_field(field: np.ndarray, ops: ChannelOps, enforce_velocity_bc: bool) -> np.ndarray:
    qhat = np.fft.fftn(np.asarray(field, dtype=np.float64), axes=(0, 2))
    d1_qy = apply_y_matrix(qhat[..., 1], ops.D1)
    proj = qhat.copy()
    for iz, kz_val in enumerate(ops.kz):
        for ix, kx_val in enumerate(ops.kx):
            rhs_mode = 1j * kx_val * qhat[iz, :, ix, 0] + d1_qy[iz, :, ix] + 1j * kz_val * qhat[iz, :, ix, 2]
            rhs_mode[0] = qhat[iz, 0, ix, 1]
            rhs_mode[-1] = qhat[iz, -1, ix, 1]
            phi = solve_poisson(rhs_mode, ops, iz, ix)
            dphi_dy = ops.D1 @ phi
            proj[iz, :, ix, 0] -= 1j * kx_val * phi
            proj[iz, :, ix, 1] -= dphi_dy
            proj[iz, :, ix, 2] -= 1j * kz_val * phi
    projected = np.fft.ifftn(proj, axes=(0, 2)).real
    return enforce_no_slip(projected) if enforce_velocity_bc else projected


def spectral_derivative_x(field: np.ndarray, ops: ChannelOps) -> np.ndarray:
    fhat = np.fft.fftn(field, axes=(0, 2))
    return np.fft.ifftn(1j * ops.KX[:, None, :, None] * fhat, axes=(0, 2)).real


def spectral_derivative_z(field: np.ndarray, ops: ChannelOps) -> np.ndarray:
    fhat = np.fft.fftn(field, axes=(0, 2))
    return np.fft.ifftn(1j * ops.KZ[:, None, :, None] * fhat, axes=(0, 2)).real


def laplacian_vector(u: np.ndarray, ops: ChannelOps) -> np.ndarray:
    uhat = np.fft.fftn(u, axes=(0, 2))
    lap_xz = np.fft.ifftn((-ops.K2[:, None, :, None]) * uhat, axes=(0, 2)).real
    lap_y = apply_y_matrix(u, ops.D2)
    return lap_xz + lap_y


def convective_term(u: np.ndarray, ops: ChannelOps) -> np.ndarray:
    dudx = spectral_derivative_x(u, ops)
    dudz = spectral_derivative_z(u, ops)
    dudy = apply_y_matrix(u, ops.D1)
    return u[..., 0:1] * dudx + u[..., 1:2] * dudy + u[..., 2:3] * dudz


def divergence_tensor_channel(tau: np.ndarray, ops: ChannelOps) -> np.ndarray:
    out = np.zeros(tau.shape[:3] + (3,), dtype=np.float64)
    for comp in range(3):
        grad_x = spectral_derivative_x(tau[..., comp, 0][..., None], ops)[..., 0]
        grad_z = spectral_derivative_z(tau[..., comp, 2][..., None], ops)[..., 0]
        grad_y = apply_y_matrix(tau[..., comp, 1], ops.D1)
        out[..., comp] = grad_x + grad_y + grad_z
    return out


def divergence_rms(u: np.ndarray, ops: ChannelOps) -> float:
    uhat = np.fft.fftn(u, axes=(0, 2))
    d1_qy = apply_y_matrix(uhat[..., 1], ops.D1)
    div_hat = 1j * ops.KX[:, None, :] * uhat[..., 0] + d1_qy + 1j * ops.KZ[:, None, :] * uhat[..., 2]
    div = np.fft.ifftn(div_hat, axes=(0, 2)).real
    return float(np.sqrt(np.mean(div ** 2)))


def load_symbolic_constants(path: str) -> dict[str, dict[str, float]]:
    constants_by_model: dict[str, dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            raw = row.get("constants", "").strip()
            if not raw:
                continue
            if model not in {"Champion", "Jaumann_hybrid", "Wstretch_hybrid"}:
                continue
            constants_by_model[model] = {key: float(value) for key, value in ast.literal_eval(raw).items()}
    return constants_by_model


def symbolic_model_catalog() -> dict[str, SymbolicModel]:
    exprs = native_model_exprs()
    models = [
        SymbolicModel("Champion", exprs["Champion"]),
        SymbolicModel("Jaumann_hybrid", exprs["Jaumann_hybrid"]),
        SymbolicModel("Wstretch_hybrid", exprs["Wstretch_hybrid"]),
    ]
    return {model.name: model for model in models}


def build_tau_predictor(
    name: str,
    engine: TensorSymbolicEngine,
    symbolic_models: dict[str, SymbolicModel],
    constants_by_model: dict[str, dict[str, float]],
    axis_coords,
    delta_eff: np.ndarray,
    axis_widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    feature_dt: float,
):
    def predict(u: np.ndarray, prev_u: np.ndarray) -> np.ndarray:
        bundle = feature_bundle(
            u,
            axis_coords=axis_coords,
            mode="nearest",
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
                mode="nearest",
                plane_average=True,
            )
            return tau
        if name == "WALE_canonical":
            return tau_wale(bundle["S"], bundle["S_d"], delta_eff)
        if name == "AMD_canonical":
            return tau_amd(bundle["grad_u"], bundle["S"], axis_widths)
        if name == "Vreman_canonical":
            return tau_vreman(bundle["grad_u"], bundle["S"], axis_widths)
        model = symbolic_models[name]
        return symbolic_tau(engine, model.expr, constants_by_model[name], bundle, delta_eff)

    return predict


def rhs(
    u: np.ndarray,
    prev_u: np.ndarray,
    tau_predictor,
    viscosity: float,
    ops: ChannelOps,
) -> np.ndarray:
    tau = tau_predictor(u, prev_u)
    conv = convective_term(u, ops)
    div_tau = divergence_tensor_channel(tau, ops)
    projected = project_field(conv + div_tau, ops, enforce_velocity_bc=False)
    return -projected + viscosity * laplacian_vector(u, ops)


def integrate_heun(
    u0: np.ndarray,
    prev_u: np.ndarray,
    tau_predictor,
    viscosity: float,
    ops: ChannelOps,
    dt: float,
    n_substeps: int,
) -> np.ndarray:
    u = project_field(u0, ops, enforce_velocity_bc=True)
    feature_prev = project_field(prev_u, ops, enforce_velocity_bc=True)
    sub_dt = dt / max(n_substeps, 1)
    for _ in range(max(n_substeps, 1)):
        k1 = rhs(u, feature_prev, tau_predictor, viscosity, ops)
        predictor = project_field(u + sub_dt * k1, ops, enforce_velocity_bc=True)
        k2 = rhs(predictor, feature_prev, tau_predictor, viscosity, ops)
        u_next = project_field(u + 0.5 * sub_dt * (k1 + k2), ops, enforce_velocity_bc=True)
        u = u_next
    return u


def calibrate_viscosity(
    frames: list[np.ndarray],
    tau_predictor,
    ops: ChannelOps,
    dt: float,
    calib_steps: int,
) -> float:
    numerator = 0.0
    denominator = 0.0
    max_step = min(calib_steps, len(frames) - 2)
    for step in range(1, max_step + 1):
        prev_u = frames[step - 1]
        curr_u = frames[step]
        next_u = frames[step + 1]
        tau = tau_predictor(curr_u, prev_u)
        explicit = project_field(convective_term(curr_u, ops) + divergence_tensor_channel(tau, ops), ops, enforce_velocity_bc=False)
        lap = laplacian_vector(curr_u, ops)
        rhs_true = (next_u - curr_u) / dt
        target = rhs_true + explicit
        numerator += float(np.sum(lap * target))
        denominator += float(np.sum(lap * lap))
    return max(numerator / (denominator + EPS), 0.0)


def profile_corr(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref_profile = np.mean(reference[..., 0], axis=(0, 2))
    pred_profile = np.mean(prediction[..., 0], axis=(0, 2))
    if np.std(ref_profile) < 1.0e-15 or np.std(pred_profile) < 1.0e-15:
        return float("nan")
    return float(np.corrcoef(ref_profile, pred_profile)[0, 1])


def wall_shear_rel(reference: np.ndarray, prediction: np.ndarray, ops: ChannelOps) -> float:
    ref_grad = apply_y_matrix(np.mean(reference[..., 0], axis=(0, 2))[None, :, None], ops.D1)[0, :, 0]
    pred_grad = apply_y_matrix(np.mean(prediction[..., 0], axis=(0, 2))[None, :, None], ops.D1)[0, :, 0]
    ref_walls = np.array([ref_grad[0], ref_grad[-1]], dtype=np.float64)
    pred_walls = np.array([pred_grad[0], pred_grad[-1]], dtype=np.float64)
    return float(np.mean(np.abs(pred_walls - ref_walls) / (np.abs(ref_walls) + EPS)))


def save_rows(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sequence = load_filtered_sequence(args.h5, filter_width=1.0, boundary_mode="nearest")
    n_frames = min(args.max_frames, len(sequence.filtered_frames))
    filtered_frames = [np.asarray(frame, dtype=np.float64) for frame in sequence.filtered_frames[:n_frames]]
    ops = build_channel_ops(sequence)
    reference_frames = [project_field(frame, ops, enforce_velocity_bc=True) for frame in filtered_frames]

    constants_by_model = load_symbolic_constants("results/modern_baselines/apriori_summary.csv")
    engine = TensorSymbolicEngine()
    symbolic_models = symbolic_model_catalog()
    default_models = [
        "No_model",
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "WALE_canonical",
        "AMD_canonical",
        "Vreman_canonical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
    ]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models.strip() else default_models

    rows: list[dict[str, object]] = []
    for model_name in models:
        tau_predictor = build_tau_predictor(
            name=model_name,
            engine=engine,
            symbolic_models=symbolic_models,
            constants_by_model=constants_by_model,
            axis_coords=sequence.axis_coords,
            delta_eff=sequence.delta_eff,
            axis_widths=sequence.axis_widths,
            feature_dt=sequence.dt,
        )
        t0 = perf_counter()
        status = "ok"
        row = {
            "model": model_name,
            "n_frames": len(reference_frames),
            "dt": sequence.dt,
            "viscosity": np.nan,
            "teacher_rhs_rmse_mean": np.nan,
            "teacher_one_step_rmse_mean": np.nan,
            "teacher_energy_rel_mean": np.nan,
            "rollout_rmse_mean": np.nan,
            "rollout_rmse_final": np.nan,
            "rollout_energy_rel_mean": np.nan,
            "rollout_energy_rel_final": np.nan,
            "final_profile_corr": np.nan,
            "final_wall_shear_rel": np.nan,
            "final_div_rms": np.nan,
            "reference_div_rms_final": float(divergence_rms(reference_frames[-1], ops)),
            "runtime_s": np.nan,
            "status": status,
        }
        try:
            viscosity = calibrate_viscosity(
                frames=reference_frames,
                tau_predictor=tau_predictor,
                ops=ops,
                dt=sequence.dt,
                calib_steps=args.calib_steps,
            )
            row["viscosity"] = viscosity

            teacher_rhs_rmse: list[float] = []
            teacher_one_step_rmse: list[float] = []
            teacher_energy_rel: list[float] = []
            rollout_rmse: list[float] = []
            rollout_energy_rel: list[float] = []

            for step in range(1, len(reference_frames) - 1):
                prev_dns = reference_frames[step - 1]
                curr_dns = reference_frames[step]
                next_dns = reference_frames[step + 1]
                rhs_true = (next_dns - curr_dns) / sequence.dt
                rhs_pred = rhs(curr_dns, prev_dns, tau_predictor, viscosity, ops)
                teacher_rhs_rmse.append(float(np.sqrt(np.mean((rhs_pred - rhs_true) ** 2))))

                teacher_next = integrate_heun(
                    u0=curr_dns,
                    prev_u=prev_dns,
                    tau_predictor=tau_predictor,
                    viscosity=viscosity,
                    ops=ops,
                    dt=sequence.dt,
                    n_substeps=args.substeps,
                )
                teacher_one_step_rmse.append(float(np.sqrt(np.mean((teacher_next - next_dns) ** 2))))
                teacher_energy_rel.append(
                    float(
                        (volume_mean_kinetic_energy(teacher_next) - volume_mean_kinetic_energy(next_dns))
                        / (volume_mean_kinetic_energy(next_dns) + EPS)
                    )
                )

            row["teacher_rhs_rmse_mean"] = float(np.mean(teacher_rhs_rmse))
            row["teacher_one_step_rmse_mean"] = float(np.mean(teacher_one_step_rmse))
            row["teacher_energy_rel_mean"] = float(np.mean(np.abs(teacher_energy_rel)))

            prev_state = reference_frames[0]
            curr_state = reference_frames[1]
            rollout_states = [curr_state]
            for horizon_idx in range(1, len(reference_frames) - 1):
                pred_next = integrate_heun(
                    u0=curr_state,
                    prev_u=prev_state,
                    tau_predictor=tau_predictor,
                    viscosity=viscosity,
                    ops=ops,
                    dt=sequence.dt,
                    n_substeps=args.substeps,
                )
                dns_next = reference_frames[horizon_idx + 1]
                rollout_states.append(pred_next)
                rollout_rmse.append(float(np.sqrt(np.mean((pred_next - dns_next) ** 2))))
                rollout_energy_rel.append(
                    float(
                        (volume_mean_kinetic_energy(pred_next) - volume_mean_kinetic_energy(dns_next))
                        / (volume_mean_kinetic_energy(dns_next) + EPS)
                    )
                )
                prev_state, curr_state = curr_state, pred_next

            final_state = rollout_states[-1]
            final_ref = reference_frames[-1]
            row["rollout_rmse_mean"] = float(np.mean(rollout_rmse))
            row["rollout_rmse_final"] = float(rollout_rmse[-1])
            row["rollout_energy_rel_mean"] = float(np.mean(np.abs(rollout_energy_rel)))
            row["rollout_energy_rel_final"] = float(rollout_energy_rel[-1])
            row["final_profile_corr"] = profile_corr(final_ref, final_state)
            row["final_wall_shear_rel"] = wall_shear_rel(final_ref, final_state, ops)
            row["final_div_rms"] = divergence_rms(final_state, ops)
        except Exception as exc:
            row["status"] = f"failed: {type(exc).__name__}"
        row["runtime_s"] = float(perf_counter() - t0)
        rows.append(row)
        save_rows(args.out, rows)
        print(row, flush=True)

    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
