#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import os
from time import perf_counter

import numpy as np

from aposteriori_isotropic_les import (
    SymbolicModel,
    build_tau_predictor,
    box_lengths_from_volume,
    calibrate_viscosity,
    divergence_rms,
    integrate_rk4,
    isotropic_shell_spectrum,
    make_kgrid,
    rhs,
    spectrum_corr,
    volume_mean_kinetic_energy,
)
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.structured_sgs import load_filtered_sequence
from sgs_discovery.symbolic_closures import native_model_exprs


def symbolic_model_catalog() -> dict[str, SymbolicModel]:
    exprs = native_model_exprs()
    models = [
        SymbolicModel("Champion", exprs["Champion"]),
        SymbolicModel("Jaumann_hybrid", exprs["Jaumann_hybrid"]),
        SymbolicModel("Wstretch_hybrid", exprs["Wstretch_hybrid"]),
    ]
    return {model.name: model for model in models}


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
            constants = {
                key: float(value)
                for key, value in ast.literal_eval(raw).items()
            }
            constants_by_model[model] = constants
    missing = {"Champion", "Jaumann_hybrid", "Wstretch_hybrid"} - set(constants_by_model)
    if missing:
        raise RuntimeError(f"missing symbolic constants for: {sorted(missing)}")
    return constants_by_model


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
    out_dir = "results/aposteriori_rollout"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "isotropic_rollout_summary.csv")

    sequence = load_filtered_sequence(
        "jhtdb_u_tensor_64_periodic_rollout.h5",
        filter_width=1.0,
        boundary_mode="wrap",
    )
    if len(sequence.filtered_frames) < 4:
        raise RuntimeError("rollout benchmark requires at least 4 frames")

    oracle_ap = JHTDBOracle("jhtdb_u_tensor_64_periodic.h5", filter_width=1.0, boundary_mode="wrap")

    box_lengths = box_lengths_from_volume(sequence)
    kgrid = make_kgrid(sequence.grid_shape, box_lengths)
    _, target_spec_final = isotropic_shell_spectrum(sequence.filtered_frames[-1], box_lengths)

    tau_true = oracle_ap.tau.reshape(sequence.grid_shape + (3, 3))
    viscosity = calibrate_viscosity(
        u0=sequence.filtered_frames[1],
        u1=sequence.filtered_frames[2],
        tau_true=tau_true,
        axis_coords=sequence.axis_coords,
        kgrid=kgrid,
        dt=sequence.dt,
    )

    engine = TensorSymbolicEngine()
    constants_by_model = load_symbolic_constants("results/modern_baselines/apriori_summary.csv")
    symbolic_models = symbolic_model_catalog()

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
    frame_count = len(sequence.filtered_frames)
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

        teacher_rhs_rmse: list[float] = []
        teacher_energy_rel: list[float] = []
        rollout_rmse: list[float] = []
        rollout_energy_rel: list[float] = []
        status = "ok"
        t0 = perf_counter()

        try:
            for step in range(1, len(sequence.filtered_frames) - 1):
                prev_dns = sequence.filtered_frames[step - 1]
                curr_dns = sequence.filtered_frames[step]
                next_dns = sequence.filtered_frames[step + 1]
                rhs_true = (next_dns - curr_dns) / sequence.dt
                rhs_pred = rhs(
                    curr_dns,
                    prev_dns,
                    tau_predictor,
                    viscosity,
                    sequence.axis_coords,
                    kgrid,
                )
                teacher_rhs_rmse.append(float(np.sqrt(np.mean((rhs_pred - rhs_true) ** 2))))
                teacher_energy_rel.append(
                    float(
                        (volume_mean_kinetic_energy(curr_dns + sequence.dt * rhs_pred) - volume_mean_kinetic_energy(next_dns))
                        / (volume_mean_kinetic_energy(next_dns) + 1.0e-30)
                    )
                )

            prev_state = sequence.filtered_frames[0]
            curr_state = sequence.filtered_frames[1]
            rollout_states = [curr_state]
            for horizon_idx in range(1, len(sequence.filtered_frames) - 1):
                pred_next = integrate_rk4(
                    u0=curr_state,
                    prev_u=prev_state,
                    tau_predictor=tau_predictor,
                    viscosity=viscosity,
                    axis_coords=sequence.axis_coords,
                    kgrid=kgrid,
                    dt=sequence.dt,
                    n_steps=4,
                )
                dns_next = sequence.filtered_frames[horizon_idx + 1]
                rollout_states.append(pred_next)
                rollout_rmse.append(float(np.sqrt(np.mean((pred_next - dns_next) ** 2))))
                rollout_energy_rel.append(
                    float(
                        (volume_mean_kinetic_energy(pred_next) - volume_mean_kinetic_energy(dns_next))
                        / (volume_mean_kinetic_energy(dns_next) + 1.0e-30)
                    )
                )
                prev_state, curr_state = curr_state, pred_next

            final_state = rollout_states[-1]
            pred_k, pred_spec_final = isotropic_shell_spectrum(final_state, box_lengths)
            row = {
                "model": model_name,
                "n_frames": frame_count,
                "viscosity": viscosity,
                "teacher_rhs_rmse_mean": float(np.mean(teacher_rhs_rmse)),
                "teacher_rhs_rmse_final": float(teacher_rhs_rmse[-1]),
                "teacher_energy_rel_mean": float(np.mean(np.abs(teacher_energy_rel))),
                "rollout_rmse_mean": float(np.mean(rollout_rmse)),
                "rollout_rmse_final": float(rollout_rmse[-1]),
                "rollout_energy_rel_mean": float(np.mean(np.abs(rollout_energy_rel))),
                "rollout_energy_rel_final": float(rollout_energy_rel[-1]),
                "final_spectrum_corr": spectrum_corr(target_spec_final, pred_spec_final),
                "final_div_rms": divergence_rms(final_state, kgrid),
                "runtime_s": float(perf_counter() - t0),
                "status": status,
            }
        except Exception as exc:
            row = {
                "model": model_name,
                "n_frames": frame_count,
                "viscosity": viscosity,
                "teacher_rhs_rmse_mean": np.nan,
                "teacher_rhs_rmse_final": np.nan,
                "teacher_energy_rel_mean": np.nan,
                "rollout_rmse_mean": np.nan,
                "rollout_rmse_final": np.nan,
                "rollout_energy_rel_mean": np.nan,
                "rollout_energy_rel_final": np.nan,
                "final_spectrum_corr": np.nan,
                "final_div_rms": np.nan,
                "runtime_s": float(perf_counter() - t0),
                "status": f"failed: {type(exc).__name__}",
            }
        rows.append(row)
        save_rows(out_path, rows)
        print(row, flush=True)

    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
