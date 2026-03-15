#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import sympy as sp

from aposteriori_isotropic_les import (
    SymbolicModel,
    box_lengths_from_volume,
    build_tau_predictor,
    calibrate_viscosity,
    divergence_rms,
    integrate_rk4,
    isotropic_shell_spectrum,
    make_kgrid,
    rhs,
    spectrum_corr,
    volume_mean_kinetic_energy,
)
from benchmark_modern_baselines import (
    dissipation_corr,
    evaluate_symbolic,
    mean_component_correlation,
)
from sgs_discovery.grid_metrics import wall_profile_corr
from sgs_discovery.mcts_agent import NeuroSymbolicMCTS
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.physics_env import (
    TensorSymbolicEngine,
    c_syms,
)
from sgs_discovery.structured_sgs import load_filtered_sequence
from sgs_discovery.symbolic_closures import corrected_catalog, corrected_root_library

OUT_DIR = "results/corrected_research_large"
SEARCH_RUNS_PATH = os.path.join(OUT_DIR, "search_runs.csv")
RAW_CANDIDATES_PATH = os.path.join(OUT_DIR, "search_candidates_raw.csv")
APRIORI_PATH = os.path.join(OUT_DIR, "apriori_screen.csv")
SOLVER_PATH = os.path.join(OUT_DIR, "solver_screen.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "top_summary.csv")

SEARCH_SEEDS = (11, 29)
SEARCH_ITERATIONS = 5
SEARCH_MAX_DEPTH = 3
SEARCH_TOP_K_PER_RUN = 12
SEARCH_OPT_MAX_ITER = 140
REFIT_MAX_ITER = 400
APRIORI_SHORTLIST = 10
SOLVER_SHORTLIST = 4


@dataclass
class RolloutContext:
    sequence: object
    viscosity: float
    kgrid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    box_lengths: tuple[float, float, float]
    target_spec_final: np.ndarray


def save_rows(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clipped(value: float) -> float:
    if np.isnan(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def canonicalize_expr(expr: sp.MatrixExpr) -> sp.MatrixExpr:
    used = [ci for ci in c_syms if ci in expr.free_symbols]
    if not used:
        return sp.expand(expr)
    mapping = {old: c_syms[idx] for idx, old in enumerate(used)}
    return sp.expand(expr.xreplace(mapping))


def expr_key(expr: sp.MatrixExpr) -> str:
    return str(canonicalize_expr(expr))


def root_library() -> list[tuple[str, sp.MatrixExpr]]:
    return corrected_root_library()


def known_catalog() -> dict[str, str]:
    catalog = corrected_catalog()
    return {expr_key(expr): name for name, expr in catalog.items()}


def evaluate_apriori_row(
    engine: TensorSymbolicEngine,
    expr: sp.MatrixExpr,
    constants: dict[str, float],
    oracle_iso: JHTDBOracle,
    oracle_chan: JHTDBOracle,
) -> dict[str, object]:
    tau_iso = evaluate_symbolic(engine, SymbolicModel("candidate", expr), constants, oracle_iso)
    tau_chan = evaluate_symbolic(engine, SymbolicModel("candidate", expr), constants, oracle_chan)
    mean_r_iso, comp_iso = mean_component_correlation(oracle_iso.tau, tau_iso)
    mean_r_chan, comp_chan = mean_component_correlation(oracle_chan.tau, tau_chan)
    pi_r_iso = dissipation_corr(oracle_iso, tau_iso)
    pi_r_chan = dissipation_corr(oracle_chan, tau_chan)
    wall_corr = wall_profile_corr(oracle_chan, tau_chan)
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
    return {
        "nmse_iso": float(oracle_iso.evaluate_mse(tau_iso)),
        "nmse_chan": float(oracle_chan.evaluate_mse(tau_chan)),
        "mean_r_iso": mean_r_iso,
        "mean_r_chan": mean_r_chan,
        "tau12_r_chan": comp_chan["12"],
        "pi_r_iso": pi_r_iso,
        "pi_r_chan": pi_r_chan,
        "wall_tau12_corr_chan": wall_corr,
        "balanced_score": balanced_score,
        "channel_score": channel_score,
    }


def make_rollout_context() -> RolloutContext:
    sequence = load_filtered_sequence(
        "jhtdb_u_tensor_64_periodic_rollout.h5",
        filter_width=1.0,
        boundary_mode="wrap",
    )
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
    return RolloutContext(
        sequence=sequence,
        viscosity=viscosity,
        kgrid=kgrid,
        box_lengths=box_lengths,
        target_spec_final=target_spec_final,
    )


def evaluate_rollout_row(
    engine: TensorSymbolicEngine,
    model: SymbolicModel,
    constants: dict[str, float],
    context: RolloutContext,
) -> dict[str, object]:
    tau_predictor = build_tau_predictor(
        name=model.name,
        engine=engine,
        symbolic_models={model.name: model},
        constants_by_model={model.name: constants},
        axis_coords=context.sequence.axis_coords,
        delta_eff=context.sequence.delta_eff,
        axis_widths=context.sequence.axis_widths,
        feature_dt=context.sequence.dt,
    )

    teacher_rhs_rmse: list[float] = []
    teacher_energy_rel: list[float] = []
    rollout_rmse: list[float] = []
    rollout_energy_rel: list[float] = []
    t0 = perf_counter()

    for step in range(1, len(context.sequence.filtered_frames) - 1):
        prev_dns = context.sequence.filtered_frames[step - 1]
        curr_dns = context.sequence.filtered_frames[step]
        next_dns = context.sequence.filtered_frames[step + 1]
        rhs_true = (next_dns - curr_dns) / context.sequence.dt
        rhs_pred = rhs(
            curr_dns,
            prev_dns,
            tau_predictor,
            context.viscosity,
            context.sequence.axis_coords,
            context.kgrid,
        )
        teacher_rhs_rmse.append(float(np.sqrt(np.mean((rhs_pred - rhs_true) ** 2))))
        teacher_energy_rel.append(
            float(
                (
                    volume_mean_kinetic_energy(curr_dns + context.sequence.dt * rhs_pred)
                    - volume_mean_kinetic_energy(next_dns)
                )
                / (volume_mean_kinetic_energy(next_dns) + 1.0e-30)
            )
        )

    prev_state = context.sequence.filtered_frames[0]
    curr_state = context.sequence.filtered_frames[1]
    rollout_states = [curr_state]
    for horizon_idx in range(1, len(context.sequence.filtered_frames) - 1):
        pred_next = integrate_rk4(
            u0=curr_state,
            prev_u=prev_state,
            tau_predictor=tau_predictor,
            viscosity=context.viscosity,
            axis_coords=context.sequence.axis_coords,
            kgrid=context.kgrid,
            dt=context.sequence.dt,
            n_steps=4,
        )
        dns_next = context.sequence.filtered_frames[horizon_idx + 1]
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
    _, pred_spec_final = isotropic_shell_spectrum(final_state, context.box_lengths)
    return {
        "n_frames": len(context.sequence.filtered_frames),
        "viscosity": context.viscosity,
        "teacher_rhs_rmse_mean": float(np.mean(teacher_rhs_rmse)),
        "teacher_rhs_rmse_final": float(teacher_rhs_rmse[-1]),
        "teacher_energy_rel_mean": float(np.mean(np.abs(teacher_energy_rel))),
        "rollout_rmse_mean": float(np.mean(rollout_rmse)),
        "rollout_rmse_final": float(rollout_rmse[-1]),
        "rollout_energy_rel_mean": float(np.mean(np.abs(rollout_energy_rel))),
        "rollout_energy_rel_final": float(rollout_energy_rel[-1]),
        "final_spectrum_corr": spectrum_corr(context.target_spec_final, pred_spec_final),
        "final_div_rms": divergence_rms(final_state, context.kgrid),
        "runtime_s": float(perf_counter() - t0),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading corrected search oracles...", flush=True)
    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")
    engine = TensorSymbolicEngine()

    search_runs: list[dict[str, object]] = []
    raw_candidates: list[dict[str, object]] = []
    candidate_pool: dict[str, dict[str, object]] = {}
    known = known_catalog()

    roots = root_library()
    for seed in SEARCH_SEEDS:
        for root_name, root_expr in roots:
            print("=" * 80, flush=True)
            print(f"[SEARCH] root={root_name} seed={seed}", flush=True)
            np.random.seed(seed)
            optimizer = LeafNodeOptimizer(
                engine,
                [oracle_iso, oracle_chan],
                lambda_pi=1.0,
                lambda_diss=1.0,
                lambda_l1=1.0e-5,
                max_iter=SEARCH_OPT_MAX_ITER,
            )
            mcts = NeuroSymbolicMCTS(
                engine=engine,
                oracles=[oracle_iso, oracle_chan],
                optimizer=optimizer,
                exploration_weight=1.414,
                max_depth=SEARCH_MAX_DEPTH,
                gemini_api_key="",
            )
            root = mcts.search(root_expr=root_expr, n_iterations=SEARCH_ITERATIONS, log_interval=1)
            top_nodes = mcts.get_top_k(SEARCH_TOP_K_PER_RUN)
            best = top_nodes[0]
            run_row = {
                "root": root_name,
                "seed": seed,
                "best_search_loss": float(best.mse),
                "best_expr": str(best.expr),
                "best_expr_key": expr_key(best.expr),
                "tree_size": mcts._tree_size(root),
                "evaluated_count": len(mcts._all_evaluated),
                "known_match": known.get(expr_key(best.expr), ""),
            }
            search_runs.append(run_row)
            save_rows(SEARCH_RUNS_PATH, search_runs)

            for rank, node in enumerate(top_nodes, start=1):
                key = expr_key(node.expr)
                raw_row = {
                    "root": root_name,
                    "seed": seed,
                    "rank_in_run": rank,
                    "search_loss": float(node.mse),
                    "expr": str(node.expr),
                    "expr_key": key,
                    "known_match": known.get(key, ""),
                    "search_constants": str(node.best_constants),
                }
                raw_candidates.append(raw_row)
                best_so_far = candidate_pool.get(key)
                if best_so_far is None or float(node.mse) < float(best_so_far["best_search_loss"]):
                    candidate_pool[key] = {
                        "expr": canonicalize_expr(node.expr),
                        "best_search_loss": float(node.mse),
                        "search_constants": str(node.best_constants),
                        "sources": [f"{root_name}:seed{seed}:rank{rank}"],
                        "known_match": known.get(key, ""),
                    }
                else:
                    best_so_far["sources"].append(f"{root_name}:seed{seed}:rank{rank}")
            save_rows(RAW_CANDIDATES_PATH, raw_candidates)

    unique_candidates = sorted(
        candidate_pool.values(),
        key=lambda row: float(row["best_search_loss"]),
    )
    print(f"[SEARCH] unique structural candidates: {len(unique_candidates)}", flush=True)

    print("Refitting shortlisted candidates on corrected full-field metrics...", flush=True)
    apriori_rows: list[dict[str, object]] = []
    shortlist_state: dict[str, tuple[sp.MatrixExpr, dict[str, float]]] = {}
    refit_optimizer = LeafNodeOptimizer(
        engine,
        [oracle_iso, oracle_chan],
        lambda_pi=1.0,
        lambda_diss=1.0,
        lambda_l1=1.0e-5,
        max_iter=REFIT_MAX_ITER,
    )

    for idx, candidate in enumerate(unique_candidates[:APRIORI_SHORTLIST], start=1):
        expr = candidate["expr"]
        model_name = f"search_{idx:02d}"
        print(f"[APRIORI] {model_name}", flush=True)
        np.random.seed(7000 + idx)
        constants, refit_loss = refit_optimizer.optimize(expr)
        metrics = evaluate_apriori_row(engine, expr, constants, oracle_iso, oracle_chan)
        row = {
            "model": model_name,
            "known_match": candidate["known_match"],
            "expr": str(expr),
            "best_search_loss": float(candidate["best_search_loss"]),
            "refit_loss": float(refit_loss),
            "constants": str(constants),
            "sources": " | ".join(candidate["sources"]),
            **metrics,
        }
        apriori_rows.append(row)
        shortlist_state[model_name] = (expr, constants)
        save_rows(APRIORI_PATH, apriori_rows)

    apriori_rows.sort(
        key=lambda row: (
            float(row["balanced_score"]),
            float(row["channel_score"]),
            -float(row["refit_loss"]),
        ),
        reverse=True,
    )
    save_rows(APRIORI_PATH, apriori_rows)

    print("Running 13-frame periodic-box rollout on top discovered candidates...", flush=True)
    rollout_context = make_rollout_context()
    solver_rows: list[dict[str, object]] = []
    for row in apriori_rows[:SOLVER_SHORTLIST]:
        expr, constants = shortlist_state[str(row["model"])]
        model = SymbolicModel(name=str(row["model"]), expr=expr)
        print(f"[SOLVER] {model.name}", flush=True)
        rollout_metrics = evaluate_rollout_row(engine, model, constants, rollout_context)
        solver_rows.append(
            {
                "model": model.name,
                "known_match": row["known_match"],
                "expr": row["expr"],
                "balanced_score": row["balanced_score"],
                "channel_score": row["channel_score"],
                **rollout_metrics,
            }
        )
        save_rows(SOLVER_PATH, solver_rows)

    solver_rows.sort(key=lambda row: float(row["rollout_rmse_final"]))
    save_rows(SOLVER_PATH, solver_rows)

    summary_rows: list[dict[str, object]] = []
    if apriori_rows:
        best_apriori = apriori_rows[0]
        summary_rows.append(
            {
                "summary": "best_apriori_candidate",
                "model": best_apriori["model"],
                "known_match": best_apriori["known_match"],
                "balanced_score": best_apriori["balanced_score"],
                "channel_score": best_apriori["channel_score"],
                "mean_r_iso": best_apriori["mean_r_iso"],
                "mean_r_chan": best_apriori["mean_r_chan"],
                "tau12_r_chan": best_apriori["tau12_r_chan"],
                "pi_r_chan": best_apriori["pi_r_chan"],
                "wall_tau12_corr_chan": best_apriori["wall_tau12_corr_chan"],
            }
        )
    if solver_rows:
        best_solver = min(solver_rows, key=lambda row: float(row["rollout_rmse_final"]))
        summary_rows.append(
            {
                "summary": "best_solver_candidate",
                "model": best_solver["model"],
                "known_match": best_solver["known_match"],
                "rollout_rmse_final": best_solver["rollout_rmse_final"],
                "teacher_rhs_rmse_mean": best_solver["teacher_rhs_rmse_mean"],
                "rollout_energy_rel_mean": best_solver["rollout_energy_rel_mean"],
                "final_spectrum_corr": best_solver["final_spectrum_corr"],
            }
        )
    save_rows(SUMMARY_PATH, summary_rows)

    print(f"Saved {SEARCH_RUNS_PATH}", flush=True)
    print(f"Saved {RAW_CANDIDATES_PATH}", flush=True)
    print(f"Saved {APRIORI_PATH}", flush=True)
    print(f"Saved {SOLVER_PATH}", flush=True)
    print(f"Saved {SUMMARY_PATH}", flush=True)

    if apriori_rows:
        print("\nTop corrected re-search candidates by balanced_score:", flush=True)
        for row in apriori_rows[:5]:
            print(
                f"  {row['model']}: known_match={row['known_match'] or 'novel'} "
                f"balanced={float(row['balanced_score']):.3f} "
                f"mean_r_chan={float(row['mean_r_chan']):.3f} "
                f"tau12={float(row['tau12_r_chan']):.3f} "
                f"pi_chan={float(row['pi_r_chan']):.3f} "
                f"wall={float(row['wall_tau12_corr_chan']):.3f}",
                flush=True,
            )
    if solver_rows:
        print("\nSolver-screened candidates by final rollout RMSE:", flush=True)
        for row in solver_rows:
            print(
                f"  {row['model']}: known_match={row['known_match'] or 'novel'} "
                f"rollout_final={float(row['rollout_rmse_final']):.8f} "
                f"teacher_rhs={float(row['teacher_rhs_rmse_mean']):.4f} "
                f"energy_bias={float(row['rollout_energy_rel_mean']):.3e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
