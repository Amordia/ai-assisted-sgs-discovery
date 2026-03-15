#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_modern_baselines import dissipation_corr, evaluate_symbolic, mean_component_correlation
from sgs_discovery.grid_metrics import wall_profile_corr
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.physics_env import TensorSymbolicEngine, c_syms
from sgs_discovery.symbolic_closures import native_model_exprs


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/subsample_sensitivity"
ROWS_PATH = OUT_DIR / "ratio_sweep_rows.csv"
SUMMARY_PATH = OUT_DIR / "ratio_sweep_summary.csv"
FIG_PATH = OUT_DIR / "subsample_stability.pdf"

RATIOS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
SEEDS = (0, 1, 2, 3)
MODELS = ("Jaumann_hybrid", "Champion")
TOP_PERCENTILES = (0.01, 0.05)
MAX_ITER = 250


def quantile_label(quantile: float) -> str:
    return f"q{int(round(quantile * 1000)):03d}"


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_sample_indices(seed: int, oracles: list[JHTDBOracle], ratio: float, min_size: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    sampled: list[np.ndarray] = []
    for oracle in oracles:
        n = oracle.S.shape[0]
        n_sub = max(min_size, int(n * ratio))
        n_sub = min(n_sub, n)
        sampled.append(rng.choice(n, size=n_sub, replace=False))
    return sampled


def backscatter_extremes(oracle: JHTDBOracle, idx: np.ndarray, quantile: float) -> tuple[int, int, float]:
    pi_true = -np.einsum("nij,nij->n", oracle.tau, oracle.S)
    cutoff = float(np.quantile(pi_true, quantile))
    extreme_mask = pi_true <= cutoff
    total = int(np.sum(extreme_mask))
    captured = int(np.sum(extreme_mask[idx]))
    coverage = captured / total if total else 0.0
    return total, captured, coverage


def summarise_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    metrics = [
        "train_loss",
        "mean_r_iso",
        "mean_r_chan",
        "tau12_r_chan",
        "pi_r_iso",
        "pi_r_chan",
        "wall_tau12_corr_chan",
        "nmse_iso",
        "nmse_chan",
        "sample_size_iso",
        "sample_size_chan",
        "coverage_q010_iso",
        "coverage_q010_chan",
        "coverage_q050_iso",
        "coverage_q050_chan",
    ]
    coeff_cols = [f"{ci.name}" for ci in c_syms[:4]]
    for model in MODELS:
        for ratio in RATIOS:
            subset = [
                row for row in rows if str(row["model"]) == model and abs(float(row["ratio"]) - ratio) < 1.0e-12
            ]
            summary = {"model": model, "ratio": ratio, "n_runs": len(subset)}
            for metric in metrics + coeff_cols:
                values = np.array([float(row.get(metric, np.nan)) for row in subset], dtype=float)
                if np.all(np.isnan(values)):
                    summary[f"{metric}_mean"] = float("nan")
                    summary[f"{metric}_std"] = float("nan")
                    continue
                summary[f"{metric}_mean"] = float(np.nanmean(values))
                summary[f"{metric}_std"] = float(np.nanstd(values))
            summary_rows.append(summary)
    return summary_rows


def plot_summary(summary_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
    colors = {"Jaumann_hybrid": "#c44e52", "Champion": "#4c72b0"}
    for model in MODELS:
        subset = [row for row in summary_rows if str(row["model"]) == model]
        x = [float(row["ratio"]) for row in subset]
        axes[0].plot(
            x,
            [float(row["mean_r_chan_mean"]) for row in subset],
            marker="o",
            lw=2.0,
            color=colors[model],
            label=model.replace("_", " "),
        )
        axes[0].fill_between(
            x,
            [float(row["mean_r_chan_mean"]) - float(row["mean_r_chan_std"]) for row in subset],
            [float(row["mean_r_chan_mean"]) + float(row["mean_r_chan_std"]) for row in subset],
            color=colors[model],
            alpha=0.15,
        )
        axes[1].plot(
            x,
            [float(row["wall_tau12_corr_chan_mean"]) for row in subset],
            marker="o",
            lw=2.0,
            color=colors[model],
        )
        axes[2].plot(
            x,
            [float(row["coverage_q010_chan_mean"]) for row in subset],
            marker="o",
            lw=2.0,
            color=colors[model],
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_title("(a) Channel stress correlation", loc="left")
    axes[0].set_xlabel("Frozen subsample ratio")
    axes[0].set_ylabel(r"$\bar r_\tau^{CHAN}$")
    axes[0].legend(frameon=False, fontsize=8.5)

    axes[1].set_title("(b) Wall-profile correlation", loc="left")
    axes[1].set_xlabel("Frozen subsample ratio")
    axes[1].set_ylabel(r"$r_{\langle \tau_{12}\rangle(y)}$")

    axes[2].set_title("(c) Coverage of 1% strongest backscatter points", loc="left")
    axes[2].set_xlabel("Frozen subsample ratio")
    axes[2].set_ylabel("coverage fraction")

    fig.savefig(FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    engine = TensorSymbolicEngine()
    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")
    oracles = [oracle_iso, oracle_chan]
    exprs = native_model_exprs()

    rows: list[dict[str, object]] = []
    for model in MODELS:
        expr = exprs[model]
        for ratio in RATIOS:
            for seed in SEEDS:
                fit_seed = 70_000 + 1000 * MODELS.index(model) + 100 * int(round(ratio * 100)) + seed
                sampled_indices = draw_sample_indices(
                    seed=fit_seed,
                    oracles=oracles,
                    ratio=ratio,
                    min_size=256,
                )
                np.random.seed(fit_seed)
                optimizer = LeafNodeOptimizer(
                    engine,
                    oracles,
                    lambda_pi=1.0,
                    lambda_diss=1.0,
                    lambda_l1=1.0e-5,
                    max_iter=MAX_ITER,
                    subsample_ratio=ratio,
                    min_subsample_size=256,
                )
                constants, train_loss = optimizer.optimize(expr)
                tau_iso = evaluate_symbolic(engine, type("Model", (), {"expr": expr, "name": model})(), constants, oracle_iso)
                tau_chan = evaluate_symbolic(engine, type("Model", (), {"expr": expr, "name": model})(), constants, oracle_chan)
                mean_r_iso, _ = mean_component_correlation(oracle_iso.tau, tau_iso)
                mean_r_chan, comp_chan = mean_component_correlation(oracle_chan.tau, tau_chan)
                row: dict[str, object] = {
                    "model": model,
                    "ratio": float(ratio),
                    "seed": seed,
                    "fit_seed": fit_seed,
                    "train_loss": float(train_loss),
                    "sample_size_iso": int(sampled_indices[0].size),
                    "sample_size_chan": int(sampled_indices[1].size),
                    "nmse_iso": float(oracle_iso.evaluate_mse(tau_iso)),
                    "nmse_chan": float(oracle_chan.evaluate_mse(tau_chan)),
                    "mean_r_iso": mean_r_iso,
                    "mean_r_chan": mean_r_chan,
                    "tau12_r_chan": comp_chan["12"],
                    "pi_r_iso": dissipation_corr(oracle_iso, tau_iso),
                    "pi_r_chan": dissipation_corr(oracle_chan, tau_chan),
                    "wall_tau12_corr_chan": wall_profile_corr(oracle_chan, tau_chan),
                }
                for ci in c_syms[:4]:
                    row[ci.name] = float(constants.get(ci.name, np.nan))
                for quantile in TOP_PERCENTILES:
                    total, captured, coverage = backscatter_extremes(oracle_iso, sampled_indices[0], quantile)
                    qlabel = quantile_label(quantile)
                    row[f"extreme_total_{qlabel}_iso"] = total
                    row[f"extreme_captured_{qlabel}_iso"] = captured
                    row[f"coverage_{qlabel}_iso"] = coverage
                    total, captured, coverage = backscatter_extremes(oracle_chan, sampled_indices[1], quantile)
                    row[f"extreme_total_{qlabel}_chan"] = total
                    row[f"extreme_captured_{qlabel}_chan"] = captured
                    row[f"coverage_{qlabel}_chan"] = coverage
                rows.append(row)
                print(
                    f"[SUBSAMPLE] model={model:>15s} ratio={ratio:>4.2f} seed={seed} "
                    f"mean_r_chan={mean_r_chan:.3f} wall={row['wall_tau12_corr_chan']:.3f}",
                    flush=True,
                )
                save_rows(ROWS_PATH, rows)

    summary_rows = summarise_rows(rows)
    save_rows(SUMMARY_PATH, summary_rows)
    plot_summary(summary_rows)
    print(f"Saved {ROWS_PATH}")
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
