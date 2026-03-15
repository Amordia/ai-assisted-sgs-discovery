from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from sgs_discovery.grid_metrics import reshape_tensor_field
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/comparison_post_dimensional_fix"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.pad_inches": 0.02,
    }
)


MODEL_LABELS = {
    "Leonard": "Leonard",
    "WALE_like": "WALE-like",
    "Smagorinsky_like": "Smagorinsky-like",
    "WALE_canonical": "Canonical WALE",
    "WALE_canonical_physical": "Canonical WALE",
    "Jaumann_hybrid": "Jaumann hybrid",
    "Wstretch_hybrid": "Vortex-stretching hybrid",
    "Champion": "Laplacian-strain hybrid",
    "Bardina_Leonard": "Bardina/Leonard",
    "Dynamic_Smagorinsky": "Dynamic Smagorinsky",
    "AMD_canonical": "AMD",
    "No_model": "No model",
}


MODEL_COLORS = {
    "Leonard": "#4C566A",
    "WALE_canonical": "#5E81AC",
    "WALE_canonical_physical": "#5E81AC",
    "Jaumann_hybrid": "#C2543A",
    "Wstretch_hybrid": "#D18F2C",
    "Champion": "#6B7280",
    "Bardina_Leonard": "#4C566A",
    "Dynamic_Smagorinsky": "#2A9D8F",
    "AMD_canonical": "#8C6C3F",
    "No_model": "#BBBBBB",
}


PLOT_LABELS = {
    "Bardina_Leonard": "Bardina/\nLeonard",
    "Dynamic_Smagorinsky": "Dyn.\nSmag.",
    "AMD_canonical": "AMD",
    "WALE_canonical": "Canonical\nWALE",
    "WALE_canonical_physical": "Canonical\nWALE",
    "Jaumann_hybrid": "Jaumann",
    "Wstretch_hybrid": "Vortex-\nstretching",
    "Champion": "Laplacian-\nstrain",
    "No_model": "No model",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "---"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1.0e3 or (0.0 < abs_value < 1.0e-3):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if pd.isna(mean) or pd.isna(std):
        return "---"
    return f"{fmt(mean, digits)} +/- {fmt(std, digits)}"


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join([header, sep] + rows)


def write_table(name: str, df: pd.DataFrame, latex_columns: dict[str, str] | None = None) -> None:
    csv_path = TABLE_DIR / f"{name}.csv"
    md_path = TABLE_DIR / f"{name}.md"
    tex_path = TABLE_DIR / f"{name}.tex"

    df.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(df) + "\n")

    latex_df = df.rename(columns=latex_columns or {})
    tex = latex_df.to_latex(index=False, escape=False)
    tex_path.write_text(tex)


def save_figure(fig: plt.Figure, name: str) -> None:
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def row_by_model(df: pd.DataFrame, model: str) -> pd.Series:
    return df[df["model"] == model].iloc[0]


def build_baseline_table() -> None:
    df = pd.read_csv(ROOT / "results/baselines/summary_metrics.csv")
    models = [
        "Leonard",
        "WALE_like",
        "Smagorinsky_like",
        "WALE_canonical",
        "Champion",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
    ]
    rows = []
    for model in models:
        row = row_by_model(df, model)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "r_tau_iso": fmt(row["mean_r_iso"]),
                "r_tau_chan": fmt(row["mean_r_chan"]),
                "r_tau12_chan": fmt(row["tau12_r_chan"]),
                "r_pi_iso": fmt(row["pi_r_iso"]),
                "r_pi_chan": fmt(row["pi_r_chan"]),
                "r_wall_tau12": fmt(row["wall_tau12_corr_chan"]),
            }
        )
    write_table(
        "tab_baseline_updated",
        pd.DataFrame(rows),
        latex_columns={
            "model": "Model",
            "r_tau_iso": r"$\overline{r}_{\tau}^{\mathrm{ISO}}$",
            "r_tau_chan": r"$\overline{r}_{\tau}^{\mathrm{CHAN}}$",
            "r_tau12_chan": r"$r_{\tau_{12}}^{\mathrm{CHAN}}$",
            "r_pi_iso": r"$r_{\Pi}^{\mathrm{ISO}}$",
            "r_pi_chan": r"$r_{\Pi}^{\mathrm{CHAN}}$",
            "r_wall_tau12": r"$r_{\langle \tau_{12} \rangle (y)}$",
        },
    )


def build_objective_ablation_table() -> None:
    df = pd.read_csv(ROOT / "results/objective_ablation/objective_ablation_summary.csv")
    label_map = {
        "NMSE_only": "NMSE only",
        "NMSE_plus_backscatter": "NMSE + backscatter",
        "NMSE_plus_dissipation": "NMSE + dissipation",
        "Full_objective": "Full objective",
    }
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "objective": label_map[row["objective"]],
                "r_tau_iso": fmt(row["mean_r_iso_mean"]),
                "r_tau_chan": fmt(row["mean_r_chan_mean"]),
                "nmse_chan": fmt(row["nmse_chan_mean"]),
                "r_pi_chan": fmt(row["pi_r_chan_mean"]),
                "r_wall_tau12": fmt(row["wall_tau12_corr_chan_mean"]),
            }
        )
    write_table(
        "tab_objective_ablation_updated",
        pd.DataFrame(rows),
        latex_columns={
            "objective": "Objective",
            "r_tau_iso": r"$\overline{r}_{\tau}^{\mathrm{ISO}}$",
            "r_tau_chan": r"$\overline{r}_{\tau}^{\mathrm{CHAN}}$",
            "nmse_chan": r"$\mathrm{NMSE}^{\mathrm{CHAN}}$",
            "r_pi_chan": r"$r_{\Pi}^{\mathrm{CHAN}}$",
            "r_wall_tau12": r"$r_{\langle \tau_{12} \rangle (y)}$",
        },
    )


def build_holdout_table() -> None:
    df = pd.read_csv(ROOT / "results/holdout/blocked_holdout_summary.csv")
    models = ["Leonard", "WALE_canonical", "Champion", "Jaumann_hybrid", "Wstretch_hybrid"]
    rows = []
    for model in models:
        row = row_by_model(df, model)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "r_tau_iso_test": fmt_pm(row["mean_r_iso_test_mean"], row["mean_r_iso_test_std"]),
                "r_tau_chan_test": fmt_pm(row["mean_r_chan_test_mean"], row["mean_r_chan_test_std"]),
                "r_tau12_chan_test": fmt_pm(row["tau12_r_chan_test_mean"], row["tau12_r_chan_test_std"]),
                "r_pi_chan_test": fmt_pm(row["pi_r_chan_test_mean"], row["pi_r_chan_test_std"]),
                "r_wall_tau12_test": fmt_pm(
                    row["wall_tau12_corr_chan_test_mean"],
                    row["wall_tau12_corr_chan_test_std"],
                ),
            }
        )
    write_table(
        "tab_holdout_updated",
        pd.DataFrame(rows),
        latex_columns={
            "model": "Model",
            "r_tau_iso_test": r"$\overline{r}_{\tau}^{\mathrm{ISO,test}}$",
            "r_tau_chan_test": r"$\overline{r}_{\tau}^{\mathrm{CHAN,test}}$",
            "r_tau12_chan_test": r"$r_{\tau_{12}}^{\mathrm{CHAN,test}}$",
            "r_pi_chan_test": r"$r_{\Pi}^{\mathrm{CHAN,test}}$",
            "r_wall_tau12_test": r"$r_{\langle \tau_{12} \rangle (y)}^{\mathrm{test}}$",
        },
    )


def build_transfer_combined_table() -> None:
    shifted = pd.read_csv(ROOT / "results/external_cutout/shifted_cutout_summary.csv")
    temporal = pd.read_csv(ROOT / "results/temporal_transfer/temporal_transfer_summary.csv")
    filt = pd.read_csv(ROOT / "results/filter_transfer/filter_width_transfer_summary.csv")
    models = ["Champion", "Jaumann_hybrid", "Wstretch_hybrid"]
    rows = []

    for model in models:
        row = row_by_model(shifted, model)
        rows.append(
            {
                "test": "Shifted spatial",
                "model": MODEL_LABELS[model],
                "r_tau_iso": fmt(row["mean_r_iso_shifted"]),
                "r_tau_chan": fmt(row["mean_r_chan_shifted"]),
                "r_tau12_chan": fmt(row["tau12_r_chan_shifted"]),
                "r_pi_chan": fmt(row["pi_r_chan_shifted"]),
                "r_wall_tau12": fmt(row["wall_tau12_corr_chan_shifted"]),
            }
        )

    for model in models:
        row = temporal[
            (temporal["model"] == model) & (temporal["case"] == "future_same_window")
        ].iloc[0]
        rows.append(
            {
                "test": "Later time (t=51)",
                "model": MODEL_LABELS[model],
                "r_tau_iso": fmt(row["mean_r_iso_transfer"]),
                "r_tau_chan": fmt(row["mean_r_chan_transfer"]),
                "r_tau12_chan": fmt(row["tau12_r_chan_transfer"]),
                "r_pi_chan": fmt(row["pi_r_chan_transfer"]),
                "r_wall_tau12": fmt(row["wall_tau12_corr_chan_transfer"]),
            }
        )

    for sigma in [1.5, 2.0]:
        for model in models:
            row = filt[(filt["model"] == model) & (filt["test_sigma"] == sigma)].iloc[0]
            rows.append(
                {
                    "test": f"Filter Delta={sigma:.1f}",
                    "model": MODEL_LABELS[model],
                    "r_tau_iso": fmt(row["mean_r_iso_transfer"]),
                    "r_tau_chan": fmt(row["mean_r_chan_transfer"]),
                    "r_tau12_chan": fmt(row["tau12_r_chan_transfer"]),
                    "r_pi_chan": fmt(row["pi_r_chan_transfer"]),
                    "r_wall_tau12": fmt(row["wall_tau12_corr_chan_transfer"]),
                }
            )

    write_table(
        "tab_transfer_combined_updated",
        pd.DataFrame(rows),
        latex_columns={
            "test": "Test",
            "model": "Model",
            "r_tau_iso": r"$\overline{r}_{\tau}^{\mathrm{ISO}}$",
            "r_tau_chan": r"$\overline{r}_{\tau}^{\mathrm{CHAN}}$",
            "r_tau12_chan": r"$r_{\tau_{12}}^{\mathrm{CHAN}}$",
            "r_pi_chan": r"$r_{\Pi}^{\mathrm{CHAN}}$",
            "r_wall_tau12": r"$r_{\langle \tau_{12} \rangle (y)}$",
        },
    )


def build_external_table() -> None:
    iso = pd.read_csv(ROOT / "results/isotropic_ood/isotropic_ood_summary.csv")
    chan = pd.read_csv(ROOT / "results/channel5200_ood/channel5200_transfer_summary.csv")
    rows = []

    for case, label in [
        ("isotropic1024coarse", "iso1024c"),
        ("isotropic4096_matched_filter", "iso4096"),
    ]:
        for model in ["Jaumann_hybrid", "Champion"]:
            row = iso[(iso["case"] == case) & (iso["model"] == model)].iloc[0]
            rows.append(
                {
                    "database": label,
                    "model": MODEL_LABELS[model],
                    "r_tau_ood": fmt(row["mean_r_iso_transfer"]),
                    "r_pi_ood": fmt(row["pi_r_iso_transfer"]),
                    "r_tau12_ood": "---",
                    "r_wall_tau12_ood": "---",
                }
            )

    for model in ["Jaumann_hybrid", "Wstretch_hybrid", "WALE_canonical"]:
        row = chan[(chan["case"] == "channel5200") & (chan["model"] == model)].iloc[0]
        rows.append(
            {
                "database": "chan5200",
                "model": MODEL_LABELS[model],
                "r_tau_ood": fmt(row["mean_r_transfer"]),
                "r_pi_ood": fmt(row["pi_r_transfer"]),
                "r_tau12_ood": fmt(row["r12_transfer"]),
                "r_wall_tau12_ood": fmt(row["wall_tau12_corr_transfer"]),
            }
        )

    write_table(
        "tab_external_updated",
        pd.DataFrame(rows),
        latex_columns={
            "database": "Database",
            "model": "Model",
            "r_tau_ood": r"$\overline{r}_{\tau}^{\mathrm{OOD}}$",
            "r_pi_ood": r"$r_{\Pi}^{\mathrm{OOD}}$",
            "r_tau12_ood": r"$r_{\tau_{12}}^{\mathrm{OOD}}$",
            "r_wall_tau12_ood": r"$r_{\langle \tau_{12} \rangle (y)}^{\mathrm{OOD}}$",
        },
    )


def build_modern_baselines_table() -> None:
    df = pd.read_csv(ROOT / "results/modern_baselines/apriori_summary.csv")
    models = [
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "WALE_canonical_physical",
        "AMD_canonical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
    ]
    rows = []
    for model in models:
        row = row_by_model(df, model)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "r_tau_iso": fmt(row["mean_r_iso"]),
                "r_tau_chan": fmt(row["mean_r_chan"]),
                "r_pi_chan": fmt(row["pi_r_chan"]),
                "r_wall_tau12": fmt(row["wall_tau12_corr_chan"]),
            }
        )
    write_table(
        "tab_modern_baselines_updated",
        pd.DataFrame(rows),
        latex_columns={
            "model": "Model",
            "r_tau_iso": r"$\overline{r}_{\tau}^{\mathrm{ISO}}$",
            "r_tau_chan": r"$\overline{r}_{\tau}^{\mathrm{CHAN}}$",
            "r_pi_chan": r"$r_{\Pi}^{\mathrm{CHAN}}$",
            "r_wall_tau12": r"$r_{\langle \tau_{12} \rangle (y)}$",
        },
    )


def build_periodic_rollout_table() -> None:
    df = pd.read_csv(ROOT / "results/aposteriori_rollout/isotropic_rollout_summary.csv")
    models = [
        "No_model",
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "AMD_canonical",
        "Jaumann_hybrid",
    ]
    rows = []
    for model in models:
        row = row_by_model(df, model)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "teacher_rmse": fmt(row["teacher_rhs_rmse_mean"]),
                "rollout_rmse": fmt(row["rollout_rmse_final"], 5),
                "mean_abs_energy_drift": fmt(abs(row["rollout_energy_rel_mean"]), 3),
            }
        )
    write_table(
        "tab_aposteriori_periodic_updated",
        pd.DataFrame(rows),
        latex_columns={
            "model": "Model",
            "teacher_rmse": "Teacher RMSE",
            "rollout_rmse": "Rollout RMSE",
            "mean_abs_energy_drift": r"$\langle |\Delta E|/E \rangle$",
        },
    )


def build_channel_rollout_table() -> None:
    df = pd.read_csv(ROOT / "results/aposteriori_channel/channel_rollout_pchip_summary_4frames.csv")
    models = [
        "No_model",
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "AMD_canonical",
        "Jaumann_hybrid",
    ]
    rows = []
    for model in models:
        row = row_by_model(df, model)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "one_step_rmse": fmt(row["teacher_one_step_rmse_mean"]),
                "two_step_rollout_rmse": fmt(row["rollout_rmse_final"]),
                "div_rms": fmt(row["final_div_rms"]),
            }
        )
    write_table(
        "tab_aposteriori_channel_updated",
        pd.DataFrame(rows),
        latex_columns={
            "model": "Model",
            "one_step_rmse": "One-step RMSE",
            "two_step_rollout_rmse": "2-step rollout RMSE",
            "div_rms": "Div. RMS",
        },
    )


def build_search_campaign_table() -> None:
    df = pd.read_csv(ROOT / "results/corrected_research_large/apriori_screen.csv").head(5)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "candidate": row["model"],
                "balanced_score": fmt(row["balanced_score"]),
                "r_tau_chan": fmt(row["mean_r_chan"]),
                "r_tau12_chan": fmt(row["tau12_r_chan"]),
                "r_pi_chan": fmt(row["pi_r_chan"]),
                "r_wall_tau12": fmt(row["wall_tau12_corr_chan"]),
            }
        )
    write_table(
        "extra_search_campaign_updated",
        pd.DataFrame(rows),
        latex_columns={
            "candidate": "Candidate",
            "balanced_score": "Balanced score",
            "r_tau_chan": r"$\overline{r}_{\tau}^{\mathrm{CHAN}}$",
            "r_tau12_chan": r"$r_{\tau_{12}}^{\mathrm{CHAN}}$",
            "r_pi_chan": r"$r_{\Pi}^{\mathrm{CHAN}}$",
            "r_wall_tau12": r"$r_{\langle \tau_{12} \rangle (y)}$",
        },
    )


def evaluate_tensor_field(
    engine: TensorSymbolicEngine,
    oracle: JHTDBOracle,
    expr,
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


def build_channel_physical_figure() -> None:
    summary = pd.read_csv(ROOT / "results/baselines/summary_metrics.csv")
    representative_models = [
        "Leonard",
        "WALE_canonical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
        "Champion",
    ]
    constants_map = {
        row.model: ast.literal_eval(row.constants)
        for row in summary.itertuples(index=False)
        if row.model in set(representative_models)
    }

    oracle = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")
    engine = TensorSymbolicEngine()
    exprs = native_model_exprs()

    tau_true_3d = reshape_tensor_field(oracle, oracle.tau)
    tau12_true = tau_true_3d[..., 0, 1]
    mean_true = tau12_true.mean(axis=(0, 2))

    interior_start = 2 if mean_true.size > 6 else 0
    interior_stop = mean_true.size - 2 if mean_true.size > 6 else mean_true.size
    interior = mean_true[interior_start:interior_stop]
    slice_idx = int(np.argmax(np.abs(interior)) + interior_start)

    predictions = {}
    profiles = {}
    for model in representative_models:
        tau_pred = evaluate_tensor_field(engine, oracle, exprs[model], constants_map[model])
        tau_pred_3d = reshape_tensor_field(oracle, tau_pred)
        predictions[model] = tau_pred_3d[..., 0, 1]
        profiles[model] = predictions[model].mean(axis=(0, 2))

    wall_distance = np.abs(oracle.y_coords[0] - oracle.y_coords)
    slice_distance = float(wall_distance[slice_idx])
    dns_slice = tau12_true[:, slice_idx, :]
    slice_fields = {"DNS": dns_slice} | {
        model: predictions[model][:, slice_idx, :] for model in representative_models
    }
    vmax = np.quantile(
        np.abs(np.concatenate([field.ravel() for field in slice_fields.values()])),
        0.995,
    )

    fig = plt.figure(figsize=(11.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.0, 1.0], hspace=0.10, wspace=0.10)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(wall_distance, mean_true, color="black", lw=2.7, label="DNS")
    ax0.plot(wall_distance, profiles["Leonard"], color="#4C566A", lw=1.8, ls="-.", label="Leonard")
    ax0.plot(wall_distance, profiles["WALE_canonical"], color="#5E81AC", lw=1.8, ls=":", label="Canonical WALE")
    ax0.plot(wall_distance, profiles["Jaumann_hybrid"], color="#C2543A", lw=2.3, label="Jaumann hybrid")
    ax0.plot(wall_distance, profiles["Wstretch_hybrid"], color="#D18F2C", lw=2.1, ls="--", label="Vortex-stretching hybrid")
    ax0.plot(
        wall_distance,
        profiles["Champion"],
        color="#6B7280",
        lw=1.8,
        ls=(0, (3, 1, 1, 1)),
        label="Laplacian-strain hybrid",
    )
    ax0.axvline(slice_distance, color="#9CA3AF", lw=1.0, ls=":")
    ax0.text(
        slice_distance + 0.001,
        np.nanmax(mean_true) * 0.92,
        "slice plane",
        fontsize=8.5,
        color="#6B7280",
        rotation=90,
        va="top",
    )
    ax0.set_title("(a) Mean SGS shear profile", loc="left")
    ax0.set_xlabel(r"Distance from wall, $1-y$")
    ax0.set_ylabel(r"$\langle \tau_{12} \rangle (y)$")
    legend = ax0.legend(fontsize=8.4, ncol=3, loc="lower left")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.88)
    ax0.grid(color="#E5E7EB", linewidth=0.8)

    extent = [
        float(oracle.x_coords.min()),
        float(oracle.x_coords.max()),
        float(oracle.z_coords.min()),
        float(oracle.z_coords.max()),
    ]
    panels = [
        ("(b) DNS", slice_fields["DNS"]),
        ("(c) Leonard", slice_fields["Leonard"]),
        ("(d) Canonical WALE", slice_fields["WALE_canonical"]),
        ("(e) Jaumann hybrid", slice_fields["Jaumann_hybrid"]),
        ("(f) Vortex-stretching hybrid", slice_fields["Wstretch_hybrid"]),
        ("(g) Laplacian-strain hybrid", slice_fields["Champion"]),
    ]
    axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]
    images = []
    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        images.append(im)
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$z/\delta$")

    cbar = fig.colorbar(images[0], ax=axes, fraction=0.020, pad=0.012)
    cbar.set_label(r"$\tau_{12}$ at peak-shear plane")
    cbar.ax.tick_params(labelsize=8)

    save_figure(fig, "fig_channel_physical_updated")


def annotate_points(ax, df, xcol: str, ycol: str, label_offsets: dict[str, tuple[float, float]]) -> None:
    for _, row in df.iterrows():
        model = row["model"]
        x = row[xcol]
        y = row[ycol]
        dx, dy = label_offsets.get(model, (0.003, 0.02))
        ax.text(
            x + dx,
            y + dy,
            PLOT_LABELS.get(model, MODEL_LABELS.get(model, model)),
            fontsize=8.1,
            color="#1F2937",
            ha="right" if dx < 0 else "left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.12},
        )


def build_validation_map_figure() -> None:
    full = pd.read_csv(ROOT / "results/baselines/summary_metrics.csv")
    holdout = pd.read_csv(ROOT / "results/holdout/blocked_holdout_summary.csv")
    shifted = pd.read_csv(ROOT / "results/external_cutout/shifted_cutout_summary.csv")
    temporal = pd.read_csv(ROOT / "results/temporal_transfer/temporal_transfer_summary.csv")
    filt = pd.read_csv(ROOT / "results/filter_transfer/filter_width_transfer_summary.csv")
    ch5200 = pd.read_csv(ROOT / "results/channel5200_ood/channel5200_transfer_summary.csv")

    models = ["Leonard", "WALE_canonical", "Jaumann_hybrid", "Wstretch_hybrid", "Champion"]
    labels = ["Leonard", "WALE", "Jaumann", "Vortex-\nstretching", "Laplacian-\nstrain"]
    scenarios = [
        "full",
        "holdout",
        "shifted",
        "future",
        "future+\nshift",
        "Delta=1.5",
        "Delta=2.0",
        "5200",
        "5200+\nshift",
    ]

    mean = np.zeros((len(models), len(scenarios)))
    wall = np.zeros((len(models), len(scenarios)))

    for i, model in enumerate(models):
        mean[i, 0] = row_by_model(full, model)["mean_r_chan"]
        wall[i, 0] = row_by_model(full, model)["wall_tau12_corr_chan"]

        mean[i, 1] = row_by_model(holdout, model)["mean_r_chan_test_mean"]
        wall[i, 1] = row_by_model(holdout, model)["wall_tau12_corr_chan_test_mean"]

        mean[i, 2] = row_by_model(shifted, model)["mean_r_chan_shifted"]
        wall[i, 2] = row_by_model(shifted, model)["wall_tau12_corr_chan_shifted"]

        for j, case in enumerate(["future_same_window", "future_shifted_window"], start=3):
            row = temporal[(temporal["model"] == model) & (temporal["case"] == case)].iloc[0]
            mean[i, j] = row["mean_r_chan_transfer"]
            wall[i, j] = row["wall_tau12_corr_chan_transfer"]

        for j, sigma in enumerate([1.5, 2.0], start=5):
            row = filt[(filt["model"] == model) & (filt["test_sigma"] == sigma)].iloc[0]
            mean[i, j] = row["mean_r_chan_transfer"]
            wall[i, j] = row["wall_tau12_corr_chan_transfer"]

        for j, case in enumerate(["channel5200", "channel5200_shifted"], start=7):
            row = ch5200[(ch5200["model"] == model) & (ch5200["case"] == case)].iloc[0]
            mean[i, j] = row["mean_r_transfer"]
            wall[i, j] = row["wall_tau12_corr_transfer"]

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 4.9), constrained_layout=True)
    configs = [
        (
            axes[0],
            mean,
            "(a) Channel mean-stress correlation",
            TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.32),
            "RdBu_r",
        ),
        (
            axes[1],
            wall,
            "(b) Wall-shear-profile correlation",
            TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
            "RdBu_r",
        ),
    ]

    for ax, data, title, norm, cmap in configs:
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks(np.arange(len(scenarios)))
        ax.set_xticklabels(scenarios)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticks(np.arange(-0.5, len(scenarios), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                text_color = "white" if abs(norm(value)) > 0.62 else "#1F2937"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8.2)
        cbar = fig.colorbar(im, ax=ax, fraction=0.024, pad=0.012)
        cbar.ax.tick_params(labelsize=8)

    save_figure(fig, "fig_validation_map_updated")


def build_solver_screen_figure() -> None:
    modern = pd.read_csv(ROOT / "results/modern_baselines/apriori_summary.csv")
    periodic = pd.read_csv(ROOT / "results/aposteriori_rollout/isotropic_rollout_summary.csv")
    periodic = periodic[periodic["n_frames"] == 13].copy()
    channel = pd.read_csv(ROOT / "results/aposteriori_channel/channel_rollout_pchip_summary_4frames.csv")

    modern_models = [
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "AMD_canonical",
        "WALE_canonical_physical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
        "Champion",
    ]
    modern = modern[modern["model"].isin(modern_models)].copy()

    periodic_models = [
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "AMD_canonical",
        "WALE_canonical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
        "Champion",
    ]
    periodic = periodic[periodic["model"].isin(["No_model"] + periodic_models)].copy()
    baseline_periodic = float(periodic[periodic["model"] == "No_model"]["rollout_rmse_final"].iloc[0])
    baseline_teacher = float(periodic[periodic["model"] == "No_model"]["teacher_rhs_rmse_mean"].iloc[0])
    periodic = periodic[periodic["model"] != "No_model"].copy()
    periodic["teacher_improvement_pct"] = 100.0 * (
        baseline_teacher - periodic["teacher_rhs_rmse_mean"]
    ) / baseline_teacher
    periodic["rollout_improvement_pct"] = 100.0 * (
        baseline_periodic - periodic["rollout_rmse_final"]
    ) / baseline_periodic
    periodic = periodic.set_index("model").loc[periodic_models].reset_index()

    channel_models = [
        "No_model",
        "Bardina_Leonard",
        "Dynamic_Smagorinsky",
        "AMD_canonical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
        "WALE_canonical",
    ]
    channel = channel.set_index("model").loc[channel_models].reset_index()

    fig = plt.figure(figsize=(10.6, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("RdBu_r")
    norm = TwoSlopeNorm(vmin=-0.40, vcenter=0.0, vmax=0.65)
    for _, row in modern.iterrows():
        model = row["model"]
        ax0.scatter(
            row["mean_r_chan"],
            row["wall_tau12_corr_chan"],
            s=118,
            c=[row["pi_r_chan"]],
            cmap=cmap,
            norm=norm,
            edgecolor="#1F2937",
            linewidth=0.7,
            zorder=3,
        )
    ax0.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    ax0.axvline(0.0, color="#9CA3AF", linewidth=0.8)
    ax0.grid(color="#E5E7EB", linewidth=0.6, alpha=0.65)
    ax0.set_xlabel(r"Full-field $\overline{r}_{\tau}^{\mathrm{CHAN}}$")
    ax0.set_ylabel(r"Wall-profile $r_{\langle \tau_{12} \rangle (y)}$")
    ax0.set_title("(a) Wall-bounded a priori trade-off", loc="left")
    ax0.set_xlim(-0.12, 0.36)
    ax0.set_ylim(-1.02, 1.08)
    offsets = {
        "Bardina_Leonard": (0.010, 0.07),
        "Dynamic_Smagorinsky": (0.010, -0.10),
        "AMD_canonical": (0.010, -0.08),
        "WALE_canonical_physical": (0.010, -0.06),
        "Jaumann_hybrid": (0.010, -0.06),
        "Wstretch_hybrid": (0.014, -0.02),
        "Champion": (0.010, -0.08),
    }
    annotate_points(ax0, modern, "mean_r_chan", "wall_tau12_corr_chan", offsets)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(modern["pi_r_chan"])
    cbar = fig.colorbar(sm, ax=ax0, fraction=0.042, pad=0.025)
    cbar.set_label(r"$r_{\Pi}^{\mathrm{CHAN}}$")
    cbar.ax.tick_params(labelsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(periodic))
    teacher = periodic["teacher_improvement_pct"].to_numpy()
    rollout = periodic["rollout_improvement_pct"].to_numpy()
    for yi, model, x0, x1 in zip(y, periodic["model"], teacher, rollout):
        ax1.plot([min(x0, x1), max(x0, x1)], [yi, yi], color="#D1D5DB", linewidth=1.5, zorder=1)
        ax1.scatter(
            x0,
            yi,
            s=62,
            color=MODEL_COLORS.get(model, "#4C566A"),
            edgecolor="#1F2937",
            linewidth=0.5,
            zorder=3,
        )
        ax1.scatter(
            x1,
            yi,
            s=52,
            facecolors="white",
            edgecolors=MODEL_COLORS.get(model, "#4C566A"),
            linewidth=1.2,
            zorder=4,
        )
    ax1.set_yticks(y)
    ax1.set_yticklabels([PLOT_LABELS.get(m, MODEL_LABELS.get(m, m)) for m in periodic["model"]])
    ax1.invert_yaxis()
    ax1.set_xlabel("Improvement over no-model (%)")
    ax1.set_title("(b) Periodic-box rollout screen", loc="left")
    ax1.set_xlim(-0.02, max(teacher.max(), rollout.max()) * 1.08)
    ax1.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#6B7280", markeredgecolor="#1F2937", markersize=6.5, label="Teacher RHS"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#6B7280", markersize=6.5, label="Final rollout"),
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )

    ax2 = fig.add_subplot(gs[1, 0])
    channel_plot_models = [
        "Dynamic_Smagorinsky",
        "Bardina_Leonard",
        "AMD_canonical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
        "WALE_canonical",
    ]
    channel_plot = channel[channel["model"].isin(channel_plot_models)].set_index("model").loc[channel_plot_models].reset_index()
    baseline_rmse = float(channel[channel["model"] == "No_model"]["rollout_rmse_final"].iloc[0])
    y = np.arange(len(channel_plot))
    ax2.axvline(baseline_rmse, color="#9CA3AF", linestyle="--", linewidth=1.1, zorder=1)
    for yi, (_, row) in enumerate(channel_plot.iterrows()):
        val = row["rollout_rmse_final"]
        ax2.plot([baseline_rmse, val], [yi, yi], color="#D1D5DB", linewidth=1.3, zorder=1)
        ax2.scatter(
            val,
            yi,
            s=64,
            color=MODEL_COLORS.get(row["model"], "#4C566A"),
            edgecolor="#1F2937",
            linewidth=0.5,
            zorder=3,
        )
    ax2.set_xscale("log")
    ax2.set_yticks(y)
    ax2.set_yticklabels([PLOT_LABELS.get(m, MODEL_LABELS.get(m, m)) for m in channel_plot["model"]])
    ax2.invert_yaxis()
    ax2.set_xlabel("Final 2-step RMSE")
    ax2.set_title("(c) Channel rollout RMSE", loc="left")
    ax2.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax2.text(
        0.98,
        0.98,
        "Dashed line = no model",
        transform=ax2.transAxes,
        fontsize=7.4,
        color="#6B7280",
        ha="right",
        va="top",
    )

    ax3 = fig.add_subplot(gs[1, 1])
    baseline_div = float(channel[channel["model"] == "No_model"]["final_div_rms"].iloc[0])
    y = np.arange(len(channel_plot))
    ax3.axvline(baseline_div, color="#9CA3AF", linestyle="--", linewidth=1.1, zorder=1)
    for yi, (_, row) in enumerate(channel_plot.iterrows()):
        val = row["final_div_rms"]
        ax3.plot([baseline_div, val], [yi, yi], color="#D1D5DB", linewidth=1.3, zorder=1)
        ax3.scatter(
            val,
            yi,
            s=64,
            color=MODEL_COLORS.get(row["model"], "#4C566A"),
            edgecolor="#1F2937",
            linewidth=0.5,
            zorder=3,
        )
    ax3.set_xscale("log")
    ax3.set_yticks(y)
    ax3.set_yticklabels([PLOT_LABELS.get(m, MODEL_LABELS.get(m, m)) for m in channel_plot["model"]])
    ax3.invert_yaxis()
    ax3.set_xlabel("Final divergence RMS")
    ax3.set_title("(d) Channel divergence growth", loc="left")
    ax3.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax3.text(
        0.98,
        0.98,
        "Dashed line = no model",
        transform=ax3.transAxes,
        fontsize=7.4,
        color="#6B7280",
        ha="right",
        va="top",
    )

    save_figure(fig, "fig_solver_screen_updated")


def build_readme() -> None:
    content = """# Post-Fix Comparison Pack

This directory records post-dimensional-consistency versions of the experimental
tables and figures that appeared in the manuscript. The original manuscript
files were left untouched on purpose. These files are a side-by-side lookup pack
for later comparison against the older paper snapshot.

## Model-name mapping

- `Champion` = current dimensionally consistent Laplacian-strain branch.
- `Jaumann_hybrid` = current dimensionally consistent Jaumann hybrid with a `Delta^2 S_j` term.
- `Wstretch_hybrid` = current dimensionally consistent vortex-stretching hybrid.
- `WALE_canonical_physical` = the modern-baseline table row corresponding to canonical WALE.

## Table mapping

- Old `tab:baseline` -> `tables/tab_baseline_updated.*`
- Old `tab:objective_ablation` -> `tables/tab_objective_ablation_updated.*`
- Old `tab:holdout` -> `tables/tab_holdout_updated.*`
- Old `tab:transfer_combined` -> `tables/tab_transfer_combined_updated.*`
- Old `tab:external` -> `tables/tab_external_updated.*`
- Old `tab:modern_baselines` -> `tables/tab_modern_baselines_updated.*`
- Old `tab:aposteriori_periodic` -> `tables/tab_aposteriori_periodic_updated.*`
- Old `tab:aposteriori_channel` -> `tables/tab_aposteriori_channel_updated.*`
- Old expanded-search discussion -> `tables/extra_search_campaign_updated.*`

## Figure mapping

- Old `fig:channel_physical` -> `figures/fig_channel_physical_updated.pdf`
- Old `fig:validation_map` -> `figures/fig_validation_map_updated.pdf`
- Old `fig:solver_screen` -> `figures/fig_solver_screen_updated.pdf`

## Source result files

- `results/baselines/summary_metrics.csv`
- `results/holdout/blocked_holdout_summary.csv`
- `results/external_cutout/shifted_cutout_summary.csv`
- `results/temporal_transfer/temporal_transfer_summary.csv`
- `results/filter_transfer/filter_width_transfer_summary.csv`
- `results/isotropic_ood/isotropic_ood_summary.csv`
- `results/channel5200_ood/channel5200_transfer_summary.csv`
- `results/modern_baselines/apriori_summary.csv`
- `results/objective_ablation/objective_ablation_summary.csv`
- `results/corrected_research_large/apriori_screen.csv`
- `results/aposteriori_isotropic/isotropic_short_horizon_summary.csv`
- `results/aposteriori_rollout/isotropic_rollout_summary.csv`
- `results/aposteriori_channel/channel_rollout_pchip_summary_4frames.csv`

## Notes

- These artifacts reflect the rerun, physically consistent benchmark chain.
- Some old narrative claims no longer hold under the updated results; this pack
  is intentionally descriptive rather than editorial.
- The objective-ablation table now corresponds to the current `Champion`
  definition used in code, not the older pre-fix naked-dimensional form.
"""
    (OUT_DIR / "README.md").write_text(content)


def main() -> None:
    ensure_dirs()
    build_baseline_table()
    build_objective_ablation_table()
    build_holdout_table()
    build_transfer_combined_table()
    build_external_table()
    build_modern_baselines_table()
    build_periodic_rollout_table()
    build_channel_rollout_table()
    build_search_campaign_table()
    build_channel_physical_figure()
    build_validation_map_figure()
    build_solver_screen_figure()
    build_readme()
    print(f"Saved comparison pack to {OUT_DIR}")


if __name__ == "__main__":
    main()
