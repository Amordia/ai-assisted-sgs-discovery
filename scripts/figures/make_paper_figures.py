from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "results" / "generated_figures"


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
    "WALE_canonical": "Canonical WALE",
    "WALE_canonical_physical": "Canonical WALE",
    "Jaumann_hybrid": "Jaumann hybrid",
    "Wstretch_hybrid": "Vortex-stretching hybrid",
    "Champion": "Laplacian-strain hybrid",
    "Archived_hybrid": "Laplacian-strain hybrid",
    "Bardina_Leonard": "Bardina/Leonard",
    "Dynamic_Smagorinsky": "Dynamic Smagorinsky",
    "AMD_canonical": "AMD",
    "WALE_canonical_periodic": "Canonical WALE",
}

MODEL_COLORS = {
    "Leonard": "#4C566A",
    "WALE_canonical": "#5E81AC",
    "WALE_canonical_physical": "#5E81AC",
    "Jaumann_hybrid": "#C2543A",
    "Wstretch_hybrid": "#D18F2C",
    "Champion": "#6B7280",
    "Archived_hybrid": "#6B7280",
    "Bardina_Leonard": "#4C566A",
    "Dynamic_Smagorinsky": "#2A9D8F",
    "AMD_canonical": "#8C6C3F",
    "WALE_canonical_periodic": "#5E81AC",
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


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _box(ax, xy, wh, title, lines, face, edge="#2E3440"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.08 * w,
        y + 0.70 * h,
        title,
        fontsize=10.2,
        fontweight="bold",
        va="top",
    )
    ax.text(
        x + 0.08 * w,
        y + 0.46 * h,
        "\n".join(lines),
        fontsize=8.3,
        va="top",
        linespacing=1.26,
    )


def _arrow(ax, p0, p1):
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#4C566A",
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def make_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 1.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        (0.015, 0.24),
        (0.22, 0.42),
        "Data oracles",
        [
            "DNS cutouts + SGS targets",
            "Explicit x/y/z metrics",
        ],
        face="#EEF4FA",
    )
    _box(
        ax,
        (0.26, 0.24),
        (0.22, 0.42),
        "Search space",
        [
            "Compact tensor library",
            "MCTS + optional LLM proposals",
        ],
        face="#F7F2EA",
    )
    _box(
        ax,
        (0.505, 0.24),
        (0.22, 0.42),
        "Numerical screening",
        [
            "Admissibility guards",
            "Coeff. fitting + dual-oracle score",
        ],
        face="#EEF7F1",
    )
    _box(
        ax,
        (0.75, 0.24),
        (0.22, 0.42),
        "Validated outputs",
        [
            "Holdout / transfer / solver screens",
            "Jaumann / vortex-stretching families",
        ],
        face="#FFF6DF",
    )

    _arrow(ax, (0.235, 0.50), (0.26, 0.50))
    _arrow(ax, (0.48, 0.50), (0.505, 0.50))
    _arrow(ax, (0.725, 0.50), (0.75, 0.50))

    _save(fig, "fig_discovery_workflow.pdf")


def _row(df: pd.DataFrame, model: str) -> pd.Series:
    return df[df["model"] == model].iloc[0]


def make_validation_map() -> None:
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
        "$\\Delta=1.5$",
        "$\\Delta=2.0$",
        "5200",
        "5200+\nshift",
    ]

    mean = np.zeros((len(models), len(scenarios)))
    wall = np.zeros((len(models), len(scenarios)))

    for i, model in enumerate(models):
        mean[i, 0] = _row(full, model)["mean_r_chan"]
        wall[i, 0] = _row(full, model)["wall_tau12_corr_chan"]

        mean[i, 1] = _row(holdout, model)["mean_r_chan_test_mean"]
        wall[i, 1] = _row(holdout, model)["wall_tau12_corr_chan_test_mean"]

        mean[i, 2] = _row(shifted, model)["mean_r_chan_shifted"]
        wall[i, 2] = _row(shifted, model)["wall_tau12_corr_chan_shifted"]

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
            TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.20),
            "RdBu_r",
        ),
        (
            axes[1],
            wall,
            "(b) Wall-shear-profile correlation",
            TwoSlopeNorm(vmin=-0.65, vcenter=0.0, vmax=1.0),
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

    _save(fig, "fig_validation_map.pdf")


def _annotate_points(ax, df, xcol, ycol, label_offsets):
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


def make_screening_figure() -> None:
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
    norm = TwoSlopeNorm(vmin=-0.40, vcenter=0.0, vmax=0.55)
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
    ax0.set_xlim(-0.12, 0.28)
    ax0.set_ylim(-1.02, 1.08)
    offsets = {
        "Bardina_Leonard": (0.010, 0.07),
        "Dynamic_Smagorinsky": (0.010, -0.10),
        "AMD_canonical": (0.010, -0.08),
        "WALE_canonical_physical": (0.010, -0.06),
        "Jaumann_hybrid": (-0.010, 0.035),
        "Wstretch_hybrid": (0.016, 0.085),
        "Champion": (0.010, -0.08),
    }
    _annotate_points(ax0, modern, "mean_r_chan", "wall_tau12_corr_chan", offsets)
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
    ax1.set_xlim(-0.01, max(teacher.max(), rollout.max()) * 1.08)
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
    ax2.text(0.98, 0.98, "Dashed line = no model", transform=ax2.transAxes, fontsize=7.4, color="#6B7280", ha="right", va="top")

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
    ax3.text(0.98, 0.98, "Dashed line = no model", transform=ax3.transAxes, fontsize=7.4, color="#6B7280", ha="right", va="top")

    _save(fig, "fig_solver_screen.pdf")


if __name__ == "__main__":
    make_workflow()
    make_validation_map()
    make_screening_figure()
