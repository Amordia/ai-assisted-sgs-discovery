from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sgs_discovery.grid_metrics import reshape_tensor_field
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import native_model_exprs


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "results" / "generated_figures"


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.pad_inches": 0.02,
    }
)


def _evaluate(engine: TensorSymbolicEngine, oracle: JHTDBOracle, expr, constants: dict[str, float]) -> np.ndarray:
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
    summary = pd.read_csv(ROOT / "results/baselines/summary_metrics.csv")
    representative_models = [
        "Leonard",
        "WALE_canonical",
        "Jaumann_hybrid",
        "Wstretch_hybrid",
        "Champion",
    ]
    model_labels = {
        "Leonard": "Leonard",
        "WALE_canonical": "Canonical WALE",
        "Jaumann_hybrid": "Jaumann hybrid",
        "Wstretch_hybrid": "Vortex-stretching hybrid",
        "Champion": "Laplacian-strain hybrid",
    }
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

    predictions: dict[str, np.ndarray] = {}
    profiles: dict[str, np.ndarray] = {}
    for model in representative_models:
        tau_pred = _evaluate(engine, oracle, exprs[model], constants_map[model])
        tau_pred_3d = reshape_tensor_field(oracle, tau_pred)
        predictions[model] = tau_pred_3d[..., 0, 1]
        profiles[model] = predictions[model].mean(axis=(0, 2))

    wall_distance = np.abs(oracle.y_coords[0] - oracle.y_coords)
    slice_distance = float(wall_distance[slice_idx])

    dns_slice = tau12_true[:, slice_idx, :]
    slice_fields = {"DNS": dns_slice} | {model: predictions[model][:, slice_idx, :] for model in representative_models}
    vmax = np.quantile(np.abs(np.concatenate([field.ravel() for field in slice_fields.values()])), 0.995)

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
        (f"(c) {model_labels['Leonard']}", slice_fields["Leonard"]),
        (f"(d) {model_labels['WALE_canonical']}", slice_fields["WALE_canonical"]),
        (f"(e) {model_labels['Jaumann_hybrid']}", slice_fields["Jaumann_hybrid"]),
        (f"(f) {model_labels['Wstretch_hybrid']}", slice_fields["Wstretch_hybrid"]),
        (f"(g) {model_labels['Champion']}", slice_fields["Champion"]),
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

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "fig_channel_physical.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
