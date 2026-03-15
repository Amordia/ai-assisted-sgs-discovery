#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from sgs_discovery.oracle import JHTDBOracle


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/commutation"
SUMMARY_PATH = OUT_DIR / "commutation_summary.csv"
PROFILE_PATH = OUT_DIR / "channel_commutation_profiles.csv"
FIG_PATH = OUT_DIR / "commutation_profiles.pdf"


def filter_vector_field(u: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    out = np.empty_like(u, dtype=np.float64)
    for i in range(3):
        out[..., i] = gaussian_filter(u[..., i], sigma=sigma, mode=mode)
    return out


def filter_tensor_field(tensor: np.ndarray, sigma: float, mode: str) -> np.ndarray:
    out = np.empty_like(tensor, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out[..., i, j] = gaussian_filter(tensor[..., i, j], sigma=sigma, mode=mode)
    return out


def load_raw_velocity(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        u = np.asarray(f["Velocity_t2"], dtype=np.float64)
        y = np.asarray(f["y_coords"], dtype=np.float64) if "y_coords" in f else None
    return u, y


def make_strain(oracle: JHTDBOracle, u: np.ndarray) -> np.ndarray:
    z, y, x = oracle.grid_shape
    grad = np.zeros((z, y, x, 3, 3), dtype=np.float64)
    for comp in range(3):
        gz, gy, gx = oracle._gradient3(u[..., comp])
        grad[..., comp, 0] = gx
        grad[..., comp, 1] = gy
        grad[..., comp, 2] = gz
    return 0.5 * (grad + np.swapaxes(grad, -1, -2))


def laplacian_tensor(oracle: JHTDBOracle, tensor: np.ndarray) -> np.ndarray:
    out = np.zeros_like(tensor, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            field = tensor[..., i, j]
            dzz = oracle._gradient_axis(oracle._gradient_axis(field, axis=0), axis=0)
            dyy = oracle._gradient_axis(oracle._gradient_axis(field, axis=1), axis=1)
            dxx = oracle._gradient_axis(oracle._gradient_axis(field, axis=2), axis=2)
            out[..., i, j] = dxx + dyy + dzz
    return out


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def relative_profile(comm: np.ndarray, ref: np.ndarray) -> np.ndarray:
    num = np.mean(np.abs(comm), axis=(0, 2))
    den = np.mean(np.abs(ref), axis=(0, 2)) + 1.0e-30
    return num / den


def summarise_commutation(
    dataset: str,
    operator: str,
    comm: np.ndarray,
    ref: np.ndarray,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "operator": operator,
        "comm_rms": rms(comm),
        "ref_rms": rms(ref),
        "relative_rms": rms(comm) / (rms(ref) + 1.0e-30),
        "mean_abs": float(np.mean(np.abs(comm))),
        "max_abs": float(np.max(np.abs(comm))),
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    configs = [
        ("isotropic", "jhtdb_u_tensor_64.h5", "wrap"),
        ("channel", "channel_u_tensor_64.h5", "nearest"),
    ]

    summary_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    channel_profiles: dict[str, np.ndarray] = {}
    channel_y: np.ndarray | None = None

    for dataset, h5_name, mode in configs:
        raw_u, _ = load_raw_velocity(h5_name)
        oracle = JHTDBOracle(h5_name, filter_width=1.0, boundary_mode=mode)
        oracle._configure_spatial_coordinates(*oracle.grid_shape)
        filt_u = filter_vector_field(raw_u, sigma=1.0, mode=mode)

        grad_comm_components = []
        grad_ref_components = []
        for axis, axis_name in enumerate(["dz", "dy", "dx"]):
            raw_grad = np.stack(
                [oracle._gradient_axis(raw_u[..., comp], axis=axis) for comp in range(3)],
                axis=-1,
            )
            filt_raw_grad = filter_vector_field(raw_grad, sigma=1.0, mode=mode)
            grad_of_filt = np.stack(
                [oracle._gradient_axis(filt_u[..., comp], axis=axis) for comp in range(3)],
                axis=-1,
            )
            comm = filt_raw_grad - grad_of_filt
            summary_rows.append(summarise_commutation(dataset, f"grad_{axis_name}", comm, grad_of_filt))
            grad_comm_components.append(comm)
            grad_ref_components.append(grad_of_filt)

            if dataset == "channel" and axis_name == "dy":
                profile = relative_profile(
                    np.linalg.norm(comm, axis=-1),
                    np.linalg.norm(grad_of_filt, axis=-1),
                )
                channel_profiles["grad_dy"] = profile

        grad_comm = np.stack(grad_comm_components, axis=-1)
        grad_ref = np.stack(grad_ref_components, axis=-1)
        summary_rows.append(
            summarise_commutation(
                dataset,
                "grad_all",
                np.linalg.norm(grad_comm, axis=(-1, -2)),
                np.linalg.norm(grad_ref, axis=(-1, -2)),
            )
        )

        raw_S = make_strain(oracle, raw_u)
        filt_raw_S = filter_tensor_field(raw_S, sigma=1.0, mode=mode)
        S_from_filt = make_strain(oracle, filt_u)
        S_comm = filt_raw_S - S_from_filt
        summary_rows.append(summarise_commutation(dataset, "strain", S_comm, S_from_filt))

        raw_lap_s = laplacian_tensor(oracle, raw_S)
        filt_raw_lap_s = filter_tensor_field(raw_lap_s, sigma=1.0, mode=mode)
        lap_s_from_filt = laplacian_tensor(oracle, S_from_filt)
        lap_comm = filt_raw_lap_s - lap_s_from_filt
        summary_rows.append(summarise_commutation(dataset, "laplacian_strain", lap_comm, lap_s_from_filt))

        if dataset == "channel":
            channel_y = np.asarray(oracle.y_coords, dtype=np.float64)
            channel_profiles["strain"] = relative_profile(
                np.linalg.norm(S_comm, axis=(-1, -2)),
                np.linalg.norm(S_from_filt, axis=(-1, -2)),
            )
            channel_profiles["laplacian_strain"] = relative_profile(
                np.linalg.norm(lap_comm, axis=(-1, -2)),
                np.linalg.norm(lap_s_from_filt, axis=(-1, -2)),
            )

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if channel_y is not None:
        for i, y in enumerate(channel_y):
            row = {"y": float(y)}
            for key, values in channel_profiles.items():
                row[key] = float(values[i])
            profile_rows.append(row)
        with open(PROFILE_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
            writer.writeheader()
            writer.writerows(profile_rows)

        wall_distance = np.abs(channel_y[0] - channel_y)
        fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), constrained_layout=True)
        axes[0].plot(wall_distance, channel_profiles["grad_dy"], color="#1f77b4", lw=2.0)
        axes[0].set_title("(a) Relative commutation error for wall-normal velocity gradients", loc="left")
        axes[0].set_ylabel(r"$\langle |C_{\partial_y u}| \rangle / \langle |\partial_y \bar u| \rangle$")
        axes[0].grid(alpha=0.3)

        axes[1].plot(wall_distance, channel_profiles["strain"], color="#d95f02", lw=1.8, label="strain")
        axes[1].plot(
            wall_distance,
            channel_profiles["laplacian_strain"],
            color="#7570b3",
            lw=1.8,
            label="laplacian strain",
        )
        axes[1].set_title("(b) Relative commutation error for derived SGS features", loc="left")
        axes[1].set_xlabel(r"Distance from wall, $1-y$")
        axes[1].set_ylabel(r"relative commutation error")
        axes[1].legend(frameon=False, fontsize=8.5)
        axes[1].grid(alpha=0.3)
        fig.savefig(FIG_PATH, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {PROFILE_PATH}")
    print(f"Saved {FIG_PATH}")


if __name__ == "__main__":
    main()
