"""
grid_metrics.py
===============
Shared helpers for recovering structured grid shapes from flattened oracle
arrays and for wall-profile metrics on wall-bounded data.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class GridLike(Protocol):
    n_samples: int
    y_coords: np.ndarray | None
    tau: np.ndarray


def infer_grid_shape(obj: GridLike) -> tuple[int, int, int]:
    grid_shape = getattr(obj, "grid_shape", None)
    if grid_shape is not None:
        z, y, x = (int(v) for v in grid_shape)
        if z * y * x != int(obj.n_samples):
            raise ValueError(
                f"grid_shape={grid_shape} is inconsistent with n_samples={obj.n_samples}"
            )
        return z, y, x

    y_coords = getattr(obj, "y_coords", None)
    z_coords = getattr(obj, "z_coords", None)
    x_coords = getattr(obj, "x_coords", None)

    if y_coords is not None:
        y = len(y_coords)
        if x_coords is not None and z_coords is not None:
            z = len(z_coords)
            x = len(x_coords)
        else:
            xz = int(obj.n_samples) // y
            x = int(round(np.sqrt(xz)))
            z = xz // x
        if z * y * x != int(obj.n_samples):
            raise ValueError(
                f"Cannot infer wall-bounded grid shape for n_samples={obj.n_samples}"
            )
        return z, y, x

    side = int(round(float(obj.n_samples) ** (1.0 / 3.0)))
    if side ** 3 != int(obj.n_samples):
        raise ValueError(f"Cannot infer cubic grid shape for n_samples={obj.n_samples}")
    return side, side, side


def reshape_tensor_field(obj: GridLike, field: np.ndarray) -> np.ndarray:
    z, y, x = infer_grid_shape(obj)
    return np.asarray(field).reshape(z, y, x, 3, 3)


def wall_profile_corr(obj: GridLike, tau_pred: np.ndarray) -> float:
    if getattr(obj, "y_coords", None) is None:
        return float("nan")

    tau_true_3d = reshape_tensor_field(obj, obj.tau)
    tau_pred_3d = reshape_tensor_field(obj, tau_pred)
    tau12_true = np.mean(tau_true_3d[..., 0, 1], axis=(0, 2))
    tau12_pred = np.mean(tau_pred_3d[..., 0, 1], axis=(0, 2))
    if np.std(tau12_true) < 1e-15 or np.std(tau12_pred) < 1e-15:
        return float("nan")
    return float(np.corrcoef(tau12_true, tau12_pred)[0, 1])


def wall_profile_rmse(obj: GridLike, tau_pred: np.ndarray) -> float:
    if getattr(obj, "y_coords", None) is None:
        return float("nan")

    tau_true_3d = reshape_tensor_field(obj, obj.tau)
    tau_pred_3d = reshape_tensor_field(obj, tau_pred)
    tau12_true = np.mean(tau_true_3d[..., 0, 1], axis=(0, 2))
    tau12_pred = np.mean(tau_pred_3d[..., 0, 1], axis=(0, 2))
    return float(np.sqrt(np.mean((tau12_pred - tau12_true) ** 2)))
