#!/usr/bin/env python3
"""
download_additional_cutouts.py
==============================
Download alternative 64^3 JHTDB cutouts for external spatial generalization.

This script preserves the original training cutouts and writes new files:
  - jhtdb_u_tensor_64_shifted.h5
  - channel_u_tensor_64_shifted.h5
"""

from __future__ import annotations

import time

import h5py
import numpy as np

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout
from sgs_discovery.jhtdb_config import get_jhtdb_token


N = 64
CHANNEL_X_SPACING = 8.0 * np.pi / 2048.0
CHANNEL_Z_SPACING = 3.0 * np.pi / 1536.0


def fetch_velocity(dataset_title: str, t: int, x0: int, y0: int, z0: int) -> np.ndarray:
    dataset_obj = turb_dataset(
        dataset_title=dataset_title,
        output_path=".",
        auth_token=get_jhtdb_token(),
    )
    axes_ranges = np.array([
        [x0, x0 + N - 1],
        [y0, y0 + N - 1],
        [z0, z0 + N - 1],
        [t, t],
    ])
    strides = np.array([1, 1, 1, 1])
    result = getCutout(dataset_obj, "velocity", axes_ranges, strides)
    return list(result.values())[0]


def channel_y_coords(ny: int, start: int, ny_full: int = 512) -> np.ndarray:
    j_full = np.arange(ny_full)
    y_full = np.cos(np.pi * j_full / (ny_full - 1))
    s0 = start - 1
    return y_full[s0:s0 + ny]


def uniform_coords(n: int, spacing: float, start: int, stride: int = 1) -> np.ndarray:
    start0 = start - 1
    return (start0 + np.arange(n, dtype=np.float64) * stride) * spacing


def save_cube(
    path: str,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    y_coords: np.ndarray | None = None,
    x_coords: np.ndarray | None = None,
    z_coords: np.ndarray | None = None,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("Velocity_t1", data=u1)
        f.create_dataset("Velocity_t2", data=u2)
        f.create_dataset("Velocity_t3", data=u3)
        if y_coords is not None:
            f.create_dataset("y_coords", data=y_coords)
        if x_coords is not None:
            f.create_dataset("x_coords", data=x_coords)
        if z_coords is not None:
            f.create_dataset("z_coords", data=z_coords)


def download_case(dataset_title: str, save_path: str, x0: int, y0: int, z0: int, include_y: bool = False) -> None:
    print("=" * 72)
    print(f"[DOWNLOAD] {dataset_title} -> {save_path}")
    print(f"ranges: x={x0}:{x0+N-1}, y={y0}:{y0+N-1}, z={z0}:{z0+N-1}")
    fields = []
    for t in [1, 2, 3]:
        t0 = time.time()
        arr = fetch_velocity(dataset_title, t=t, x0=x0, y0=y0, z0=z0)
        print(f"  t={t} shape={arr.shape} mean={float(arr.mean()):.6e} std={float(arr.std()):.6e} dt={time.time()-t0:.1f}s")
        fields.append(arr)
    y_coords = channel_y_coords(N, start=y0) if include_y else None
    x_coords = uniform_coords(N, CHANNEL_X_SPACING, start=x0) if include_y else None
    z_coords = uniform_coords(N, CHANNEL_Z_SPACING, start=z0) if include_y else None
    save_cube(
        save_path,
        fields[0],
        fields[1],
        fields[2],
        y_coords=y_coords,
        x_coords=x_coords,
        z_coords=z_coords,
    )
    print(f"saved {save_path}")


def main() -> None:
    # Use shifted domains that do not overlap the original [1, 64]^3 window in x and z.
    download_case(
        dataset_title="isotropic1024fine",
        save_path="jhtdb_u_tensor_64_shifted.h5",
        x0=129,
        y0=129,
        z0=129,
        include_y=False,
    )
    # Keep the same near-wall y band for channel flow, but shift streamwise/spanwise windows.
    download_case(
        dataset_title="channel",
        save_path="channel_u_tensor_64_shifted.h5",
        x0=129,
        y0=1,
        z0=129,
        include_y=True,
    )


if __name__ == "__main__":
    main()
