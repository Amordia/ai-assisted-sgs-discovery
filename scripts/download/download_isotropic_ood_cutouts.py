#!/usr/bin/env python3
"""
download_isotropic_ood_cutouts.py
=================================
Download external isotropic JHTDB cutouts for out-of-database transfer tests.

Outputs:
  - jhtdb_u_tensor_64_coarse_ood.h5
  - jhtdb_u_tensor_64_4096_ood.h5
"""

from __future__ import annotations

import time

import h5py
import numpy as np

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout
from sgs_discovery.jhtdb_config import get_jhtdb_token


N = 64


def fetch_velocity(
    dataset_title: str,
    t: int,
    x0: int,
    y0: int,
    z0: int,
) -> np.ndarray:
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
    ], dtype=np.int64)
    strides = np.array([1, 1, 1, 1], dtype=np.int64)
    result = getCutout(dataset_obj, "velocity", axes_ranges, strides)
    return list(result.values())[0]


def save_cube(path: str, u1: np.ndarray, u2: np.ndarray, u3: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("Velocity_t1", data=u1)
        f.create_dataset("Velocity_t2", data=u2)
        f.create_dataset("Velocity_t3", data=u3)


def save_single_snapshot(path: str, u: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("Velocity", data=u)


def download_case(
    dataset_title: str,
    save_path: str,
    time_indices: tuple[int, int, int],
    x0: int,
    y0: int,
    z0: int,
) -> None:
    print("=" * 72)
    print(f"[DOWNLOAD] {dataset_title} -> {save_path}")
    print(f"time indices: {time_indices}")
    print(f"ranges: x={x0}:{x0+N-1}, y={y0}:{y0+N-1}, z={z0}:{z0+N-1}")
    fields = []
    for t in time_indices:
        t0 = time.time()
        arr = fetch_velocity(dataset_title, t=t, x0=x0, y0=y0, z0=z0)
        print(
            f"  t={t} shape={arr.shape} mean={float(arr.mean()):.6e} "
            f"std={float(arr.std()):.6e} dt={time.time()-t0:.1f}s"
        )
        fields.append(arr)
    save_cube(save_path, fields[0], fields[1], fields[2])
    print(f"saved {save_path}")


def download_single_snapshot(
    dataset_title: str,
    save_path: str,
    t: int,
    x0: int,
    y0: int,
    z0: int,
) -> None:
    print("=" * 72)
    print(f"[DOWNLOAD] {dataset_title} -> {save_path}")
    print(f"time index: {t}")
    print(f"ranges: x={x0}:{x0+N-1}, y={y0}:{y0+N-1}, z={z0}:{z0+N-1}")
    t0 = time.time()
    arr = fetch_velocity(dataset_title, t=t, x0=x0, y0=y0, z0=z0)
    print(
        f"  t={t} shape={arr.shape} mean={float(arr.mean()):.6e} "
        f"std={float(arr.std()):.6e} dt={time.time()-t0:.1f}s"
    )
    save_single_snapshot(save_path, arr)
    print(f"saved {save_path}")


def main() -> None:
    download_case(
        dataset_title="isotropic1024coarse",
        save_path="jhtdb_u_tensor_64_coarse_ood.h5",
        time_indices=(100, 101, 102),
        x0=129,
        y0=129,
        z0=129,
    )
    download_single_snapshot(
        dataset_title="isotropic4096",
        save_path="jhtdb_u_tensor_64_4096_ood.h5",
        t=1,
        x0=513,
        y0=513,
        z0=513,
    )


if __name__ == "__main__":
    main()
