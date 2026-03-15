#!/usr/bin/env python3
"""
download_mixing_cutout.py
=========================
Download a 64^3 cutout from the JHTDB homogeneous buoyancy-driven mixing
dataset for out-of-training-distribution transfer tests.

Output:
  - mixing_u_tensor_64_ood.h5
"""

from __future__ import annotations

import time

import h5py
import numpy as np

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout
from sgs_discovery.jhtdb_config import get_jhtdb_token, mask_token


DATASET = "mixing"
N = 64
T1, T2, T3 = 100, 101, 102


def fetch_velocity(dataset_obj: turb_dataset, t: int, x0: int, y0: int, z0: int) -> np.ndarray:
    axes_ranges = np.array([
        [x0, x0 + N - 1],
        [y0, y0 + N - 1],
        [z0, z0 + N - 1],
        [t, t],
    ], dtype=np.int64)
    strides = np.array([1, 1, 1, 1], dtype=np.int64)
    result = getCutout(dataset_obj, "velocity", axes_ranges, strides)
    return list(result.values())[0]


def main() -> None:
    auth_token = get_jhtdb_token()
    dataset_obj = turb_dataset(
        dataset_title=DATASET,
        output_path=".",
        auth_token=auth_token,
    )
    x0 = y0 = z0 = 129

    print("=" * 72)
    print(f"[DOWNLOAD] {DATASET} -> mixing_u_tensor_64_ood.h5")
    print(f"token={mask_token(auth_token)}")
    print(f"time indices: {T1}, {T2}, {T3}")
    print(f"ranges: x={x0}:{x0+N-1}, y={y0}:{y0+N-1}, z={z0}:{z0+N-1}")

    fields = []
    for t in [T1, T2, T3]:
        t0 = time.time()
        arr = fetch_velocity(dataset_obj, t=t, x0=x0, y0=y0, z0=z0)
        print(
            f"  t={t} shape={arr.shape} mean={float(arr.mean()):.6e} "
            f"std={float(arr.std()):.6e} dt={time.time()-t0:.1f}s"
        )
        fields.append(arr)

    with h5py.File("mixing_u_tensor_64_ood.h5", "w") as f:
        f.create_dataset("Velocity_t1", data=fields[0])
        f.create_dataset("Velocity_t2", data=fields[1])
        f.create_dataset("Velocity_t3", data=fields[2])

    print("saved mixing_u_tensor_64_ood.h5")


if __name__ == "__main__":
    main()
