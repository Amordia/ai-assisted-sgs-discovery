#!/usr/bin/env python3
from __future__ import annotations

import time

import h5py
import numpy as np

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData
from sgs_discovery.jhtdb_config import get_jhtdb_token, mask_token

DATASET = "isotropic1024fine"
N = 64
STRIDE = 16
DX = 2.0 * np.pi / 1024.0
TIMEPOINTS = (0.0, 0.0002, 0.0004)
BATCH_SIZE = 32768
OUT_PATH = "jhtdb_u_tensor_64_periodic.h5"


def sample_coords() -> np.ndarray:
    idx = np.arange(N, dtype=np.float64) * STRIDE
    return idx * DX


def sample_points(coords: np.ndarray) -> np.ndarray:
    z = np.repeat(coords, N * N)
    y = np.tile(np.repeat(coords, N), N)
    x = np.tile(coords, N * N)
    return np.column_stack([x, y, z]).astype(np.float32)


def download_timestep(dataset_obj, timepoint: float, points: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_points = len(points)
    t0 = time.time()
    for start in range(0, n_points, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n_points)
        batch = points[start:stop]
        result = getData(
            dataset_obj,
            "velocity",
            timepoint,
            "none",
            "none",
            "field",
            batch,
            verbose=False,
        )
        values = result[0][["ux", "uy", "uz"]].to_numpy(dtype=np.float32)
        chunks.append(values)
        print(
            f"  t={timepoint:.4f} points {start}:{stop} / {n_points} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    flat = np.concatenate(chunks, axis=0)
    return flat.reshape(N, N, N, 3)


def main() -> None:
    auth_token = get_jhtdb_token()
    coords = sample_coords()
    points = sample_points(coords)
    dataset_obj = turb_dataset(dataset_title=DATASET, output_path=".", auth_token=auth_token)

    print("=" * 72, flush=True)
    print("  Download periodic isotropic 64^3 coarse box via getData", flush=True)
    print("=" * 72, flush=True)
    print(f"dataset={DATASET} stride={STRIDE} points={len(points)}", flush=True)
    print(f"token={mask_token(auth_token)}", flush=True)

    fields = []
    for timepoint in TIMEPOINTS:
        t0 = time.time()
        field = download_timestep(dataset_obj, timepoint, points)
        print(
            f"t={timepoint:.4f} shape={field.shape} mean={float(field.mean()):.6e} "
            f"std={float(field.std()):.6e} dt={time.time() - t0:.1f}s",
            flush=True,
        )
        fields.append(field)

    with h5py.File(OUT_PATH, "w") as f:
        f.create_dataset("Velocity_t1", data=fields[0])
        f.create_dataset("Velocity_t2", data=fields[1])
        f.create_dataset("Velocity_t3", data=fields[2])
        f.create_dataset("x_coords", data=coords)
        f.create_dataset("y_coords", data=coords)
        f.create_dataset("z_coords", data=coords)

    print(f"Saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
