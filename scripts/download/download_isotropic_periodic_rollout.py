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
DT = 0.0002
NUM_FRAMES = 13
TIMEPOINTS = tuple(i * DT for i in range(NUM_FRAMES))
BATCH_SIZE = 32768
OUT_PATH = "jhtdb_u_tensor_64_periodic_rollout.h5"
MAX_RETRIES = 5


def sample_coords() -> np.ndarray:
    idx = np.arange(N, dtype=np.float64) * STRIDE
    return idx * DX


def sample_points(coords: np.ndarray) -> np.ndarray:
    z = np.repeat(coords, N * N)
    y = np.tile(np.repeat(coords, N), N)
    x = np.tile(coords, N * N)
    return np.column_stack([x, y, z]).astype(np.float32)


def fetch_velocity(dataset_obj, timepoint: float, points: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n_points = len(points)
    t0 = time.time()
    for start in range(0, n_points, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, n_points)
        batch = points[start:stop]
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
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
                break
            except Exception as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    raise
                wait_s = 2.0 * attempt
                print(
                    f"  retry {attempt}/{MAX_RETRIES} for t={timepoint:.4f} "
                    f"batch {start}:{stop} after {wait_s:.1f}s ({type(exc).__name__}: {exc})",
                    flush=True,
                )
                time.sleep(wait_s)
        print(
            f"  t={timepoint:.4f} points {start}:{stop} / {n_points} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    field = np.concatenate(chunks, axis=0)
    return field.reshape(N, N, N, 3)


def main() -> None:
    auth_token = get_jhtdb_token()
    coords = sample_coords()
    points = sample_points(coords)
    dataset_obj = turb_dataset(dataset_title=DATASET, output_path=".", auth_token=auth_token)

    print("=" * 72, flush=True)
    print("  Download periodic isotropic rollout sequence via getData", flush=True)
    print("=" * 72, flush=True)
    print(f"dataset={DATASET} stride={STRIDE} points={len(points)} frames={len(TIMEPOINTS)}", flush=True)
    print(f"token={mask_token(auth_token)}", flush=True)

    with h5py.File(OUT_PATH, "a") as f:
        if "x_coords" not in f:
            f.create_dataset("x_coords", data=coords)
        if "y_coords" not in f:
            f.create_dataset("y_coords", data=coords)
        if "z_coords" not in f:
            f.create_dataset("z_coords", data=coords)
        for idx, timepoint in enumerate(TIMEPOINTS):
            dataset_name = f"Velocity_t{idx:04d}"
            if dataset_name in f:
                print(f"{dataset_name} already exists, skipping", flush=True)
                continue
            t0 = time.time()
            field = fetch_velocity(dataset_obj, timepoint, points)
            f.create_dataset(dataset_name, data=field)
            print(
                f"{dataset_name} @ t={timepoint:.4f} shape={field.shape} "
                f"mean={float(field.mean()):.6e} std={float(field.std()):.6e} "
                f"dt={time.time() - t0:.1f}s",
                flush=True,
            )
        f.attrs["dt"] = DT
        f.attrs["timepoints"] = np.array(TIMEPOINTS, dtype=np.float64)

    print(f"Saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
