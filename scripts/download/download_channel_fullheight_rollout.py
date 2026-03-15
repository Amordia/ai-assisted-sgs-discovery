#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_gizmos.variable_dy_grid_ys import get_channel_ys
from givernylocal.turbulence_toolkit import getData
from sgs_discovery.jhtdb_config import get_jhtdb_token, mask_token


DATASET = "channel"
X_FULL = 2048
Y_FULL = 512
Z_FULL = 1536
X_SPACING = 8.0 * np.pi / X_FULL
Z_SPACING = 3.0 * np.pi / Z_FULL
DEFAULT_TIME_DT = 25.9935 / 3999.0
DEFAULT_START_TIME = 2.0
DEFAULT_NUM_FRAMES = 13
DEFAULT_BATCH_SIZE = 8192
DEFAULT_OUT_PATH = "channel_fullheight_u_tensor_rollout.h5"
MAX_RETRIES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a full-height coarse channel rollout sequence from JHTDB.",
    )
    parser.add_argument("--nx", type=int, default=64, help="Number of streamwise samples.")
    parser.add_argument("--ny", type=int, default=64, help="Number of wall-normal samples.")
    parser.add_argument("--nz", type=int, default=64, help="Number of spanwise samples.")
    parser.add_argument(
        "--start-time",
        type=float,
        default=DEFAULT_START_TIME,
        help="Starting physical time for the channel rollout.",
    )
    parser.add_argument(
        "--time-dt",
        type=float,
        default=DEFAULT_TIME_DT,
        help="Time spacing between frames.",
    )
    parser.add_argument(
        "--temporal-method",
        default="none",
        help="Temporal interpolation method passed to getData (e.g. none, pchip).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of consecutive snapshots to download.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of points per REST batch.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUT_PATH),
        help="Output HDF5 path.",
    )
    return parser.parse_args()


def evenly_spaced_periodic_indices(full_size: int, n: int) -> np.ndarray:
    if full_size % n != 0:
        raise ValueError(f"full_size={full_size} must be divisible by n={n}")
    stride = full_size // n
    return np.arange(n, dtype=np.int64) * stride


def evenly_spaced_wall_indices(full_size: int, n: int) -> np.ndarray:
    idx = np.rint(np.linspace(0, full_size - 1, n)).astype(np.int64)
    idx = np.unique(idx)
    if len(idx) != n:
        raise ValueError(f"unable to create {n} unique wall-normal indices from {full_size}")
    return idx


def sample_geometry(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_indices = evenly_spaced_periodic_indices(X_FULL, nx)
    z_indices = evenly_spaced_periodic_indices(Z_FULL, nz)
    y_indices = evenly_spaced_wall_indices(Y_FULL, ny)

    y_full = np.asarray(get_channel_ys(), dtype=np.float64)
    if y_full.shape != (Y_FULL,):
        raise RuntimeError(f"unexpected full channel y-grid shape: {y_full.shape}")

    x_coords = x_indices.astype(np.float64) * X_SPACING
    z_coords = z_indices.astype(np.float64) * Z_SPACING
    y_coords = y_full[y_indices]
    return x_indices, y_indices, z_indices, x_coords, y_coords, z_coords


def sample_points(x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray) -> np.ndarray:
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def fetch_timeseries(
    dataset_obj,
    points: np.ndarray,
    start_time: float,
    time_dt: float,
    num_frames: int,
    batch_size: int,
    temporal_method: str,
) -> list[np.ndarray]:
    end_time = start_time + (num_frames - 1) * time_dt
    n_points = len(points)
    per_frame = [np.empty((n_points, 3), dtype=np.float32) for _ in range(num_frames)]
    t0 = time.time()

    for batch_start in range(0, n_points, batch_size):
        batch_stop = min(batch_start + batch_size, n_points)
        batch = points[batch_start:batch_stop]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if num_frames == 1:
                    results = getData(
                        dataset_obj,
                        "velocity",
                        float(start_time),
                        temporal_method,
                        "none",
                        "field",
                        batch,
                        verbose=False,
                    )
                else:
                    results = getData(
                        dataset_obj,
                        "velocity",
                        float(start_time),
                        temporal_method,
                        "none",
                        "field",
                        batch,
                        option=[float(end_time), float(time_dt)],
                        verbose=False,
                    )
                if len(results) != num_frames:
                    raise RuntimeError(f"expected {num_frames} frames, got {len(results)}")
                for frame_idx, df in enumerate(results):
                    per_frame[frame_idx][batch_start:batch_stop] = df[["ux", "uy", "uz"]].to_numpy(dtype=np.float32)
                break
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise
                wait_s = 2.0 * attempt
                print(
                    f"retry {attempt}/{MAX_RETRIES} batch {batch_start}:{batch_stop} "
                    f"after {wait_s:.1f}s ({type(exc).__name__}: {exc})",
                    flush=True,
                )
                time.sleep(wait_s)
        elapsed = time.time() - t0
        print(
            f"batch {batch_start}:{batch_stop} / {n_points} "
            f"frames={num_frames} elapsed={elapsed:.1f}s",
            flush=True,
        )

    return per_frame


def main() -> None:
    args = parse_args()
    auth_token = get_jhtdb_token()
    if args.num_frames < 1:
        raise ValueError("--num-frames must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    x_indices, y_indices, z_indices, x_coords, y_coords, z_coords = sample_geometry(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
    )
    points = sample_points(x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)
    n_points = len(points)
    total_queries = n_points * args.num_frames
    timepoints = args.start_time + np.arange(args.num_frames, dtype=np.float64) * args.time_dt
    time_indices = np.rint(timepoints / DEFAULT_TIME_DT).astype(np.int64)

    print("=" * 72, flush=True)
    print("  Download full-height channel rollout sequence via getData", flush=True)
    print("=" * 72, flush=True)
    print(
        f"grid=({args.nz}, {args.ny}, {args.nx}) points={n_points:,} "
        f"frames={args.num_frames} point*time={total_queries:,}",
        flush=True,
    )
    print(
        f"time range={timepoints[0]:.6f}..{timepoints[-1]:.6f} "
        f"physical dt={args.time_dt:.8f} tint={args.temporal_method}",
        flush=True,
    )
    print(
        f"x stride={X_FULL // args.nx} z stride={Z_FULL // args.nz} "
        f"y indices sample={y_indices[:4].tolist()}...{y_indices[-4:].tolist()}",
        flush=True,
    )
    print(f"token={mask_token(auth_token)}", flush=True)

    dataset_obj = turb_dataset(dataset_title=DATASET, output_path=".", auth_token=auth_token)
    frames = fetch_timeseries(
        dataset_obj=dataset_obj,
        points=points,
        start_time=args.start_time,
        time_dt=args.time_dt,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        temporal_method=args.temporal_method,
    )

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for frame_idx, frame in enumerate(frames):
            volume = frame.reshape(args.nz, args.ny, args.nx, 3)
            dataset_name = f"Velocity_t{frame_idx:04d}"
            f.create_dataset(dataset_name, data=volume)
        f.create_dataset("x_coords", data=x_coords)
        f.create_dataset("y_coords", data=y_coords)
        f.create_dataset("z_coords", data=z_coords)
        f.create_dataset("x_indices", data=x_indices)
        f.create_dataset("y_indices", data=y_indices)
        f.create_dataset("z_indices", data=z_indices)
        f.attrs["dataset"] = DATASET
        f.attrs["dt"] = args.time_dt
        f.attrs["timepoints"] = timepoints
        f.attrs["time_indices"] = time_indices
        f.attrs["temporal_method"] = args.temporal_method

    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
