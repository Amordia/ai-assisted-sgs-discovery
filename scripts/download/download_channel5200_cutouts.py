#!/usr/bin/env python3
"""
download_channel5200_cutouts.py
===============================
Download strided 64^3 cutouts from the JHTDB channel5200 dataset for
cross-Reynolds transfer tests.

This script intentionally bypasses ``givernylocal`` and calls the JHTDB
REST cutout endpoint directly. In this environment the direct endpoint is
materially more reliable for legacy datasets such as ``channel5200`` than
the library wrapper.

Outputs:
  - channel5200_u_tensor_64_ood.h5
  - channel5200_u_tensor_64_ood_shifted.h5
"""

from __future__ import annotations

import time

import h5py
import numpy as np
import requests
from sgs_discovery.jhtdb_config import get_jhtdb_token, mask_token


DATASET = "channel5200"
API_URL = "https://web.idies.jhu.edu/turbulence-svc/cutout/api/local"
N = 64
TX = (5, 6, 7)
STRIDE_X = 5
STRIDE_Y = 3
STRIDE_Z = 5
NY_FULL = 1536
X_CHUNK = 16
Z_CHUNK = 16
MAX_RETRIES = 5
TIMEOUT_S = 60
BASE_X_SPACING = 8.0 * np.pi / 10240.0
BASE_Z_SPACING = 3.0 * np.pi / 7680.0


def sampled_channel5200_y_coords(ny: int, start: int, stride: int, ny_full: int = NY_FULL) -> np.ndarray:
    j_full = np.arange(ny_full)
    y_full = np.cos(np.pi * j_full / (ny_full - 1))
    start_idx = start - 1
    sample_idx = start_idx + np.arange(ny) * stride
    return y_full[sample_idx]


def sampled_uniform_coords(n: int, start: int, stride: int, base_spacing: float) -> np.ndarray:
    start_idx = start - 1
    sample_idx = start_idx + np.arange(n, dtype=np.float64) * stride
    return sample_idx * base_spacing


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({
        "Connection": "close",
        "User-Agent": "ai-assisted-sgs-discovery/1.0",
    })
    return session


def fetch_velocity_block(
    session: requests.Session,
    t: int,
    x0: int,
    y0: int,
    z0: int,
    nx_out: int,
    nz_out: int,
) -> np.ndarray:
    x1 = x0 + (nx_out - 1) * STRIDE_X
    y1 = y0 + (N - 1) * STRIDE_Y
    z1 = z0 + (nz_out - 1) * STRIDE_Z
    params = {
        "token": get_jhtdb_token(),
        "function": "velocity",
        "dataset": DATASET,
        "xs": x0,
        "xe": x1,
        "ys": y0,
        "ye": y1,
        "zs": z0,
        "ze": z1,
        "ts": t,
        "te": t,
        "stridet": 1,
        "stridex": STRIDE_X,
        "stridey": STRIDE_Y,
        "stridez": STRIDE_Z,
        "filter_width": 1,
    }
    expected_shape = (nz_out, N, nx_out, 3)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(API_URL, params=params, timeout=TIMEOUT_S)
            response.raise_for_status()
            payload = response.json()
            data_var = next(iter(payload["data_vars"].values()))
            block = np.array(data_var["data"], dtype=np.float32)
            if block.shape != expected_shape:
                raise RuntimeError(
                    f"unexpected block shape {block.shape}, expected {expected_shape}"
                )
            return block
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait_s = 2.0 * attempt
            print(
                f"  retry {attempt}/{MAX_RETRIES} for block "
                f"x={x0}:{x1}, y={y0}:{y1}, z={z0}:{z1}, t={t} "
                f"after {wait_s:.1f}s ({type(exc).__name__}: {exc})"
            )
            time.sleep(wait_s)

    raise RuntimeError(
        f"failed to download block x={x0}:{x1}, y={y0}:{y1}, z={z0}:{z1}, t={t}: {last_error}"
    )


def fetch_velocity(session: requests.Session, t: int, x0: int, y0: int, z0: int) -> np.ndarray:
    z_blocks = []
    for z_offset in range(0, N, Z_CHUNK):
        nz_out = min(Z_CHUNK, N - z_offset)
        z_start = z0 + z_offset * STRIDE_Z
        x_blocks = []
        for x_offset in range(0, N, X_CHUNK):
            nx_out = min(X_CHUNK, N - x_offset)
            x_start = x0 + x_offset * STRIDE_X
            block = fetch_velocity_block(
                session,
                t=t,
                x0=x_start,
                y0=y0,
                z0=z_start,
                nx_out=nx_out,
                nz_out=nz_out,
            )
            x_blocks.append(block)
        z_blocks.append(np.concatenate(x_blocks, axis=2))
    return np.concatenate(z_blocks, axis=0)


def save_cube(
    path: str,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("Velocity_t1", data=u1)
        f.create_dataset("Velocity_t2", data=u2)
        f.create_dataset("Velocity_t3", data=u3)
        f.create_dataset("y_coords", data=y_coords)
        f.create_dataset("x_coords", data=x_coords)
        f.create_dataset("z_coords", data=z_coords)


def download_case(save_path: str, x0: int, y0: int, z0: int) -> None:
    session = make_session()
    print("=" * 72)
    print(f"[DOWNLOAD] {DATASET} -> {save_path}")
    print(f"time indices: {TX}")
    print(
        f"ranges: x={x0}:{x0 + (N - 1) * STRIDE_X}, "
        f"y={y0}:{y0 + (N - 1) * STRIDE_Y}, "
        f"z={z0}:{z0 + (N - 1) * STRIDE_Z}"
    )
    print(f"strides: x={STRIDE_X}, y={STRIDE_Y}, z={STRIDE_Z}")

    fields = []
    for t in TX:
        t0 = time.time()
        arr = fetch_velocity(session, t=t, x0=x0, y0=y0, z0=z0)
        print(
            f"  t={t} shape={arr.shape} mean={float(arr.mean()):.6e} "
            f"std={float(arr.std()):.6e} dt={time.time() - t0:.1f}s"
        )
        fields.append(arr)

    y_coords = sampled_channel5200_y_coords(N, start=y0, stride=STRIDE_Y)
    x_coords = sampled_uniform_coords(N, start=x0, stride=STRIDE_X, base_spacing=BASE_X_SPACING)
    z_coords = sampled_uniform_coords(N, start=z0, stride=STRIDE_Z, base_spacing=BASE_Z_SPACING)
    save_cube(save_path, fields[0], fields[1], fields[2], y_coords, x_coords, z_coords)
    print(f"saved {save_path}")


def main() -> None:
    print(f"Using JHTDB token {mask_token(get_jhtdb_token())}")
    download_case(
        save_path="channel5200_u_tensor_64_ood.h5",
        x0=1,
        y0=1,
        z0=1,
    )
    download_case(
        save_path="channel5200_u_tensor_64_ood_shifted.h5",
        x0=513,
        y0=1,
        z0=513,
    )


if __name__ == "__main__":
    main()
