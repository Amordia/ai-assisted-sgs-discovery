"""
download_channel.py — Download multi-timestep channel flow from JHTDB
=====================================================================
Uses the personal API token (2M-point limit) to download a 64^3 velocity
field at t=1, t=2, t=3 in 3 API calls.

The channel dataset uses Chebyshev collocation in the wall-normal (y) direction.
We compute and save the physical y-coordinates for each grid point.

This multi-timestep sequence enables central time-differencing for the
calculation of Objective Rates (e.g. the Jaumann derivative).

Output: channel_u_tensor_64.h5 with:
  - 'Velocity_t1': shape (64, 64, 64, 3)
  - 'Velocity_t2': shape (64, 64, 64, 3)
  - 'Velocity_t3': shape (64, 64, 64, 3)
  - 'y_coords':    shape (64,) — physical wall-normal coordinates
"""

import numpy as np
import h5py
import time

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout
from sgs_discovery.jhtdb_config import get_jhtdb_token, mask_token

# ── Configuration ──
DATASET = "channel"
N = 64  # 64^3 volume
CHANNEL_X_SPACING = 8.0 * np.pi / 2048.0
CHANNEL_Z_SPACING = 3.0 * np.pi / 1536.0


def download_timestep(dataset_obj, t: int) -> np.ndarray:
    """Download a full 64^3 velocity field at timestep t in a single API call."""
    axes_ranges = np.array([
        [1, N],   # X
        [1, N],   # Y
        [1, N],   # Z
        [t, t],   # T
    ])
    strides = np.array([1, 1, 1, 1])

    result = getCutout(dataset_obj, "velocity", axes_ranges, strides)

    key_name = f"velocity_{str(t).zfill(4)}"
    if key_name in result:
        return result[key_name]
    else:
        return list(result.values())[0]


def compute_channel_y_coords(ny: int, ny_full: int = 512) -> np.ndarray:
    """
    Compute the physical y-coordinates for the first `ny` grid points
    of the JHTDB channel flow dataset.

    The channel dataset uses Chebyshev-Gauss-Lobatto collocation points
    in the wall-normal direction with `ny_full` total points spanning [-1, 1].
    We extract the first `ny` points.

    Parameters
    ----------
    ny : int
        Number of y-grid points we downloaded (64 in our case).
    ny_full : int
        Total number of y-grid points in the full DNS (512 for JHTDB channel).

    Returns
    -------
    np.ndarray, shape (ny,)
        Physical y-coordinates in [-1, 1] for the downloaded subset.
    """
    # Chebyshev-Gauss-Lobatto points: y_j = cos(pi * j / (N-1)), j=0..N-1
    # These go from +1 (near top wall) to -1 (near bottom wall)
    j_full = np.arange(ny_full)
    y_full = np.cos(np.pi * j_full / (ny_full - 1))
    # Take the first ny points (corresponding to indices 0..ny-1)
    return y_full[:ny]


def compute_uniform_coords(n: int, spacing: float, start_index: int = 1) -> np.ndarray:
    start = start_index - 1
    return (start + np.arange(n, dtype=np.float64)) * spacing


def main():
    auth_token = get_jhtdb_token()
    print("=" * 60)
    print("  JHTDB Channel Flow — Multi-Timestep Download")
    print("=" * 60)
    print(f"  Dataset:  {DATASET}")
    print(f"  Volume:   {N}×{N}×{N} = {N**3:,} points")
    print(f"  Token:    {mask_token(auth_token)}")
    print()

    dataset_obj = turb_dataset(
        dataset_title=DATASET,
        output_path='.',
        auth_token=auth_token,
    )

    # ── Download t=1 (previous timestep → S_prev) ──
    print("📥 Downloading t=1 (for S_prev temporal memory)...")
    t0 = time.time()
    u_t1 = download_timestep(dataset_obj, t=1)
    print(f"   ✅ t=1 done in {time.time() - t0:.1f}s — shape: {u_t1.shape}")

    # ── Download t=2 (current timestep → S, Omega, L, tau) ──
    print("📥 Downloading t=2 (current features + ground truth)...")
    t0 = time.time()
    u_t2 = download_timestep(dataset_obj, t=2)
    print(f"   ✅ t=2 done in {time.time() - t0:.1f}s — shape: {u_t2.shape}")

    # ── Download t=3 (future timestep → forward time finite difference) ──
    print("📥 Downloading t=3 (future timestep)...")
    t0 = time.time()
    u_t3 = download_timestep(dataset_obj, t=3)
    print(f"   ✅ t=3 done in {time.time() - t0:.1f}s — shape: {u_t3.shape}")

    # ── Compute Wall-Normal Heights ──
    y_coords = compute_channel_y_coords(ny=N)
    x_coords = compute_uniform_coords(n=N, spacing=CHANNEL_X_SPACING, start_index=1)
    z_coords = compute_uniform_coords(n=N, spacing=CHANNEL_Z_SPACING, start_index=1)

    # ── Save to HDF5 ──
    save_file = "channel_u_tensor_64.h5"
    print(f"\n💾 Saving multi-timestep data to: {save_file}")
    with h5py.File(save_file, 'w') as f:
        f.create_dataset('Velocity_t1', data=u_t1)
        f.create_dataset('Velocity_t2', data=u_t2)
        f.create_dataset('Velocity_t3', data=u_t3)
        f.create_dataset('y_coords', data=y_coords)
        f.create_dataset('x_coords', data=x_coords)
        f.create_dataset('z_coords', data=z_coords)

    print(f"🎉 Done! Channel multi-timestep data saved ({save_file})")


if __name__ == "__main__":
    main()
