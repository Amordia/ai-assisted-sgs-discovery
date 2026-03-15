"""
download_jhtdb.py — Download multi-timestep isotropic turbulence from JHTDB
===========================================================================
Uses the personal API token (2M-point limit) to download a 64^3 velocity
field at t=1, t=2, t=3 in 3 API calls.

This multi-timestep sequence enables central time-differencing for the
calculation of Objective Rates (e.g. the Jaumann derivative).

Output: jhtdb_u_tensor_64.h5 with datasets 'Velocity_t1', 'Velocity_t2', 'Velocity_t3'
"""

import numpy as np
import h5py
import time

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout
from sgs_discovery.jhtdb_config import get_jhtdb_token, mask_token

# ── Configuration ──
DATASET = "isotropic1024fine"
N = 64  # 64^3 = 262,144 points — well within 2M limit


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

    # Extract velocity array from result dict
    key_name = f"velocity_{str(t).zfill(4)}"
    if key_name in result:
        return result[key_name]
    else:
        return list(result.values())[0]


def main():
    auth_token = get_jhtdb_token()
    print("=" * 60)
    print("  JHTDB Isotropic Turbulence — Multi-Timestep Download")
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

    # ── Save to HDF5 ──
    save_file = "jhtdb_u_tensor_64.h5"
    print(f"\n💾 Saving multi-timestep data to: {save_file}")
    with h5py.File(save_file, 'w') as f:
        f.create_dataset('Velocity_t1', data=u_t1)
        f.create_dataset('Velocity_t2', data=u_t2)
        f.create_dataset('Velocity_t3', data=u_t3)

    print(f"🎉 Done! Isotropic multi-timestep data saved ({save_file})")


if __name__ == "__main__":
    main()
