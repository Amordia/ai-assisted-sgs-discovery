#!/usr/bin/env python3
"""
Regression test for non-uniform wall-normal derivatives in oracle.py.
"""

from __future__ import annotations

import numpy as np

from sgs_discovery.oracle import JHTDBOracle


def main() -> None:
    z_coords = np.linspace(0.0, 2.0, 17)
    x_coords = np.linspace(-1.0, 1.5, 19)
    eta = np.linspace(-1.0, 1.0, 21)
    y_coords = np.tanh(1.4 * eta)

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    field = np.sin(1.3 * zz) + yy**3 - 0.4 * yy + 0.5 * xx**2

    oracle = JHTDBOracle.__new__(JHTDBOracle)
    oracle.h5_path = "<synthetic>"
    oracle.dx = float(x_coords[1] - x_coords[0])
    oracle.y_coords = y_coords
    oracle.x_coords = x_coords
    oracle.z_coords = z_coords
    oracle._axis_coords = None
    oracle._configure_spatial_coordinates(Z=len(z_coords), Y=len(y_coords), X=len(x_coords))

    grad_z, grad_y, grad_x = oracle._gradient3(field)
    grad_yy = oracle._gradient_axis(grad_y, axis=1)

    grad_z_true = 1.3 * np.cos(1.3 * zz)
    grad_y_true = 3.0 * yy**2 - 0.4
    grad_x_true = xx
    grad_yy_true = 6.0 * yy

    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    checks = {
        "dz": np.max(np.abs(grad_z[interior] - grad_z_true[interior])),
        "dy": np.max(np.abs(grad_y[interior] - grad_y_true[interior])),
        "dx": np.max(np.abs(grad_x[interior] - grad_x_true[interior])),
        "dyy": np.max(np.abs(grad_yy[interior] - grad_yy_true[interior])),
    }

    tolerances = {"dz": 3.0e-2, "dy": 3.5e-2, "dx": 1.0e-12, "dyy": 3.0e-1}
    failed = {
        name: (error, tolerances[name])
        for name, error in checks.items()
        if not np.isfinite(error) or error > tolerances[name]
    }

    for name, error in checks.items():
        print(f"{name}: max interior abs error = {error:.6e}")

    if failed:
        details = ", ".join(
            f"{name} error {error:.3e} > tol {tol:.3e}"
            for name, (error, tol) in failed.items()
        )
        raise SystemExit(f"non-uniform gradient regression failed: {details}")

    descending_oracle = JHTDBOracle.__new__(JHTDBOracle)
    descending_oracle.h5_path = "<synthetic-descending>"
    descending_oracle.dx = float(x_coords[1] - x_coords[0])
    descending_oracle.y_coords = y_coords[::-1]
    descending_oracle.x_coords = x_coords
    descending_oracle.z_coords = z_coords
    descending_oracle._axis_coords = None
    descending_oracle._configure_spatial_coordinates(Z=len(z_coords), Y=len(y_coords), X=len(x_coords))
    field_desc = field[:, ::-1, :]
    grad_y_desc = descending_oracle._gradient_axis(field_desc, axis=1)
    grad_y_desc_true = grad_y_true[:, ::-1, :]
    desc_error = np.max(np.abs(grad_y_desc[interior] - grad_y_desc_true[interior]))
    print(f"dy(desc): max interior abs error = {desc_error:.6e}")
    if not np.isfinite(desc_error) or desc_error > tolerances["dy"]:
        raise SystemExit(
            f"descending-coordinate regression failed: dy error {desc_error:.3e} > tol {tolerances['dy']:.3e}"
        )

    print("non-uniform gradient regression passed")


if __name__ == "__main__":
    main()
