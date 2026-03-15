"""
oracle.py — Data Oracle for Neuro-Symbolic-SGS
================================================
Provides flow-field tensor data from JHTDB DNS databases for
automated SGS closure discovery.

Tensors (from timestep t=2)
-------
S_ij      : Strain-rate tensor          (symmetric,      3x3)
Omega_ij  : Rotation-rate tensor        (anti-symmetric, 3x3)
L_ij      : Leonard stress tensor       (symmetric,      3x3)
S_d_ij    : WALE tensor                 (symmetric,      3x3)
tau_ij    : SGS stress tensor (target)  (symmetric,      3x3)

Scalar
------
Delta     : Effective physical filter width (scalar)
var_tau   : Local variance of tau_ij    (N,) -> Local NMSE
"""

from __future__ import annotations

import os
import numpy as np
from numpy.typing import NDArray
import h5py
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatArray = NDArray[np.float64]

_DEFAULT_UNIFORM_DX = 2.0 * np.pi / 1024.0
_CHANNEL_X_SPACING = 8.0 * np.pi / 2048.0
_CHANNEL_Z_SPACING = 3.0 * np.pi / 1536.0
_CACHE_VERSION = "v10"


class JHTDBOracle:
    """
    Oracle that loads real DNS velocity data from JHTDB cutouts
    and extracts physics features via 3D spatial filtering.

    Features: S, Omega, L, S_d (WALE tensor).
    Target: tau_exact.
    Normalization: var_tau (local Y-plane variance for channel, global for ISO).
    """

    def __init__(
        self,
        h5_path: str,
        filter_width: float = 1.0,
        dx: float = _DEFAULT_UNIFORM_DX,
        boundary_mode: str = 'wrap',
    ) -> None:
        self.h5_path = h5_path
        self.filter_width = filter_width
        self.dx = dx
        self.boundary_mode = boundary_mode
        self.x_coords: FloatArray | None = None
        self.z_coords: FloatArray | None = None
        self._axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray] | None = None
        self.grid_shape: tuple[int, int, int] | None = None

        # Tensor attributes — shape (n_samples, 3, 3)
        self.S: FloatArray
        self.Omega: FloatArray
        self.tau: FloatArray
        self.L: FloatArray
        self.S_d: FloatArray
        self.Delta: FloatArray
        self.omega_vec: FloatArray  # Vorticity vector (n_samples, 3)
        self.W_vec: FloatArray      # Vortex stretching vector (n_samples, 3)
        self.h_scalar: FloatArray   # Helicity pseudo-scalar (n_samples, 1)
        self.S_jaumann: FloatArray  # Jaumann (objective) derivative of S (n_samples, 3, 3)
        self.Lap_S: FloatArray      # Laplacian of Strain tensor (n_samples, 3, 3)
        
        # Local NMSE array — shape (n_samples,)
        self.var_tau: FloatArray
        self.var_pi: float
        self.mean_pi: float

        self.y_coords: FloatArray | None = None  # only for channel

        # Try to load from cache first.
        cache_name = os.path.basename(self.h5_path).replace('.h5', '')
        self.cache_path = (
            f"{cache_name}_processed_{_CACHE_VERSION}_sigma{self.filter_width}_{self.boundary_mode}.h5"
        )

        if os.path.exists(self.cache_path):
            print(f"⚡ Loading processed tensors from cache: {self.cache_path}")
            with h5py.File(self.cache_path, 'r') as f:
                self.S = np.array(f['S'])
                self.Omega = np.array(f['Omega'])
                self.tau = np.array(f['tau'])
                self.L = np.array(f['L'])
                self.S_d = np.array(f['S_d'])
                self.Delta = np.array(f['Delta'])
                self.omega_vec = np.array(f['omega_vec'])
                self.W_vec = np.array(f['W_vec'])
                self.h_scalar = np.array(f['h_scalar'])
                self.S_jaumann = np.array(f['S_jaumann'])
                self.Lap_S = np.array(f['Lap_S'])
                self.var_tau = np.array(f['var_tau'])
                self.var_pi = float(f['var_pi'][()])
                self.mean_pi = float(f['mean_pi'][()])
                if 'y_coords' in f:
                    self.y_coords = np.array(f['y_coords'])
                if 'x_coords' in f:
                    self.x_coords = np.array(f['x_coords'])
                if 'z_coords' in f:
                    self.z_coords = np.array(f['z_coords'])
        else:
            self._process_data()

        self._finalize_grid_metadata()

        self.n_samples = self.S.shape[0]

        mean_var = float(np.mean(self.var_tau))
        print(f"  mean(var_tau_local) = {mean_var:.4e}  |  mean(Pi) = {self.mean_pi:.4e}")

    def _process_data(self) -> None:
        """Process raw velocity data into physics tensors."""
        with h5py.File(self.h5_path, 'r') as f:
            # We now require multi-timestep velocity fields: t1, t2, t3
            if 'Velocity_t2' in f:
                u_t1_raw = np.array(f['Velocity_t1'])
                u_t2_raw = np.array(f['Velocity_t2'])
                u_t3_raw = np.array(f['Velocity_t3'])
                dt = 0.002  # JHTDB standard time step between stored frames
                is_multi_time = True
            else:
                u_t2_raw = np.array(f['Velocity'])
                is_multi_time = False

            # Load y_coords if present (channel dataset)
            if 'y_coords' in f:
                self.y_coords = np.array(f['y_coords'])
            if 'x_coords' in f:
                self.x_coords = np.array(f['x_coords'])
            if 'z_coords' in f:
                self.z_coords = np.array(f['z_coords'])

        Z, Y, X, _ = u_t2_raw.shape
        self._configure_spatial_coordinates(Z=Z, Y=Y, X=X)
        self.grid_shape = (Z, Y, X)
        print(f"  Processing {Z}×{Y}×{X} volume (boundary={self.boundary_mode})...")

        # 1. Spatial Filtering (Only filter t2 for the core variables, filter t1/t3 for unsteadiness)
        u_bar = np.zeros_like(u_t2_raw)
        u_bar_t1 = np.zeros_like(u_t2_raw) if is_multi_time else None
        u_bar_t3 = np.zeros_like(u_t2_raw) if is_multi_time else None

        for i in range(3):
            u_bar[..., i] = gaussian_filter(
                u_t2_raw[..., i], sigma=self.filter_width, mode=self.boundary_mode
            )
            if is_multi_time:
                u_bar_t1[..., i] = gaussian_filter(
                    u_t1_raw[..., i], sigma=self.filter_width, mode=self.boundary_mode
                )
                u_bar_t3[..., i] = gaussian_filter(
                    u_t3_raw[..., i], sigma=self.filter_width, mode=self.boundary_mode
                )

        # 2. Ground Truth tau & Leonard Stress L (Computed at t=2)
        tau_exact = np.zeros((Z, Y, X, 3, 3))
        L_exact = np.zeros((Z, Y, X, 3, 3))
        for i in range(3):
            for j in range(3):
                uu = u_t2_raw[..., i] * u_t2_raw[..., j]
                uu_bar = gaussian_filter(uu, sigma=self.filter_width, mode=self.boundary_mode)
                ubar_ubar = u_bar[..., i] * u_bar[..., j]
                ubar_ubar_filtered = gaussian_filter(
                    ubar_ubar, sigma=self.filter_width, mode=self.boundary_mode
                )
                tau_exact[..., i, j] = uu_bar - ubar_ubar
                L_exact[..., i, j] = ubar_ubar_filtered - ubar_ubar

        # 3. Features: S, Omega, and g (velocity gradient)
        grad_u = np.zeros((Z, Y, X, 3, 3))
        for i in range(3):
            grad_z, grad_y, grad_x = self._gradient3(u_bar[..., i])
            grad_u[..., i, 0] = grad_x
            grad_u[..., i, 1] = grad_y
            grad_u[..., i, 2] = grad_z

        S = 0.5 * (grad_u + grad_u.transpose(0, 1, 2, 4, 3))
        Omega = 0.5 * (grad_u - grad_u.transpose(0, 1, 2, 4, 3))

        # 4. WALE Tensor S_d
        # g^2 = grad_u @ grad_u
        g_sq = np.matmul(grad_u, grad_u)
        S_d = 0.5 * (g_sq + g_sq.transpose(0, 1, 2, 4, 3))
        trace_g_sq = np.trace(g_sq, axis1=3, axis2=4)
        for i in range(3):
            S_d[..., i, i] -= trace_g_sq / 3.0

        # 5. Topological Vectors (omega, W, h)
        # Vorticity omega_i = epsilon_ijk * Omega_jk.  Since Omega_jk = 0.5*(du_j/dx_k - du_k/dx_j)
        # Then omega_x = 2*Omega_32, omega_y = 2*Omega_13, omega_z = 2*Omega_21
        omega_vec = np.zeros((Z, Y, X, 3))
        omega_vec[..., 0] = 2.0 * Omega[..., 2, 1]  # z, y
        omega_vec[..., 1] = 2.0 * Omega[..., 0, 2]  # x, z
        omega_vec[..., 2] = 2.0 * Omega[..., 1, 0]  # y, x

        # Vortex stretching W_i = S_ij * omega_j
        # omega_vec shape is (Z, Y, X, 3), add dim to matmul
        W_vec = np.matmul(S, omega_vec[..., np.newaxis])[..., 0] # shape (Z, Y, X, 3)

        # Helicity h = u_i * omega_i
        h_scalar = np.sum(u_bar * omega_vec, axis=-1, keepdims=True) # shape (Z, Y, X, 1)

        # 6. Objective Rate (Jaumann Derivative of Strain S)
        S_jaumann = np.zeros((Z, Y, X, 3, 3))
        if is_multi_time:
            # 6a. Unsteady term: dS/dt using central difference (u_t3 - u_t1)/(2*dt)
            du_dt = (u_bar_t3 - u_bar_t1) / (2.0 * dt)
            grad_du_dt = np.zeros((Z, Y, X, 3, 3))
            for i in range(3):
                grad_z, grad_y, grad_x = self._gradient3(du_dt[..., i])
                grad_du_dt[..., i, 0] = grad_x
                grad_du_dt[..., i, 1] = grad_y
                grad_du_dt[..., i, 2] = grad_z
            dS_dt_unsteady = 0.5 * (grad_du_dt + grad_du_dt.transpose(0, 1, 2, 4, 3))

            # 6b. Convective term: u_k * dS_ij/dx_k
            dS_dx = np.zeros((Z, Y, X, 3, 3, 3)) # (z, y, x, i, j, k)
            for i in range(3):
                for j in range(3):
                    gr_z, gr_y, gr_x = self._gradient3(S[..., i, j])
                    dS_dx[..., i, j, 0] = gr_x
                    dS_dx[..., i, j, 1] = gr_y
                    dS_dx[..., i, j, 2] = gr_z
            
            # einsum notation:
            # u_bar (..., k)
            # dS_dx (..., i, j, k) -> multiply and sum over k
            convective_S = np.einsum('...k,...ijk->...ij', u_bar, dS_dx)

            # 6c. Co-rotational spin coupling: S_ik O_kj - O_ik S_kj = S*Omega - Omega*S
            S_Om = np.matmul(S, Omega)
            Om_S = np.matmul(Omega, S)
            spin_coupling = S_Om - Om_S

            # Assembly: dS/dt_total(material) + Spin
            S_jaumann = dS_dt_unsteady + convective_S + spin_coupling
        else:
            # Fallback if no multi-time: just spin coupling
            S_Om = np.matmul(S, Omega)
            Om_S = np.matmul(Omega, S)
            S_jaumann = S_Om - Om_S

        # 6.5 Laplacian of Strain (Lap_S = d^2S/dx^2 + d^2S/dy^2 + d^2S/dz^2)
        Lap_S = np.zeros((Z, Y, X, 3, 3))
        for i in range(3):
            for j in range(3):
                # 1st spatial derivative of the strain field component S_{ij}
                gr_z, gr_y, gr_x = self._gradient3(S[..., i, j])
                # 2nd spatial derivatives along the principal axes
                gr_xx = self._gradient_axis(gr_x, axis=2)
                gr_yy = self._gradient_axis(gr_y, axis=1)
                gr_zz = self._gradient_axis(gr_z, axis=0)
                
                # Laplacian is the trace of the spatial Hessian
                Lap_S[..., i, j] = gr_xx + gr_yy + gr_zz

        delta_eff = self._delta_eff_field(Z=Z, Y=Y, X=X)

        # 7. Local Variance (NMSE Denominator)
        # We need var_tau to match the flattened shape (Z*Y*X,)
        # For Channel, variance is computed per Y-plane.
        # For Isotropic, it's global.
        var_tau_flat = np.zeros(Z * Y * X)
        if self.y_coords is not None:
            # Channel flow: group by Y
            for j in range(Y):
                plane_tau = tau_exact[:, j, :, :, :]  # shape: (Z, X, 3, 3)
                plane_var = np.var(plane_tau)
                # Map to flat array: the flattened indices for y=j are all i where (i // X) % Y == j
                # Better approach: reshape var_tau_flat to match (Z, Y, X)
                var_tau_3d = var_tau_flat.reshape(Z, Y, X)
                var_tau_3d[:, j, :] = plane_var
        else:
            # Isotropic: global variance
            global_var = np.var(tau_exact)
            var_tau_flat.fill(global_var)

        # Global variance for physics loss (Pi)
        Pi_true = -np.einsum('nij,nij->n', tau_exact.reshape(-1, 3, 3), S.reshape(-1, 3, 3))
        var_pi = float(np.var(Pi_true))
        mean_pi = float(np.mean(Pi_true))

        # ══════════════════════════════════════════════════════════════
        # Flatten to (n_points, 3, 3)
        # ══════════════════════════════════════════════════════════════
        n_points = Z * Y * X
        self.S = S.reshape(-1, 3, 3)
        self.Omega = Omega.reshape(-1, 3, 3)
        self.tau = tau_exact.reshape(-1, 3, 3)
        self.L = L_exact.reshape(-1, 3, 3)
        self.S_d = S_d.reshape(-1, 3, 3)
        self.Delta = delta_eff.reshape(-1)
        self.omega_vec = omega_vec.reshape(-1, 3)
        self.W_vec = W_vec.reshape(-1, 3)
        self.h_scalar = h_scalar.reshape(-1, 1)
        self.S_jaumann = S_jaumann.reshape(-1, 3, 3)
        self.Lap_S = Lap_S.reshape(-1, 3, 3)
        self.var_tau = var_tau_flat
        self.var_pi = var_pi
        self.mean_pi = mean_pi

        # ── Cache to disk ──
        print(f"💾 Caching processed tensors to: {self.cache_path}")
        with h5py.File(self.cache_path, 'w') as f:
            f.create_dataset('S', data=self.S)
            f.create_dataset('Omega', data=self.Omega)
            f.create_dataset('tau', data=self.tau)
            f.create_dataset('L', data=self.L)
            f.create_dataset('S_d', data=self.S_d)
            f.create_dataset('Delta', data=self.Delta)
            f.create_dataset('omega_vec', data=self.omega_vec)
            f.create_dataset('W_vec', data=self.W_vec)
            f.create_dataset('h_scalar', data=self.h_scalar)
            f.create_dataset('S_jaumann', data=self.S_jaumann)
            f.create_dataset('Lap_S', data=self.Lap_S)
            f.create_dataset('var_tau', data=self.var_tau)
            f.create_dataset('var_pi', data=self.var_pi)
            f.create_dataset('mean_pi', data=self.mean_pi)
            if self.y_coords is not None:
                f.create_dataset('y_coords', data=self.y_coords)
            if self.x_coords is not None:
                f.create_dataset('x_coords', data=self.x_coords)
            if self.z_coords is not None:
                f.create_dataset('z_coords', data=self.z_coords)

    def _configure_spatial_coordinates(self, Z: int, Y: int, X: int) -> None:
        """
        Construct coordinate arrays used by the finite-difference operators.

        For wall-bounded datasets, the wall-normal coordinates are non-uniform and
        are provided explicitly. Lateral coordinates are inferred from the
        repository's channel-style cutouts when they are not stored in the raw HDF5.
        """
        if self.y_coords is not None:
            self.y_coords = np.asarray(self.y_coords, dtype=np.float64)
            if self.y_coords.shape != (Y,):
                raise ValueError(
                    f"y_coords shape {self.y_coords.shape} does not match Y={Y} for {self.h5_path}"
                )

            if self.x_coords is None:
                self.x_coords = np.arange(X, dtype=np.float64) * _CHANNEL_X_SPACING
            else:
                self.x_coords = np.asarray(self.x_coords, dtype=np.float64)
            if self.z_coords is None:
                self.z_coords = np.arange(Z, dtype=np.float64) * _CHANNEL_Z_SPACING
            else:
                self.z_coords = np.asarray(self.z_coords, dtype=np.float64)

            if self.x_coords.shape != (X,):
                raise ValueError(
                    f"x_coords shape {self.x_coords.shape} does not match X={X} for {self.h5_path}"
                )
            if self.z_coords.shape != (Z,):
                raise ValueError(
                    f"z_coords shape {self.z_coords.shape} does not match Z={Z} for {self.h5_path}"
                )
            self._validate_axis_coordinates(self.z_coords, axis_name="z")
            self._validate_axis_coordinates(self.y_coords, axis_name="y")
            self._validate_axis_coordinates(self.x_coords, axis_name="x")

            self._axis_coords = (self.z_coords, self.y_coords, self.x_coords)
            return

        self._axis_coords = (float(self.dx), float(self.dx), float(self.dx))

    def _gradient_axis(self, field: FloatArray, axis: int) -> FloatArray:
        if self._axis_coords is None:
            raise RuntimeError("spatial coordinates are not configured")
        edge_order = 2 if field.shape[axis] >= 3 else 1
        return np.gradient(field, self._axis_coords[axis], axis=axis, edge_order=edge_order)

    def _gradient3(self, field: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        return (
            self._gradient_axis(field, axis=0),
            self._gradient_axis(field, axis=1),
            self._gradient_axis(field, axis=2),
        )

    def _delta_eff_field(self, Z: int, Y: int, X: int) -> FloatArray:
        if self.y_coords is None:
            return np.full((Z, Y, X), float(self.filter_width) * float(self.dx), dtype=np.float64)

        dx_1d = np.abs(np.gradient(self.x_coords)).astype(np.float64)
        dy_1d = np.abs(np.gradient(self.y_coords)).astype(np.float64)
        dz_1d = np.abs(np.gradient(self.z_coords)).astype(np.float64)
        dx = np.broadcast_to(dx_1d[None, None, :], (Z, Y, X))
        dy = np.broadcast_to(dy_1d[None, :, None], (Z, Y, X))
        dz = np.broadcast_to(dz_1d[:, None, None], (Z, Y, X))
        return float(self.filter_width) * np.cbrt(dx * dy * dz)

    @staticmethod
    def _validate_axis_coordinates(coords: FloatArray, axis_name: str) -> None:
        diffs = np.diff(coords)
        if np.any(~np.isfinite(coords)):
            raise ValueError(f"{axis_name}_coords contain non-finite values")
        if np.all(diffs > 0.0) or np.all(diffs < 0.0):
            return
        raise ValueError(
            f"{axis_name}_coords must be strictly monotone; found mixed ordering in {axis_name}_coords"
        )

    def _finalize_grid_metadata(self) -> None:
        if self.grid_shape is not None:
            return

        if self.y_coords is not None:
            y = len(self.y_coords)
            if self.x_coords is not None:
                x = len(self.x_coords)
            else:
                x = int(round(np.sqrt(self.S.shape[0] / y)))
            if self.z_coords is not None:
                z = len(self.z_coords)
            else:
                z = self.S.shape[0] // (y * x)
            if z * y * x != self.S.shape[0]:
                raise ValueError(
                    f"Cannot infer wall-bounded grid shape from cached metadata for {self.h5_path}"
                )
            self.grid_shape = (z, y, x)
            return

        side = int(round(self.S.shape[0] ** (1.0 / 3.0)))
        if side ** 3 != self.S.shape[0]:
            raise ValueError(
                f"Cannot infer cubic grid shape from cached metadata for {self.h5_path}"
            )
        self.grid_shape = (side, side, side)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sample(self, idx: int) -> dict[str, FloatArray]:
        if not (0 <= idx < self.n_samples):
            raise IndexError(f"Sample index {idx} out of range [0, {self.n_samples}).")
        return {
            "S":      self.S[idx],
            "Omega":  self.Omega[idx],
            "L":      self.L[idx],
            "S_d":    self.S_d[idx],
            "tau":    self.tau[idx],
            "Delta":  self.Delta[idx],
            "omega_vec": self.omega_vec[idx],
            "W_vec":     self.W_vec[idx],
            "h_scalar":  self.h_scalar[idx],
            "S_jaumann": self.S_jaumann[idx],
            "Lap_S":     self.Lap_S[idx],
        }

    def evaluate_mse(
        self,
        predicted_tau: FloatArray,
        true_tau: FloatArray | None = None,
    ) -> float:
        """Local Variance-normalized MSE (NMSE)."""
        if true_tau is None:
            true_tau = self.tau

        predicted_tau = np.asarray(predicted_tau, dtype=np.float64)
        true_tau = np.asarray(true_tau, dtype=np.float64)

        residual = predicted_tau - true_tau
        
        # Div by local var_tau (shape N), broadcasted to (N, 3, 3)
        # To avoid ZeroDivisionError if var_tau is exactly 0:
        safe_var = np.where(self.var_tau > 1e-12, self.var_tau, 1e-12)
        local_mse = (residual ** 2) / safe_var[:, None, None]
        
        return float(np.mean(local_mse))

    def evaluate_physics_loss(self, predicted_tau: FloatArray) -> float:
        """
        Normalized backscatter constraint.
        Capped to -MAX_REWARD to prevent reward hacking.
        """
        predicted_tau = np.asarray(predicted_tau, dtype=np.float64)
        dissipation = -np.einsum('nij,nij->n', predicted_tau, self.S)
        backscatter = np.maximum(0, -dissipation)
        
        mean_bs = float(np.mean(backscatter))
        norm_bs = mean_bs / self.var_pi
        
        # Reward backscatter up to a limit (e.g., max reward = 0.5)
        # Anything beyond the limit provides no marginal returns.
        MAX_REWARD = 0.5
        bounded_reward = min(norm_bs, MAX_REWARD)
        
        # Return as a negative loss (reward)
        return -bounded_reward

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"JHTDBOracle(h5_path='{self.h5_path}', "
            f"filter_width={self.filter_width}, "
            f"n_samples={self.n_samples})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (run with:  python oracle.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if os.path.exists("jhtdb_u_tensor_64.h5"):
        print("=" * 50)
        print("🚀 Testing JHTDBOracle with Isotropic DNS Data")
        print("=" * 50)
        oracle_iso = JHTDBOracle(
            h5_path="jhtdb_u_tensor_64.h5",
            filter_width=1.0,
            boundary_mode='wrap',
        )
        print(oracle_iso)
        print(f"Samples: {oracle_iso.n_samples}")
        print(f"S shape: {oracle_iso.S.shape}")
        print(f"S_d shape: {oracle_iso.S_d.shape}")

        zero_pred = np.zeros_like(oracle_iso.tau)
        nmse_zero = oracle_iso.evaluate_mse(zero_pred)
        print(f"\nNMSE (zero prediction): {nmse_zero:.6f} (should be ≈1.0)")

    if os.path.exists("channel_u_tensor_64.h5"):
        print("\n" + "=" * 50)
        print("🚀 Testing JHTDBOracle with Channel DNS Data")
        print("=" * 50)
        oracle_chan = JHTDBOracle(
            h5_path="channel_u_tensor_64.h5",
            filter_width=1.0,
            boundary_mode='nearest',
        )
        print(oracle_chan)
        print(f"Samples: {oracle_chan.n_samples}")
        print(f"S_d shape: {oracle_chan.S_d.shape}")
        if oracle_chan.y_coords is not None:
            print(f"y_coords: {oracle_chan.y_coords[:5]}... (shape {oracle_chan.y_coords.shape})")

        zero_pred = np.zeros_like(oracle_chan.tau)
        nmse_zero = oracle_chan.evaluate_mse(zero_pred)
        print(f"\nNMSE (zero prediction): {nmse_zero:.6f} (should be ≈1.0)")
