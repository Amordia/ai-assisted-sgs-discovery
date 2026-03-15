"""
optimizer.py — Leaf-Node Numerical Optimizer for Neuro-Symbolic-SGS
====================================================================
Given a SymPy tensor expression containing unknown scalar constants
(c1..c5), fit them via L-BFGS-B to minimise NMSE against the
oracle's ground-truth τ_ij.

Key features:
  - Variance-normalized MSE (NMSE) across dual oracles
  - 10% sub-sampling: indices drawn ONCE per optimize() call
    (frozen for all L-BFGS-B iterations so gradients are consistent)
  - Physics backscatter constraint (optional)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, OptimizeResult
import sympy as sp

from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.physics_env import (
    TensorSymbolicEngine,
    S_sym,
    Omega_sym,
    c_syms,
)

FloatArray = NDArray[np.float64]

# Default sub-sampling ratio: use 10% of data per oracle
_SUBSAMPLE_RATIO = 0.10
_MIN_SUBSAMPLE_SIZE = 256


class LeafNodeOptimizer:
    """
    Optimise scalar constants in a candidate SGS closure expression
    using L-BFGS-B with deterministic 10% sub-sampling.

    The sub-sample indices are drawn ONCE at the start of each
    optimize() call and frozen throughout all L-BFGS-B iterations.
    This ensures the objective function is deterministic (same x →
    same output), which is required for L-BFGS-B's finite-difference
    gradient estimation to work correctly.

    Parameters
    ----------
    engine : TensorSymbolicEngine
        The symbolic engine used for numerical evaluation.
    oracles : list[JHTDBOracle]
        Data oracles providing ground-truth tensors.
    bounds : tuple[float, float]
        Elementwise bounds for every constant.  Default (-10, 10).
    max_iter : int
        Maximum iterations for the L-BFGS-B solver.
    lambda_pi : float
        Weight for the backscatter physics loss term.
    """

    def __init__(
        self,
        engine: TensorSymbolicEngine,
        oracles: list[JHTDBOracle],
        bounds: tuple[float, float] = (-1000.0, 1000.0),
        max_iter: int = 200,
        lambda_pi: float = 0.0,
        lambda_diss: float = 1.0,
        lambda_l1: float = 1e-5,
        subsample_ratio: float = _SUBSAMPLE_RATIO,
        min_subsample_size: int = _MIN_SUBSAMPLE_SIZE,
    ) -> None:
        self.engine = engine
        self.oracles = oracles
        self.bounds = bounds
        self.max_iter = max_iter
        self.lambda_pi = lambda_pi
        self.lambda_diss = lambda_diss
        self.lambda_l1 = lambda_l1
        self.subsample_ratio = float(subsample_ratio)
        self.min_subsample_size = int(min_subsample_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        expr: sp.MatrixExpr,
        initial_guess: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], float]:
        """
        Fit scalar constants in *expr* to minimise NMSE(τ_pred, τ_true)
        across all oracles using deterministic sub-sampling.

        Returns
        -------
        best_constants : dict[str, float]
        best_loss : float
        """
        # 1. Identify which constants appear in the expression
        free = expr.free_symbols
        active_consts: list[sp.Symbol] = [
            ci for ci in c_syms if ci in free
        ]

        # No free constants → evaluate once on full data
        if not active_consts:
            return self._evaluate_constant_free(expr)

        # 2. Build initial guess vector — start near zero, NOT 0.1
        if initial_guess is None:
            initial_guess = {}
        x0 = np.array(
            [initial_guess.get(ci.name, 0.0) for ci in active_consts],
            dtype=np.float64,
        )

        # 3. Draw sub-sample indices ONCE (frozen for all L-BFGS-B iters)
        #    This is CRITICAL: L-BFGS-B needs a deterministic objective
        #    to estimate gradients via finite differences.
        frozen_indices: list[np.ndarray] = []
        frozen_data: list[dict] = []
        for oracle in self.oracles:
            n = oracle.S.shape[0]
            n_sub = max(self.min_subsample_size, int(n * self.subsample_ratio))
            n_sub = min(n_sub, n)
            idx = np.random.choice(n, size=n_sub, replace=False)
            frozen_indices.append(idx)
            frozen_data.append({
                'S':       oracle.S[idx],
                'Omega':   oracle.Omega[idx],
                'L':       oracle.L[idx],
                'S_d':     oracle.S_d[idx],
                'S_j':     oracle.S_jaumann[idx],
                'Lap_S':   oracle.Lap_S[idx],
                'Delta':   oracle.Delta[idx],
                'tau':     oracle.tau[idx],
                'var_tau': oracle.var_tau[idx],
                'omega_vec': oracle.omega_vec[idx],
                'W_vec':     oracle.W_vec[idx],
                'h_scalar':  oracle.h_scalar[idx],
            })

        linear_warm_start = self._linear_warm_start(expr, active_consts, frozen_data)
        if linear_warm_start is not None and np.all(np.isfinite(linear_warm_start)):
            x0 = np.clip(linear_warm_start, self.bounds[0], self.bounds[1])

        # 4. Deterministic objective function
        def objective(x: FloatArray) -> float:
            const_dict = {
                ci.name: float(x[i]) for i, ci in enumerate(active_consts)
            }
            total_loss = 0.0

            for oracle, data in zip(self.oracles, frozen_data):
                try:
                    tau_pred = self.engine.lambdify_tensor_expr(
                        expr,
                        data['S'], data['Omega'], data['L'],
                        data['S_d'], data['S_j'], data['Lap_S'], data['Delta'],
                        data['omega_vec'], data['W_vec'], data['h_scalar'],
                        const_dict,
                    )
                    if np.any(np.isnan(tau_pred)) or np.any(np.isinf(tau_pred)):
                        return float("inf")

                    # NMSE on sub-sampled data using local variance
                    residual = tau_pred - data['tau']
                    # Use safe variance to avoid division by zero
                    safe_var = np.where(data['var_tau'] > 1e-12, data['var_tau'], 1e-12)
                    local_mse = (residual ** 2) / safe_var[:, None, None]
                    mse = float(np.mean(local_mse))
                    loss = mse

                    # Physics backscatter constraint
                    if self.lambda_pi > 0.0:
                        dissipation = -np.einsum('nij,nij->n', tau_pred, data['S'])
                        backscatter = np.maximum(0, -dissipation)
                        mean_bs = float(np.mean(backscatter))
                        norm_bs = mean_bs / oracle.var_pi
                        
                        # Apply the same bound as oracle.py (MAX_REWARD = 0.5)
                        MAX_REWARD = 0.5
                        phys_loss = -min(norm_bs, MAX_REWARD)
                        
                        loss += self.lambda_pi * phys_loss
                        
                    # Global Energy Dissipation Budget constraint
                    if self.lambda_diss > 0.0:
                        dissipation = -np.einsum('nij,nij->n', tau_pred, data['S'])
                        mean_pi_pred = float(np.mean(dissipation))
                        
                        # Use asymptotically stable logarithmic physical prior penalty
                        rel_err = (mean_pi_pred - oracle.mean_pi) / (abs(oracle.mean_pi) + 1e-12)
                        diss_penalty = self.lambda_diss * (np.log1p(abs(rel_err)) ** 2)
                        loss += diss_penalty

                    total_loss += loss

                except Exception:
                    return float("inf")
                    
            # ── L1 Sparse Regularization (LASSO-like) ──
            # Encourages the optimization engine to zero out physically unneeded coefficients
            if self.lambda_l1 > 0.0:
                total_loss += self.lambda_l1 * float(np.sum(np.abs(x)))

            return total_loss

        # 5. Run L-BFGS-B
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result: OptimizeResult = minimize(
                    objective,
                    x0,
                    method="L-BFGS-B",
                    bounds=[self.bounds] * len(active_consts),
                    options={"maxiter": self.max_iter, "ftol": 1e-12},
                )
            if not np.isfinite(result.fun):
                return self._fail_result(active_consts)

            best_constants: dict[str, float] = {
                ci.name: float(result.x[i])
                for i, ci in enumerate(active_consts)
            }
            return best_constants, float(result.fun)

        except Exception:
            return self._fail_result(active_consts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_constant_free(
        self, expr: sp.MatrixExpr,
    ) -> tuple[dict[str, float], float]:
        """Evaluate loss for an expression with no free constants (full data)."""
        total_loss = 0.0
        try:
            for oracle in self.oracles:
                tau_pred = self.engine.lambdify_tensor_expr(
                    expr,
                    oracle.S, oracle.Omega, oracle.L,
                    oracle.S_d, oracle.S_jaumann, oracle.Lap_S, oracle.Delta,
                    oracle.omega_vec, oracle.W_vec, oracle.h_scalar, {},
                )
                if np.any(np.isnan(tau_pred)) or np.any(np.isinf(tau_pred)):
                    return {}, float("inf")

                mse = oracle.evaluate_mse(tau_pred)
                loss = mse

                if self.lambda_pi > 0.0:
                    loss += self.lambda_pi * oracle.evaluate_physics_loss(tau_pred)
                    
                if self.lambda_diss > 0.0:
                    dissipation = -np.einsum('nij,nij->n', tau_pred, oracle.S)
                    mean_pi_pred = float(np.mean(dissipation))
                    rel_err = (mean_pi_pred - oracle.mean_pi) / (abs(oracle.mean_pi) + 1e-12)
                    diss_penalty = self.lambda_diss * (np.log1p(abs(rel_err)) ** 2)
                    loss += diss_penalty

                total_loss += loss

            return {}, total_loss
        except Exception:
            return {}, float("inf")

    @staticmethod
    def _fail_result(
        active_consts: list[sp.Symbol],
    ) -> tuple[dict[str, float], float]:
        """Return a sentinel result when optimisation fails."""
        return {ci.name: 0.0 for ci in active_consts}, float("inf")

    def _linear_warm_start(
        self,
        expr: sp.MatrixExpr,
        active_consts: list[sp.Symbol],
        frozen_data: list[dict[str, Any]],
    ) -> FloatArray | None:
        """
        Build a weighted least-squares initialization when the expression is
        linear in its active scalar coefficients.
        """
        if not active_consts:
            return None

        zero_constants = {ci.name: 0.0 for ci in active_consts}
        offset_expr = expr.subs({ci: 0.0 for ci in active_consts})
        basis_exprs: list[sp.MatrixExpr] = []
        for ci in active_consts:
            basis_expr = expr.diff(ci)
            if any(cj in basis_expr.free_symbols for cj in active_consts):
                return None
            basis_exprs.append(basis_expr)

        if any(cj in offset_expr.free_symbols for cj in active_consts):
            return None

        design_blocks: list[FloatArray] = []
        target_blocks: list[FloatArray] = []

        for data in frozen_data:
            try:
                offset = self.engine.lambdify_tensor_expr(
                    offset_expr,
                    data['S'], data['Omega'], data['L'],
                    data['S_d'], data['S_j'], data['Lap_S'], data['Delta'],
                    data['omega_vec'], data['W_vec'], data['h_scalar'],
                    zero_constants,
                )
                if np.any(~np.isfinite(offset)):
                    return None

                basis_values = []
                for basis_expr in basis_exprs:
                    basis = self.engine.lambdify_tensor_expr(
                        basis_expr,
                        data['S'], data['Omega'], data['L'],
                        data['S_d'], data['S_j'], data['Lap_S'], data['Delta'],
                        data['omega_vec'], data['W_vec'], data['h_scalar'],
                        zero_constants,
                    )
                    if np.any(~np.isfinite(basis)):
                        return None
                    basis_values.append(basis.reshape(len(data['tau']) * 9))

                residual_target = (data['tau'] - offset).reshape(len(data['tau']) * 9)
                safe_var = np.where(data['var_tau'] > 1e-12, data['var_tau'], 1e-12)
                weights = np.repeat(1.0 / np.sqrt(safe_var), 9)
                design = np.stack(basis_values, axis=1)
                design_blocks.append(design * weights[:, None])
                target_blocks.append(residual_target * weights)
            except Exception:
                return None

        if not design_blocks:
            return None

        design_matrix = np.concatenate(design_blocks, axis=0)
        target_vector = np.concatenate(target_blocks, axis=0)

        try:
            solution, *_ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)
        except np.linalg.LinAlgError:
            return None

        return np.asarray(solution, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"LeafNodeOptimizer(bounds={self.bounds}, "
            f"max_iter={self.max_iter})"
        )


# -----------------------------------------------------------------------
# Quick smoke-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import os, time
    from sgs_discovery.physics_env import c1, c2, S_d_sym

    if os.path.exists("jhtdb_u_tensor_64.h5"):
        oracle1 = JHTDBOracle(h5_path="jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode='wrap')
        oracle2 = JHTDBOracle(h5_path="channel_u_tensor_64.h5", filter_width=1.0, boundary_mode='nearest')
        engine = TensorSymbolicEngine()
        opt = LeafNodeOptimizer(engine, [oracle1, oracle2], lambda_pi=1.0)

        exprs = [
            ("c1*S",             c1 * S_sym),
            ("c1*S + c2*S_d",    c1 * S_sym + c2 * S_d_sym),
        ]
        for name, expr in exprs:
            t0 = time.time()
            consts, loss = opt.optimize(expr)
            print(f"{name}: loss={loss:.4e}, consts={consts}, time={time.time()-t0:.1f}s")
    else:
        print("⚠ No data file found. Run download_jhtdb.py first.")
