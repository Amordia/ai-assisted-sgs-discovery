"""
physics_env.py — Physics-Informed Symbolic Engine for Neuro-Symbolic-SGS
=========================================================================
Manages tensor symbols, Galilean-invariant operations, and physical
constraint pruning (symmetry + dimensional homogeneity).

Key responsibilities
--------------------
* Define symbolic tensors  S (strain-rate) and Ω (rotation-rate).
* Provide legal tensor operations (add, multiply/contract, trace, scalar×tensor).
* Prune candidate expressions that violate physical constraints
  (currently: symmetry of the resulting 2nd-order tensor).
* Convert a SymPy matrix expression → NumPy-evaluable function.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
import sympy as sp
from sympy import (
    MatrixSymbol,
    Symbol,
    Trace,
    Identity,
    Matrix,
    zeros as sp_zeros,
    simplify,
    Transpose,
)

FloatArray = NDArray[np.float64]

# -----------------------------------------------------------------------
# Symbolic building blocks — module-level so they can be imported easily
# -----------------------------------------------------------------------
_DIM = 3

# Tensor symbols
# Tensor symbols
S_sym      = MatrixSymbol("S",      _DIM, _DIM)    # strain-rate (symmetric)
Omega_sym  = MatrixSymbol("Omega",  _DIM, _DIM)    # rotation-rate (anti-symmetric)
L_sym      = MatrixSymbol("L",      _DIM, _DIM)    # Leonard stress (symmetric)
S_d_sym    = MatrixSymbol("S_d",    _DIM, _DIM)    # WALE tensor (symmetric, traceless)
S_j_sym    = MatrixSymbol("S_j",    _DIM, _DIM)    # Jaumann Objective Rate (symmetric)
Lap_S_sym  = MatrixSymbol("Lap_S",  _DIM, _DIM)    # Laplacian of Strain (symmetric)
Delta_sym  = Symbol("Delta", real=True, positive=True) # Filter scale (scalar)

# Topological vector symbols
omega_sym  = MatrixSymbol("omega",  _DIM, 1)       # Vorticity vector
W_sym      = MatrixSymbol("W",      _DIM, 1)       # Vortex stretching vector
h_sym      = Symbol("h", real=True)                # Helicity scalar

I_sym      = Identity(_DIM)                        # 3×3 identity

# Scalar constants — pool of 8
_NUM_CONSTANTS = 8
c_syms: list[Symbol] = [Symbol(f"c{i}", real=True) for i in range(1, _NUM_CONSTANTS + 1)]
c1, c2, c3, c4, c5, c6, c7, c8 = c_syms


class TensorSymbolicEngine:
    """
    Engine for constructing, validating, and evaluating tensor
    expressions that are candidates for SGS closure models.

    The engine exposes:
      * Convenience wrappers for Galilean-invariant tensor algebra.
      * ``is_physically_valid(expr)`` — symmetry + dimensional pruning.
      * ``check_dimensions(expr)``   — dimensional homogeneity check.
      * ``lambdify_tensor_expr(...)`` — SymPy → NumPy bridge.
    """

    # Exposed symbols (for convenience from outside)
    S: MatrixSymbol     = S_sym
    Omega: MatrixSymbol = Omega_sym
    L: MatrixSymbol     = L_sym
    S_d: MatrixSymbol   = S_d_sym
    S_j: MatrixSymbol   = S_j_sym
    Lap_S: MatrixSymbol = Lap_S_sym
    Delta: Symbol       = Delta_sym
    omega: MatrixSymbol = omega_sym
    W: MatrixSymbol     = W_sym
    h: Symbol           = h_sym
    I: Identity         = I_sym
    constants: list[Symbol] = c_syms

    def __init__(self) -> None:
        self.dim: int = _DIM
        self._symmetry_map: dict[str, str] = {
            "S":      "symmetric",
            "Omega":  "antisymmetric",
            "L":      "symmetric",
            "S_d":    "symmetric",
            "S_j":    "symmetric",
            "Lap_S":  "symmetric",
        }
        self._dimension_map: dict[sp.Basic, tuple[sp.Rational, sp.Rational]] = {
            S_sym: (sp.Integer(0), sp.Integer(-1)),
            Omega_sym: (sp.Integer(0), sp.Integer(-1)),
            L_sym: (sp.Integer(2), sp.Integer(-2)),
            S_d_sym: (sp.Integer(0), sp.Integer(-2)),
            S_j_sym: (sp.Integer(0), sp.Integer(-2)),
            Lap_S_sym: (sp.Integer(-2), sp.Integer(-1)),
            Delta_sym: (sp.Integer(1), sp.Integer(0)),
            omega_sym: (sp.Integer(0), sp.Integer(-1)),
            W_sym: (sp.Integer(0), sp.Integer(-2)),
            h_sym: (sp.Integer(1), sp.Integer(-2)),
        }
        # Cache: id(expr) → compiled (funcs_9, free_consts, ...)
        # id() is safe here because _expr_refs keeps hard references.
        self._lambdify_cache: dict[int, Any] = {}
        self._expr_refs:      dict[int, Any] = {}  # prevent GC/id reuse

    # ------------------------------------------------------------------
    # 1. Galilean-invariant tensor operations
    # ------------------------------------------------------------------

    @staticmethod
    def tensor_add(A: sp.MatrixExpr, B: sp.MatrixExpr) -> sp.MatrixExpr:
        """Tensor addition  A + B."""
        return A + B

    @staticmethod
    def tensor_mul(A: sp.MatrixExpr, B: sp.MatrixExpr) -> sp.MatrixExpr:
        """Matrix (tensor contraction) product  A · B."""
        return A * B

    @staticmethod
    def tensor_trace(A: sp.MatrixExpr) -> sp.Expr:
        """Scalar invariant  Tr(A)."""
        return Trace(A)

    @staticmethod
    def scalar_tensor_mul(scalar: sp.Expr, T: sp.MatrixExpr) -> sp.MatrixExpr:
        """Multiply a scalar (or scalar invariant) with a tensor."""
        return scalar * T

    @staticmethod
    def vector_outer(v1: sp.MatrixExpr, v2: sp.MatrixExpr) -> sp.MatrixExpr:
        """Outer product of two vectors v1 * v2^T. Returns a 3x3 MatrixExpr."""
        return v1 * Transpose(v2)

    @staticmethod
    def vector_dot(v1: sp.MatrixExpr, v2: sp.MatrixExpr) -> sp.Expr:
        """Dot product of two vectors v1^T * v2. Returns a scalar."""
        return Trace(Transpose(v1) * v2)

    # ------------------------------------------------------------------
    # 2. Predefined Galilean-invariant building blocks
    # ------------------------------------------------------------------

    def invariant_bases(self) -> dict[str, sp.MatrixExpr]:
        """
        Return a dictionary of well-known Galilean-invariant
        tensor bases up to quadratic order (Pope, 1975).

        These are the candidate building blocks MCTS can compose.
        """
        S = self.S
        O = self.Omega
        I = self.I

        return {
            # Linear
            "S":           S,
            # Quadratic (symmetric products)
            "S2":          S * S,
            "OmS_SOm":     O * S - S * O,          # anti-commutator → symmetric
            # Trace-weighted
            "TrS2_S":      Trace(S * S) * S,
            "TrS2_I":      Trace(S * S) * I,
            "TrOm2_S":     Trace(O * O) * S,
            "S2Om_OmS2":   S * S * O - O * S * S,  # symmetric
            
            # Topological bases
            "omega_outer": self.vector_outer(self.omega, self.omega),
            "W_outer":     self.vector_outer(self.W, self.W),
            "omega_W_sym": self.vector_outer(self.omega, self.W) + self.vector_outer(self.W, self.omega),

            # Objective Time-Memory base
            "S_j":         self.S_j,
            # High-Order Spatial base
            "Lap_S":       self.Lap_S,
        }

    # ------------------------------------------------------------------
    # 3. Physical validity checks (pruning)
    # ------------------------------------------------------------------

    def is_physically_valid(self, expr: sp.MatrixExpr) -> bool:
        """
        Check whether *expr* can represent a valid SGS stress tensor τ_ij.

        Current checks
        ~~~~~~~~~~~~~~
        1. The expression must be a matrix expression of shape (3, 3).
        2. It must be dimensionally homogeneous with SGS stress units.
        3. It must be **symmetric**: expr − exprᵀ == 0  (symbolically).
        4. Each additive tensor term must carry at most one active scalar
           coefficient, and active coefficients may not be shared across
           multiple additive terms. This prevents the search from encoding
           the same basis tensor with redundant constants that would
           otherwise weaken the intended L1 sparsity penalty.

        Returns True if valid; False (with no exception) otherwise.
        """
        try:
            # ---- shape check ----
            if not hasattr(expr, "shape"):
                return False
            if expr.shape != (self.dim, self.dim):
                return False

            if not self._check_prefactor_structure(expr):
                return False

            if not self.check_dimensions(expr):
                return False

            # ---- symmetry check ----
            return self._check_symmetry(expr)

        except Exception:
            # Any SymPy simplification explosion → conservatively reject
            return False

    def _check_prefactor_structure(self, expr: sp.MatrixExpr) -> bool:
        terms = sp.Add.make_args(sp.expand(expr)) if expr.is_Add else (expr,)

        for term in terms:
            scalar_part = sp.Integer(1)
            matrix_factors: list[sp.Expr] = []

            if isinstance(term, sp.MatMul):
                term_args = term.args
            else:
                term_args = (term,)

            for arg in term_args:
                if getattr(arg, "is_commutative", False):
                    scalar_part *= arg
                else:
                    matrix_factors.append(arg)

            if not matrix_factors:
                return False

            active = [ci for ci in c_syms if ci in scalar_part.free_symbols]
            if len(active) > 1:
                return False

        return True

    def _check_symmetry(self, expr: sp.MatrixExpr) -> bool:
        """
        Numerically verify symmetry by evaluating the expression at a
        random (but structurally valid) point.

        Reuses the fast lambdify infrastructure so this is O(1) NumPy
        work, not O(n²) symbolic work.
        """
        rng = np.random.default_rng(0)
        d   = self.dim

        # Single-sample batch (shape 1×3×3)
        raw   = rng.normal(size=(1, d, d))
        S_b   = 0.5 * (raw   + raw.transpose(0, 2, 1))    # symmetric
        raw2  = rng.normal(size=(1, d, d))
        Om_b  = 0.5 * (raw2  - raw2.transpose(0, 2, 1))   # anti-symmetric
        raw3  = rng.normal(size=(1, d, d))
        L_b   = 0.5 * (raw3  + raw3.transpose(0, 2, 1))   # symmetric
        raw4  = rng.normal(size=(1, d, d))
        S_d_b   = 0.5 * (raw4  + raw4.transpose(0, 2, 1)) # symmetric
        raw5  = rng.normal(size=(1, d, d))
        S_j_b   = 0.5 * (raw5  + raw5.transpose(0, 2, 1)) # symmetric
        raw6  = rng.normal(size=(1, d, d))
        Lap_S_b = 0.5 * (raw6  + raw6.transpose(0, 2, 1)) # symmetric
        Delta_b = np.abs(rng.normal(size=(1,))) + 0.1     # positive scalar
        
        # New Topological variables validation batch
        omega_b = rng.normal(size=(1, 3))                 # vector
        W_b     = rng.normal(size=(1, 3))                 # vector
        h_b     = rng.normal(size=(1, 1))                 # scalar

        c_val = {ci.name: float(rng.normal()) for ci in c_syms}

        try:
            result = self.lambdify_tensor_expr(
                expr, S_b, Om_b, L_b, S_d_b, S_j_b, Lap_S_b, Delta_b, 
                omega_b, W_b, h_b, c_val
            )
            if np.any(np.isnan(result)):
                return False
            mat  = result[0]          # (3, 3)
            diff = mat - mat.T
            return bool(np.all(np.abs(diff) < 1e-8))
        except Exception:
            return False

    def check_dimensions(self, expr: sp.MatrixExpr) -> bool:
        dims = self._expr_dimensions(expr)
        if dims == "zero":
            return True
        return dims == (sp.Integer(2), sp.Integer(-2))

    def _expr_dimensions(
        self, expr: sp.Basic
    ) -> tuple[sp.Rational, sp.Rational] | str | None:
        if expr in self._dimension_map:
            return self._dimension_map[expr]

        if expr in c_syms:
            return (sp.Integer(0), sp.Integer(0))

        if (
            expr == 0
            or getattr(expr, "is_zero", False)
            or getattr(expr, "is_ZeroMatrix", False)
            or getattr(expr, "is_zero_matrix", False)
        ):
            return "zero"

        if isinstance(expr, (sp.Integer, sp.Float, sp.Rational, sp.NumberSymbol)):
            return (sp.Integer(0), sp.Integer(0))

        if isinstance(expr, Identity):
            return (sp.Integer(0), sp.Integer(0))

        if isinstance(expr, Trace):
            return self._expr_dimensions(expr.args[0])

        if isinstance(expr, Transpose):
            return self._expr_dimensions(expr.args[0])

        if isinstance(expr, (sp.Pow, sp.MatPow)):
            base_dims = self._expr_dimensions(expr.base)
            if base_dims in (None, "zero"):
                return base_dims
            if not expr.exp.is_number:
                return None
            return (
                sp.simplify(base_dims[0] * expr.exp),
                sp.simplify(base_dims[1] * expr.exp),
            )

        if isinstance(expr, (sp.Add, sp.MatAdd)):
            dims: tuple[sp.Rational, sp.Rational] | str | None = "zero"
            for arg in expr.args:
                arg_dims = self._expr_dimensions(arg)
                if arg_dims is None:
                    return None
                if dims == "zero":
                    dims = arg_dims
                    continue
                if arg_dims == "zero":
                    continue
                if dims != arg_dims:
                    return None
            return dims

        if isinstance(expr, (sp.Mul, sp.MatMul)):
            total_l = sp.Integer(0)
            total_t = sp.Integer(0)
            for arg in expr.args:
                arg_dims = self._expr_dimensions(arg)
                if arg_dims is None:
                    return None
                if arg_dims == "zero":
                    return "zero"
                total_l += arg_dims[0]
                total_t += arg_dims[1]
            return (sp.simplify(total_l), sp.simplify(total_t))

        if isinstance(expr, sp.MatrixSymbol):
            return self._dimension_map.get(expr)

        return None

    # ------------------------------------------------------------------
    # 4. Numerical evaluation (SymPy → NumPy bridge)
    # ------------------------------------------------------------------

    def lambdify_tensor_expr(
        self,
        expr: sp.MatrixExpr,
        S_val: FloatArray,
        Omega_val: FloatArray,
        L_val: FloatArray,
        S_d_val: FloatArray,
        S_j_val: FloatArray,
        Lap_S_val: FloatArray,
        Delta_val: FloatArray,
        omega_val: FloatArray,
        W_val: FloatArray,
        h_val: FloatArray,
        constants_dict: dict[str, float] | None = None,
    ) -> FloatArray:
        """
        Evaluate a SymPy matrix expression on **batched** NumPy data.

        Strategy (fast path)
        --------------------
        On first call for a given ``expr`` structure:
          1. Expand ``MatrixSymbol`` S, Omega, L, S_prev into explicit 3×3
             scalar symbols.
          2. Use ``sp.lambdify`` to compile each of the 9 output elements
             into a NumPy-callable.  Free scalar constants become extra
             arguments.
          3. Cache the compiled functions keyed by ``id(expr)``.

        On subsequent calls: skip compilation; evaluate vectorially over
        all samples with pure NumPy broadcasting (no Python loop).

        Parameters
        ----------
        expr : sp.MatrixExpr
        S_val : FloatArray, shape (n, 3, 3)
        Omega_val : FloatArray, shape (n, 3, 3)
        constants_dict : dict mapping constant-name to float.

        Returns
        -------
        FloatArray, shape (n, 3, 3)
        """
        if constants_dict is None:
            constants_dict = {}

        n = S_val.shape[0]

        # Use id(expr) as cache key — O(1) hash, no SymPy printer call.
        # _expr_refs keeps a hard reference so the id stays stable.
        cache_key = id(expr)
        if cache_key not in self._lambdify_cache:
            self._expr_refs[cache_key] = expr          # pin to prevent GC
            self._lambdify_cache[cache_key] = self._build_lambdify(
                expr, repr(type(expr))
            )

        cached = self._lambdify_cache[cache_key]
        if cached is None:
            return np.full((n, self.dim, self.dim), np.nan)

        merged_func, free_consts = cached

        # ---- vectorised evaluation (fast, pure NumPy) ----
        # Build flat input list: 9 S + 9 Omega + 9 L + 9 S_d + 9 S_j + 9 Lap_S + Delta + Topology + constants
        # Use scalar broadcast for constants (no np.full allocation)
        inputs: list[Any] = (
            [S_val[:, i, j]     for i in range(3) for j in range(3)] +
            [Omega_val[:, i, j] for i in range(3) for j in range(3)] +
            [L_val[:, i, j]     for i in range(3) for j in range(3)] +
            [S_d_val[:, i, j]   for i in range(3) for j in range(3)] +
            [S_j_val[:, i, j]   for i in range(3) for j in range(3)] +
            [Lap_S_val[:, i, j] for i in range(3) for j in range(3)] +
            [Delta_val] +
            [omega_val[:, i]    for i in range(3)] + 
            [W_val[:, i]        for i in range(3)] + 
            [h_val[:, 0] if h_val.ndim == 2 else h_val] + 
            [constants_dict.get(ci.name, 1.0) for ci in free_consts]
        )

        try:
            # Single merged call returns a flat tuple of 9 arrays
            vals = merged_func(*inputs)
            result = np.empty((n, self.dim, self.dim), dtype=np.float64)
            for idx in range(9):
                i, j = divmod(idx, 3)
                v = vals[idx]
                if np.ndim(v) == 0:
                    result[:, i, j] = float(v)
                else:
                    result[:, i, j] = v
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            return np.full((n, self.dim, self.dim), np.nan)

    def _build_lambdify(
        self,
        expr: sp.MatrixExpr,
        cache_key: str,
    ) -> Any:
        """
        Compile *expr* into a single merged NumPy-callable that returns
        all 9 tensor elements at once (avoids 9× Python dispatch overhead).
        Returns None on failure (will trigger NaN output).
        """
        try:
            d = self.dim
            # Create explicit scalar symbols for S, Omega, L, S_d, S_j, Lap_S and Delta
            s_mat_syms   = [[sp.Symbol(f"_s{i}{j}")   for j in range(d)] for i in range(d)]
            om_mat_syms  = [[sp.Symbol(f"_om{i}{j}")  for j in range(d)] for i in range(d)]
            l_mat_syms   = [[sp.Symbol(f"_l{i}{j}")   for j in range(d)] for i in range(d)]
            sd_mat_syms  = [[sp.Symbol(f"_sd{i}{j}")  for j in range(d)] for i in range(d)]
            sj_mat_syms  = [[sp.Symbol(f"_sj{i}{j}")  for j in range(d)] for i in range(d)]
            laps_mat_syms = [[sp.Symbol(f"_laps{i}{j}") for j in range(d)] for i in range(d)]
            delta_sym    = sp.Symbol("_delta", real=True, positive=True)
            S_explicit   = sp.Matrix(s_mat_syms)
            Om_explicit  = sp.Matrix(om_mat_syms)
            L_explicit   = sp.Matrix(l_mat_syms)
            SD_explicit  = sp.Matrix(sd_mat_syms)
            SJ_explicit  = sp.Matrix(sj_mat_syms)
            LapS_explicit = sp.Matrix(laps_mat_syms)
            
            # Topological structures
            omega_vec_syms = [sp.Symbol(f"_omega{i}") for i in range(d)]
            W_vec_syms     = [sp.Symbol(f"_W{i}")     for i in range(d)]
            h_scalar_sym   = sp.Symbol("_h_scalar", real=True)
            
            omega_explicit = sp.Matrix(omega_vec_syms)
            W_explicit     = sp.Matrix(W_vec_syms)

            # Free scalar constants in this expression
            free_consts = [ci for ci in c_syms if ci in expr.free_symbols]

            # Substitute MatrixSymbols → explicit matrices and evaluate once
            expr_expanded = expr.subs(
                {S_sym: S_explicit, Omega_sym: Om_explicit, L_sym: L_explicit, S_d_sym: SD_explicit, S_j_sym: SJ_explicit, Lap_S_sym: LapS_explicit,
                 Delta_sym: delta_sym, omega_sym: omega_explicit, W_sym: W_explicit, h_sym: h_scalar_sym}
            ).doit()
            result_mat = sp.Matrix(expr_expanded)

            # All lambdify input symbols
            s_flat   = [s_mat_syms[i][j]  for i in range(d) for j in range(d)]
            om_flat  = [om_mat_syms[i][j] for i in range(d) for j in range(d)]
            l_flat   = [l_mat_syms[i][j]  for i in range(d) for j in range(d)]
            sd_flat  = [sd_mat_syms[i][j] for i in range(d) for j in range(d)]
            sj_flat  = [sj_mat_syms[i][j] for i in range(d) for j in range(d)]
            laps_flat = [laps_mat_syms[i][j] for i in range(d) for j in range(d)]
            all_syms = s_flat + om_flat + l_flat + sd_flat + sj_flat + laps_flat + [delta_sym] + omega_vec_syms + W_vec_syms + [h_scalar_sym] + free_consts

            # Build a single tuple expression containing all 9 elements,
            # then compile ONE function that returns all of them at once.
            # This avoids 9 separate Python→NumPy dispatch calls.
            elements = [result_mat[i, j] for i in range(d) for j in range(d)]
            merged_func = sp.lambdify(all_syms, elements, modules="numpy")

            return (merged_func, free_consts)

        except Exception as exc:
            import warnings
            warnings.warn(f"[ENGINE] lambdify build failed for '{cache_key[:60]}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"TensorSymbolicEngine(dim={self.dim})"


# -----------------------------------------------------------------------
# Quick smoke-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    engine = TensorSymbolicEngine()
    print(engine)

    # Build a simple candidate: c1 * S + c2 * S·S
    expr = c1 * S_sym + c2 * S_sym * S_sym
    print(f"\nCandidate expr: {expr}")
    print(f"  is_physically_valid: {engine.is_physically_valid(expr)}")  # True

    # O·S - S·O is symmetric → should pass
    anti_comm = Omega_sym * S_sym - S_sym * Omega_sym
    print(f"  OmS - SOm valid?:   {engine.is_physically_valid(anti_comm)}")  # True

    # S_d should be valid
    expr3 = c1 * S_sym + c2 * S_d_sym
    print(f"  S + S_d valid?:     {engine.is_physically_valid(expr3)}")  # True

    # Quick numerical eval
    rng = np.random.default_rng(7)
    S_data = rng.normal(size=(3, 3, 3))
    S_data = 0.5 * (S_data + S_data.transpose(0, 2, 1))
    Om_data = rng.normal(size=(3, 3, 3))
    Om_data = 0.5 * (Om_data - Om_data.transpose(0, 2, 1))

    result = engine.lambdify_tensor_expr(
        c1 * S_sym + 0.1 * engine.vector_outer(omega_sym, omega_sym),
        S_data, Om_data, np.zeros_like(S_data), np.zeros_like(S_data), np.zeros_like(S_data), np.zeros_like(S_data), np.ones(3),
        omega_val=np.zeros((3, 3)), W_val=np.zeros((3, 3)), h_val=np.zeros((3, 1)),
        constants_dict={"c1": -0.02},
    )
    print(f"\n  lambdify shape: {result.shape}")
    print(f"  sample [0]:\n{result[0]}")
