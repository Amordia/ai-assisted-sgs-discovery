from __future__ import annotations

import sympy as sp

from sgs_discovery.physics_env import (
    Delta_sym,
    L_sym,
    Lap_S_sym,
    S_d_sym,
    S_j_sym,
    S_sym,
    W_sym,
    omega_sym,
    c1,
    c2,
    c3,
    c4,
)


def strain_sq() -> sp.Expr:
    return sp.Trace(S_sym * S_sym)


def strain_mag() -> sp.Expr:
    return sp.sqrt(2 * strain_sq())


def wale_viscosity() -> sp.Expr:
    s_sq = strain_sq()
    sd_sq = sp.Trace(S_d_sym * S_d_sym)
    base = sd_sq ** sp.Rational(3, 2) / (
        s_sq ** sp.Rational(5, 2) + sd_sq ** sp.Rational(5, 4)
    )
    return (Delta_sym ** 2) * base


def leonard_term() -> sp.MatrixExpr:
    return L_sym


def smagorinsky_like_term() -> sp.MatrixExpr:
    return (Delta_sym ** 2) * strain_mag() * S_sym


def wale_like_term() -> sp.MatrixExpr:
    return (Delta_sym ** 2) * S_d_sym


def wale_canonical_term() -> sp.MatrixExpr:
    return wale_viscosity() * S_sym


def jaumann_term() -> sp.MatrixExpr:
    return (Delta_sym ** 2) * S_j_sym


def laplacian_strain_term() -> sp.MatrixExpr:
    return (Delta_sym ** 4) * (Lap_S_sym * S_sym + S_sym * Lap_S_sym)


def omega_outer_term() -> sp.MatrixExpr:
    return (Delta_sym ** 2) * (omega_sym * omega_sym.T)


def wstretch_term() -> sp.MatrixExpr:
    return (Delta_sym ** 2 / strain_sq()) * (W_sym * W_sym.T)


def omega_w_term() -> sp.MatrixExpr:
    return (Delta_sym ** 2 / sp.sqrt(strain_sq())) * (
        omega_sym * W_sym.T + W_sym * omega_sym.T
    )


def native_model_exprs() -> dict[str, sp.MatrixExpr]:
    return {
        "Zero": sp.zeros(3),
        "Leonard": c1 * leonard_term(),
        "WALE_like": c1 * wale_like_term(),
        "Smagorinsky_like": c1 * smagorinsky_like_term(),
        "WALE_canonical": c1 * wale_canonical_term(),
        "Mixed_L_WALE_like": c1 * leonard_term() + c2 * wale_like_term(),
        "Jaumann_hybrid": c1 * leonard_term() + c2 * wale_canonical_term() + c3 * jaumann_term(),
        "Champion": c1 * leonard_term() + c2 * wale_like_term() + c3 * laplacian_strain_term(),
        "Wstretch_hybrid": c1 * leonard_term() + c2 * wale_canonical_term() + c3 * wale_like_term() + c4 * wstretch_term(),
    }


def corrected_root_library() -> list[tuple[str, sp.MatrixExpr]]:
    return [
        ("Leonard", c1 * leonard_term()),
        ("L_WALE", c1 * leonard_term() + c2 * wale_canonical_term()),
        ("L_WALE_Jaumann", c1 * leonard_term() + c2 * wale_canonical_term() + c3 * jaumann_term()),
        ("L_WALE_Sd", c1 * leonard_term() + c2 * wale_canonical_term() + c3 * wale_like_term()),
        ("L_WALE_Wstretch", c1 * leonard_term() + c2 * wale_canonical_term() + c3 * wstretch_term()),
    ]


def corrected_catalog() -> dict[str, sp.MatrixExpr]:
    return {
        "Champion": native_model_exprs()["Champion"],
        "Jaumann_hybrid": native_model_exprs()["Jaumann_hybrid"],
        "Wstretch_hybrid": native_model_exprs()["Wstretch_hybrid"],
        "L_WALE_Sd_Jaumann": c1 * leonard_term() + c2 * wale_canonical_term() + c3 * wale_like_term() + c4 * jaumann_term(),
        "L_WALE_Sd_omegaW": c1 * leonard_term() + c2 * wale_canonical_term() + c3 * wale_like_term() + c4 * omega_w_term(),
    }


def corrected_term_dictionary() -> dict[str, sp.MatrixExpr]:
    return {
        "L": leonard_term(),
        "WALE_like": wale_like_term(),
        "Smagorinsky_like": smagorinsky_like_term(),
        "WALE_canonical": wale_canonical_term(),
        "Jaumann": jaumann_term(),
        "Laplacian_strain": laplacian_strain_term(),
        "omega_outer": omega_outer_term(),
        "Wstretch": wstretch_term(),
        "omega_W": omega_w_term(),
    }
