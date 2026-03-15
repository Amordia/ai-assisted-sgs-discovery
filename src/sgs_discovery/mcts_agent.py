"""
mcts_agent.py — Neuro-Symbolic Monte Carlo Tree Search for SGS Closure
=======================================================================
Combines UCT-based tree search, optional Gemini proposals, deterministic
fallback proposals, and physics-informed pruning to discover SGS closure
expressions.

LLM backend
-----------
Pass ``gemini_api_key=...`` to ``NeuroSymbolicMCTS`` or set the
environment variable ``GEMINI_API_KEY``. When neither is available
the agent falls back to hard-coded proposals automatically; this is the
canonical reproducible path used in the tracked paper artifacts.

Node value:  V = 1 / (1 + MSE)

Expansion candidates that fail ``physics_env.is_physically_valid()``
are pruned immediately.
"""

from __future__ import annotations

import math
import os
import re
import json
import warnings
import ssl
from dataclasses import dataclass, field
from typing import Any

# Bypass strict SSL checking for proxy environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import numpy as np
import sympy as sp
from sympy import Trace

from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.physics_env import (
    TensorSymbolicEngine,
    S_sym,
    Omega_sym,
    L_sym,
    S_d_sym,
    S_j_sym,
    Lap_S_sym,
    omega_sym,
    W_sym,
    h_sym,
    c_syms,
    c1, c2, c3, c4, c5,
)
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.symbolic_closures import corrected_term_dictionary

# New official Gemini SDK (google-genai)
try:
    from google import genai as genai_sdk
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# -----------------------------------------------------------------------
# MCTS Node
# -----------------------------------------------------------------------

@dataclass
class Node:
    """
    A single node in the MCTS search tree.

    Attributes
    ----------
    expr : sp.MatrixExpr
        The current candidate SymPy tensor expression.
    parent : Node or None
        Parent node (None for the root).
    children : list[Node]
        Expanded child nodes.
    N : int
        Visit count.
    Q : float
        Cumulative reward (value sum).
    best_constants : dict[str, float]
        Best-fit constants found by the optimiser for this expression.
    mse : float
        MSE achieved with `best_constants`. ``inf`` if not yet evaluated.
    is_terminal : bool
        Whether the node has been fully expanded.
    """
    expr:            sp.MatrixExpr
    parent:          Node | None      = None
    children:        list[Node]       = field(default_factory=list)
    N:               int              = 0
    Q:               float            = 0.0
    best_constants:  dict[str, float] = field(default_factory=dict)
    mse:             float            = float("inf")
    is_terminal:     bool             = False

    # convenience
    @property
    def value(self) -> float:
        """V = -loss  (higher is better, preserves ranking).
        
        Since the optimizer returns total_loss where lower = better,
        negating it gives a reward where higher = better, which is
        what UCT needs.  No saturation — UCT can distinguish
        -26.3 (better) from -26.1 (worse).
        """
        if self.mse == float("inf"):
            return 0.0
        return -self.mse

    @property
    def avg_Q(self) -> float:
        return self.Q / self.N if self.N > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"Node(expr={self.expr}, N={self.N}, "
            f"Q={self.Q:.4f}, MSE={self.mse:.4e})"
        )


# -----------------------------------------------------------------------
# MCTS Agent
# -----------------------------------------------------------------------

class NeuroSymbolicMCTS:
    """
    Monte Carlo Tree Search for symbolic regression of SGS closures.

    Parameters
    ----------
    engine : TensorSymbolicEngine
    oracle : FluidDataOracle
    optimizer : LeafNodeOptimizer
    exploration_weight : float
        UCT exploration constant  C_p  (default √2).
    max_depth : int
        Maximum expression-tree depth.  Deeper nodes become terminal.
    gemini_api_key : str or None
        Gemini API key.  Falls back to env var ``GEMINI_API_KEY``.
        If neither is set, uses hard-coded fallback proposals.
    gemini_model : str
        Gemini model ID (default: ``gemini-3.1-flash-lite-preview``).
    http_proxy : str or None
        Optional HTTP/HTTPS proxy URL, e.g. ``"http://127.0.0.1:7890"``.
        Falls back to env vars ``HTTPS_PROXY`` / ``HTTP_PROXY``.
        Required when running from a geo-restricted region (e.g. mainland
        China).  Set in ``.env``::  HTTPS_PROXY=http://127.0.0.1:7890
    """

    def __init__(
        self,
        engine: TensorSymbolicEngine,
        oracles: list[JHTDBOracle],
        optimizer: LeafNodeOptimizer,
        exploration_weight: float = 1.414,
        max_depth: int = 4,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-3.1-flash-lite-preview",
        http_proxy: str | None = None,
    ) -> None:
        self.engine = engine
        self.oracles = oracles
        self.optimizer = optimizer
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth

        # ── Gemini LLM setup ──────────────────────────────────────────
        api_key   = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        proxy_url = (
            http_proxy
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY")
            or ""
        )
        self._llm_failures: int = 0    # always init, regardless of path
        self._llm_enabled:  bool = False
        self._gemini_client: Any = None
        self._gemini_model_name: str = gemini_model

        if api_key and _GENAI_AVAILABLE:
            try:
                # google-genai respects HTTPS_PROXY env var automatically.
                if proxy_url:
                    os.environ["HTTPS_PROXY"] = proxy_url
                    os.environ["HTTP_PROXY"]  = proxy_url
                    print(f"[LLM]  Routing via proxy: {proxy_url}")

                self._gemini_client = genai_sdk.Client(api_key=api_key, http_options={'timeout': 120000.0})
                self._llm_enabled = True
                print(f"[LLM]  Gemini '{gemini_model}' ready ✓")
            except Exception as e:
                warnings.warn(f"[LLM]  Gemini init failed ({e}). Using fallback.")
        else:
            if not _GENAI_AVAILABLE:
                warnings.warn("[LLM]  google-genai not installed. Using fallback.")
            else:
                print("[LLM]  No API key — using hard-coded fallback proposals.")

        # Global record of all evaluated nodes (for top-k retrieval)
        self._all_evaluated: list[Node] = []

    # ------------------------------------------------------------------
    # 1. Selection — UCT
    # ------------------------------------------------------------------

    def _uct_score(self, node: Node, parent_visits: int) -> float:
        """Upper Confidence Bound for Trees."""
        if node.N == 0:
            return float("inf")  # always explore unvisited
        exploit = node.avg_Q
        explore = self.exploration_weight * math.sqrt(
            math.log(parent_visits) / node.N
        )
        return exploit + explore

    def _select(self, node: Node) -> Node:
        """Walk down the tree, choosing the child with highest UCT."""
        while node.children and not node.is_terminal:
            node = max(
                node.children,
                key=lambda ch: self._uct_score(ch, node.N),
            )
        return node

    # ------------------------------------------------------------------
    # 2. Expansion — Gemini LLM (with hard-coded fallback)
    # ------------------------------------------------------------------

    # System context sent once per session
    _SYSTEM_PROMPT = (
        "You are an expert in computational fluid dynamics (CFD), turbulence "
        "physics, and applied mathematics. Your task is to assist a Monte Carlo "
        "Tree Search (MCTS) algorithm in discovering analytical closure equations "
        "for the subgrid-scale (SGS) stress tensor (tau_ij) in Large Eddy Simulation (LES).\n\n"
        "Known physical quantities (Input features as SymPy MatrixSymbols):\n"
        "  S      : Strain rate tensor (symmetric, 3x3) — represents local shear\n"
        "  Omega  : Rotation rate tensor (antisymmetric, 3x3) — represents local rotation rate\n"
        "  L      : Leonard stress tensor (symmetric, 3x3) — represents SPATIAL inter-scale structure\n"
        "  S_d    : WALE tensor (symmetric, traceless, 3x3) — built from velocity gradient square. "
        "It naturally damps out as O(y^3) near solid walls. Represents intrinsic WALL-DISTANCE sensing.\n"
        "  S_j    : Jaumann Objective Rate (symmetric, 3x3) — The materially-objective time derivative of S. "
        "It incorporates the unsteady rate dS/dt, the convective transport u*dS/dx, and co-rotational spin S*Omega - Omega*S. "
        "It provides true Galilean-invariant TEMPORAL MEMORY mapping flow history.\n"
        "  Lap_S  : Laplacian of Strain Rate (symmetric, 3x3) — High-order spatial derivative defined as div(grad(S)). "
        "It captures spatial curvature and non-local momentum diffusion accurately.\n"
        "  omega  : Vorticity vector (3x1) — spatial curl of velocity, independent from pure Omega\n"
        "  W      : Vortex stretching vector (3x1) — defined as W = S * omega\n"
        "  h      : Helicity scalar — topological knotting defined as u * omega\n"
        "  Delta  : Filter width (positive scalar)\n"
        "  c1..c8 : Unknown scalar constants (MAXIMUM 8 allowed)\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "CRITICAL CONSTRAINTS (VIOLATION = AUTOMATIC REJECTION):\n"
        "═══════════════════════════════════════════════════════════════\n"
        "1. CONSTANT LIMIT: You MUST use AT MOST 8 distinct constants (c1 to c8 ONLY).\n"
        "2. TOPOLOGICAL BREAKTHROUGH: You are highly encouraged to invent new tensors by combining vectors! "
        "According to the Cayley-Hamilton theorem, S and Omega can only produce 10 unique bases. To discover truly "
        "new turbulence closures, use outer products of vorticity and vortex stretching vectors. Construct terms like "
        "`omega * omega.T` or `W * W.T` or `(omega * W.T + W * omega.T)`. Note that in SymPy, `v * v.T` creates a 3x3 matrix.\n"
        "3. HIGH-ORDER INVARIANTS: You are encouraged to invent new invariants. Beyond Tr(S**2) and Tr(Omega**2), "
        "consider high-order scalar multipliers like Tr(S**3), Tr(L*S), h, or Tr(S*Omega**2).\n"
        "4. BEYOND-HUMAN NONLINEAR TENSORS: Combine core tensors in high-order ways. E.g., L*S*Omega - Omega*S*L, "
        "or S**2*L + L*S**2, or W * W.T * h. The goal is to discover terms human physicists haven't thought of.\n"
        "5. SYMMETRY: The target tau_ij is symmetric. Outer products `v * v.T` are symmetric. If you use Omega, ensure "
        "the term is symmetric (e.g., Omega*S - S*Omega).\n"
        "6. DIMENSIONAL HOMOGENEITY: Every additive term must scale like SGS stress. Use explicit powers of Delta when needed, "
        "for example `Delta**2 * S_d`, `Delta**2 * S_j`, `Delta**4 * (Lap_S*S + S*Lap_S)`, or "
        "`Delta**2/(Tr(S**2)+eps) * (W * W.T)`. Bare `S_j`, bare `Lap_S`, or bare `W*W.T` are invalid.\n"
        "7. SymPy SYNTAX: Use * for multiplication, Tr() for trace. For outer products of vectors, use `.T` for transpose (e.g., `omega * omega.T`). No @ operator.\n"
        "8. OUTPUT: Return ONLY a strict JSON array of 5 SymPy expression strings.\n"
    )

    _USER_TEMPLATE = (
        "The current best candidate formula in our search tree is:\n"
        "  {expr}\n\n"
        "REMINDER: You have AT MOST 8 constants (c1-c8). No more.\n\n"
        "Please propose exactly 5 different mathematical mutations. Do not be afraid to generate\n"
        "highly complex, high-order 'BEYOND HUMAN' tensor combinations. You can anchor on L,\n"
        "use S_d for wall damping, and explicitly incorporate outer products like `W * W.T` or `omega * omega.T`\n"
        "to break the Cayley-Hamilton algebraic constraints.\n\n"
        "Example Output Format:\n"
        '[\n'
        '  "c1 * L + c2 * Delta**2 * S_d",\n'
        '  "c1 * L + c2 * Delta**2 * S_j + c3 * Delta**2 * S_d",\n'
        '  "c1 * L + c2 * Delta**2 * (((Trace(S_d*S_d) + 1.0e-30)**(3/2))/((Trace(S*S) + 1.0e-30)**(5/2) + (Trace(S_d*S_d) + 1.0e-30)**(5/4))) * S + c3 * Delta**2 * S_j",\n'
        '  "c1 * L + c2 * Delta**2 * (((Trace(S_d*S_d) + 1.0e-30)**(3/2))/((Trace(S*S) + 1.0e-30)**(5/2) + (Trace(S_d*S_d) + 1.0e-30)**(5/4))) * S + c3 * Delta**2/(Trace(S*S) + 1.0e-30) * (W * W.T)",\n'
        '  "c1 * L + c2 * Delta**2 * S_d + c3 * Delta**4 * (Lap_S * S + S * Lap_S)"\n'
        ']\n\n'
        "Now, please provide your 5 proposed derivations as a strict JSON array:"
    )

    def llm_propose_next_steps(
        self, current_expr: sp.MatrixExpr, depth: int,
    ) -> list[sp.MatrixExpr]:
        """
        Ask Gemini to propose new Galilean-invariant tensor additions.

        When ``self._llm_enabled`` is True, sends a structured prompt to
        the Gemini API and parses the returned JSON list of SymPy
        expression strings.  Falls back to hard-coded proposals on any
        error (network failure, parse error, timeout).

        Parameters
        ----------
        current_expr : sp.MatrixExpr
            The current candidate closure formula.
        depth : int
            Current depth in the MCTS tree (used to pick fresh constants).

        Returns
        -------
        list of sp.MatrixExpr
            Candidate expressions (may be empty if all fail validity).
        """
        if self._llm_enabled:
            try:
                result = self._gemini_propose(current_expr, depth)
                self._llm_failures = 0              # reset on success
                return result
            except Exception as e:
                self._llm_failures += 1
                err_str = str(e)

                # Categorise error for user-friendly messages
                if "location" in err_str.lower() or "FAILED_PRECONDITION" in err_str:
                    msg = ("[LLM]  Geo-restriction: Gemini API unavailable without a "
                           "working proxy.  Add/fix HTTPS_PROXY in .env.")
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    msg = (f"[LLM]  Model not found (404).  "
                           f"Check model name '{self._gemini_model_name}'.")
                else:
                    msg = f"[LLM]  API error: {err_str[:120]}"

                # Disable permanently after 10 consecutive failures
                if self._llm_failures >= 10:
                    self._llm_enabled = False
                    warnings.warn(msg + f"  ({self._llm_failures} failures → LLM disabled permanently)")
                else:
                    warnings.warn(msg + f"  (failure {self._llm_failures}/10, using fallback)")

        return self._fallback_proposals(current_expr, depth)

    def _gemini_propose(
        self, current_expr: sp.MatrixExpr, depth: int,
    ) -> list[sp.MatrixExpr]:
        """
        Core Gemini API call + SymPy parse + physics validation.
        """
        # Determine next free constant index
        used_indices = [
            i for i, ci in enumerate(c_syms, start=1)
            if ci in current_expr.free_symbols
        ]

        prompt = self._USER_TEMPLATE.format(
            expr=str(current_expr),
        )

        # New google-genai SDK call with High Thinking Level
        response = self._gemini_client.models.generate_content(
            model=self._gemini_model_name,
            contents=[self._SYSTEM_PROMPT + "\n\n" + prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.7,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_level="high"  # 或使用 genai_types.ThinkingLevel.HIGH
                ),
                max_output_tokens=8192, 
            ),
        )
        raw_text: str = response.text.strip()
        print(f"[LLM]  Gemini raw ({len(raw_text)} chars):\n{raw_text}")

        # ── Strip thinking-mode XML blocks (<thinking>...</thinking>)
        clean = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL)
        # ── Strip markdown code fences (```json ... ``` or ``` ... ```)
        clean = re.sub(r'```[a-z]*\n?(.*?)```', r'\1', clean, flags=re.DOTALL)
        clean = clean.strip()

        # ── Extract the JSON array
        
        # First try strict json parsing
        json_match = re.search(r'\[.*\]', clean, re.DOTALL)
        candidates_str: list[str] = []
        if json_match:
            try:
                candidates_str = json.loads(json_match.group())
            except Exception:
                pass
        
        # Fallback: if json parsing failed (e.g. truncated response string)
        if not candidates_str:
            matches = re.finditer(r'"(.*?)"(?=\s*,|\s*$|\s*\])', clean, re.DOTALL)
            for m in matches:
                expr = m.group(1).strip()
                if expr and len(expr) > 2:
                    candidates_str.append(expr)

        if not candidates_str:
            raise ValueError("No valid math expression strings found in Gemini response.")
        from sgs_discovery.physics_env import L_sym, Delta_sym, S_d_sym, S_j_sym, Lap_S_sym, omega_sym, W_sym, h_sym
        # Symbols available in eval namespace
        S = S_sym
        Omega = Omega_sym
        L = L_sym
        S_d = S_d_sym
        S_j = S_j_sym
        Lap_S = Lap_S_sym
        Delta = Delta_sym
        omega = omega_sym
        W = W_sym
        h = h_sym
        def Tr(x: sp.MatrixExpr) -> sp.Expr:
            return Trace(x)
        ns = {
            "S": S, "Omega": Omega, "L": L, "S_d": S_d, "S_j": S_j, "Lap_S": Lap_S, "Delta": Delta, 
            "omega": omega, "W": W, "h": h, 
            "Tr": Tr, "sp": sp,
            "sqrt": sp.sqrt,
            "Abs": sp.Abs,
            "exp": sp.exp,
            "log": sp.log,
            "Matrix": getattr(sp, "Matrix", sp.Matrix),
            "Trace": Trace,
            "eye": getattr(sp, "eye", sp.eye),
            "Rational": getattr(sp, "Rational", sp.Rational),
            **{ci.name: ci for ci in c_syms},
        }

        valid_exprs: list[sp.MatrixExpr] = []
        for expr_str in candidates_str:
            try:
                # Replace @ (Python matrix mul) with * for SymPy
                sympy_str = expr_str.replace("@", "*")
                candidate = eval(sympy_str, ns)  # noqa: S307
                valid_exprs.append(candidate)
            except Exception as parse_err:
                warnings.warn(f"[LLM]  Parse error for '{expr_str}': {parse_err}")

        if not valid_exprs:
            raise ValueError("No parseable expressions from Gemini.")

        return valid_exprs

    def _fallback_proposals(
        self, current_expr: sp.MatrixExpr, depth: int,
    ) -> list[sp.MatrixExpr]:
        """
        Hard-coded corrected-aware proposals.

        The archived fallback space was too narrow and largely incapable of
        rediscovering the mixed Leonard/WALE family that remains promising
        after the corrected wall-normal metric handling. We therefore bias the
        fallback towards:
          * structural anchoring via Leonard stress,
          * wall-aware eddy-viscosity via canonical WALE,
          * trace-WALE and objective-memory add-ons,
          * symmetric topological outer products for isotropic recovery.
        """
        terms = corrected_term_dictionary()
        wale_like = terms["WALE_like"]
        wale_canonical = terms["WALE_canonical"]
        jaumann = terms["Jaumann"]
        omega_outer = terms["omega_outer"]
        W_outer = terms["Wstretch"]
        omega_W_sym = terms["omega_W"]
        lap_term = terms["Laplacian_strain"]

        unused = [ci for ci in c_syms if ci not in current_expr.free_symbols]
        if not unused:
            unused = [c_syms[-1]]

        def pick(idx: int) -> sp.Symbol:
            return unused[min(idx, len(unused) - 1)]

        c_a = pick(0)
        c_b = pick(1)
        c_c = pick(2)

        proposals = [
            current_expr + c_a * L_sym,
            current_expr + c_a * wale_canonical,
            current_expr + c_a * wale_like,
            current_expr + c_a * jaumann,
            current_expr + c_a * omega_outer,
            current_expr + c_a * W_outer,
            current_expr + c_a * omega_W_sym,
            current_expr + c_a * lap_term,
            current_expr + c_a * wale_canonical + c_b * wale_like,
            current_expr + c_a * wale_canonical + c_b * jaumann,
            current_expr + c_a * wale_canonical + c_b * omega_outer,
            current_expr + c_a * wale_canonical + c_b * W_outer,
            current_expr + c_a * wale_canonical + c_b * omega_W_sym,
            current_expr + c_a * wale_canonical + c_b * wale_like + c_c * jaumann,
        ]

        unique: list[sp.MatrixExpr] = []
        seen: set[str] = set()
        for expr in proposals:
            key = str(sp.expand(expr))
            if key not in seen:
                seen.add(key)
                unique.append(expr)
        return unique

    def _expand(self, node: Node, depth: int) -> list[Node]:
        """
        Generate child nodes.  Only candidates that pass the
        physics validity check become children.
        """
        if depth >= self.max_depth:
            node.is_terminal = True
            return []

        proposals = self.llm_propose_next_steps(node.expr, depth)
        new_children: list[Node] = []

        for prop in proposals:
            if self.engine.is_physically_valid(prop):
                child = Node(expr=prop, parent=node)
                new_children.append(child)

        node.children.extend(new_children)
        if not new_children:
            node.is_terminal = True

        return new_children

    # ------------------------------------------------------------------
    # 3. Simulation (evaluate leaf)
    # ------------------------------------------------------------------

    def _simulate(self, node: Node) -> float:
        """
        Evaluate the candidate expression:
            1. Optimise scalar constants via LeafNodeOptimizer.
            2. Compute value V = 1/(1 + MSE).
        """
        best_consts, mse = self.optimizer.optimize(node.expr)
        node.best_constants = best_consts
        node.mse = mse
        self._all_evaluated.append(node)
        return node.value

    # ------------------------------------------------------------------
    # 4. Back-propagation
    # ------------------------------------------------------------------

    @staticmethod
    def _backpropagate(node: Node, value: float) -> None:
        """Propagate the simulation value up to the root."""
        current: Node | None = node
        while current is not None:
            current.N += 1
            current.Q += value
            current = current.parent

    # ------------------------------------------------------------------
    # 5. Main search loop
    # ------------------------------------------------------------------

    def search(
        self,
        root_expr: sp.MatrixExpr,
        n_iterations: int = 50,
        log_interval: int = 10,
    ) -> Node:
        """
        Run *n_iterations* of MCTS starting from *root_expr*.

        Parameters
        ----------
        root_expr : sp.MatrixExpr
            Initial closure expression (e.g. c1 * S).
        n_iterations : int
            Number of MCTS iterations.
        log_interval : int
            Print a log line every *log_interval* iterations.

        Returns
        -------
        root : Node
            The root of the search tree (full tree accessible via children).
        """
        root = Node(expr=root_expr)

        # Evaluate the root itself
        self._simulate(root)
        root.N = 1
        root.Q = root.value
        print(
            f"[MCTS] Root | expr: {root.expr} | "
            f"Loss: {root.mse:.6e} | V: {root.value:.4f}"
        )

        for it in range(1, n_iterations + 1):
            # 1) Select
            leaf = self._select(root)

            # 2) Expand
            depth = self._node_depth(leaf)
            new_children = self._expand(leaf, depth)

            # 3) Simulate all newly expanded children so proposal ordering
            # does not dominate the search. If expansion produced no new
            # children, fall back to re-evaluating the selected leaf.
            if new_children:
                for child in new_children:
                    value = self._simulate(child)
                    self._backpropagate(child, value)
            else:
                value = self._simulate(leaf)
                self._backpropagate(leaf, value)

            # Logging
            if it % log_interval == 0 or it == 1:
                best = self.get_top_k(1)[0]
                print(
                    f"[MCTS] Iter {it:>3d}/{n_iterations} | "
                    f"best Loss so far: {best.mse:.6e} | "
                    f"tree size: {self._tree_size(root)} | "
                    f"expr: {best.expr}"
                )

        return root

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _node_depth(node: Node) -> int:
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    @staticmethod
    def _tree_size(node: Node) -> int:
        count = 1
        for ch in node.children:
            count += NeuroSymbolicMCTS._tree_size(ch)
        return count

    def get_top_k(
        self,
        k: int = 3,
        only_valid: bool = True,
    ) -> list[Node]:
        """
        Return the *k* nodes with the lowest MSE from all evaluated
        candidates.  If *only_valid* is True (default), only nodes
        whose expression passes ``is_physically_valid`` are included.
        """
        candidates = [
            n for n in self._all_evaluated
            if n.mse < float("inf")
            and (not only_valid or self.engine.is_physically_valid(n.expr))
        ]
        candidates.sort(key=lambda n: n.mse)
        return candidates[:k]

    def __repr__(self) -> str:
        llm_status = "Gemini" if self._llm_enabled else "fallback"
        return (
            f"NeuroSymbolicMCTS("
            f"exploration={self.exploration_weight}, "
            f"max_depth={self.max_depth}, "
            f"llm={llm_status})"
        )


# -----------------------------------------------------------------------
# Quick smoke-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    # Load .env if present
    _env = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env):
        for _ln in open(_env):
            _ln = _ln.strip()
            if _ln and not _ln.startswith("#") and "=" in _ln:
                _k, _v = _ln.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    oracle = FluidDataOracle(n_samples=50, seed=42)
    engine = TensorSymbolicEngine()
    opt = LeafNodeOptimizer(engine, oracle)
    mcts = NeuroSymbolicMCTS(
        engine, oracle, opt,
        max_depth=2,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        # Uncomment and set your proxy port if in a geo-restricted region:
        http_proxy="http://127.0.0.1:7897",
    )
    print(mcts)

    root = mcts.search(c1 * S_sym, n_iterations=5, log_interval=1)

    print("\n=== Top-3 closures ===")
    for rank, node in enumerate(mcts.get_top_k(3), start=1):
        print(
            f"  #{rank}  MSE={node.mse:.6e}  "
            f"constants={node.best_constants}  "
            f"expr={node.expr}"
        )
