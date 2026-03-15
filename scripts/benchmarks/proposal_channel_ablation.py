#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import sympy as sp

from corrected_research_screen import expr_key
from sgs_discovery.mcts_agent import NeuroSymbolicMCTS
from sgs_discovery.optimizer import LeafNodeOptimizer
from sgs_discovery.oracle import JHTDBOracle
from sgs_discovery.physics_env import TensorSymbolicEngine
from sgs_discovery.symbolic_closures import corrected_catalog, corrected_root_library


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/llm_ablation"
ROWS_PATH = OUT_DIR / "search_rows.csv"
SUMMARY_PATH = OUT_DIR / "search_summary.csv"

SEARCH_SEEDS = (11, 29)
ROOT_NAMES = ("Leonard", "L_WALE")
SEARCH_ITERATIONS = 3
SEARCH_MAX_DEPTH = 3
SEARCH_OPT_MAX_ITER = 140


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for channel in sorted({str(row["channel"]) for row in rows}):
        subset = [row for row in rows if str(row["channel"]) == channel]
        summary_rows.append(
            {
                "channel": channel,
                "n_runs": len(subset),
                "runtime_s_mean": float(np.mean([float(r["runtime_s"]) for r in subset])),
                "best_search_loss_mean": float(np.mean([float(r["best_search_loss"]) for r in subset])),
                "best_search_loss_std": float(np.std([float(r["best_search_loss"]) for r in subset])),
                "tree_size_mean": float(np.mean([float(r["tree_size"]) for r in subset])),
                "evaluated_count_mean": float(np.mean([float(r["evaluated_count"]) for r in subset])),
                "llm_enabled_fraction": float(np.mean([float(r["llm_enabled"]) for r in subset])),
                "llm_success_fraction": float(np.mean([float(r["llm_success"]) for r in subset])),
                "unique_best_expr_count": len({str(r["best_expr_key"]) for r in subset}),
                "known_match_fraction": float(
                    np.mean([1.0 if str(r["known_match"]).strip() else 0.0 for r in subset])
                ),
            }
        )
    return summary_rows


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    load_env_file(ROOT / ".env")

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
    http_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None

    engine = TensorSymbolicEngine()
    oracle_iso = JHTDBOracle("jhtdb_u_tensor_64.h5", filter_width=1.0, boundary_mode="wrap")
    oracle_chan = JHTDBOracle("channel_u_tensor_64.h5", filter_width=1.0, boundary_mode="nearest")
    known = {expr_key(expr): name for name, expr in corrected_catalog().items()}
    roots = [(name, expr) for name, expr in corrected_root_library() if name in ROOT_NAMES]

    channels = [
        {"name": "deterministic", "api_key": ""},
        {"name": "gemini", "api_key": gemini_api_key},
    ]

    rows: list[dict[str, object]] = []
    for channel in channels:
        for seed in SEARCH_SEEDS:
            for root_name, root_expr in roots:
                print("=" * 80, flush=True)
                print(f"[ABLATION] channel={channel['name']} root={root_name} seed={seed}", flush=True)
                np.random.seed(seed)
                prev_key = os.environ.get("GEMINI_API_KEY")
                if channel["api_key"]:
                    os.environ["GEMINI_API_KEY"] = channel["api_key"]
                else:
                    os.environ.pop("GEMINI_API_KEY", None)
                try:
                    optimizer = LeafNodeOptimizer(
                        engine,
                        [oracle_iso, oracle_chan],
                        lambda_pi=1.0,
                        lambda_diss=1.0,
                        lambda_l1=1.0e-5,
                        max_iter=SEARCH_OPT_MAX_ITER,
                    )
                    mcts = NeuroSymbolicMCTS(
                        engine=engine,
                        oracles=[oracle_iso, oracle_chan],
                        optimizer=optimizer,
                        exploration_weight=1.414,
                        max_depth=SEARCH_MAX_DEPTH,
                        gemini_api_key=channel["api_key"],
                        gemini_model=gemini_model,
                        http_proxy=http_proxy,
                    )
                    t0 = perf_counter()
                    root = mcts.search(root_expr=root_expr, n_iterations=SEARCH_ITERATIONS, log_interval=1)
                    runtime_s = perf_counter() - t0
                finally:
                    if prev_key:
                        os.environ["GEMINI_API_KEY"] = prev_key
                    else:
                        os.environ.pop("GEMINI_API_KEY", None)
                best = mcts.get_top_k(1)[0]
                row = {
                    "channel": channel["name"],
                    "root": root_name,
                    "seed": seed,
                    "runtime_s": float(runtime_s),
                    "best_search_loss": float(best.mse),
                    "best_expr": str(best.expr),
                    "best_expr_key": expr_key(best.expr),
                    "tree_size": int(mcts._tree_size(root)),
                    "evaluated_count": len(mcts._all_evaluated),
                    "llm_enabled": int(bool(mcts._llm_enabled)),
                    "llm_failures": int(mcts._llm_failures),
                    "llm_success": int(bool(channel["api_key"]) and mcts._llm_failures == 0 and mcts._llm_enabled),
                    "known_match": known.get(expr_key(best.expr), ""),
                }
                rows.append(row)
                save_rows(ROWS_PATH, rows)

    summary_rows = aggregate_rows(rows)
    save_rows(SUMMARY_PATH, summary_rows)
    print(f"Saved {ROWS_PATH}")
    print(f"Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
