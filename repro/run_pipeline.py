#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = ["tests", "benchmarks", "solver", "figures"]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical reproduction pipeline.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "repro" / "manifest.json"),
        help="Path to the reproduction manifest.",
    )
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help="Comma-separated group list, or 'all'.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a step if all of its declared outputs already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the selected steps without executing them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed step instead of stopping immediately.",
    )
    return parser.parse_args()


def selected_groups(raw: str, manifest: dict) -> list[str]:
    available = []
    for step in manifest["steps"]:
        group = step["group"]
        if group not in available:
            available.append(group)
    if raw.strip().lower() == "all":
        return available
    groups = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [group for group in groups if group not in available]
    if unknown:
        raise ValueError(f"unknown groups: {unknown}; available groups: {available}")
    return groups


def outputs_exist(step: dict) -> bool:
    outputs = [ROOT / rel for rel in step.get("outputs", [])]
    return bool(outputs) and all(path.exists() for path in outputs)


def materialize_command(step: dict) -> list[str]:
    cmd = list(step["command"])
    if cmd and cmd[0] == "python":
        return [sys.executable, *cmd[1:]]
    return cmd


def run_step(step: dict) -> None:
    for env_name in step.get("requires_env", []):
        if not os.environ.get(env_name, "").strip():
            raise RuntimeError(f"{step['name']} requires environment variable {env_name}")
    cmd = materialize_command(step)
    cwd = ROOT / step.get("cwd", ".")
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    print(f"[run] {step['name']}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    groups = selected_groups(args.groups, manifest)
    steps = [step for step in manifest["steps"] if step["group"] in groups]

    if args.list:
        for step in steps:
            print(f"{step['group']:>10}  {step['name']}")
        return 0

    failures: list[str] = []
    for step in steps:
        if args.skip_existing and outputs_exist(step):
            print(f"[skip] {step['name']} (all outputs present)")
            continue
        try:
            run_step(step)
        except Exception as exc:  # pragma: no cover - CLI failure path
            print(f"[fail] {step['name']}: {exc}", file=sys.stderr)
            failures.append(step["name"])
            if not args.keep_going:
                return 1

    if failures:
        print(f"Completed with failures: {', '.join(failures)}", file=sys.stderr)
        return 1

    print("Reproduction pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
