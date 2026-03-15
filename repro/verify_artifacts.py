#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify canonical reproduction artifacts.")
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "repro" / "manifest.json"),
        help="Path to the reproduction manifest.",
    )
    parser.add_argument(
        "--groups",
        default="all",
        help="Comma-separated group list, or 'all'.",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Also require the raw input HDF5 files listed in required_inputs.",
    )
    return parser.parse_args()


def selected_groups(raw: str, manifest: dict) -> list[str]:
    available: list[str] = []
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


def csv_row_count(path: Path) -> tuple[list[str], int]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0
        return header, sum(1 for _ in reader)


def artifact_paths_for_groups(manifest: dict, groups: list[str]) -> set[str]:
    outputs: set[str] = set()
    for step in manifest["steps"]:
        if step["group"] in groups:
            outputs.update(step.get("outputs", []))
    return outputs


def main() -> int:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    groups = selected_groups(args.groups, manifest)

    failures: list[str] = []
    checked = 0

    if args.check_inputs:
        for rel in manifest.get("required_inputs", []):
            checked += 1
            path = ROOT / rel
            if path.exists():
                print(f"[ok] input {rel}")
            else:
                failures.append(f"missing input: {rel}")
                print(f"[missing] input {rel}")

    selected_outputs = artifact_paths_for_groups(manifest, groups)
    for step in manifest["steps"]:
        if step["group"] not in groups:
            continue
        for rel in step.get("outputs", []):
            checked += 1
            path = ROOT / rel
            if path.exists():
                print(f"[ok] output {rel}")
            else:
                failures.append(f"missing output: {rel}")
                print(f"[missing] output {rel}")

    for rel, spec in manifest.get("artifacts", {}).items():
        if rel not in selected_outputs:
            continue
        checked += 1
        path = ROOT / rel
        if not path.exists():
            failures.append(f"artifact missing: {rel}")
            print(f"[missing] artifact {rel}")
            continue

        if spec.get("type") != "csv":
            print(f"[skip] artifact {rel} (unsupported type {spec.get('type')})")
            continue

        header, rows = csv_row_count(path)
        missing_cols = [name for name in spec.get("required_columns", []) if name not in header]
        too_short = rows < int(spec.get("min_rows", 0))
        if missing_cols or too_short:
            detail = []
            if missing_cols:
                detail.append(f"missing columns {missing_cols}")
            if too_short:
                detail.append(f"rows {rows} < {spec['min_rows']}")
            failures.append(f"artifact invalid: {rel} ({'; '.join(detail)})")
            print(f"[invalid] artifact {rel}: {'; '.join(detail)}")
            continue

        print(f"[ok] artifact {rel} ({rows} rows)")

    if failures:
        print("\nVerification failed:")
        for failure in failures:
            print(f"  - {failure}")
        print(f"\nChecked {checked} items, {len(failures)} failed.")
        return 1

    print(f"\nVerification passed. Checked {checked} items across groups: {', '.join(groups)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
