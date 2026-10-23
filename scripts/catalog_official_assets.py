#!/usr/bin/env python3
"""Inventory / validate official challenge assets without importing OmniGibson.

The default operation reads every registered TRO and scene template as JSON.
It never starts Isaac, modifies assets, or claims that a scene was simulated.
Use --emit-catalog when deliberately refreshing the checked-in inventory.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION = "2026-challenge-task-instances"
METADATA_FILES = ("task.jsonl", "B100_task_misc.csv", "available_tasks.yaml")
SPLIT_DIRECTORIES = {"train": "scenes", "public_test": "scene_test/public"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_metadata(asset_root: Path) -> tuple[list[dict], dict, dict]:
    """Cross-check all three official metadata sources; never guess task IDs."""
    import yaml  # Already a dependency of the supported OmniGibson runtime.

    metadata_root = asset_root / COLLECTION / "metadata"
    tasks = [json.loads(line) for line in (metadata_root / "task.jsonl").read_text().splitlines() if line.strip()]
    rows = list(csv.DictReader((metadata_root / "B100_task_misc.csv").open(encoding="utf-8", newline="")))
    available = yaml.safe_load((metadata_root / "available_tasks.yaml").read_text())
    task_mapping = {row["task_name"]: row["task_index"] for row in tasks}
    csv_mapping = {row["Task"]: int(row["Task ID"]) for row in rows}
    if len(tasks) != 100 or len(rows) != 100 or len(task_mapping) != 100 or len(csv_mapping) != 100:
        raise ValueError("Expected exactly 100 unique official challenge tasks in both metadata sources")
    if any(type(value) is not int for value in task_mapping.values()) or set(task_mapping.values()) != set(range(100)):
        raise ValueError("Official task_index values must be the unique integers 0..99")
    if task_mapping != csv_mapping:
        raise ValueError("task.jsonl and B100_task_misc.csv disagree on task IDs / names")
    if not isinstance(available, dict) or set(available) != set(task_mapping):
        raise ValueError("available_tasks.yaml does not cover exactly the same 100 tasks")
    for name, definitions in available.items():
        if not isinstance(definitions, dict) or set(definitions) != {0}:
            raise ValueError(f"Unsupported official definition IDs for {name}: {list(definitions)}")
        if not isinstance(definitions[0].get("scene_model"), str):
            raise ValueError(f"Missing official scene_model for {name}")
    fingerprints = {
        name: {"path": f"{COLLECTION}/metadata/{name}", "sha256": sha256(metadata_root / name)}
        for name in METADATA_FILES
    }
    return sorted(tasks, key=lambda row: row["task_index"]), available, fingerprints


def instance_relative_path(task: dict, split: str, instance_id: int) -> str:
    scene, name, definition = task["scene_model"], task["task_name"], task["activity_definition_id"]
    return (
        f"{COLLECTION}/{SPLIT_DIRECTORIES[split]}/{scene}/json/"
        f"{scene}_task_{name}_instances/{scene}_task_{name}_{definition}_{instance_id}_template-tro_state.json"
    )


def build_catalog(asset_root: Path) -> dict:
    tasks, available, fingerprints = read_metadata(asset_root)
    source = json.loads((PROJECT_ROOT / "behavior_ap_sim/configs/simulator-source.json").read_text())
    entries, missing_optional = [], []
    for row in tasks:
        name, task_index = row["task_name"], row["task_index"]
        scene = available[name][0]["scene_model"]
        entry = {
            "task_index": task_index,
            "task_name": name,
            "scene_model": scene,
            "activity_definition_id": 0,
            "instances": {},
            "templates": {},
        }
        for split, folder in SPLIT_DIRECTORIES.items():
            json_directory = Path(COLLECTION) / folder / scene / "json"
            instance_directory = asset_root / json_directory / f"{scene}_task_{name}_instances"
            expression = re.compile(re.escape(f"{scene}_task_{name}_0_") + r"(\d+)_template-tro_state\.json")
            ids = []
            for path in instance_directory.glob("*-tro_state.json"):
                match = expression.fullmatch(path.name)
                if not match:
                    raise ValueError(f"Unrecognized instance filename: {path.relative_to(asset_root)}")
                ids.append(int(match.group(1)))
            ids.sort()
            expected = list(range(301, 321)) if split == "public_test" else list(range(0, 300) if task_index < 50 else range(1, 301))
            if ids != expected:
                raise ValueError(f"Incomplete / unexpected {split} instances for {name}: {ids}")
            entry["instances"][split] = ids
            full = (json_directory / f"{scene}_task_{name}_0_0_template.json").as_posix()
            partial = (json_directory / f"{scene}_task_{name}_0_0_template-partial_rooms.json").as_posix()
            if not (asset_root / full).is_file():
                raise FileNotFoundError(f"Required full scene template missing: {full}")
            partial_available = (asset_root / partial).is_file()
            entry["templates"][split] = {
                "full": full,
                "partial_rooms": partial if partial_available else None,
                "partial_rooms_expected": partial,
            }
            if not partial_available:
                missing_optional.append({"task_name": name, "split": split, "path": partial, "required_for_evaluator": False})
        # The pinned official evaluator constructs the base scene from the train
        # full template, then loads the selected split's TRO state separately.
        # partial_scene_load filters rooms; it does not select a -partial_rooms
        # file. Missing optional derived caches must not change that behavior.
        entry["runtime_scene_template"] = entry["templates"]["train"]["full"]
        entries.append(entry)
    scene_counts = collections.Counter(task["scene_model"] for task in entries)
    return {
        "schema_version": 1,
        "scope": "BEHAVIOR-1K 2026 challenge public assets: 100 tasks; not all BEHAVIOR-1K activity definitions",
        "source": {
            "repository": source["repo"],
            "omnigibson_commit": source["commit"],
            "asset_collection": COLLECTION,
            "metadata": fingerprints,
            "assets_included": False,
            "scene_loading": "Official evaluator uses the train full template for the base scene, then applies the selected split TRO; partial_scene_load filters rooms independently",
        },
        "counts": {
            "task_count": len(entries),
            "scene_model_count": len(scene_counts),
            "tasks_by_scene_model": dict(sorted(scene_counts.items())),
            "instances_by_split": {split: sum(len(task["instances"][split]) for task in entries) for split in SPLIT_DIRECTORIES},
            "total_instances": sum(len(ids) for task in entries for ids in task["instances"].values()),
            "full_templates": len(entries) * len(SPLIT_DIRECTORIES),
            "partial_room_templates": len(entries) * len(SPLIT_DIRECTORIES) - len(missing_optional),
        },
        "default_original": {"split": "public_test", "instance_id": 301},
        "unavailable_splits": {
            "hidden_test": "Official IDs 321..340 are defined by the evaluator, but private instances are not distributed here",
            "val": "No separate val split exists in this asset package",
        },
        "missing_optional_templates": missing_optional,
        "tasks": entries,
    }


def validate_catalog(asset_root: Path, catalog: dict, *, parse_payloads: bool = True) -> dict:
    """Check every catalogued path and optionally parse all 32,000 TRO payloads."""
    missing_required, missing_optional, invalid, extra = [], [], [], []
    rows, checked_json_count = [], 0
    metadata_errors = []
    try:
        official_rows, available, fingerprints = read_metadata(asset_root)
        official = {row["task_name"]: row["task_index"] for row in official_rows}
        registered = {row["task_name"]: row["task_index"] for row in catalog["tasks"]}
        if official != registered or len(catalog["tasks"]) != 100:
            metadata_errors.append("Catalog task IDs / names differ from official metadata")
        if fingerprints != catalog["source"]["metadata"]:
            metadata_errors.append("Official metadata SHA256 differs from the registered catalog")
        for task in catalog["tasks"]:
            if available[task["task_name"]][0]["scene_model"] != task["scene_model"]:
                metadata_errors.append(f"Scene model differs: {task['task_name']}")
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        metadata_errors.append(f"{type(exc).__name__}: {exc}")

    def inspect(relative: str, *, optional: bool = False, tro: bool = False) -> bool:
        nonlocal checked_json_count
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            invalid.append({"path": relative, "reason": "Catalog paths must be asset-root-relative"})
            return False
        path = asset_root / relative_path
        if not path.is_file():
            (missing_optional if optional else missing_required).append(relative)
            return False
        if parse_payloads:
            try:
                payload = json.loads(path.read_text())
                if not isinstance(payload, dict) or not payload:
                    raise ValueError("Expected a nonempty JSON object")
                if tro and not isinstance(payload.get("robot_poses"), dict):
                    raise ValueError("TRO payload has no robot_poses object")
                checked_json_count += 1
            except (OSError, ValueError) as exc:
                invalid.append({"path": relative, "reason": f"{type(exc).__name__}: {exc}"})
                return False
        return True

    for task in catalog["tasks"]:
        row = {"task_index": task["task_index"], "task_name": task["task_name"], "scene_model": task["scene_model"], "splits": {}}
        for split, ids in task["instances"].items():
            expected_paths = {instance_relative_path(task, split, instance_id) for instance_id in ids}
            valid_count = sum(inspect(path, tro=True) for path in sorted(expected_paths))
            if expected_paths:
                directory = (asset_root / next(iter(expected_paths))).parent
                extra.extend(sorted(str(path.relative_to(asset_root)) for path in directory.glob("*-tro_state.json") if str(path.relative_to(asset_root)) not in expected_paths))
            template = task["templates"][split]
            full_valid = inspect(template["full"])
            partial_expected = template["partial_rooms_expected"]
            partial_valid = inspect(partial_expected, optional=template["partial_rooms"] is None)
            row["splits"][split] = {
                "expected_instances": len(ids), "valid_instances": valid_count,
                "full_template_valid": full_valid, "partial_rooms_template_valid": partial_valid,
            }
        row["runtime_scene_template"] = task["runtime_scene_template"]
        rows.append(row)
    return {
        "valid": not (missing_required or invalid or metadata_errors or extra),
        "validation_level": "all_registered_json_parsed" if parse_payloads else "registered_file_existence_only",
        "simulator_started": False,
        "scene_execution_verified": False,
        "task_count": len(rows),
        "instance_count": sum(len(ids) for task in catalog["tasks"] for ids in task["instances"].values()),
        "json_payloads_checked": checked_json_count,
        "metadata_errors": metadata_errors,
        "missing_required_files": missing_required,
        "missing_optional_files": missing_optional,
        "invalid_files": invalid,
        "unexpected_instance_files": extra,
        "tasks": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path(os.environ.get("BEHAVIOR_ASSET_ROOT", os.environ.get("OMNIGIBSON_DATA_PATH", "/root/download/datasets/BEHAVIOR-1K"))))
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "behavior_ap_sim/configs/official_tasks.json")
    parser.add_argument("--emit-catalog", action="store_true", help="Print a fresh inventory to stdout; does not alter any file")
    parser.add_argument("--skip-payload-check", action="store_true", help="Only check metadata and paths; do not parse scene / TRO payloads")
    parser.add_argument("--report", type=Path, help="Additionally save the validation JSON at this path")
    args = parser.parse_args()
    try:
        if args.emit_catalog:
            print(json.dumps(build_catalog(args.asset_root), indent=2, ensure_ascii=False))
            return 0
        catalog = json.loads(args.catalog.read_text())
        report = validate_catalog(args.asset_root, catalog, parse_payloads=not args.skip_payload_check)
        serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(serialized)
        print(serialized, end="")
        return 0 if report["valid"] else 1
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(json.dumps({"valid": False, "error": f"{type(exc).__name__}: {exc}", "simulator_started": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
