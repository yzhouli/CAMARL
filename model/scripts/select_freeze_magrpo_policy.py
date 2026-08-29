#!/usr/bin/env python3
"""Select the best validation-dev MA-GRPO checkpoint and freeze its LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--pool-size",
        type=int,
        choices=(20, 50, 100, 500, 1000, 1500, 2000),
        required=True,
    )
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates: list[dict[str, Any]] = []
    for path in args.audits:
        report = json.loads(path.read_text())
        audit = report["audit"]
        policy = report["selected_policy"]
        if audit.get("split") != "validation-dev" or audit.get("test_records_used"):
            raise ValueError(f"Refusing non-validation-dev audit: {path}")
        if not audit.get("fail_on_request_error") or int(
            audit.get("request_errors", -1)
        ) != 0:
            raise ValueError(f"Refusing audit with non-strict or failed requests: {path}")
        if int(audit.get("pool_size", -1)) != args.pool_size:
            raise ValueError(
                f"Refusing audit for another task: expected N={args.pool_size}, "
                f"got {audit.get('pool_size')} in {path}"
            )
        candidates.append(
            {
                "audit_path": str(path),
                "pool_size": args.pool_size,
                "audit_sha256": sha256_file(path),
                "adapter_path": str(audit["adapter_path"]),
                "selection_score": float(audit["selection_score"]),
                "ranking_utility": float(policy["ranking_utility"]),
                "normalized_tool_cost": float(policy["normalized_tool_cost"]),
                "valid_action_rate": float(audit["valid_action_rate"]),
                "oracle_action_accuracy": float(audit["oracle_action_accuracy"]),
            }
        )
    if not candidates:
        raise ValueError("No checkpoint audits supplied")
    selected = max(
        candidates,
        key=lambda value: (
            value["selection_score"],
            value["ranking_utility"],
            -value["normalized_tool_cost"],
            value["adapter_path"],
        ),
    )
    source = Path(selected["adapter_path"])
    missing = [name for name in REQUIRED_ADAPTER_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Selected adapter missing files {missing}: {source}")
    args.frozen_dir.mkdir(parents=True, exist_ok=True)
    if any(args.frozen_dir.iterdir()):
        raise FileExistsError(
            f"Frozen directory is not empty; refusing to overwrite: {args.frozen_dir}"
        )
    copied: dict[str, str] = {}
    for name in REQUIRED_ADAPTER_FILES:
        destination = args.frozen_dir / name
        shutil.copy2(source / name, destination)
        copied[name] = sha256_file(destination)
    for optional in ("README.md", "tokenizer_config.json", "special_tokens_map.json"):
        if (source / optional).is_file():
            shutil.copy2(source / optional, args.frozen_dir / optional)
            copied[optional] = sha256_file(args.frozen_dir / optional)

    manifest = {
        "selection_split": "validation-dev",
        "task": f"N={args.pool_size}",
        "pool_size": args.pool_size,
        "cross_pool_policy_sharing": False,
        "test_records_used": False,
        "selection_rule": (
            "max selection_score, then ranking_utility, then lower normalized cost"
        ),
        "selected": selected,
        "candidates": sorted(candidates, key=lambda value: value["adapter_path"]),
        "frozen_adapter": str(args.frozen_dir),
        "frozen_files_sha256": copied,
        "policy_frozen_before_test": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
