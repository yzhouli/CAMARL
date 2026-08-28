#!/usr/bin/env python3
"""Audit and aggregate five independent CAMARL train/validation/test seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = {13, 21, 34, 55, 89}
EFFICIENCY_KEYS = (
    "call_rate_any_expert",
    "expert_calls_per_query",
    "model_requests_per_query",
    "prompt_tokens_per_query",
    "completion_tokens_per_query",
    "total_tokens_per_query",
    "latency_median_seconds",
    "latency_p95_seconds",
    "profile_memory_hit_rate_mean",
    "normalized_unit_tool_cost",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, choices=(20, 50, 100, 500), required=True)
    parser.add_argument("--inputs", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pool_key = str(args.pool_size)
    seed_summaries: list[dict[str, Any]] = []
    seen: set[int] = set()
    protocol_hashes: set[str] = set()
    for path in args.inputs:
        source = json.loads(path.read_text())
        audit = source["audit"]
        if audit["split"] != "test":
            raise ValueError(f"Non-test result: {path}")
        if int(audit["trained_pool_size"]) != args.pool_size:
            raise ValueError(f"Wrong trained pool in {path}")
        if list(map(int, audit["pool_sizes"])) != [args.pool_size]:
            raise ValueError(f"Cross-pool result in {path}")
        if audit["relation_scope"] != "prefix_aggregate":
            raise ValueError(f"Wrong relation scope in {path}")
        if not audit["policy_frozen_before_test"]:
            raise ValueError(f"Unfrozen policy in {path}")
        if audit["test_records_used_for_training_or_selection"]:
            raise ValueError(f"Test leakage flag in {path}")
        seeds = source["seeds"]
        if len(seeds) != 1:
            raise ValueError(f"Expected one seed in {path}")
        seed = int(seeds[0]["seed"])
        if seed in seen:
            raise ValueError(f"Duplicate seed {seed}")
        seen.add(seed)
        pool = seeds[0]["pools"][pool_key]
        if len(pool["ranks"]) != 1856:
            raise ValueError(f"Incomplete ranks in {path}")
        diagnostics = pool["diagnostics"]
        if int(diagnostics["request_errors"]) != 0:
            raise ValueError(f"Request errors in {path}")
        protocol_hashes.add(str(audit["graphhard_report_sha256"]))
        efficiency = {
            key: float(diagnostics[key]) for key in EFFICIENCY_KEYS
        }
        for expert, value in diagnostics["expert_call_rate"].items():
            efficiency[f"{expert}_call_rate"] = float(value)
        seed_summaries.append(
            {
                "seed": seed,
                "source": str(path),
                "source_sha256": sha256_file(path),
                "metrics": {key: float(value) for key, value in pool["metrics"].items()},
                "efficiency": efficiency,
            }
        )

    if seen != EXPECTED_SEEDS:
        raise ValueError(f"Seed mismatch: got {sorted(seen)}")
    if len(protocol_hashes) != 1:
        raise ValueError(f"Protocol mismatch: {protocol_hashes}")
    seed_summaries.sort(key=lambda item: item["seed"])
    metric_names = seed_summaries[0]["metrics"]
    efficiency_names = seed_summaries[0]["efficiency"]
    output = {
        "audit": {
            "method": "CAMARL",
            "pool_size": args.pool_size,
            "seeds": sorted(seen),
            "sample_standard_deviation": True,
            "records_per_seed": 1856,
            "split": "test",
            "relation_scope": "prefix_aggregate",
            "graphhard_report_sha256": next(iter(protocol_hashes)),
            "independent_training_and_validation_selection_per_seed": True,
        },
        "seed_summaries": seed_summaries,
        "aggregate": {
            "metrics": {
                key: mean_std([seed["metrics"][key] for seed in seed_summaries])
                for key in metric_names
            },
            "efficiency": {
                key: mean_std([seed["efficiency"][key] for seed in seed_summaries])
                for key in efficiency_names
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
