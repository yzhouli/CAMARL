#!/usr/bin/env python3
"""Independently verify a completed CAMARL/MA-GRPO evaluation artifact.

This verifier deliberately does not import the evaluator.  Ranking metrics and
five-seed aggregates are recomputed from the stored per-query ranks so that a
bug in the evaluation/aggregation implementation cannot validate itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


CUTOFFS = (1, 2, 3, 5, 10)
ALL_EXPERTS = ("semantic", "profile", "topology")
ABS_TOLERANCE = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: Any, expected: float, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: expected a finite number, got {actual!r}") from error
    require(math.isfinite(value), f"{label}: value is not finite: {value}")
    require(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=ABS_TOLERANCE),
        f"{label}: reported={value!r}, recomputed={expected!r}",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranking_metrics(ranks: list[int]) -> dict[str, float]:
    """Metrics for one relevant item per query, independently implemented."""
    count = len(ranks)
    output: dict[str, float] = {}
    for cutoff in CUTOFFS:
        output[f"H@{cutoff}"] = sum(rank <= cutoff for rank in ranks) / count
        output[f"MAP@{cutoff}"] = (
            sum(1.0 / rank for rank in ranks if rank <= cutoff) / count
        )
        output[f"NDCG@{cutoff}"] = (
            sum(1.0 / math.log2(rank + 1) for rank in ranks if rank <= cutoff)
            / count
        )
    return output


def verify(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(args.input.read_text(encoding="utf-8"))
    audit = data.get("audit")
    require(isinstance(audit, dict), "missing audit object")
    expected_seeds = list(args.expected_seeds)
    require(audit.get("seeds") == expected_seeds, "audit seed contract mismatch")
    require(audit.get("pool_sizes") == [args.pool_size], "audit pool contract mismatch")
    require(audit.get("split") == "test", "artifact is not a test result")
    require(int(audit.get("records", -1)) == args.expected_records, "record contract mismatch")
    require(audit.get("fail_on_request_error") is True, "strict request failure is disabled")
    require(
        int(audit.get("trained_pool_size", -1)) == args.pool_size,
        "trained pool does not match evaluated pool",
    )
    require(
        audit.get("test_records_used_for_training_or_selection") is False,
        "test-leakage audit flag is not false",
    )
    require(audit.get("policy_frozen_before_test") is True, "policy was not frozen before test")
    if args.expected_protocol_sha256 is not None:
        require(
            audit.get("graphhard_report_sha256") == args.expected_protocol_sha256,
            "graphhard protocol hash mismatch",
        )
    if args.disabled_expert is not None:
        require(audit.get("disabled_expert") == args.disabled_expert, "disabled expert mismatch")
        require(audit.get("ablation_type") == "leave_one_expert_out", "ablation type mismatch")

    enabled = set(ALL_EXPERTS) - ({args.disabled_expert} if args.disabled_expert else set())
    seed_results = data.get("seeds")
    require(isinstance(seed_results, list), "missing seed results")
    require(
        [seed.get("seed") for seed in seed_results] == expected_seeds,
        "seed result order/content mismatch",
    )

    seed_metrics: list[dict[str, float]] = []
    seed_efficiency: list[dict[str, float]] = []
    pool_key = str(args.pool_size)
    for seed_result in seed_results:
        seed = int(seed_result["seed"])
        pool = seed_result.get("pools", {}).get(pool_key)
        require(isinstance(pool, dict), f"seed={seed}: missing pool {pool_key}")
        ranks = pool.get("ranks")
        require(isinstance(ranks, list), f"seed={seed}: ranks are missing")
        require(len(ranks) == args.expected_records, f"seed={seed}: incomplete ranks")
        require(
            all(type(rank) is int and (1 <= rank <= 10 or rank == args.pool_size) for rank in ranks),
            f"seed={seed}: ranks violate top-10/fallback encoding",
        )
        top10 = pool.get("ranked_top10")
        require(isinstance(top10, list) and len(top10) == args.expected_records, f"seed={seed}: incomplete top-10 outputs")
        require(
            all(isinstance(row, list) and len(row) == 10 and len(set(row)) == 10 for row in top10),
            f"seed={seed}: malformed or duplicate top-10 output",
        )
        per_query = pool.get("per_query_diagnostics")
        require(
            isinstance(per_query, list) and len(per_query) == args.expected_records,
            f"seed={seed}: incomplete per-query diagnostics",
        )

        recomputed = ranking_metrics(ranks)
        reported_metrics = pool.get("metrics", {})
        for metric, expected in recomputed.items():
            close(reported_metrics.get(metric), expected, f"seed={seed}.{metric}")

        expert_calls = 0
        request_count = 0
        request_errors = 0
        any_expert = 0
        expert_counts = {expert: 0 for expert in ALL_EXPERTS}
        latencies: list[float] = []
        for index, row in enumerate(per_query):
            called = row.get("called_experts")
            require(isinstance(called, list), f"seed={seed}, query={index}: missing called_experts")
            require(len(called) == len(set(called)), f"seed={seed}, query={index}: duplicate expert call")
            require(set(called) <= enabled, f"seed={seed}, query={index}: invalid/disabled expert call")
            calls = row.get("calls")
            require(isinstance(calls, list), f"seed={seed}, query={index}: missing calls")
            expert_calls += len(called)
            any_expert += bool(called)
            request_count += len(calls)
            request_errors += sum(bool(call.get("error")) for call in calls)
            for expert in called:
                expert_counts[expert] += 1
            latency = float(row.get("latency_seconds", float("nan")))
            require(math.isfinite(latency) and latency >= 0.0, f"seed={seed}, query={index}: invalid latency")
            latencies.append(latency)

        diagnostics = pool.get("diagnostics", {})
        require(int(diagnostics.get("records", -1)) == args.expected_records, f"seed={seed}: diagnostic record mismatch")
        require(int(diagnostics.get("request_errors", -1)) == 0, f"seed={seed}: reported request errors")
        require(request_errors == 0, f"seed={seed}: raw calls contain request errors")
        count = args.expected_records
        efficiency = {
            "call_rate_any_expert": any_expert / count,
            "expert_calls_per_query": expert_calls / count,
            "model_requests_per_query": request_count / count,
            "normalized_unit_tool_cost": expert_calls / (len(ALL_EXPERTS) * count),
        }
        for key, expected in efficiency.items():
            close(diagnostics.get(key), expected, f"seed={seed}.{key}")
        reported_rates = diagnostics.get("expert_call_rate", {})
        for expert, calls in expert_counts.items():
            close(reported_rates.get(expert), calls / count, f"seed={seed}.{expert}_call_rate")
        seed_metrics.append(recomputed)
        seed_efficiency.append(efficiency)

    aggregate = data.get("aggregate", {}).get(pool_key, {})
    aggregate_metrics = aggregate.get("metrics", {})
    for metric in seed_metrics[0]:
        values = [row[metric] for row in seed_metrics]
        close(aggregate_metrics.get(metric, {}).get("mean"), statistics.mean(values), f"aggregate.{metric}.mean")
        close(aggregate_metrics.get(metric, {}).get("std"), statistics.stdev(values), f"aggregate.{metric}.std")
    aggregate_efficiency = aggregate.get("efficiency", {})
    for key in seed_efficiency[0]:
        values = [row[key] for row in seed_efficiency]
        close(aggregate_efficiency.get(key, {}).get("mean"), statistics.mean(values), f"aggregate.{key}.mean")
        close(aggregate_efficiency.get(key, {}).get("std"), statistics.stdev(values), f"aggregate.{key}.std")

    return {
        "status": "verified",
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "seeds": expected_seeds,
        "pool_size": args.pool_size,
        "records_per_seed": args.expected_records,
        "total_queries": args.expected_records * len(expected_seeds),
        "request_errors": 0,
        "metric_definition": {
            "H@K": "mean(rank <= K)",
            "MAP@K": "mean(1/rank if rank <= K else 0); one relevant item per query",
            "NDCG@K": "mean(1/log2(rank+1) if rank <= K else 0); one relevant item per query",
        },
        "aggregate_metrics": aggregate_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-records", type=int, default=1856)
    parser.add_argument("--disabled-expert", choices=ALL_EXPERTS)
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    report = verify(args)
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
