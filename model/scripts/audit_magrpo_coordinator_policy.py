#!/usr/bin/env python3
"""Audit one full-coordinator MA-GRPO checkpoint on validation-dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


TOOLS = ("semantic", "profile", "topology")
SELECTION_SPEC = {
    "ranking_utility_weight": 0.65,
    "oracle_action_accuracy_weight": 0.30,
    "valid_action_rate_weight": 0.05,
    "normalized_tool_cost_penalty": 0.10,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_key(tools: list[str]) -> str:
    selected = set(tools)
    return "+".join(tool for tool in TOOLS if tool in selected) or "none"


def parse_payload(text: str) -> dict[str, Any]:
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end]) if start >= 0 and end > start else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def parse_decision(
    text: str, record: dict[str, Any]
) -> tuple[dict[str, Any], str, list[str], bool]:
    payload = parse_payload(text)
    action = str(payload.get("action", "")).strip().lower()
    table = json.loads(record["action_scores"])
    if action == "call":
        expert = str(payload.get("expert", "")).strip().lower()
        key = f"call:{expert}"
        if expert in TOOLS and key in table:
            return {"action": "call", "expert": expert}, key, [], True
    if action == "stop":
        raw_top = payload.get("top_user_ids", payload.get("top", []))
        ranking = [str(value) for value in raw_top] if isinstance(raw_top, list) else []
        candidates = set(map(str, record["candidate_ids_reward_only"]))
        top_k = int(record["top_k_reward_only"])
        valid = (
            len(ranking) == top_k
            and len(set(ranking)) == top_k
            and all(user in candidates for user in ranking)
        )
        if valid:
            return {"action": "stop"}, "stop", ranking, True
    return {}, "", [], False


def rank_metrics(ranks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cutoff in (1, 5, 10):
        metrics[f"H@{cutoff}"] = statistics.mean(rank <= cutoff for rank in ranks)
        metrics[f"MAP@{cutoff}"] = statistics.mean(
            1.0 / rank if rank <= cutoff else 0.0 for rank in ranks
        )
        metrics[f"NDCG@{cutoff}"] = statistics.mean(
            1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0 for rank in ranks
        )
    return metrics


def ranking_utility_from_metrics(metrics: dict[str, float]) -> float:
    return (
        0.45 * metrics["H@1"]
        + 0.25 * metrics["NDCG@10"]
        + 0.20 * metrics["H@5"]
        + 0.10 * metrics["H@10"]
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = rank_metrics([int(row["rank"]) for row in rows])
    return {
        "query_records": len(rows),
        "metrics": metrics,
        "ranking_utility": ranking_utility_from_metrics(metrics),
        "average_expert_calls": statistics.mean(
            len(row["executed_tools"]) for row in rows
        ),
        "normalized_tool_cost": statistics.mean(
            len(row["executed_tools"]) / len(TOOLS) for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--coordinator-model-name", default="CAMARL")
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--pool-size",
        type=int,
        choices=(20, 50, 100, 500, 1000, 1500, 2000),
        required=True,
    )
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--server-max-model-len",
        type=int,
        help="Audited max-model-len used by every backing inference server.",
    )
    parser.add_argument(
        "--fail-on-request-error",
        action="store_true",
        help="Reject checkpoint audits containing any model request error.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    records = [
        record
        for record in records
        if record["partition"] == "dev"
        and int(record["pool_size"]) == args.pool_size
    ]
    if not records:
        raise ValueError(f"No validation-dev records found for N={args.pool_size}")
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["query_id"]), {})[
            subset_key(list(map(str, record["state_tools"])))
        ] = record
    queries = sorted(grouped.items())
    clients = [
        OpenAI(
            api_key="EMPTY",
            base_url=f"http://127.0.0.1:{port}/v1",
            timeout=300,
            max_retries=2,
        )
        for port in args.ports
    ]
    for client in clients:
        models = {model.id for model in client.models.list().data}
        if args.coordinator_model_name not in models:
            raise ValueError(
                f"{args.coordinator_model_name} absent from served models {models}"
            )

    def run(
        index: int, query_id: str, states: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        executed: list[str] = []
        trace: list[dict[str, Any]] = []
        ranking: list[str] = []
        terminal_record = states["none"]
        for step in range(4):
            state = subset_key(executed)
            record = states[state]
            terminal_record = record
            text = ""
            error = ""
            try:
                response = clients[index % len(clients)].chat.completions.create(
                    model=args.coordinator_model_name,
                    temperature=0.2,
                    top_p=0.9,
                    seed=args.seed,
                    max_tokens=160,
                    messages=record["prompt"],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content or ""
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
            action, selected_key, proposed_ranking, valid = parse_decision(text, record)
            oracle_key = str(record["oracle_action_key"])
            trace.append(
                {
                    "step": step,
                    "state_tools": list(executed),
                    "selected_action": action,
                    "oracle_action": record["oracle_action"],
                    "exact_oracle": valid and selected_key == oracle_key,
                    "valid": valid,
                    "error": error,
                    "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
            if not valid:
                ranking = list(map(str, record["state_fallback_reward_only"]))
                break
            if action["action"] == "stop":
                ranking = proposed_ranking
                break
            executed.append(str(action["expert"]))
            executed = [tool for tool in TOOLS if tool in set(executed)]
        if not ranking:
            state = subset_key(executed)
            terminal_record = states[state]
            ranking = list(map(str, terminal_record["state_fallback_reward_only"]))
        positive = str(terminal_record["positive_id_reward_only"])
        pool_size = int(terminal_record["pool_size"])
        rank = ranking.index(positive) + 1 if positive in ranking else pool_size
        return {
            "query_id": query_id,
            "pool_size": pool_size,
            "rank": rank,
            "top_user_ids": ranking,
            "executed_tools": executed,
            "trace": trace,
        }

    rows: list[dict[str, Any] | None] = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run, index, query_id, states): index
            for index, (query_id, states) in enumerate(queries)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows[futures[future]] = future.result()
            if completed % 50 == 0 or completed == len(queries):
                print(f"validation-dev coordinator audit {completed}/{len(queries)}", flush=True)
    final_rows = [row for row in rows if row is not None]
    decisions = [decision for row in final_rows for decision in row["trace"]]
    request_errors = sum(bool(value["error"]) for value in decisions)
    if args.fail_on_request_error and request_errors:
        raise RuntimeError(
            f"Validation checkpoint audit produced {request_errors} request errors"
        )
    overall = summarize(final_rows)
    valid_action_rate = statistics.mean(value["valid"] for value in decisions)
    oracle_action_accuracy = statistics.mean(
        value["exact_oracle"] for value in decisions
    )
    selection_score = (
        SELECTION_SPEC["ranking_utility_weight"] * overall["ranking_utility"]
        + SELECTION_SPEC["oracle_action_accuracy_weight"] * oracle_action_accuracy
        + SELECTION_SPEC["valid_action_rate_weight"] * valid_action_rate
        - SELECTION_SPEC["normalized_tool_cost_penalty"]
        * overall["normalized_tool_cost"]
    )
    pools = {
        str(pool_size): summarize(
            [row for row in final_rows if int(row["pool_size"]) == pool_size]
        )
        for pool_size in sorted(set(int(row["pool_size"]) for row in final_rows))
    }
    report = {
        "audit": {
            "split": "validation-dev",
            "task": f"N={args.pool_size}",
            "pool_size": args.pool_size,
            "cross_pool_policy_sharing": False,
            "test_records_used": False,
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "coordinator_model_name": args.coordinator_model_name,
            "adapter_path": str(args.adapter_path),
            "seed": args.seed,
            "query_records": len(final_rows),
            "decision_records": len(decisions),
            "request_errors": request_errors,
            "fail_on_request_error": args.fail_on_request_error,
            "server_max_model_len": args.server_max_model_len,
            "valid_action_rate": valid_action_rate,
            "oracle_action_accuracy": oracle_action_accuracy,
            "selection_spec": SELECTION_SPEC,
            "selection_score": selection_score,
        },
        "selected_policy": overall,
        "pools": pools,
        "per_query": final_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {"audit": report["audit"], "selected_policy": overall, "pools": pools},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
