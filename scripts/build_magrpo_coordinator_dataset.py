#!/usr/bin/env python3
"""Build validation-only states for the full CAMARL coordinator.

The coordinator is trained at every reachable expert state.  It either calls
one unused expert or stops and emits the final top-10 ranking.  Expert returns
come from the completed fixed-all validation cache.  The hidden validation
target is stored only in a reward-only column and is never serialized into the
model prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from build_grpo_router_dataset import (
    DEFAULT_UTILITY_WEIGHTS,
    all_subsets,
    deterministic_partition,
    sequential_action_tables,
    subset_key,
)
from eval_no_grpo_graphhard import (
    PROTOCOL_SHA256,
    TOOLS,
    consensus_ranking,
    coordinator_holistic_evidence,
    coordinator_prompt,
    fallback_ranking,
    load_pickle,
    sha256_file,
)


DATASET_VERSION = (
    "camarl-full-coordinator-validation-v3-disabled-expert-ablation"
)
COORDINATOR_SYSTEM = """You are the CAMARL evidence coordinator. At the current state, either call exactly one unused semantic, profile, or topology expert, or stop and rank the candidates. Candidate order is random, the hidden next participant is unknown, and extra calls have a cost. Use only the supplied label-free context and observations from experts already executed. For a tool call return JSON only as {\"action\":\"call\",\"expert\":\"semantic|profile|topology\"}. To stop return JSON only as {\"action\":\"stop\",\"top_user_ids\":[...]}, with exactly the requested number of unique candidate IDs. Never call an executed expert. Do not output chain-of-thought."""


def available_subsets(available_experts: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Return reachable subsets while retaining the canonical expert order."""
    available = set(available_experts)
    canonical = tuple(tool for tool in TOOLS if tool in available)
    if len(canonical) != len(available_experts):
        raise ValueError(f"Invalid available expert set: {available_experts}")
    return [
        state
        for state in all_subsets()
        if all(tool in available for tool in state)
    ]


def coordinator_system(available_experts: tuple[str, ...]) -> str:
    """Keep the default prompt exact and constrain leave-one-out prompts."""
    if available_experts == tuple(TOOLS):
        return COORDINATOR_SYSTEM
    expert_list = ", ".join(available_experts)
    expert_choices = "|".join(available_experts)
    return (
        "You are the CAMARL evidence coordinator. At the current state, "
        "either call exactly one unused expert from the available set "
        f"({expert_list}), "
        "or stop and rank the candidates. Candidate order is random, the hidden "
        "next participant is unknown, and extra calls have a cost. Use only the "
        "supplied label-free context and observations from experts already executed. "
        "For a tool call return JSON only as "
        f'{{"action":"call","expert":"{expert_choices}"}}. To stop return JSON '
        'only as {"action":"stop","top_user_ids":[...]}, with exactly the '
        "requested number of unique candidate IDs. Never call an executed or "
        "unavailable expert. Do not output chain-of-thought."
    )


def sequential_action_tables_for_experts(
    candidates: list[str],
    fallback: list[str],
    observations: dict[str, dict[str, Any]],
    positive: str,
    pool_size: int,
    available_experts: tuple[str, ...],
    prior_anchor: float,
    rank_score_balance: float,
    confidence_floor: float,
    oracle_cost_weight: float,
    utility_weights: tuple[float, float, float, float],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    """Recompute the oracle over only the available expert actions.

    Terminal ranking utilities are identical to the full-state calculation for
    each retained subset. Backward induction is then repeated after removing
    the disabled action. Cost remains normalized by the original three-expert
    action space so leave-one-out variants are directly comparable.
    """
    states = available_subsets(available_experts)
    if available_experts == tuple(TOOLS):
        return sequential_action_tables(
            candidates,
            fallback,
            observations,
            positive,
            pool_size,
            prior_anchor=prior_anchor,
            rank_score_balance=rank_score_balance,
            confidence_floor=confidence_floor,
            oracle_cost_weight=oracle_cost_weight,
            utility_weights=utility_weights,
        )

    full_tables, _ = sequential_action_tables(
        candidates,
        fallback,
        observations,
        positive,
        pool_size,
        prior_anchor=prior_anchor,
        rank_score_balance=rank_score_balance,
        confidence_floor=confidence_floor,
        oracle_cost_weight=oracle_cost_weight,
        utility_weights=utility_weights,
    )
    tables: dict[str, dict[str, dict[str, Any]]] = {}
    oracle: dict[str, dict[str, Any]] = {}

    def solve(state: tuple[str, ...]) -> dict[str, Any]:
        key = subset_key(state)
        if key in oracle:
            return oracle[key]["path"]
        stop = dict(full_tables[key]["stop"])
        actions: dict[str, dict[str, Any]] = {"stop": stop}
        executed = set(state)
        for tool in available_experts:
            if tool in executed:
                continue
            child_state = tuple(
                value for value in TOOLS if value in executed | {tool}
            )
            child = solve(child_state)
            future_calls = 1 + int(child["future_calls"])
            path = {
                "action": {"action": "call", "expert": tool},
                "terminal_tools": child["terminal_tools"],
                "rank": child["rank"],
                "rank_utility": child["rank_utility"],
                "future_calls": future_calls,
                "normalized_tool_cost": future_calls / len(TOOLS),
                "oracle_objective": (
                    child["rank_utility"]
                    - oracle_cost_weight * future_calls / len(TOOLS)
                ),
            }
            actions[f"call:{tool}"] = path
        best_key, best = max(
            actions.items(),
            key=lambda item: (
                item[1]["oracle_objective"],
                item[1]["rank_utility"],
                -item[1]["future_calls"],
                item[0] == "stop",
                item[0],
            ),
        )
        tables[key] = actions
        oracle[key] = {
            "action_key": best_key,
            "action": best["action"],
            "path": best,
        }
        return best

    solve(tuple())
    expected_keys = {subset_key(state) for state in states}
    if set(tables) != expected_keys:
        raise RuntimeError("Restricted backward induction omitted a reachable state")
    return tables, oracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--disabled-expert",
        choices=TOOLS,
        default=None,
        help="Build a leave-one-expert-out coordinator dataset.",
    )
    parser.add_argument(
        "--pool-sizes", type=int, nargs="+", default=[20, 50, 100, 500]
    )
    parser.add_argument("--relation-scope", default="prefix_aggregate")
    parser.add_argument("--max-train-queries-per-pool", type=int, default=512)
    parser.add_argument("--prior-anchor", type=float, default=0.20)
    parser.add_argument("--rank-score-balance", type=float, default=0.75)
    parser.add_argument("--confidence-floor", type=float, default=0.20)
    parser.add_argument("--oracle-cost-weight", type=float, default=0.08)
    parser.add_argument(
        "--utility-weights",
        type=float,
        nargs=4,
        metavar=("H1", "NDCG10", "H5", "H10"),
        default=DEFAULT_UTILITY_WEIGHTS,
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--graphhard-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-protocol-sha256",
        required=True,
        help="Expected hash of the frozen candidate protocol report.",
    )
    args = parser.parse_args()
    utility_weights = tuple(map(float, args.utility_weights))
    if any(value < 0.0 for value in utility_weights):
        raise ValueError("utility weights must be non-negative")
    if abs(sum(utility_weights) - 1.0) > 1e-8:
        raise ValueError("utility weights must sum to one")

    report = json.loads(args.validation_report.read_text())
    audit = report.get("audit", {})
    if audit.get("method") != "fixed_all":
        raise ValueError("MA-GRPO states require a fixed_all validation report")
    if audit.get("split") != "validation":
        raise ValueError("MA-GRPO states must be validation-only")
    if audit.get("relation_scope") != args.relation_scope:
        raise ValueError(
            f"Relation-scope mismatch: {audit.get('relation_scope')} != "
            f"{args.relation_scope}"
        )
    protocol_path = args.graphhard_dir / "graphhard_protocol_report.json"
    protocol_hash = sha256_file(protocol_path)
    if protocol_hash != args.expected_protocol_sha256:
        raise ValueError(f"Unexpected graphhard protocol hash: {protocol_hash}")
    seed_report = next(
        (
            value
            for value in report.get("seeds", [])
            if int(value.get("seed")) == args.seed
        ),
        None,
    )
    if seed_report is None:
        raise ValueError(f"Seed {args.seed} absent from validation report")

    users: dict[str, Any] = load_pickle(args.data_dir / "users_all.pkl")
    news: dict[str, Any] = load_pickle(args.data_dir / "news_all.pkl")
    available_experts = tuple(
        tool for tool in TOOLS if tool != args.disabled_expert
    )
    states = available_subsets(available_experts)
    system_prompt = coordinator_system(available_experts)
    rows: list[dict[str, Any]] = []

    for pool_size in args.pool_sizes:
        records = load_pickle(
            args.graphhard_dir / f"validation_graphhard_pools_N{pool_size}.pkl"
        )
        diagnostics = seed_report["pools"][str(pool_size)]["per_query_diagnostics"]
        if len(records) != len(diagnostics):
            raise ValueError(
                f"N={pool_size} record/report mismatch: "
                f"{len(records)} != {len(diagnostics)}"
            )
        train_indices = [
            index
            for index, record in enumerate(records)
            if deterministic_partition(str(record["news_id"])) == "train"
        ]
        train_indices = sorted(
            train_indices,
            key=lambda value: hashlib.sha256(
                f"N{pool_size}:{value}".encode()
            ).hexdigest(),
        )[: args.max_train_queries_per_pool]
        retained_train_indices = set(train_indices)
        for index, (record, diag) in enumerate(zip(records, diagnostics)):
            positive = str(record["next_user"])
            candidates = list(map(str, record["neg_users"])) + [positive]
            random.Random(
                args.seed * 1_000_003 + pool_size * 10_007 + index
            ).shuffle(candidates)
            news_id = str(record["news_id"])
            topic = str((news.get(news_id, {}) or {}).get("text", ""))
            fallback = fallback_ranking(candidates, topic, users)
            observations = diag.get("expert_observations", {})
            missing = [tool for tool in TOOLS if tool not in observations]
            if missing:
                raise ValueError(
                    f"N={pool_size} index={index} missing experts {missing}"
                )
            action_tables, oracle_by_state = sequential_action_tables_for_experts(
                candidates,
                fallback,
                observations,
                positive,
                pool_size,
                available_experts=available_experts,
                prior_anchor=args.prior_anchor,
                rank_score_balance=args.rank_score_balance,
                confidence_floor=args.confidence_floor,
                oracle_cost_weight=args.oracle_cost_weight,
                utility_weights=utility_weights,
            )
            query_id = f"N{pool_size}:{index}"
            partition = deterministic_partition(news_id)
            if partition == "train" and index not in retained_train_indices:
                continue
            top_k = min(10, len(candidates))
            for state in states:
                key = subset_key(state)
                selected = {tool: observations[tool] for tool in state}
                unused = [
                    tool for tool in available_experts if tool not in state
                ]
                holistic = coordinator_holistic_evidence(
                    record,
                    candidates,
                    users,
                    topic,
                    fallback,
                    selected,
                )
                prompt = coordinator_prompt(
                    topic,
                    candidates,
                    fallback,
                    selected,
                    unused,
                    holistic,
                )
                consensus, _, _ = consensus_ranking(
                    candidates,
                    fallback,
                    selected,
                    prior_anchor=args.prior_anchor,
                    rank_score_balance=args.rank_score_balance,
                    confidence_floor=args.confidence_floor,
                )
                state_oracle = oracle_by_state[key]
                rows.append(
                    {
                        "example_id": f"{query_id}:S={key}",
                        "query_id": query_id,
                        "partition": partition,
                        "pool_size": pool_size,
                        "relation_scope": args.relation_scope,
                        "available_experts": list(available_experts),
                        "disabled_expert": args.disabled_expert,
                        "state_tools": list(state),
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                        "action_scores": json.dumps(
                            action_tables[key],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "oracle_action": state_oracle["action"],
                        "oracle_action_key": state_oracle["action_key"],
                        "oracle_terminal_rank": state_oracle["path"]["rank"],
                        "candidate_ids_reward_only": candidates,
                        "positive_id_reward_only": positive,
                        "top_k_reward_only": top_k,
                        "state_fallback_reward_only": consensus[:top_k],
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    counts = {
        partition: sum(row["partition"] == partition for row in rows)
        for partition in ("train", "dev")
    }
    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_validation_report": str(args.validation_report),
        "source_validation_report_sha256": sha256_file(args.validation_report),
        "graphhard_report_sha256": protocol_hash,
        "split": "validation",
        "seed": args.seed,
        "pool_sizes": args.pool_sizes,
        "relation_scope": args.relation_scope,
        "available_experts": list(available_experts),
        "disabled_expert": args.disabled_expert,
        "max_train_queries_per_pool": args.max_train_queries_per_pool,
        "paper_hyperparameters": {
            "prior_anchor": args.prior_anchor,
            "rank_score_balance": args.rank_score_balance,
            "confidence_floor": args.confidence_floor,
            "oracle_cost_weight": args.oracle_cost_weight,
            "utility_weights_h1_ndcg10_h5_h10": list(utility_weights),
        },
        "state_records": len(rows),
        "query_records": len(rows) // len(states),
        "partition_counts": counts,
        "states_per_query": len(states),
        "actions": ["stop+ranking"]
        + [f"call:{tool}" for tool in available_experts],
        "tool_cost_normalization_denominator": len(TOOLS),
        "policy": (
            "full coordinator: sequential tool calls and final ranking"
            if args.disabled_expert is None
            else (
                "leave-one-expert-out coordinator: sequential tool calls and "
                "final ranking"
            )
        ),
        "coordinator_card_protocol": {
            "limit": 64,
            "selection": "label-free fallback, executed-expert, and public-structure retrieval",
            "all_candidate_ids_visible": True,
            "experts_receive_full_candidate_pool": True,
        },
        "target_visible_to_model": False,
        "reward_only_columns": [
            "candidate_ids_reward_only",
            "positive_id_reward_only",
            "top_k_reward_only",
        ],
        "test_records_used": False,
        "output_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
