#!/usr/bin/env python3
"""Build a validation-only GRPO routing dataset for CAMARL.

The input is a completed ``fixed_all`` validation report.  Each expert is
executed exactly once by that upstream job.  This script then evaluates all
eight expert subsets with the label-free consensus ranker and uses the hidden
validation target only to construct rewards.  The target ID is never written
to the model prompt or the output JSONL.

The resulting task is a sequential tool-selection problem.  For each of the
eight reachable expert states, the policy sees only observations already
executed in that state and emits one unused tool call or stop.  Backward
induction over cached validation outcomes supplies the ranking/cost return and
the minimum-cost oracle action.  The target is never part of a policy prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from eval_no_grpo_graphhard import (
    PROTOCOL_SHA256,
    TOOLS,
    RelationIndex,
    compact,
    consensus_ranking,
    containment,
    fallback_ranking,
    grams,
    load_pickle,
    media_for,
    sha256_file,
)


DATASET_VERSION = "camarl-grpo-sequential-router-validation-v2"
ROUTER_SYSTEM = """You are the CAMARL coordinator's sequential tool router. At the current state, either call exactly one unused semantic, profile, or topology expert, or stop. Use only the supplied label-free context and observations from experts already executed. Candidate order is random, the hidden next participant is unknown, and extra calls have a cost. Return JSON only as {\"action\":\"call\",\"expert\":\"semantic|profile|topology\"} or {\"action\":\"stop\"}. Never call an executed expert and do not output a ranking or chain-of-thought."""
ORACLE_COST_WEIGHT = 0.08
DEFAULT_UTILITY_WEIGHTS = (0.45, 0.25, 0.20, 0.10)


def all_subsets() -> list[tuple[str, ...]]:
    return [
        tuple(tool for tool, enabled in zip(TOOLS, bits) if enabled)
        for bits in itertools.product((False, True), repeat=len(TOOLS))
    ]


def subset_key(tools: tuple[str, ...] | list[str]) -> str:
    selected = set(tools)
    return "+".join(tool for tool in TOOLS if tool in selected) or "none"


def ranking_utility(
    rank: int,
    utility_weights: tuple[float, float, float, float] = DEFAULT_UTILITY_WEIGHTS,
) -> float:
    """Paper-oriented scalar reward with explicit top-1/5/10 components."""
    if rank <= 0:
        return 0.0
    ndcg10 = 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
    h1_weight, ndcg_weight, h5_weight, h10_weight = utility_weights
    return (
        h1_weight * float(rank == 1)
        + ndcg_weight * ndcg10
        + h5_weight * float(rank <= 5)
        + h10_weight * float(rank <= 10)
    )


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def routing_context(
    record: dict[str, Any],
    candidates: list[str],
    users: dict[str, Any],
    news: dict[str, Any],
    relations: RelationIndex,
    media_dir: Path,
    relation_scope: str,
) -> str:
    """Create a short router input without exposing any expert observation."""
    news_id = str(record["news_id"])
    topic = str((news.get(news_id, {}) or {}).get("text", ""))
    observed = list(map(str, record["history_users"]))
    observed_set = set(observed)
    query_grams = grams(topic)

    descriptions = 0
    histories = 0
    lexical_scores: list[float] = []
    out_degrees: list[float] = []
    in_degrees: list[float] = []
    direct_links: list[float] = []
    other_cascade_degrees: list[float] = []
    for user in candidates:
        data = users.get(user, {}) or {}
        description = str(data.get("description", "") or "")
        history = list(data.get("history", []) or [])
        descriptions += bool(description.strip())
        histories += bool(history)
        lexical_text = f"{description} {' '.join(map(str, history[:3]))}"
        lexical_scores.append(containment(query_grams, grams(lexical_text)))
        outgoing = relations.out.get(user, set())
        incoming = relations.incoming.get(user, set())
        out_degrees.append(float(len(outgoing)))
        in_degrees.append(float(len(incoming)))
        direct_links.append(float(bool((outgoing | incoming) & observed_set)))
        other_cascade_degrees.append(
            float(len(relations.user_cascades.get(user, set()) - {news_id}))
        )

    top_lexical = sorted(lexical_scores, reverse=True)
    lexical_margin = (
        top_lexical[0] - top_lexical[min(4, len(top_lexical) - 1)]
        if top_lexical
        else 0.0
    )
    prefix_bios = [
        compact((users.get(user, {}) or {}).get("description", ""), 28)
        for user in observed[-6:]
    ]
    n = max(1, len(candidates))
    visual = media_for(news_id, news, media_dir) is not None
    return (
        f"Topic text: {compact(topic, 300)}\n"
        f"Observed strict-prefix users: {len(observed)}; recent bios: "
        f"{' | '.join(value for value in prefix_bios if value) or 'unavailable'}\n"
        f"Candidate pool size: {len(candidates)}; relation scope: {relation_scope}; "
        f"visual available: {str(visual).lower()}\n"
        f"Candidate evidence coverage: bio={descriptions/n:.3f}, "
        f"history={histories/n:.3f}, direct-prefix-relation={statistics.mean(direct_links):.3f}\n"
        f"Lexical compatibility: max={max(lexical_scores, default=0.0):.3f}, "
        f"mean={statistics.mean(lexical_scores) if lexical_scores else 0.0:.3f}, "
        f"top1-top5-margin={lexical_margin:.3f}\n"
        f"Static relation summary: out-degree median/p90={quantile(out_degrees, .5):.1f}/"
        f"{quantile(out_degrees, .9):.1f}, in-degree median/p90={quantile(in_degrees, .5):.1f}/"
        f"{quantile(in_degrees, .9):.1f}, other-cascade median/p90="
        f"{quantile(other_cascade_degrees, .5):.1f}/{quantile(other_cascade_degrees, .9):.1f}\n"
        "Decide whether another independent expert is worth its cost."
    )


def deterministic_partition(news_id: str) -> str:
    # Keep every prefix and pool size of one news topic in the same partition.
    digest = hashlib.sha256(news_id.encode()).digest()
    return "dev" if int.from_bytes(digest[:4], "big") % 10 == 0 else "train"


def action_key(action: dict[str, str]) -> str:
    if action.get("action") == "stop":
        return "stop"
    return f"call:{action.get('expert', '')}"


def state_prompt(
    context: str,
    observations: dict[str, dict[str, Any]],
    state: tuple[str, ...] | list[str],
) -> str:
    executed = set(state)
    observation_text = (
        "\n".join(
            f"{tool}|confidence={float(observations[tool]['confidence']):.3f}|"
            f"ranking={','.join(observations[tool]['top_user_ids'])}|"
            f"scores={json.dumps(observations[tool]['candidate_scores'], ensure_ascii=False, separators=(',', ':'))}|"
            f"evidence={compact(observations[tool].get('evidence', ''), 160) or 'not supplied'}"
            for tool in TOOLS
            if tool in executed
        )
        or "none"
    )
    unused = [tool for tool in TOOLS if tool not in executed]
    return (
        f"{context}\n"
        f"Executed experts: {','.join(tool for tool in TOOLS if tool in executed) or 'none'}\n"
        f"Unused experts: {','.join(unused) or 'none'}\n"
        f"Executed observations:\n{observation_text}\n"
        "Choose one valid next action."
    )


def sequential_action_tables(
    candidates: list[str],
    fallback: list[str],
    observations: dict[str, dict[str, Any]],
    positive: str,
    pool_size: int,
    prior_anchor: float = 0.20,
    rank_score_balance: float = 0.75,
    confidence_floor: float = 0.20,
    oracle_cost_weight: float = ORACLE_COST_WEIGHT,
    utility_weights: tuple[float, float, float, float] = DEFAULT_UTILITY_WEIGHTS,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, Any]],
]:
    """Return every state's action table and minimum-cost optimal action."""
    subsets = all_subsets()
    terminal: dict[str, dict[str, Any]] = {}
    for subset in subsets:
        selected = {tool: observations[tool] for tool in subset}
        ranking, _, _ = consensus_ranking(
            candidates,
            fallback,
            selected,
            prior_anchor=prior_anchor,
            rank_score_balance=rank_score_balance,
            confidence_floor=confidence_floor,
        )
        full_rank = ranking.index(positive) + 1
        evaluation_rank = full_rank if full_rank <= 10 else pool_size
        terminal[subset_key(subset)] = {
            "terminal_tools": list(subset),
            "rank": evaluation_rank,
            "rank_utility": ranking_utility(evaluation_rank, utility_weights),
            "future_calls": 0,
            "normalized_tool_cost": 0.0,
        }

    tables: dict[str, dict[str, dict[str, Any]]] = {}
    oracle: dict[str, dict[str, Any]] = {}

    def solve(state: tuple[str, ...]) -> dict[str, Any]:
        key = subset_key(state)
        if key in oracle:
            return oracle[key]["path"]
        stop = {
            **terminal[key],
            "action": {"action": "stop"},
            "oracle_objective": terminal[key]["rank_utility"],
        }
        actions: dict[str, dict[str, Any]] = {"stop": stop}
        executed = set(state)
        for tool in TOOLS:
            if tool in executed:
                continue
            child_state = tuple(value for value in TOOLS if value in executed | {tool})
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
        oracle[key] = {"action_key": best_key, "action": best["action"], "path": best}
        return best

    solve(tuple())
    return tables, oracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=[20, 50, 100, 500])
    parser.add_argument("--relation-scope", default="prefix_aggregate")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graphhard-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-protocol-sha256",
        required=True,
        help="Expected hash of the frozen candidate protocol report.",
    )
    args = parser.parse_args()

    report = json.loads(args.validation_report.read_text())
    audit = report.get("audit", {})
    if audit.get("method") != "fixed_all":
        raise ValueError("GRPO rewards require a fixed_all expert report")
    if audit.get("split") != "validation":
        raise ValueError("GRPO dataset must be built from validation, never test")
    if audit.get("relation_scope") != args.relation_scope:
        raise ValueError(
            f"Relation-scope mismatch: {audit.get('relation_scope')} != {args.relation_scope}"
        )
    report_hash = sha256_file(args.graphhard_dir / "graphhard_protocol_report.json")
    if report_hash != args.expected_protocol_sha256:
        raise ValueError(f"Unexpected graphhard protocol hash: {report_hash}")
    seed_report = next(
        (value for value in report.get("seeds", []) if int(value.get("seed")) == args.seed),
        None,
    )
    if seed_report is None:
        raise ValueError(f"Seed {args.seed} absent from validation report")

    users = load_pickle(args.data_dir / "users_all.pkl")
    news = load_pickle(args.data_dir / "news_all.pkl")
    relations = RelationIndex(users, args.data_dir / "cascades.txt")
    subsets = all_subsets()
    rows: list[dict[str, Any]] = []

    for pool_size in args.pool_sizes:
        records = load_pickle(
            args.graphhard_dir / f"validation_graphhard_pools_N{pool_size}.pkl"
        )
        pool_report = seed_report["pools"][str(pool_size)]
        diagnostics = pool_report["per_query_diagnostics"]
        if len(records) != len(diagnostics):
            raise ValueError(
                f"N={pool_size} record/report mismatch: {len(records)} != {len(diagnostics)}"
            )
        for index, (record, diag) in enumerate(zip(records, diagnostics)):
            positive = str(record["next_user"])
            candidates = list(map(str, record["neg_users"])) + [positive]
            random.Random(
                args.seed * 1_000_003 + pool_size * 10_007 + index
            ).shuffle(candidates)
            topic = str((news.get(str(record["news_id"]), {}) or {}).get("text", ""))
            fallback = fallback_ranking(candidates, topic, users)
            observations = diag.get("expert_observations", {})
            missing = [tool for tool in TOOLS if tool not in observations]
            if missing:
                raise ValueError(f"N={pool_size} index={index} missing experts {missing}")

            action_tables, oracle_by_state = sequential_action_tables(
                candidates, fallback, observations, positive, pool_size
            )
            context = routing_context(
                record, candidates, users, news, relations, args.media_dir,
                args.relation_scope,
            )
            query_id = f"N{pool_size}:{index}"
            for state in subsets:
                key = subset_key(state)
                state_oracle = oracle_by_state[key]
                rows.append(
                    {
                        "example_id": f"{query_id}:S={key}",
                        "query_id": query_id,
                        "partition": deterministic_partition(str(record["news_id"])),
                        "pool_size": pool_size,
                        "relation_scope": args.relation_scope,
                        "state_tools": list(state),
                        "prompt": [
                            {"role": "system", "content": ROUTER_SYSTEM},
                            {
                                "role": "user",
                                "content": state_prompt(context, observations, state),
                            },
                        ],
                        "action_scores": json.dumps(
                            action_tables[key],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "oracle_action": state_oracle["action"],
                        "oracle_action_key": state_oracle["action_key"],
                        "oracle_terminal_rank": state_oracle["path"]["rank"],
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
        "graphhard_report_sha256": report_hash,
        "split": "validation",
        "seed": args.seed,
        "pool_sizes": args.pool_sizes,
        "relation_scope": args.relation_scope,
        "state_records": len(rows),
        "query_records": len(rows) // len(subsets),
        "partition_counts": counts,
        "states_per_query": len(subsets),
        "actions": ["stop"] + [f"call:{tool}" for tool in TOOLS],
        "oracle_cost_weight": ORACLE_COST_WEIGHT,
        "ranking_utility": "0.45*H@1 + 0.25*NDCG@10 + 0.20*H@5 + 0.10*H@10",
        "target_visible_to_model": False,
        "test_records_used": False,
        "output_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
