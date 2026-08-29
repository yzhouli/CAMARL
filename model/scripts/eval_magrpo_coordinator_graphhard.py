#!/usr/bin/env python3
"""Evaluate a frozen MA-GRPO coordinator policy on graphhard-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import eval_no_grpo_graphhard as base
from build_magrpo_coordinator_dataset import COORDINATOR_SYSTEM, coordinator_system


EVALUATOR_VERSION = (
    "camarl-full-coordinator-graphhard-v2-v1-disabled-expert-ablation"
)
EXPERT_COST_DENOMINATOR = 3


def available_experts(disabled_expert: str | None) -> tuple[str, ...]:
    """Return the ordered inference-time expert space for an ablation."""
    if disabled_expert is not None and disabled_expert not in base.TOOLS:
        raise ValueError(f"Unknown disabled expert: {disabled_expert}")
    return tuple(tool for tool in base.TOOLS if tool != disabled_expert)


def _metadata_expert_space(
    metadata: dict[str, Any], source: str
) -> tuple[set[str] | None, dict[str, Any]]:
    """Extract explicit expert-space declarations from an audit/manifest."""
    cost_denominator = metadata.get(
        "tool_cost_normalization_denominator",
        metadata.get("expert_cost_denominator"),
    )
    if (
        cost_denominator is not None
        and int(cost_denominator) != EXPERT_COST_DENOMINATOR
    ):
        raise ValueError(
            f"{source} uses expert-cost denominator {cost_denominator}; "
            f"expected {EXPERT_COST_DENOMINATOR}"
        )
    declarations: list[tuple[str, set[str]]] = []
    available = metadata.get("available_experts")
    if available is not None:
        if not isinstance(available, list) or any(
            not isinstance(tool, str) for tool in available
        ):
            raise ValueError(f"{source} has malformed available_experts")
        declarations.append(("available_experts", set(available)))

    disabled = metadata.get("disabled_expert")
    if disabled is not None:
        if disabled not in base.TOOLS:
            raise ValueError(f"{source} declares unknown disabled expert {disabled}")
        declarations.append(
            ("disabled_expert", set(base.TOOLS) - {str(disabled)})
        )

    actions = metadata.get("actions")
    if isinstance(actions, list):
        called = {
            str(action).split(":", 1)[1]
            for action in actions
            if str(action).startswith("call:")
        }
        if called:
            declarations.append(("actions", called))

    unknown = sorted(
        set().union(*(value for _, value in declarations)) - set(base.TOOLS)
    ) if declarations else []
    if unknown:
        raise ValueError(f"{source} declares unknown experts: {unknown}")
    if declarations and any(value != declarations[0][1] for _, value in declarations[1:]):
        raise ValueError(
            f"Conflicting expert-space declarations in {source}: {declarations}"
        )
    space = declarations[0][1] if declarations else None
    return space, {
        "source": source,
        "explicit_expert_space": sorted(space) if space is not None else None,
        "declarations": [name for name, _ in declarations],
        "expert_cost_denominator": cost_denominator,
    }


def validate_policy_expert_space(
    policy_manifest: dict[str, Any],
    enabled_experts: tuple[str, ...],
    disabled_expert: str | None,
) -> dict[str, Any]:
    """Audit that training and inference expose the same expert action space."""
    spaces: list[tuple[str, set[str]]] = []
    details: list[dict[str, Any]] = []

    policy_space, policy_detail = _metadata_expert_space(
        policy_manifest, "policy manifest"
    )
    details.append(policy_detail)
    if policy_space is not None:
        spaces.append(("policy manifest", policy_space))

    dataset_manifest_path: Path | None = None
    dataset_path_value = policy_manifest.get("dataset")
    if dataset_path_value:
        dataset_path = Path(str(dataset_path_value))
        dataset_manifest_path = dataset_path.with_suffix(
            dataset_path.suffix + ".manifest.json"
        )
        if dataset_manifest_path.is_file():
            expected_hash = policy_manifest.get("dataset_manifest_sha256")
            actual_hash = base.sha256_file(dataset_manifest_path)
            if expected_hash is not None and actual_hash != expected_hash:
                raise ValueError(
                    "Dataset manifest hash does not match the trained policy audit"
                )
            dataset_manifest = json.loads(dataset_manifest_path.read_text())
            dataset_space, dataset_detail = _metadata_expert_space(
                dataset_manifest, "training dataset manifest"
            )
            dataset_detail.update(
                {
                    "path": str(dataset_manifest_path),
                    "sha256": actual_hash,
                }
            )
            details.append(dataset_detail)
            if dataset_space is not None:
                spaces.append(("training dataset manifest", dataset_space))

    if spaces and any(space != spaces[0][1] for _, space in spaces[1:]):
        raise ValueError(f"Policy/dataset expert-space mismatch: {spaces}")
    trained_space = spaces[0][1] if spaces else None
    expected_space = set(enabled_experts)
    if disabled_expert is not None:
        if trained_space is None:
            raise ValueError(
                "Leave-one-expert-out evaluation requires explicit trained expert-"
                "space metadata in the policy or its dataset manifest"
            )
        if trained_space != expected_space:
            raise ValueError(
                "Training/inference expert-space mismatch: "
                f"trained={sorted(trained_space)}, inference={list(enabled_experts)}"
            )
    elif trained_space is not None and trained_space != expected_space:
        raise ValueError(
            "Training/inference expert-space mismatch for the default evaluation: "
            f"trained={sorted(trained_space)}, inference={list(enabled_experts)}"
        )

    return {
        "validation": "passed",
        "trained_expert_space": (
            sorted(trained_space) if trained_space is not None else None
        ),
        "inference_expert_space": list(enabled_experts),
        "training_space_explicitly_declared": trained_space is not None,
        "dataset_manifest_checked": bool(
            dataset_manifest_path is not None and dataset_manifest_path.is_file()
        ),
        "sources": details,
    }


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(base.sha256_file(file_path).encode())
    return digest.hexdigest()


class CoordinatorEndpointPool(base.EndpointPool):
    def __init__(
        self,
        ports: list[int],
        base_model: str,
        coordinator_model: str,
        seed: int,
        coordinator_system: str = COORDINATOR_SYSTEM,
    ):
        super().__init__(ports, base_model, seed, load_balance=True)
        self.coordinator_model = coordinator_model
        self.coordinator_system = coordinator_system

    def check(self) -> None:
        for client in self.clients:
            available = {model.id for model in client.models.list().data}
            missing = {self.served_model, self.coordinator_model} - available
            if missing:
                raise ValueError(f"Models {sorted(missing)} absent from {available}")

    def call_coordinator(
        self,
        prompt: str,
        key: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, base.ModelCall]:
        client_index, client = self._acquire_client(key)
        started = time.time()
        text = ""
        error = ""
        prompt_tokens = completion_tokens = 0
        try:
            response = client.chat.completions.create(
                model=self.coordinator_model,
                temperature=temperature,
                top_p=top_p,
                seed=self.seed,
                max_tokens=160,
                messages=[
                    {"role": "system", "content": self.coordinator_system},
                    {"role": "user", "content": prompt},
                ],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "repetition_penalty": 1.0,
                },
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or ""
            if response.usage is not None:
                prompt_tokens = int(response.usage.prompt_tokens or 0)
                completion_tokens = int(response.usage.completion_tokens or 0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            self._release_client(client_index)
        return text, base.ModelCall(
            role="coordinator_magrpo_v5",
            latency_seconds=time.time() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
            response_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


def evaluate_query(
    index: int,
    record: dict[str, Any],
    pool_size: int,
    seed: int,
    users: dict[str, Any],
    news: dict[str, Any],
    relations: base.RelationIndex,
    memory: base.InterestMemory,
    endpoints: CoordinatorEndpointPool,
    media_dir: Path,
    temperature: float,
    top_p: float,
    relation_scope: str,
    enabled_experts: tuple[str, ...] = tuple(base.TOOLS),
    disabled_expert: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    positive = str(record["next_user"])
    candidates = list(map(str, record["neg_users"])) + [positive]
    random.Random(seed * 1_000_003 + pool_size * 10_007 + index).shuffle(candidates)
    news_id = str(record["news_id"])
    topic = str((news.get(news_id, {}) or {}).get("text", ""))
    fallback = base.fallback_ranking(candidates, topic, users)
    top_k = min(10, pool_size)
    calls: list[base.ModelCall] = []
    observations: dict[str, dict[str, Any]] = {}
    expert_diagnostics: list[dict[str, Any]] = []
    coordinator_trace: list[dict[str, Any]] = []
    invalid_actions = 0
    final: list[str] = []
    final_valid = 0
    final_exact = False

    for step in range(len(enabled_experts) + 1):
        unused = [tool for tool in enabled_experts if tool not in observations]
        consensus, _, _ = base.consensus_ranking(candidates, fallback, observations)
        holistic = base.coordinator_holistic_evidence(
            record,
            candidates,
            users,
            topic,
            fallback,
            observations,
        )
        prompt = base.coordinator_prompt(
            topic, candidates, fallback, observations, unused, holistic
        )
        text, call = endpoints.call_coordinator(
            prompt, index * 17 + step, temperature, top_p
        )
        calls.append(call)
        payload = base.parse_json(text)
        action = str(payload.get("action", "")).strip().lower()
        tool = str(payload.get("expert", "")).strip().lower()
        trace: dict[str, Any] = {
            "step": step,
            "state_tools": list(observations),
            "available_experts": list(enabled_experts),
            "action": action,
            "expert": tool,
            "response_sha256": call.response_sha256,
            "valid": True,
        }
        if action == "call" and tool in unused:
            observation, new_calls, diag = base.execute_expert(
                tool,
                record,
                candidates,
                users,
                news,
                relations,
                memory,
                endpoints,
                index * 17 + step + 5,
                media_dir,
                temperature,
                top_p,
                fallback,
                relation_scope,
            )
            observations[tool] = observation
            calls.extend(new_calls)
            expert_diagnostics.append(diag)
            coordinator_trace.append(trace)
            continue
        if action == "stop":
            final, final_valid, final_exact = base.parse_ranking(
                text, candidates, top_k, consensus
            )
            coordinator_trace.append(trace)
            break
        trace["valid"] = False
        trace["error"] = (
            "expert unavailable under leave-one-out evaluation"
            if action == "call" and tool not in enabled_experts
            else "invalid action or repeated expert"
        )
        coordinator_trace.append(trace)
        invalid_actions += 1
        final = consensus[:top_k]
        break
    if not final:
        consensus, _, _ = base.consensus_ranking(candidates, fallback, observations)
        final = consensus[:top_k]
    rank = final.index(positive) + 1 if positive in final else pool_size
    expert_diagnostics.append(
        {
            "tool": "coordinator_magrpo_v5",
            "trace": coordinator_trace,
            "selected_tools": list(observations),
        }
    )
    return {
        "rank": rank,
        "top_user_ids": final,
        "called_experts": list(observations),
        "calls": [call.__dict__ for call in calls],
        "expert_diagnostics": expert_diagnostics,
        "expert_observations": observations,
        "fusion_diagnostics": {
            "coordinator_trace": coordinator_trace,
            "router_valid": invalid_actions == 0,
            "parsed_valid_ids": final_valid,
            "exact_top10": final_exact,
            "disabled_expert": disabled_expert,
            "available_experts": list(enabled_experts),
        },
        "invalid_router_actions": invalid_actions,
        "latency_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-model-name", default="CAMARL")
    parser.add_argument("--base-model-name", default="Qwen3.5_4B")
    parser.add_argument(
        "--disabled-expert",
        choices=tuple(base.TOOLS),
        help=(
            "Run a leave-one-expert-out evaluation by removing this expert from "
            "the coordinator prompt, valid action space, and executable calls."
        ),
    )
    parser.add_argument("--coordinator-adapter", type=Path, required=True)
    parser.add_argument("--policy-manifest", type=Path, required=True)
    parser.add_argument(
        "--trained-pool-size",
        type=int,
        choices=(20, 50, 100, 500, 1000, 1500, 2000),
        required=True,
    )
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13])
    parser.add_argument(
        "--pool-sizes", type=int, nargs="+", default=[20, 50, 100, 500]
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--server-max-model-len",
        type=int,
        help="Audited max-model-len used by every backing inference server.",
    )
    parser.add_argument(
        "--fail-on-request-error",
        action="store_true",
        help="Abort immediately when any model request fails.",
    )
    parser.add_argument(
        "--resume-partial",
        type=Path,
        help=(
            "Resume already completed inference seeds from an audited partial "
            "result. Completed seeds must contain the full test set and zero "
            "request errors."
        ),
    )
    parser.add_argument("--interest-threshold", type=float, default=0.20)
    parser.add_argument(
        "--relation-scope",
        choices=tuple(base.TOPOLOGY_SYSTEMS),
        default="prefix_aggregate",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument(
        "--media-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if len(base.TOOLS) != EXPERT_COST_DENOMINATOR:
        raise ValueError(
            "This ablation defines normalized expert cost against the original "
            f"three-expert system, but base.TOOLS={base.TOOLS}"
        )
    enabled_experts = available_experts(args.disabled_expert)
    evaluation_system_prompt = coordinator_system(enabled_experts)

    if args.pool_sizes != [args.trained_pool_size]:
        raise ValueError(
            "An independently trained MA-GRPO policy may only be evaluated on "
            f"its own pool: trained N={args.trained_pool_size}, requested "
            f"{args.pool_sizes}"
        )

    started = time.time()
    policy_manifest = json.loads(args.policy_manifest.read_text())
    if policy_manifest.get("test_records_used"):
        raise ValueError("Refusing a policy whose selection used test records")
    if not policy_manifest.get("policy_frozen_before_test"):
        raise ValueError("Policy manifest does not declare pre-test freezing")
    manifest_pool_size = policy_manifest.get(
        "pool_size", policy_manifest.get("requested_pool_size", -1)
    )
    if int(manifest_pool_size) != args.trained_pool_size:
        raise ValueError(
            "Policy manifest pool does not match the independently trained task: "
            f"manifest={manifest_pool_size} "
            f"requested={args.trained_pool_size}"
        )
    expert_space_audit = validate_policy_expert_space(
        policy_manifest, enabled_experts, args.disabled_expert
    )
    protocol_hash = base.sha256_file(
        args.graphhard_dir / "graphhard_protocol_report.json"
    )
    if protocol_hash != args.expected_protocol_sha256:
        raise ValueError(f"Unexpected graphhard protocol hash: {protocol_hash}")

    users = base.load_pickle(args.data_dir / "users_all.pkl")
    news = base.load_pickle(args.data_dir / "news_all.pkl")
    pools = {
        size: base.load_pickle(
            args.graphhard_dir / f"{args.split}_graphhard_pools_N{size}.pkl"
        )[: args.limit]
        for size in args.pool_sizes
    }
    counts = {len(values) for values in pools.values()}
    if len(counts) != 1:
        raise ValueError(f"Unaligned pool counts: {counts}")
    relations = base.RelationIndex(users, args.data_dir / "cascades.txt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "magrpo_coordinator_results.json"
    partial_path = output_path.with_suffix(".partial.json")

    seed_results: list[dict[str, Any]] = []
    resumed_seeds: list[int] = []
    if args.resume_partial is not None:
        partial = json.loads(args.resume_partial.read_text())
        partial_audit = partial.get("audit")
        if args.disabled_expert is not None and not isinstance(partial_audit, dict):
            raise ValueError(
                "A leave-one-out run cannot resume a legacy partial without "
                "expert-space audit metadata"
            )
        if isinstance(partial_audit, dict):
            expected_partial_metadata = {
                "disabled_expert": args.disabled_expert,
                "available_experts": list(enabled_experts),
                "trained_pool_size": args.trained_pool_size,
                "split": args.split,
                "expert_cost_denominator": EXPERT_COST_DENOMINATOR,
            }
            mismatches = {
                key: (partial_audit.get(key), expected)
                for key, expected in expected_partial_metadata.items()
                if partial_audit.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Resume partial belongs to a different evaluation: {mismatches}"
                )
        seen: set[int] = set()
        for seed_result in partial.get("seeds", []):
            seed = int(seed_result["seed"])
            if seed not in args.seeds:
                raise ValueError(f"Resume partial contains unexpected seed {seed}")
            if seed in seen:
                raise ValueError(f"Resume partial contains duplicate seed {seed}")
            pool = seed_result.get("pools", {}).get(str(args.trained_pool_size))
            if pool is None:
                raise ValueError(
                    f"Resume partial seed {seed} lacks N={args.trained_pool_size}"
                )
            if len(pool.get("ranks", [])) != next(iter(counts)):
                raise ValueError(
                    f"Resume partial seed {seed} is incomplete: "
                    f"{len(pool.get('ranks', []))}/{next(iter(counts))}"
                )
            if int(pool.get("diagnostics", {}).get("request_errors", -1)) != 0:
                raise ValueError(
                    f"Resume partial seed {seed} has request errors"
                )
            seed_results.append(seed_result)
            resumed_seeds.append(seed)
            seen.add(seed)

    monitor = base.GPUMemoryMonitor(gpu_ids=args.gpu_ids)
    monitor.start()
    inference_started = time.time()
    for seed in args.seeds:
        if seed in resumed_seeds:
            print(f"MA-GRPO seed={seed} resumed from {args.resume_partial}", flush=True)
            continue
        # An inference seed is an independent evaluation replicate.  Profile
        # memory may persist across queries within that replicate, but it must
        # never carry over to a later seed.
        static_interest_cache: dict[tuple[str, str], tuple[float, str, str]] = {}
        static_interest_cache_lock = threading.Lock()
        endpoints = CoordinatorEndpointPool(
            args.ports,
            args.base_model_name,
            args.coordinator_model_name,
            seed,
            evaluation_system_prompt,
        )
        endpoints.check()
        seed_pools: dict[str, Any] = {}
        for pool_size in args.pool_sizes:
            memory = base.InterestMemory(
                args.interest_threshold,
                static_interest_cache,
                static_interest_cache_lock,
            )
            records = pools[pool_size]
            outputs: list[dict[str, Any] | None] = [None] * len(records)
            pool_started = time.time()
            grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
            for index, record in enumerate(records):
                grouped[str(record["news_id"])].append((index, record))

            def run_group(
                values: list[tuple[int, dict[str, Any]]]
            ) -> list[tuple[int, dict[str, Any]]]:
                return [
                    (
                        index,
                        evaluate_query(
                            index,
                            record,
                            pool_size,
                            seed,
                            users,
                            news,
                            relations,
                            memory,
                            endpoints,
                            args.media_dir,
                            args.temperature,
                            args.top_p,
                            args.relation_scope,
                            enabled_experts,
                            args.disabled_expert,
                        ),
                    )
                    for index, record in values
                ]

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(run_group, values) for values in grouped.values()
                ]
                completed = 0
                for future in as_completed(futures):
                    before = completed
                    finished_values = future.result()
                    if args.fail_on_request_error:
                        request_errors = [
                            call["error"]
                            for _, value in finished_values
                            for call in value["calls"]
                            if call["error"]
                        ]
                        if request_errors:
                            raise RuntimeError(
                                "Model request failed under strict evaluation: "
                                f"{request_errors[0]}"
                            )
                    for index, value in finished_values:
                        outputs[index] = value
                        completed += 1
                    if completed // 50 != before // 50 or completed == len(records):
                        ranks = [int(value["rank"]) for value in outputs if value]
                        metric = base.ranking_metrics(ranks)
                        print(
                            f"MA-GRPO seed={seed} N={pool_size} "
                            f"{completed}/{len(records)} H@1={metric['H@1']:.4f} "
                            f"H@5={metric['H@5']:.4f} "
                            f"elapsed={time.time()-pool_started:.1f}s",
                            flush=True,
                        )
            rows = [value for value in outputs if value is not None]
            seed_pools[str(pool_size)] = base.summarize_rows(rows)
            diagnostics = seed_pools[str(pool_size)]["diagnostics"]
            diagnostics.update(
                {
                    "disabled_expert": args.disabled_expert,
                    "available_experts": list(enabled_experts),
                    "expert_cost_denominator": EXPERT_COST_DENOMINATOR,
                }
            )
            memory_users, memory_entries = memory.size()
            diagnostics.update(
                {"memory_users": memory_users, "memory_entries": memory_entries}
            )
            partial_path.write_text(
                json.dumps(
                    {
                        "audit": {
                            "evaluator_version": EVALUATOR_VERSION,
                            "disabled_expert": args.disabled_expert,
                            "available_experts": list(enabled_experts),
                            "trained_pool_size": args.trained_pool_size,
                            "split": args.split,
                            "expert_cost_denominator": EXPERT_COST_DENOMINATOR,
                        },
                        "seeds": seed_results
                        + [{"seed": seed, "pools": seed_pools}],
                    },
                    ensure_ascii=False,
                )
            )
        seed_results.append({"seed": seed, "pools": seed_pools})

    gpu = monitor.stop()
    report = {
        "audit": {
            "method": (
                "magrpo_v5_leave_one_expert_out"
                if args.disabled_expert is not None
                else "magrpo_v5_full_coordinator"
            ),
            "evaluator_version": EVALUATOR_VERSION,
            "base_model_name": args.base_model_name,
            "coordinator_model_name": args.coordinator_model_name,
            "ablation_type": (
                "leave_one_expert_out" if args.disabled_expert is not None else None
            ),
            "disabled_expert": args.disabled_expert,
            "available_experts": list(enabled_experts),
            "original_expert_space": list(base.TOOLS),
            "expert_cost_denominator": EXPERT_COST_DENOMINATOR,
            "policy_action_space_restricted_at_evaluation": (
                args.disabled_expert is not None
            ),
            "policy_expert_space_audit": expert_space_audit,
            "coordinator_adapter": str(args.coordinator_adapter),
            "coordinator_adapter_sha256_tree": sha256_tree(args.coordinator_adapter),
            "policy_manifest": str(args.policy_manifest),
            "policy_manifest_sha256": base.sha256_file(args.policy_manifest),
            "test_records_used_for_training_or_selection": False,
            "policy_frozen_before_test": True,
            "split": args.split,
            "relation_scope": args.relation_scope,
            "seeds": args.seeds,
            "pool_sizes": args.pool_sizes,
            "trained_pool_size": args.trained_pool_size,
            "cross_pool_policy_evaluation": False,
            "fail_on_request_error": args.fail_on_request_error,
            "server_max_model_len": args.server_max_model_len,
            "endpoint_scheduling": "least-inflight-round-robin-tiebreak",
            "resume_partial": (
                str(args.resume_partial) if args.resume_partial is not None else None
            ),
            "resumed_seeds": resumed_seeds,
            "records": next(iter(counts)),
            "candidate_pool": "frozen coverage-matched graph-hard protocol; unchanged",
            "graphhard_report_sha256": protocol_hash,
            "coordinator_prompt_sha256": hashlib.sha256(
                evaluation_system_prompt.encode()
            ).hexdigest(),
            "coordinator_prompt_source": (
                "build_magrpo_coordinator_dataset.coordinator_system"
            ),
            "full_coordinator_prompt_sha256": hashlib.sha256(
                COORDINATOR_SYSTEM.encode()
            ).hexdigest(),
            "action_space": (
                f"call one unused expert from {list(enabled_experts)}, or stop "
                "with the final top-10 ranking"
            ),
            "uses_test_label_in_prompt_or_features": False,
            "gpu": gpu,
        },
        "seeds": seed_results,
        "aggregate": base.aggregate(seed_results),
        "elapsed_seconds": time.time() - started,
        "inference_wall_seconds": time.time() - inference_started,
        "gpu_hours": (
            (time.time() - inference_started)
            * len(gpu["peak_memory_mib_per_gpu"])
            / 3600
        ),
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
