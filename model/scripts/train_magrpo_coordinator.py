#!/usr/bin/env python3
"""Validation-only group-relative training for the CAMARL coordinator.

The policy emits either one tool call or a stop action with the final ranking.
Tool observations are part of the state prompt, so environment-return tokens
are naturally excluded from the policy loss.  Reward-only target columns are
never formatted into the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


TOOLS = ("semantic", "profile", "topology")
ACTIVE_TOOLS = TOOLS
TRAINING_VERSION = "camarl-full-coordinator-lora-v1"
DEFAULT_REWARD_SPEC = {
    "ranking_utility_weight": 0.65,
    "oracle_action_weight": 0.30,
    "valid_format_bonus": 0.05,
    "normalized_future_tool_cost_penalty": 0.10,
    "invalid_format_reward": -0.50,
}
REWARD_SPEC = dict(DEFAULT_REWARD_SPEC)


def configure_reward_spec(args: argparse.Namespace) -> dict[str, float]:
    """Validate and install one auditable reward mixture.

    The three positive reward coefficients are constrained to the unit simplex.
    This removes an otherwise unidentified global reward scale and makes trials
    comparable.  The tool-cost coefficient remains independent because it is a
    penalty trading ranking quality against inference cost.
    """
    spec = {
        "ranking_utility_weight": float(args.ranking_utility_weight),
        "oracle_action_weight": float(args.oracle_action_weight),
        "valid_format_bonus": float(args.valid_format_bonus),
        "normalized_future_tool_cost_penalty": float(args.tool_cost_penalty),
        "invalid_format_reward": float(args.invalid_format_reward),
    }
    if not all(math.isfinite(value) for value in spec.values()):
        raise ValueError(f"Reward coefficients must be finite: {spec}")
    positive_keys = (
        "ranking_utility_weight",
        "oracle_action_weight",
        "valid_format_bonus",
    )
    if any(spec[key] < 0.0 for key in positive_keys):
        raise ValueError(f"Positive reward coefficients must be non-negative: {spec}")
    positive_sum = sum(spec[key] for key in positive_keys)
    if not math.isclose(positive_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "ranking/oracle/format coefficients must sum to 1.0; "
            f"got {positive_sum:.12g} from {spec}"
        )
    if spec["normalized_future_tool_cost_penalty"] < 0.0:
        raise ValueError(f"Tool-cost penalty must be non-negative: {spec}")
    if spec["invalid_format_reward"] > 0.0:
        raise ValueError(f"Invalid-format reward must be non-positive: {spec}")
    global REWARD_SPEC
    REWARD_SPEC = spec
    return dict(spec)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_chunked_generation(
    model: Any, chunk_size: int, pad_token_id: int
) -> None:
    """Run a large local generation batch through smaller forward chunks.

    TRL still receives the complete generation batch, so reward grouping,
    batch-level advantage scaling, and the optimizer batch are unchanged. The
    split only keeps Qwen3.5's Conv1d prefill tensors below CUDA's 32-bit index
    limit when a four-GPU training layout is reproduced on one GPU.
    """
    original_generate = model.generate

    def generate_in_chunks(*args: Any, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs.get("input_ids")
        if (
            args
            or not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim == 0
            or input_ids.shape[0] <= chunk_size
        ):
            return original_generate(*args, **kwargs)

        batch_size = int(input_ids.shape[0])
        outputs: list[torch.Tensor] = []
        for start in range(0, batch_size, chunk_size):
            stop = min(start + chunk_size, batch_size)
            chunk_kwargs = {
                key: (
                    value[start:stop]
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and int(value.shape[0]) == batch_size
                    else value
                )
                for key, value in kwargs.items()
            }
            output = original_generate(**chunk_kwargs)
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise TypeError(
                    "Chunked coordinator generation requires a rank-2 token tensor"
                )
            outputs.append(output)

        max_length = max(int(output.shape[1]) for output in outputs)
        padded = []
        for output in outputs:
            if int(output.shape[1]) < max_length:
                suffix = torch.full(
                    (int(output.shape[0]), max_length - int(output.shape[1])),
                    pad_token_id,
                    dtype=output.dtype,
                    device=output.device,
                )
                output = torch.cat((output, suffix), dim=1)
            padded.append(output)
        return torch.cat(padded, dim=0)

    model.generate = generate_in_chunks


def completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"] or "")
    if isinstance(value, dict):
        return str(value.get("content", value))
    return str(value or "")


def parse_payload(value: Any) -> dict[str, Any]:
    text = completion_text(value)
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end]) if start >= 0 and end > start else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def ranking_utility(rank: int) -> float:
    if rank <= 0:
        return 0.0
    ndcg10 = 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
    return (
        0.45 * float(rank == 1)
        + 0.25 * ndcg10
        + 0.20 * float(rank <= 5)
        + 0.10 * float(rank <= 10)
    )


def coordinator_reward(
    completions: list[Any],
    action_scores: list[str],
    oracle_action: list[dict[str, str]],
    candidate_ids_reward_only: list[list[str]],
    positive_id_reward_only: list[str],
    top_k_reward_only: list[int],
    **_: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, raw_scores, raw_oracle, candidates, positive, top_k in zip(
        completions,
        action_scores,
        oracle_action,
        candidate_ids_reward_only,
        positive_id_reward_only,
        top_k_reward_only,
    ):
        payload = parse_payload(completion)
        action = str(payload.get("action", "")).strip().lower()
        table = json.loads(raw_scores)
        selected_key = ""
        utility = 0.0
        future_cost = 0.0
        valid = False

        if action == "call":
            expert = str(payload.get("expert", "")).strip().lower()
            selected_key = f"call:{expert}"
            valid = expert in ACTIVE_TOOLS and selected_key in table
            if valid:
                utility = float(table[selected_key]["rank_utility"])
                future_cost = float(table[selected_key]["normalized_tool_cost"])
        elif action == "stop":
            selected_key = "stop"
            raw_top = payload.get("top_user_ids", payload.get("top", []))
            ranking = [str(value) for value in raw_top] if isinstance(raw_top, list) else []
            candidate_set = set(map(str, candidates))
            top_k = int(top_k)
            valid = (
                selected_key in table
                and len(ranking) == top_k
                and len(set(ranking)) == top_k
                and all(user in candidate_set for user in ranking)
            )
            if valid:
                positive = str(positive)
                rank = ranking.index(positive) + 1 if positive in ranking else len(candidates)
                utility = ranking_utility(rank)
                future_cost = 0.0

        if not valid:
            rewards.append(REWARD_SPEC["invalid_format_reward"])
            continue
        oracle_key = (
            "stop"
            if raw_oracle.get("action") == "stop"
            else f"call:{raw_oracle.get('expert', '')}"
        )
        exact = float(selected_key == oracle_key)
        reward = (
            REWARD_SPEC["ranking_utility_weight"] * utility
            + REWARD_SPEC["oracle_action_weight"] * exact
            + REWARD_SPEC["valid_format_bonus"]
            - REWARD_SPEC["normalized_future_tool_cost_penalty"] * future_cost
        )
        rewards.append(float(reward))
    return rewards


def supported_config(**values: Any) -> tuple[GRPOConfig, dict[str, Any]]:
    signature = inspect.signature(GRPOConfig.__init__)
    supported = {key: value for key, value in values.items() if key in signature.parameters}
    return GRPOConfig(**supported), supported


def select_partition(dataset: Any, max_queries_per_pool: int) -> Any:
    selected = dataset.filter(lambda value: value["partition"] == "train")
    chosen: set[str] = set()
    pools = sorted(set(map(int, selected["pool_size"])))
    for pool_size in pools:
        values = {
            str(query_id)
            for query_id, current_pool in zip(
                selected["query_id"], selected["pool_size"]
            )
            if int(current_pool) == pool_size
        }
        ordered = sorted(
            values, key=lambda value: hashlib.sha256(value.encode()).hexdigest()
        )
        chosen.update(ordered[:max_queries_per_pool])
    indices = [
        index
        for index, query_id in enumerate(selected["query_id"])
        if str(query_id) in chosen
    ]
    return selected.select(indices)


def deterministic_pool_order(
    dataset: Any, prompts_per_step: int, seed: int
) -> Any:
    """Order one pool deterministically without semantic curriculum.

    Independent-N training must never interleave examples from other pool
    sizes.  Chunking by optimizer step keeps reproducible batching while the
    seed hash prevents source-file ordering from acting as a curriculum.
    """
    if prompts_per_step <= 0:
        raise ValueError("prompts_per_step must be positive")
    pools = sorted(set(map(int, dataset["pool_size"])))
    if len(pools) != 1:
        raise ValueError(
            f"Independent-pool trainer requires exactly one pool, got {pools}"
        )
    ordered = list(range(len(dataset)))
    ordered.sort(
        key=lambda index: hashlib.sha256(
            f"{seed}:{dataset[index]['example_id']}".encode()
        ).hexdigest()
    )
    return dataset.select(ordered)


def disable_thinking_prompts(dataset: Any, tokenizer: Any) -> Any:
    """Render Qwen chat prompts once with thinking disabled.

    GRPOTrainer 0.23 does not forward per-example chat-template kwargs.  If we
    leave the prompt conversational, Qwen3.5 enters its long reasoning mode
    and can consume the whole completion budget before emitting coordinator
    JSON.  A pre-rendered string bypasses that second template application.
    """

    def render(example: dict[str, Any]) -> dict[str, str]:
        prompt = example["prompt"]
        if isinstance(prompt, str):
            return {"prompt": prompt}
        return {
            "prompt": tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        }

    return dataset.map(render, desc="Rendering coordinator prompts without thinking")


def main() -> None:
    global ACTIVE_TOOLS
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pool-size",
        type=int,
        choices=(20, 50, 100, 500, 1000, 1500, 2000),
        required=True,
    )
    parser.add_argument("--max-train-queries-per-pool", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=160)
    parser.add_argument("--rollout-temperature", type=float, default=0.9)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--grpo-beta", type=float, default=0.02)
    parser.add_argument("--clip-epsilon-low", type=float, default=0.20)
    parser.add_argument("--clip-epsilon-high", type=float, default=0.28)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--generation-chunk-size",
        type=int,
        default=0,
        help=(
            "systems-only local generation chunk size; zero keeps the native "
            "TRL generation batch"
        ),
    )
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--ranking-utility-weight",
        type=float,
        default=DEFAULT_REWARD_SPEC["ranking_utility_weight"],
    )
    parser.add_argument(
        "--oracle-action-weight",
        type=float,
        default=DEFAULT_REWARD_SPEC["oracle_action_weight"],
    )
    parser.add_argument(
        "--valid-format-bonus",
        type=float,
        default=DEFAULT_REWARD_SPEC["valid_format_bonus"],
    )
    parser.add_argument(
        "--tool-cost-penalty",
        type=float,
        default=DEFAULT_REWARD_SPEC["normalized_future_tool_cost_penalty"],
    )
    parser.add_argument(
        "--invalid-format-reward",
        type=float,
        default=DEFAULT_REWARD_SPEC["invalid_format_reward"],
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    reward_spec = configure_reward_spec(args)
    if args.save_steps <= 0:
        raise ValueError("--save-steps must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not 0.0 < args.rollout_temperature:
        raise ValueError("--rollout-temperature must be positive")
    if not 0.0 < args.rollout_top_p <= 1.0:
        raise ValueError("--rollout-top-p must be in (0, 1]")
    if args.grpo_beta < 0.0:
        raise ValueError("--grpo-beta must be non-negative")
    if not 0.0 < args.clip_epsilon_low < 1.0:
        raise ValueError("--clip-epsilon-low must be in (0, 1)")
    if not args.clip_epsilon_low <= args.clip_epsilon_high < 1.0:
        raise ValueError("--clip-epsilon-high must be in [clip-epsilon-low, 1)")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in [0, 1)")
    if args.generation_chunk_size < 0:
        raise ValueError("--generation-chunk-size must be non-negative")
    if args.per_device_batch_size <= 0:
        raise ValueError("--per-device-batch-size must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = world_size * args.per_device_batch_size
    manifest_path = args.dataset.with_suffix(args.dataset.suffix + ".manifest.json")
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("split") != "validation" or source_manifest.get(
        "test_records_used"
    ):
        raise ValueError("Refusing a MA-GRPO dataset that is not validation-only")
    if source_manifest.get("target_visible_to_model"):
        raise ValueError("Refusing a dataset that exposes the target to the model")
    disabled_expert = source_manifest.get("disabled_expert")
    declared_experts = source_manifest.get("available_experts")
    if disabled_expert is not None and disabled_expert not in TOOLS:
        raise ValueError(f"Dataset manifest disables unknown expert {disabled_expert}")
    if declared_experts is None:
        available_tools = TOOLS
    else:
        if not isinstance(declared_experts, list) or any(
            not isinstance(tool, str) or tool not in TOOLS
            for tool in declared_experts
        ):
            raise ValueError(
                f"Dataset manifest has invalid available_experts={declared_experts}"
            )
        available_tools = tuple(tool for tool in TOOLS if tool in declared_experts)
        if len(available_tools) != len(declared_experts):
            raise ValueError("Dataset manifest contains duplicate available experts")
    expected_tools = tuple(tool for tool in TOOLS if tool != disabled_expert)
    if available_tools != expected_tools:
        raise ValueError(
            "Dataset manifest expert-space mismatch: "
            f"available={available_tools}, disabled={disabled_expert}"
        )
    cost_denominator = int(
        source_manifest.get("tool_cost_normalization_denominator", len(TOOLS))
    )
    if cost_denominator != len(TOOLS):
        raise ValueError(
            "Expert cost must remain normalized by the original three-expert space"
        )
    ACTIVE_TOOLS = available_tools
    manifest_pools = set(map(int, source_manifest.get("pool_sizes", [])))
    if args.pool_size not in manifest_pools:
        raise ValueError(
            f"Requested N={args.pool_size} absent from dataset manifest {sorted(manifest_pools)}"
        )

    dataset = load_dataset("json", data_files=str(args.dataset), split="train")
    dataset = dataset.filter(lambda value: int(value["pool_size"]) == args.pool_size)
    train_dataset = select_partition(dataset, args.max_train_queries_per_pool)
    if not len(train_dataset):
        raise ValueError("No MA-GRPO training examples selected")
    prompts_per_step_numerator = (
        global_batch * args.gradient_accumulation_steps
    )
    if prompts_per_step_numerator % args.num_generations:
        raise ValueError(
            "The accumulated global completion batch must be divisible by "
            "num_generations"
        )
    prompts_per_step = prompts_per_step_numerator // args.num_generations
    train_dataset = deterministic_pool_order(
        train_dataset, prompts_per_step, args.seed
    )
    selected_pools = sorted(set(map(int, train_dataset["pool_size"])))
    if selected_pools != [args.pool_size]:
        raise ValueError(
            f"Pool isolation failed: requested {[args.pool_size]}, selected {selected_pools}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config, config_values = supported_config(
        output_dir=str(args.output_dir),
        run_name=(
            f"{TRAINING_VERSION}-N{args.pool_size}"
            if disabled_expert is None
            else f"{TRAINING_VERSION}-N{args.pool_size}-no-{disabled_expert}"
        ),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=prompts_per_step_numerator,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        beta=args.grpo_beta,
        epsilon=args.clip_epsilon_low,
        epsilon_high=args.clip_epsilon_high,
        loss_type="dapo",
        scale_rewards="batch",
        mask_truncated_completions=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=4,
        report_to="none",
        remove_unused_columns=False,
        shuffle_dataset=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    target_modules = (
        r"^model\.language_model\.layers\.\d+\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(gate_proj|up_proj|down_proj))$"
    )
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    # The coordinator policy is text-only.  Passing the tokenizer explicitly
    # prevents TRL from instantiating Qwen3.5's multimodal AutoProcessor (and
    # its optional image/video dependency chain) for a text-only GRPO run.
    processing_class = AutoTokenizer.from_pretrained(str(args.model))
    train_dataset = disable_thinking_prompts(train_dataset, processing_class)
    model_config = transformers.AutoConfig.from_pretrained(str(args.model))
    architecture = getattr(transformers, model_config.architectures[0])
    policy_model = architecture.from_pretrained(
        str(args.model), dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # TRL 0.23 uses this legacy Trainer warning registry; Transformers 5.7 no
    # longer creates it on PreTrainedModel, so restore the harmless dictionary
    # explicitly instead of patching either installed package.
    policy_model.warnings_issued = {}
    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=coordinator_reward,
        args=config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    if args.generation_chunk_size:
        if processing_class.pad_token_id is None:
            raise ValueError("Chunked generation requires a tokenizer pad token")
        install_chunked_generation(
            trainer.model,
            args.generation_chunk_size,
            int(processing_class.pad_token_id),
        )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))

    import accelerate
    import datasets
    import peft
    import trl

    audit = {
        "training_version": TRAINING_VERSION,
        "task": f"N={args.pool_size}",
        "requested_pool_size": args.pool_size,
        "base_model": str(args.model),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "source_split": source_manifest["split"],
        "test_records_used": False,
        "target_visible_to_model": False,
        "training_state_records": len(train_dataset),
        "training_query_records": len(set(map(str, train_dataset["query_id"]))),
        "states_per_query": source_manifest["states_per_query"],
        "policy": (
            "full coordinator: call one unused expert or stop with ranking"
            if disabled_expert is None
            else (
                "leave-one-expert-out coordinator: call one available unused "
                "expert or stop"
            )
        ),
        "available_experts": list(available_tools),
        "disabled_expert": disabled_expert,
        "tool_cost_normalization_denominator": cost_denominator,
        "pool_sizes": selected_pools,
        "reward_spec": reward_spec,
        "reward_positive_coefficients_sum": sum(
            reward_spec[key]
            for key in (
                "ranking_utility_weight",
                "oracle_action_weight",
                "valid_format_bonus",
            )
        ),
        "group_relative_completions": args.num_generations,
        "independent_pool_training": True,
        "cross_pool_policy_sharing": False,
        "deterministic_pool_order": True,
        "prompts_per_optimizer_step": prompts_per_step,
        "world_size": world_size,
        "generation_execution": {
            "chunk_size": args.generation_chunk_size,
            "loss_microbatch_size": global_batch,
            "generation_batch_size": prompts_per_step_numerator,
            "statistical_batch_unchanged": True,
        },
        "grpo_config": config_values,
        "lora": peft_config.to_dict(),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "trl": trl.__version__,
            "accelerate": accelerate.__version__,
        },
        "final_adapter": str(final_dir),
    }
    (args.output_dir / "training_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
