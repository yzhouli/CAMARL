#!/usr/bin/env python3
"""Build coverage-matched graph-hard validation/test candidate pools.

Negative candidates are ordered by the mean score of five frozen, train-only
user--cascade LightGCN checkpoints. Candidate coverage is matched using only
train-graph membership: warm targets receive warm negatives and cold targets
receive cold negatives. ``next_user`` determines only the coverage stratum and
is never used in score ordering.

This creates a topology-hard stress test.  It is intentionally stored beside,
not over, the existing text-hard protocol because every compared method must be
rerun before results from the new pools can share a table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


SEEDS = (13, 21, 34, 55, 89)
DEFAULT_POOL_SIZES = (20, 50, 100, 500)
DEFAULT_NEGATIVE_UNIVERSE_SIZE = 999


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranking_metrics(ranks: list[int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in (1, 2, 3, 5, 10):
        result[f"H@{k}"] = float(np.mean([rank <= k for rank in ranks]))
        result[f"MAP@{k}"] = float(
            np.mean([1.0 / rank if rank <= k else 0.0 for rank in ranks])
        )
        result[f"NDCG@{k}"] = float(
            np.mean(
                [1.0 / math.log2(rank + 1) if rank <= k else 0.0 for rank in ranks]
            )
        )
    return result


def train_graph_users(train_records: list[dict[str, Any]]) -> set[str]:
    users: set[str] = set()
    for record in train_records:
        users.update(map(str, record["history_users"]))
        users.add(str(record["next_user"]))
    return users


def coverage_matched_records(
    records: list[dict[str, Any]],
    all_users: list[str],
    warm_users: set[str],
    split: str,
    negative_universe_size: int,
) -> list[dict[str, Any]]:
    populations = {
        True: [user for user in all_users if user in warm_users],
        False: [user for user in all_users if user not in warm_users],
    }
    output: list[dict[str, Any]] = []
    for row, record in enumerate(records):
        positive = str(record["next_user"])
        is_warm = positive in warm_users
        population = populations[is_warm]
        forbidden = set(map(str, record["history_users"])) | {positive}
        protocol_seed = (
            "graphhard-v2"
            if negative_universe_size == 999
            else f"graphhard-v3-{negative_universe_size}"
        )
        digest = hashlib.sha256(
            f"{protocol_seed}\0{split}\0{record['news_id']}\0{row}".encode()
        ).digest()
        generator = random.Random(int.from_bytes(digest[:8], "big"))
        selected: set[str] = set()
        while len(selected) < negative_universe_size:
            candidate = population[generator.randrange(len(population))]
            if candidate not in forbidden:
                selected.add(candidate)
        updated = dict(record)
        updated["news_id"] = str(record["news_id"])
        updated["history_users"] = list(map(str, record["history_users"]))
        updated["next_user"] = positive
        updated["neg_users"] = sorted(selected)
        output.append(updated)
    return output


def build_index_tensors(
    records: list[dict[str, Any]],
    user_to_index: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    negative_rows: list[list[int]] = []
    positives: list[int] = []
    expected_negatives: int | None = None

    for row, record in enumerate(records):
        history = list(map(str, record["history_users"]))
        if history:
            rows.extend([row] * len(history))
            columns.extend(user_to_index[user] for user in history)
            values.extend([1.0 / len(history)] * len(history))
        negatives = list(map(str, record["neg_users"]))
        if expected_negatives is None:
            expected_negatives = len(negatives)
        if len(negatives) != expected_negatives:
            raise ValueError("Negative universes do not have a fixed size")
        if len(set(negatives)) != len(negatives):
            raise ValueError(f"Duplicate negative at record {row}")
        positive = str(record["next_user"])
        if positive in set(negatives):
            raise ValueError(f"Positive present in negative universe at record {row}")
        missing = [
            user
            for user in [*history, *negatives, positive]
            if user not in user_to_index
        ]
        if missing:
            raise KeyError(f"Users absent from full embedding table: {missing[:3]}")
        negative_rows.append([user_to_index[user] for user in negatives])
        positives.append(user_to_index[positive])

    query_matrix = torch.sparse_coo_tensor(
        torch.tensor([rows, columns], dtype=torch.long, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
        (len(records), len(user_to_index)),
        device=device,
    ).coalesce()
    return (
        query_matrix,
        torch.tensor(negative_rows, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
    )


@torch.no_grad()
def score_checkpoint(
    user_embedding: torch.Tensor,
    query_matrix: torch.Tensor,
    negative_indices: torch.Tensor,
    positive_indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding = user_embedding.to(device)
    query = torch.sparse.mm(query_matrix, embedding)
    negative_scores: list[torch.Tensor] = []
    positive_scores: list[torch.Tensor] = []
    for start in range(0, query.shape[0], batch_size):
        stop = min(start + batch_size, query.shape[0])
        batch_query = query[start:stop]
        batch_negatives = negative_indices[start:stop].to(device)
        scores = (
            embedding[batch_negatives] * batch_query.unsqueeze(1)
        ).sum(dim=-1)
        negative_scores.append(scores.cpu())
        batch_positives = positive_indices[start:stop].to(device)
        positive_scores.append(
            (embedding[batch_positives] * batch_query).sum(dim=-1).cpu()
        )
    return torch.cat(negative_scores), torch.cat(positive_scores)


def load_and_score(
    records: list[dict[str, Any]],
    checkpoint_paths: list[Path],
    device: torch.device,
    batch_size: int,
) -> tuple[
    dict[str, int],
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
]:
    first = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    user_to_index = {str(key): int(value) for key, value in first["user_to_index"].items()}
    query_matrix, negative_indices, positive_indices = build_index_tensors(
        records, user_to_index, device
    )
    seed_negative_scores: list[torch.Tensor] = []
    seed_positive_scores: list[torch.Tensor] = []

    for checkpoint_index, path in enumerate(checkpoint_paths):
        checkpoint = first if checkpoint_index == 0 else torch.load(
            path, map_location="cpu", weights_only=False
        )
        current_mapping = checkpoint["user_to_index"]
        if len(current_mapping) != len(user_to_index) or any(
            int(current_mapping[key]) != value
            for key, value in list(user_to_index.items())[:1000]
        ):
            raise ValueError(f"User mapping mismatch in {path}")
        negative_scores, positive_scores = score_checkpoint(
            checkpoint["user_embedding"],
            query_matrix,
            negative_indices,
            positive_indices,
            device,
            batch_size,
        )
        seed_negative_scores.append(negative_scores)
        seed_positive_scores.append(positive_scores)
        print(f"scored checkpoint {path.name}", flush=True)
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ensemble = torch.stack(seed_negative_scores).double().mean(dim=0)
    return user_to_index, seed_negative_scores, seed_positive_scores, ensemble


def materialize_split(
    split: str,
    records: list[dict[str, Any]],
    checkpoint_paths: list[Path],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    metadata: dict[str, Any],
    pool_sizes: tuple[int, ...],
) -> dict[str, Any]:
    _, seed_negative_scores, seed_positive_scores, ensemble = load_and_score(
        records, checkpoint_paths, device, batch_size
    )
    pools: dict[int, list[dict[str, Any]]] = {size: [] for size in pool_sizes}
    seed_ranks: dict[int, dict[int, list[int]]] = {
        seed: {size: [] for size in pool_sizes} for seed in SEEDS
    }
    ensemble_positive_ranks: list[int] = []

    for row, record in enumerate(records):
        negative_ids = list(map(str, record["neg_users"]))
        ordered_indices = sorted(
            range(len(negative_ids)),
            key=lambda index: (-float(ensemble[row, index]), negative_ids[index]),
        )
        ensemble_positive = float(
            torch.stack([scores[row] for scores in seed_positive_scores]).mean()
        )
        ensemble_positive_ranks.append(
            1 + sum(float(ensemble[row, index]) > ensemble_positive for index in ordered_indices)
        )

        for size in pool_sizes:
            selected_indices = ordered_indices[: size - 1]
            selected_ids = [negative_ids[index] for index in selected_indices]
            pools[size].append(
                {
                    "news_id": str(record["news_id"]),
                    "history_users": list(map(str, record["history_users"])),
                    "next_user": str(record["next_user"]),
                    "neg_users": selected_ids,
                    "graph_miner_scores": {
                        candidate: float(ensemble[row, index])
                        for candidate, index in zip(selected_ids, selected_indices)
                    },
                    "metadata": metadata,
                }
            )
            candidate_ids = [*selected_ids, str(record["next_user"])]
            for seed_index, seed in enumerate(SEEDS):
                pairs = [
                    (
                        negative_ids[index],
                        float(seed_negative_scores[seed_index][row, index]),
                    )
                    for index in selected_indices
                ]
                pairs.append(
                    (
                        str(record["next_user"]),
                        float(seed_positive_scores[seed_index][row]),
                    )
                )
                ordered = sorted(pairs, key=lambda pair: (-pair[1], pair[0]))
                positive = str(record["next_user"])
                seed_ranks[seed][size].append(
                    1 + next(index for index, pair in enumerate(ordered) if pair[0] == positive)
                )
            if len(set(candidate_ids)) != size:
                raise AssertionError(f"Malformed N={size} pool at record {row}")
        if (row + 1) % 100 == 0:
            print(f"{split}: materialized {row + 1}/{len(records)}", flush=True)

    for size, values in pools.items():
        dump_pickle(values, output_dir / f"{split}_graphhard_pools_N{size}.pkl")

    seed_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_results.append(
            {
                "seed": seed,
                "pools": {
                    str(size): {
                        "ranks": seed_ranks[seed][size],
                        "metrics": ranking_metrics(seed_ranks[seed][size]),
                    }
                    for size in pool_sizes
                },
            }
        )
    aggregate: dict[str, Any] = {}
    for size in pool_sizes:
        aggregate[str(size)] = {}
        for key in seed_results[0]["pools"][str(size)]["metrics"]:
            values = np.asarray(
                [result["pools"][str(size)]["metrics"][key] for result in seed_results]
            )
            aggregate[str(size)][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    return {
        "records": len(records),
        "negative_universe_size": len(records[0]["neg_users"]),
        "positive_rank_in_ensemble_universe": {
            "mean": float(np.mean(ensemble_positive_ranks)),
            "median": float(np.median(ensemble_positive_ranks)),
        },
        "seed_results": seed_results,
        "aggregate": aggregate,
    }


def audit_nested_pools(
    output_dir: Path,
    split: str,
    warm_users: set[str],
    pool_sizes: tuple[int, ...],
) -> dict[str, Any]:
    loaded = {
        size: load_pickle(output_dir / f"{split}_graphhard_pools_N{size}.pkl")
        for size in pool_sizes
    }
    report: dict[str, Any] = {"records": len(loaded[pool_sizes[0]])}
    for size, records in loaded.items():
        malformed = 0
        positive_in_negatives = 0
        duplicate_negatives = 0
        coverage_mismatches = 0
        for record in records:
            negatives = list(map(str, record["neg_users"]))
            malformed += len(negatives) != size - 1
            positive_in_negatives += str(record["next_user"]) in set(negatives)
            duplicate_negatives += len(negatives) != len(set(negatives))
            positive_is_warm = str(record["next_user"]) in warm_users
            coverage_mismatches += sum(
                (negative in warm_users) != positive_is_warm
                for negative in negatives
            )
        report[str(size)] = {
            "malformed": malformed,
            "positive_in_negatives": positive_in_negatives,
            "duplicate_negatives": duplicate_negatives,
            "coverage_mismatches": coverage_mismatches,
        }
    nesting_violations = 0
    for row in range(report["records"]):
        previous: list[str] = []
        for size in pool_sizes:
            current = list(map(str, loaded[size][row]["neg_users"]))
            if previous and current[: len(previous)] != previous:
                nesting_violations += 1
            previous = current
    report["nesting_violations"] = nesting_violations
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--negative-universe-size",
        type=int,
        default=DEFAULT_NEGATIVE_UNIVERSE_SIZE,
    )
    parser.add_argument(
        "--pool-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_POOL_SIZES),
        help="Nested candidate-pool sizes to materialize.",
    )
    args = parser.parse_args()
    pool_sizes = tuple(sorted(set(args.pool_sizes)))
    if not pool_sizes or pool_sizes[0] < 2:
        raise ValueError("Every pool size must be at least 2")
    if pool_sizes[-1] - 1 > args.negative_universe_size:
        raise ValueError(
            f"N={pool_sizes[-1]} needs {pool_sizes[-1] - 1} negatives, "
            f"but the sampled universe has only {args.negative_universe_size}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_paths = [
        args.protocol_dir / f"interaction_twotower_seed{seed}.pt" for seed in SEEDS
    ]
    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    started = time.time()
    train_records = load_pickle(args.protocol_dir / "train_records_N1000.pkl")
    warm_users = train_graph_users(train_records)
    first_checkpoint = torch.load(
        checkpoint_paths[0], map_location="cpu", weights_only=False
    )
    all_users = sorted(map(str, first_checkpoint["user_to_index"]))
    del first_checkpoint
    metadata = {
        "protocol": (
            "graphhard-v2-coverage-matched"
            if args.negative_universe_size == 999
            else f"graphhard-v3-coverage-matched-{args.negative_universe_size}"
        ),
        "negative_ordering": "mean raw dot product over frozen train-only bipartite LightGCN seeds",
        "candidate_universe": (
            f"deterministic {args.negative_universe_size}-user sample from the "
            "target's train-coverage stratum"
        ),
        "coverage_definition": "user appears in a train history or train next-user field",
        "uses_next_user_for_coverage_stratum": True,
        "seeds": list(SEEDS),
        "uses_next_user_for_negative_ordering": False,
        "uses_user_user_edges": False,
        "pool_sizes": list(pool_sizes),
        "checkpoint_sha256": {
            path.name: sha256_file(path) for path in checkpoint_paths
        },
    }
    validation_records = coverage_matched_records(
        load_pickle(args.protocol_dir / "validation_records_N1000.pkl"),
        all_users,
        warm_users,
        "validation",
        args.negative_universe_size,
    )
    test_records = coverage_matched_records(
        load_pickle(args.data_dir / "test_aligned.pkl"),
        all_users,
        warm_users,
        "test",
        args.negative_universe_size,
    )
    validation_report = materialize_split(
        "validation",
        validation_records,
        checkpoint_paths,
        args.output_dir,
        device,
        args.batch_size,
        metadata,
        pool_sizes,
    )
    test_report = materialize_split(
        "test",
        test_records,
        checkpoint_paths,
        args.output_dir,
        device,
        args.batch_size,
        metadata,
        pool_sizes,
    )
    report = {
        "metadata": metadata,
        "validation": validation_report,
        "test": test_report,
        "audit": {
            "validation": audit_nested_pools(
                args.output_dir, "validation", warm_users, pool_sizes
            ),
            "test": audit_nested_pools(
                args.output_dir, "test", warm_users, pool_sizes
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    output = args.output_dir / "graphhard_protocol_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["test"]["aggregate"], ensure_ascii=False, indent=2))
    print(json.dumps(report["audit"], ensure_ascii=False, indent=2))
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
