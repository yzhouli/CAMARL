#!/usr/bin/env python3
"""Train and evaluate the train-only user--cascade interaction baseline.

The graph is strictly bipartite: user nodes are linked only to the 4,657
training cascade nodes in which they participated.  No user--user edge,
profile/text feature, validation/test target edge, or legacy embedding is read.

The embedding table covers every dataset/candidate user.  Users without a
positive training interaction are isolated during propagation but can be drawn
as BPR negatives.  At validation/test time an unseen cascade is represented by
the mean of all ``history_users`` embeddings.  Ranking uses the raw
cascade--user embedding dot product with deterministic ID tie breaking.
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
import torch.nn as nn
import torch.nn.functional as F


SEEDS = (13, 21, 34, 55, 89)
POOL_SIZES = (20, 50, 100, 500)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_participants(record: dict[str, Any]) -> list[str]:
    return sorted(
        set(map(str, record["history_users"])) | {str(record["next_user"])}
    )


def build_train_graph(
    records: list[dict[str, Any]], all_user_ids: set[str], device: torch.device
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor, torch.Tensor]:
    participants = [unique_participants(record) for record in records]
    interaction_users = {user for users in participants for user in users}
    user_ids = sorted(all_user_ids | interaction_users)
    user_to_index = {user: index for index, user in enumerate(user_ids)}

    edge_users: list[int] = []
    edge_cascades: list[int] = []
    for cascade_index, users in enumerate(participants):
        edge_users.extend(user_to_index[user] for user in users)
        edge_cascades.extend([cascade_index] * len(users))

    user_index = torch.tensor(edge_users, dtype=torch.long, device=device)
    cascade_index = torch.tensor(edge_cascades, dtype=torch.long, device=device)
    number_users = len(user_ids)
    number_cascades = len(records)
    user_degree = torch.bincount(user_index, minlength=number_users).float()
    cascade_degree = torch.bincount(
        cascade_index, minlength=number_cascades
    ).float()
    weights = torch.rsqrt(
        user_degree[user_index].clamp_min(1)
        * cascade_degree[cascade_index].clamp_min(1)
    )
    cascade_nodes = cascade_index + number_users
    adjacency = torch.sparse_coo_tensor(
        torch.stack(
            [
                torch.cat([user_index, cascade_nodes]),
                torch.cat([cascade_nodes, user_index]),
            ]
        ),
        torch.cat([weights, weights]),
        (number_users + number_cascades, number_users + number_cascades),
        device=device,
    ).coalesce()
    return user_to_index, adjacency, user_index, cascade_index


class BipartiteLightGCN(nn.Module):
    def __init__(
        self,
        number_users: int,
        number_cascades: int,
        dimension: int,
        layers: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.number_users = number_users
        self.number_cascades = number_cascades
        self.layers = layers
        self.register_buffer("adjacency", adjacency)
        self.user_embedding = nn.Embedding(number_users, dimension)
        self.cascade_embedding = nn.Embedding(number_cascades, dimension)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.cascade_embedding.weight)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.cat(
            [self.user_embedding.weight, self.cascade_embedding.weight], dim=0
        )
        values = [value]
        for _ in range(self.layers):
            value = torch.sparse.mm(self.adjacency, value)
            values.append(value)
        final = torch.stack(values, dim=0).mean(dim=0)
        return final[: self.number_users], final[self.number_users :]


def sampled_bpr_loss(
    model: BipartiteLightGCN,
    edge_users: torch.Tensor,
    edge_cascades: torch.Tensor,
    sample_count: int,
    generator: torch.Generator,
    regularization: float,
) -> torch.Tensor:
    if edge_users.numel() > sample_count:
        selected = torch.randperm(
            edge_users.numel(), generator=generator, device=edge_users.device
        )[:sample_count]
        positive_users = edge_users[selected]
        cascades = edge_cascades[selected]
    else:
        positive_users = edge_users
        cascades = edge_cascades
    negative_users = torch.randint(
        0,
        model.number_users,
        positive_users.shape,
        generator=generator,
        device=positive_users.device,
    )
    negative_users = torch.where(
        negative_users == positive_users,
        (negative_users + 1) % model.number_users,
        negative_users,
    )
    user_embedding, cascade_embedding = model()
    query = cascade_embedding[cascades]
    positive_score = (query * user_embedding[positive_users]).sum(dim=-1)
    negative_score = (query * user_embedding[negative_users]).sum(dim=-1)
    bpr = -F.logsigmoid(positive_score - negative_score).mean()
    raw_regularization = (
        model.user_embedding(positive_users).square().sum(dim=-1)
        + model.user_embedding(negative_users).square().sum(dim=-1)
        + model.cascade_embedding(cascades).square().sum(dim=-1)
    ).mean()
    return bpr + regularization * raw_regularization


def embedding_for_user(
    user_id: str, user_embedding: torch.Tensor, user_to_index: dict[str, int]
) -> torch.Tensor:
    index = user_to_index.get(str(user_id))
    if index is None:
        return torch.zeros(
            user_embedding.shape[1], dtype=user_embedding.dtype, device=user_embedding.device
        )
    return user_embedding[index]


def cascade_from_history(
    history_users: list[Any],
    user_embedding: torch.Tensor,
    user_to_index: dict[str, int],
) -> tuple[torch.Tensor, int]:
    indices = [
        user_to_index[str(user)]
        for user in history_users
        if str(user) in user_to_index
    ]
    if not indices:
        return (
            torch.zeros(
                user_embedding.shape[1],
                dtype=user_embedding.dtype,
                device=user_embedding.device,
            ),
            0,
        )
    index_tensor = torch.tensor(indices, dtype=torch.long, device=user_embedding.device)
    return user_embedding[index_tensor].mean(dim=0), len(indices)


def build_validation_tensors(
    records: list[dict[str, Any]],
    user_to_index: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(2027)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    positives: list[int] = []
    negative_rows: list[list[int]] = []
    for row, record in enumerate(records):
        history = [
            user_to_index[str(user)]
            for user in record["history_users"]
            if str(user) in user_to_index
        ]
        if history:
            rows.extend([row] * len(history))
            columns.extend(history)
            values.extend([1.0 / len(history)] * len(history))
        positives.append(user_to_index[str(record["next_user"])])
        negatives = list(map(str, record["neg_users"]))
        if len(negatives) > 128:
            negatives = rng.sample(negatives, 128)
        negative_rows.append([user_to_index[user] for user in negatives])
    query_matrix = torch.sparse_coo_tensor(
        torch.tensor([rows, columns], dtype=torch.long, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
        (len(records), len(user_to_index)),
        device=device,
    ).coalesce()
    return (
        query_matrix,
        torch.tensor(positives, dtype=torch.long, device=device),
        torch.tensor(negative_rows, dtype=torch.long, device=device),
    )


@torch.no_grad()
def validation_bpr(
    user_embedding: torch.Tensor,
    query_matrix: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
) -> float:
    query = torch.sparse.mm(query_matrix, user_embedding)
    positive_score = (query * user_embedding[positive_indices]).sum(dim=-1, keepdim=True)
    negative_score = (
        query.unsqueeze(1) * user_embedding[negative_indices]
    ).sum(dim=-1)
    return float((-F.logsigmoid(positive_score - negative_score)).mean().cpu())


def train_one_seed(
    seed: int,
    number_users: int,
    adjacency: torch.Tensor,
    edge_users: torch.Tensor,
    edge_cascades: torch.Tensor,
    validation_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dimension: int,
    layers: int,
    epochs: int,
    sample_count: int,
    regularization: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, float]], int]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = BipartiteLightGCN(
        number_users,
        int(edge_cascades.max().item()) + 1,
        dimension,
        layers,
        adjacency,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_user_embedding: torch.Tensor | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss = sampled_bpr_loss(
            model,
            edge_users,
            edge_cascades,
            sample_count,
            generator,
            regularization,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            user_embedding, _ = model()
            value = validation_bpr(user_embedding, *validation_tensors)
        row = {
            "epoch": float(epoch),
            "train_bpr": float(loss.detach().cpu()),
            "validation_bpr": value,
        }
        history.append(row)
        print(f"seed={seed} {json.dumps(row)}", flush=True)
        if value < best_validation:
            best_validation = value
            best_epoch = epoch
            best_state = {
                "user_embedding.weight": model.user_embedding.weight.detach().cpu().clone(),
                "cascade_embedding.weight": model.cascade_embedding.weight.detach().cpu().clone(),
            }
            best_user_embedding = user_embedding.detach().cpu().clone()

    if best_state is None or best_user_embedding is None:
        raise RuntimeError(f"No valid checkpoint for seed {seed}")
    return best_user_embedding, best_state, history, best_epoch


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


@torch.no_grad()
def evaluate_pool(
    records: list[dict[str, Any]],
    user_embedding: torch.Tensor,
    user_to_index: dict[str, int],
    interaction_user_ids: set[str],
) -> dict[str, Any]:
    ranks: list[int] = []
    warm_positive: list[bool] = []
    context_covered: list[bool] = []
    target_repeated: list[bool] = []
    candidate_coverage: list[float] = []
    zero = torch.zeros(user_embedding.shape[1], dtype=user_embedding.dtype)

    for record in records:
        positive = str(record["next_user"])
        candidates = list(map(str, record["neg_users"])) + [positive]
        query, covered_history = cascade_from_history(
            record["history_users"], user_embedding, user_to_index
        )
        embeddings = torch.stack(
            [
                zero if user_to_index.get(user) is None else user_embedding[user_to_index[user]]
                for user in candidates
            ]
        )
        scores = (embeddings * query.unsqueeze(0)).sum(dim=-1).tolist()
        pairs = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]))
        ranks.append(1 + next(i for i, pair in enumerate(pairs) if pair[0] == positive))
        warm_positive.append(positive in interaction_user_ids)
        context_covered.append(
            any(str(user) in interaction_user_ids for user in record["history_users"])
        )
        target_repeated.append(positive in set(map(str, record["history_users"])))
        candidate_coverage.append(
            sum(user in user_to_index for user in candidates) / len(candidates)
        )

    def subgroup(mask: list[bool]) -> dict[str, Any]:
        selected = [rank for rank, keep in zip(ranks, mask) if keep]
        return {
            "records": len(selected),
            "metrics": ranking_metrics(selected) if selected else {},
        }

    return {
        "ranks": ranks,
        "metrics": ranking_metrics(ranks),
        "coverage": {
            "warm_positive_records": int(sum(warm_positive)),
            "context_covered_records": int(sum(context_covered)),
            "target_repeated_records": int(sum(target_repeated)),
            "mean_candidate_coverage": float(np.mean(candidate_coverage)),
        },
        "subgroups": {
            "warm_positive": subgroup(warm_positive),
            "cold_positive": subgroup([not value for value in warm_positive]),
            "context_covered": subgroup(context_covered),
            "context_cold": subgroup([not value for value in context_covered]),
            "novel_target": subgroup([not value for value in target_repeated]),
            "repeated_target": subgroup(target_repeated),
        },
    }


def aggregate_seed_metrics(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for pool_size in POOL_SIZES:
        metrics = seed_results[0]["pools"][str(pool_size)]["metrics"]
        aggregate[str(pool_size)] = {}
        for key in metrics:
            values = np.asarray(
                [
                    result["pools"][str(pool_size)]["metrics"][key]
                    for result in seed_results
                ]
            )
            aggregate[str(pool_size)][key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--users-path",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=200000)
    parser.add_argument("--regularization", type=float, default=1e-5)
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_path = args.protocol_dir / "train_records_N1000.pkl"
    validation_path = args.protocol_dir / "validation_records_N1000.pkl"
    train_records = load_pickle(train_path)
    validation_records = load_pickle(validation_path)
    test_pools = {
        size: load_pickle(args.protocol_dir / f"test_pools_N{size}.pkl")
        for size in POOL_SIZES
    }
    print("Loading the complete user inventory...", flush=True)
    users_all = load_pickle(args.users_path)
    inventory_user_ids = set(map(str, users_all.keys()))
    del users_all
    referenced_user_ids: set[str] = set()
    for records in [train_records, validation_records, *test_pools.values()]:
        for record in records:
            referenced_user_ids.update(map(str, record["history_users"]))
            referenced_user_ids.add(str(record["next_user"]))
            referenced_user_ids.update(map(str, record["neg_users"]))
    all_user_ids = inventory_user_ids | referenced_user_ids
    interaction_user_ids = {
        user for record in train_records for user in unique_participants(record)
    }
    print("Building train-only user--cascade interaction graph...", flush=True)
    user_to_index, adjacency, edge_users, edge_cascades = build_train_graph(
        train_records, all_user_ids, device
    )
    validation_tensors = build_validation_tensors(
        validation_records, user_to_index, device
    )
    graph_audit = {
        "train_cascades": len(train_records),
        "train_users": len(user_to_index),
        "inventory_users": len(inventory_user_ids),
        "referenced_users_outside_inventory": len(
            referenced_user_ids - inventory_user_ids
        ),
        "users_with_train_interactions": len(interaction_user_ids),
        "unique_interaction_edges": int(edge_users.numel()),
        "adjacency_entries": int(adjacency._nnz()),
        "uses_user_user_edges": False,
        "uses_text_or_profile_features": False,
        "uses_validation_or_test_target_edges": False,
        "train_manifest_sha256": sha256_file(train_path),
        "validation_manifest_sha256": sha256_file(validation_path),
    }
    print(json.dumps(graph_audit, ensure_ascii=False), flush=True)

    seed_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        user_embedding, state, history, best_epoch = train_one_seed(
            seed,
            len(user_to_index),
            adjacency,
            edge_users,
            edge_cascades,
            validation_tensors,
            args.dimension,
            args.layers,
            args.epochs,
            args.sample_count,
            args.regularization,
            device,
        )
        checkpoint_path = args.protocol_dir / f"interaction_twotower_seed{seed}.pt"
        torch.save(
            {
                "state_dict": {key: value.cpu() for key, value in state.items() if key != "adjacency"},
                "user_embedding": user_embedding,
                "user_to_index": user_to_index,
                "seed": seed,
                "best_epoch": best_epoch,
                "graph_audit": graph_audit,
            },
            checkpoint_path,
        )
        pool_results: dict[str, Any] = {}
        for size, records in test_pools.items():
            pool_results[str(size)] = evaluate_pool(
                records, user_embedding, user_to_index, interaction_user_ids
            )
            print(
                f"seed={seed} N={size} "
                f"{json.dumps(pool_results[str(size)]['metrics'])}",
                flush=True,
            )
        seed_results.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "history": history,
                "checkpoint": str(checkpoint_path),
                "pools": pool_results,
            }
        )
        (args.protocol_dir / "interaction_twotower_results.partial.json").write_text(
            json.dumps(
                {"graph_audit": graph_audit, "seeds": seed_results},
                ensure_ascii=False,
            )
        )

    report = {
        "method": "Train-only user--cascade bipartite LightGCN; unseen cascade is mean(history user embeddings); raw dot product",
        "graph_audit": graph_audit,
        "isolated_user_policy": "Every inventory/candidate user has an ID embedding. Users without a positive training interaction remain isolated in propagation but participate in BPR as unobserved negatives.",
        "seeds": seed_results,
        "aggregate": aggregate_seed_metrics(seed_results),
        "elapsed_seconds": time.time() - started,
    }
    output = args.protocol_dir / "interaction_twotower_results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
