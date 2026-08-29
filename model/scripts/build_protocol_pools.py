#!/usr/bin/env python3
"""Build leakage-audited MosaicDiff/CAMARL manifests and hard-negative pools.

This script implements E1--E3 of the paper protocol without reading test labels
during model fitting or negative scoring.  The character-bigram towers are
trained on one deterministic next-participant record per training cascade,
selected on validation, frozen, and then applied to the supplied test-negative
universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPLIT_SALT = "MosaicDiff-CAST-AAAI2027-v1"
POOL_SIZES = (20, 50, 100, 500)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(value: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def parse_cascades(path: Path) -> dict[str, list[tuple[str, int]]]:
    cascades: dict[str, list[tuple[str, int]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                news_id, payload = line.split(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed cascade line {line_no}") from exc
            events: list[tuple[str, int]] = []
            for raw_event in payload.split(","):
                raw_event = raw_event.strip()
                if not raw_event:
                    continue
                try:
                    user_id, timestamp = raw_event.rsplit(" ", 1)
                    events.append((str(user_id), int(timestamp)))
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed event at cascade line {line_no}: {raw_event!r}"
                    ) from exc
            # Stable chronological order, retaining duplicates as observations.
            events.sort(key=lambda item: (item[1], item[0]))
            cascades[str(news_id)] = events
    return cascades


def split_manifests(
    all_news_ids: Iterable[str], test_news_ids: set[str]
) -> tuple[list[str], list[str], list[str]]:
    all_ids = set(map(str, all_news_ids))
    if not test_news_ids <= all_ids:
        missing = sorted(test_news_ids - all_ids)[:5]
        raise ValueError(f"Test news IDs absent from cascades: {missing}")
    remaining = all_ids - test_news_ids
    if len(all_ids) != 6861 or len(test_news_ids) != 207 or len(remaining) != 6654:
        raise ValueError(
            f"Unexpected split cardinalities: total={len(all_ids)}, "
            f"test={len(test_news_ids)}, remaining={len(remaining)}"
        )

    def split_key(news_id: str) -> str:
        return hashlib.sha256(f"{SPLIT_SALT}:{news_id}".encode()).hexdigest()

    ordered = sorted(remaining, key=split_key)
    train_ids = ordered[:4657]
    val_ids = ordered[4657:]
    test_ids = sorted(test_news_ids)
    assert len(train_ids) == 4657 and len(val_ids) == 1997
    return train_ids, val_ids, test_ids


def cascade_record(news_id: str, events: list[tuple[str, int]]) -> dict[str, Any]:
    # One deterministic record per cascade: the final *new* participant is the
    # target and all earlier observations form its prefix.  This preserves the
    # paper's cascade-level 4,657/1,997 counts.
    if len(events) < 2:
        raise ValueError(f"Cascade {news_id} has fewer than two events")
    target_index = len(events) - 1
    next_user = events[target_index][0]
    history = [user for user, _ in events[:target_index]]
    return {
        "news_id": news_id,
        "history_users": history,
        "next_user": next_user,
    }


def deterministic_negatives(
    record: dict[str, Any], universe: list[str], count: int = 999
) -> list[str]:
    excluded = set(map(str, record["history_users"])) | {str(record["next_user"])}
    seed_bytes = hashlib.sha256(
        f"{SPLIT_SALT}:neg:{record['news_id']}".encode()
    ).digest()[:8]
    rng = random.Random(int.from_bytes(seed_bytes, "big"))
    # Sampling indices avoids copying/shuffling the complete user universe.
    selected: list[str] = []
    seen: set[str] = set()
    while len(selected) < count:
        user = universe[rng.randrange(len(universe))]
        if user not in excluded and user not in seen:
            seen.add(user)
            selected.append(user)
    return selected


def bigram_vector(text: str, dim: int = 128) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    text = str(text)
    for index in range(max(0, len(text) - 1)):
        token = text[index : index + 2].encode("utf-8", errors="ignore")
        vector[zlib.crc32(token) % dim] += 1.0
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


class FeatureStore:
    def __init__(self, news: dict[str, Any], users: dict[str, Any]) -> None:
        self.news = news
        self.users = users
        self.news_cache: dict[str, np.ndarray] = {}
        self.user_cache: dict[str, np.ndarray] = {}

    def news_vector(self, news_id: str) -> np.ndarray:
        news_id = str(news_id)
        if news_id not in self.news_cache:
            item = self.news.get(news_id, {}) or {}
            self.news_cache[news_id] = bigram_vector(str(item.get("text", "")))
        return self.news_cache[news_id]

    def user_vector(self, user_id: str) -> np.ndarray:
        user_id = str(user_id)
        if user_id not in self.user_cache:
            item = self.users.get(user_id, {}) or {}
            description = str(item.get("description", ""))
            history = item.get("history", []) or []
            text = description + " " + " ".join(map(str, history[:8]))
            self.user_cache[user_id] = bigram_vector(text)
        return self.user_cache[user_id]


class BigramTwoTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.news_projection = nn.Linear(128, 64, bias=False)
        self.user_projection = nn.Linear(128, 64, bias=False)
        nn.init.xavier_uniform_(self.news_projection.weight)
        nn.init.xavier_uniform_(self.user_projection.weight)

    def score(
        self, news_features: torch.Tensor, user_features: torch.Tensor
    ) -> torch.Tensor:
        news_embedding = F.normalize(self.news_projection(news_features), dim=-1)
        user_embedding = F.normalize(self.user_projection(user_features), dim=-1)
        return (news_embedding * user_embedding).sum(dim=-1)


def batch_loss(
    model: BigramTwoTower,
    store: FeatureStore,
    records: list[dict[str, Any]],
    device: torch.device,
    negatives_per_record: int,
    rng: random.Random,
) -> torch.Tensor:
    news_features: list[np.ndarray] = []
    positive_features: list[np.ndarray] = []
    negative_features: list[np.ndarray] = []
    for record in records:
        negatives = record["neg_users"]
        if negatives_per_record >= len(negatives):
            sampled = negatives
        else:
            sampled = rng.sample(negatives, negatives_per_record)
        news_vector = store.news_vector(record["news_id"])
        positive = store.user_vector(record["next_user"])
        for negative in sampled:
            news_features.append(news_vector)
            positive_features.append(positive)
            negative_features.append(store.user_vector(negative))
    news_tensor = torch.from_numpy(np.stack(news_features)).to(device)
    positive_tensor = torch.from_numpy(np.stack(positive_features)).to(device)
    negative_tensor = torch.from_numpy(np.stack(negative_features)).to(device)
    positive_score = model.score(news_tensor, positive_tensor)
    negative_score = model.score(news_tensor, negative_tensor)
    return -F.logsigmoid(positive_score - negative_score).mean()


@torch.no_grad()
def validation_loss(
    model: BigramTwoTower,
    store: FeatureStore,
    records: list[dict[str, Any]],
    device: torch.device,
    negatives_per_record: int = 128,
) -> float:
    model.eval()
    losses: list[float] = []
    rng = random.Random(2701)
    for start in range(0, len(records), 32):
        loss = batch_loss(
            model,
            store,
            records[start : start + 32],
            device,
            negatives_per_record,
            rng,
        )
        losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def train_miner(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    store: FeatureStore,
    output_dir: Path,
    device: torch.device,
    epochs: int,
) -> tuple[BigramTwoTower, list[dict[str, float]]]:
    torch.manual_seed(2027)
    model = BigramTwoTower().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_path = output_dir / "miner_best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_records)))
        random.Random(2027 + epoch).shuffle(order)
        rng = random.Random(7200 + epoch)
        train_losses: list[float] = []
        for start in range(0, len(order), 32):
            batch = [train_records[index] for index in order[start : start + 32]]
            loss = batch_loss(model, store, batch, device, 64, rng)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = validation_loss(model, store, val_records, device)
        row = {
            "epoch": float(epoch),
            "train_bpr": float(np.mean(train_losses)),
            "validation_bpr": val_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_bpr": val_loss,
                    "split_salt": SPLIT_SALT,
                },
                best_path,
            )
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, history


@torch.no_grad()
def rank_negative_universe(
    model: BigramTwoTower,
    store: FeatureStore,
    record: dict[str, Any],
    device: torch.device,
) -> tuple[list[str], dict[str, float], float]:
    candidate_ids = [str(value) for value in record["neg_users"]]
    news = torch.from_numpy(store.news_vector(record["news_id"])).to(device)
    news = news.unsqueeze(0).expand(len(candidate_ids), -1)
    scores: list[float] = []
    for start in range(0, len(candidate_ids), 2048):
        ids = candidate_ids[start : start + 2048]
        users = torch.from_numpy(
            np.stack([store.user_vector(user_id) for user_id in ids])
        ).to(device)
        batch_news = news[start : start + len(ids)]
        scores.extend(model.score(batch_news, users).cpu().tolist())
    ordered_pairs = sorted(zip(candidate_ids, scores), key=lambda pair: (-pair[1], pair[0]))
    positive_feature = torch.from_numpy(store.user_vector(record["next_user"])).to(device)
    positive_score = float(model.score(news[:1], positive_feature.unsqueeze(0)).cpu()[0])
    return (
        [candidate for candidate, _ in ordered_pairs],
        {candidate: float(score) for candidate, score in ordered_pairs},
        positive_score,
    )


def materialize_pools(
    model: BigramTwoTower,
    store: FeatureStore,
    records: list[dict[str, Any]],
    split: str,
    output_dir: Path,
    device: torch.device,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    pools: dict[int, list[dict[str, Any]]] = {size: [] for size in POOL_SIZES}
    positive_ranks: list[int] = []
    for index, record in enumerate(records):
        ordered_negatives, negative_scores, positive_score = rank_negative_universe(
            model, store, record, device
        )
        positive_rank = 1 + sum(score > positive_score for score in negative_scores.values())
        positive_ranks.append(positive_rank)
        for size in POOL_SIZES:
            selected = ordered_negatives[: size - 1]
            pools[size].append(
                {
                    "news_id": str(record["news_id"]),
                    "history_users": list(map(str, record["history_users"])),
                    "next_user": str(record["next_user"]),
                    "neg_users": selected,
                    "miner_scores": {
                        **{candidate: negative_scores[candidate] for candidate in selected},
                        str(record["next_user"]): positive_score,
                    },
                    "metadata": metadata,
                }
            )
        if (index + 1) % 100 == 0:
            print(f"{split} pools: {index + 1}/{len(records)}", flush=True)
    for size, values in pools.items():
        dump_pickle(values, output_dir / f"{split}_pools_N{size}.pkl")
    return {
        "records": len(records),
        "positive_rank_mean_in_universe": float(np.mean(positive_ranks)),
        "positive_rank_median_in_universe": float(np.median(positive_ranks)),
    }


def audit_pool_records(records: list[dict[str, Any]], expected_negatives: int) -> dict[str, Any]:
    duplicate_negative_records = 0
    positive_in_negative_records = 0
    history_overlap_records = 0
    malformed_records = 0
    for record in records:
        negatives = list(map(str, record.get("neg_users", [])))
        positive = str(record.get("next_user", ""))
        history = set(map(str, record.get("history_users", [])))
        if len(negatives) != expected_negatives:
            malformed_records += 1
        if len(set(negatives)) != len(negatives):
            duplicate_negative_records += 1
        if positive in set(negatives):
            positive_in_negative_records += 1
        if history & set(negatives):
            history_overlap_records += 1
    return {
        "records": len(records),
        "expected_negatives": expected_negatives,
        "malformed_records": malformed_records,
        "duplicate_negative_records": duplicate_negative_records,
        "positive_in_negative_records": positive_in_negative_records,
        "history_overlap_records": history_overlap_records,
        "unique_news_ids": len({str(record["news_id"]) for record in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cascades_path = args.data_dir / "cascades.txt"
    test_path = args.data_dir / "test_aligned.pkl"
    cascades = parse_cascades(cascades_path)
    test_records = load_pickle(test_path)
    test_news_ids = {str(record["news_id"]) for record in test_records}
    train_ids, val_ids, test_ids = split_manifests(cascades, test_news_ids)

    manifests = {"train": train_ids, "validation": val_ids, "test": test_ids}
    manifest_report = {
        "split_salt": SPLIT_SALT,
        "counts": {key: len(value) for key, value in manifests.items()},
        "hashes": {key: manifest_hash(value) for key, value in manifests.items()},
        "cascades_sha256": sha256_file(cascades_path),
        "test_aligned_sha256": sha256_file(test_path),
    }
    (args.output_dir / "manifests.json").write_text(
        json.dumps({**manifest_report, "ids": manifests}, ensure_ascii=False, indent=2)
    )

    train_records = [cascade_record(news_id, cascades[news_id]) for news_id in train_ids]
    val_records = [cascade_record(news_id, cascades[news_id]) for news_id in val_ids]
    train_user_universe = sorted(
        {
            user
            for news_id in train_ids
            for user, _ in cascades[news_id]
        }
    )
    for index, record in enumerate(train_records):
        record["neg_users"] = deterministic_negatives(record, train_user_universe)
        if (index + 1) % 500 == 0:
            print(f"train negatives: {index + 1}/{len(train_records)}", flush=True)
    for index, record in enumerate(val_records):
        record["neg_users"] = deterministic_negatives(record, train_user_universe)
        if (index + 1) % 500 == 0:
            print(f"validation negatives: {index + 1}/{len(val_records)}", flush=True)

    dump_pickle(train_records, args.output_dir / "train_records_N1000.pkl")
    dump_pickle(val_records, args.output_dir / "validation_records_N1000.pkl")
    audit = {
        "manifest": manifest_report,
        "train": audit_pool_records(train_records, 999),
        "validation": audit_pool_records(val_records, 999),
        "test": audit_pool_records(test_records, 999),
        "train_user_universe": len(train_user_universe),
        "cascade_length": {
            "minimum": min(len(events) for events in cascades.values()),
            "maximum": max(len(events) for events in cascades.values()),
            "mean": float(np.mean([len(events) for events in cascades.values()])),
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)

    print("Loading news/users feature stores...", flush=True)
    news = load_pickle(args.data_dir / "news_all.pkl")
    users = load_pickle(args.data_dir / "users_all.pkl")
    store = FeatureStore(news, users)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, history = train_miner(
        train_records, val_records, store, args.output_dir, device, args.epochs
    )
    (args.output_dir / "miner_history.json").write_text(json.dumps(history, indent=2))
    checkpoint_hash = sha256_file(args.output_dir / "miner_best.pt")
    metadata = {
        "miner": "stable-char-bigram-128-linear-64-bpr",
        "checkpoint_sha256": checkpoint_hash,
        "manifest_hashes": manifest_report["hashes"],
        "split_salt": SPLIT_SALT,
    }
    val_summary = materialize_pools(
        model, store, val_records, "validation", args.output_dir, device, metadata
    )
    test_summary = materialize_pools(
        model, store, test_records, "test", args.output_dir, device, metadata
    )
    final_report = {
        "audit": audit,
        "miner_history": history,
        "validation_pools": val_summary,
        "test_pools": test_summary,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "protocol_report.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
