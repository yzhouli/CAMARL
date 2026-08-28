#!/usr/bin/env python3
"""Evaluate the inference-only (no-GRPO) CAMARL router on graphhard-v2.

The evaluator deliberately keeps expert execution separate from the coordinator:
the coordinator can call semantic, profile, and topology tools at most once each,
and every returned observation is placed in the next coordinator prompt.  No
gradient update or test label is used.  The profile tool owns a per-seed dynamic
interest memory and can answer from retrieval without an LLM request.  The
topology tool sees the full static user-user graph and all *other* cascades; the
current news cascade is excluded because the frozen protocol does not retain a
per-prefix timestamp with which future edges could otherwise be removed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import re
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


SEEDS = (13, 21, 34, 55, 89)
POOLS = (20, 50, 100, 500)
TOOLS = ("semantic", "profile", "topology")
PROTOCOL_SHA256 = "1756d69e089c99e736c2bf787827a40057155b07e4f8703c176ef2778b58f746"
PROMPT_VERSION = "camarl-no-grpo-graphhard-v2-multimodal-v8-scope-controlled-relations"

SEMANTIC_SYSTEM = """You are the semantic expert for next-participant reranking in an information cascade. Judge immediate compatibility between the multimodal topic and user information. Use observed-cascade context, candidate descriptions, and pre-event public content, but not persistent memory or graph topology. Candidate order is random and the true next participant is unknown. Return compact JSON only: {\"top\":[...],\"scores\":[...],\"conf\":0.0,\"why\":\"brief evidence\"}. Return exactly {top_k} unique IDs with aligned 0-to-1 scores; keep why under 12 words. No chain-of-thought."""

PROFILE_SYSTEM = """You are the persistent dynamic-interest profile expert for next-participant reranking. Judge whether the multimodal topic matches stable or recently observed interests. Prioritize retrieved memory and pre-event history over one-off lexical similarity; descriptions are supporting evidence. Candidate order is random, the true next participant is unknown, and missing fields are not negative. Do not use graph topology. Return compact JSON only: {\"top\":[...],\"scores\":[...],\"conf\":0.0,\"why\":\"brief memory evidence\"}. Return exactly {top_k} unique IDs with aligned 0-to-1 scores; keep why under 12 words. No chain-of-thought."""

FULL_PREFIX_TOPOLOGY_SYSTEM = """You are the full-relation topology expert for next-participant reranking. You may inspect the complete static social relations and historical cross-cascade relations of every observed-prefix user and every candidate, including the unknown true target candidate. Use the supplied root multimodal content to interpret the propagation context, but base ranking primarily on exact directed prefix links, full-prefix shared neighbors, and other-cascade co-participation. Candidate order is random and the true next participant is unknown. Never infer or request users/events after the supplied current-cascade prefix; current-news cross-cascade statistics are excluded. Return compact JSON only: {\"top\":[...],\"scores\":[...],\"conf\":0.0,\"why\":\"brief relation evidence\"}. Return exactly {top_k} unique IDs with aligned 0-to-1 scores; keep why under 12 words. No chain-of-thought."""

PREFIX_LOCAL_TOPOLOGY_SYSTEM = """You are the prefix-local topology expert for next-participant reranking. You may use only the supplied strict-prefix induced edges, exact directed links between each candidate and observed-prefix users, and candidate-to-prefix two-hop shared-neighbor counts. Global degree, full neighbor lists, other-cascade participation, and cross-cascade co-participation are unavailable. The unknown true target is treated exactly like every other candidate. Use root multimodal content only to interpret the local propagation context. Candidate order is random and the true next participant is unknown. Never infer or request users/events after the supplied current-cascade prefix. Return compact JSON only: {\"top\":[...],\"scores\":[...],\"conf\":0.0,\"why\":\"brief local-relation evidence\"}. Return exactly {top_k} unique IDs with aligned 0-to-1 scores; keep why under 12 words. No chain-of-thought."""

PREFIX_AGGREGATE_TOPOLOGY_SYSTEM = """You are the prefix-aware aggregate topology expert for next-participant reranking. You may use strict-prefix induced edges, exact directed candidate-prefix links, per-prefix-user two-hop counts, full candidate in/out degree, and aggregate historical cross-cascade participation/co-participation. Full neighbor ID lists, individual shared-cascade IDs, and users/events after the supplied current-cascade prefix are unavailable. The unknown true target is treated exactly like every other candidate. Use root multimodal content only to interpret propagation context. Candidate order is random and the true next participant is unknown. Return compact JSON only: {\"top\":[...],\"scores\":[...],\"conf\":0.0,\"why\":\"brief aggregate-relation evidence\"}. Return exactly {top_k} unique IDs with aligned 0-to-1 scores; keep why under 12 words. No chain-of-thought."""

TOPOLOGY_SYSTEMS = {
    "full_prefix": FULL_PREFIX_TOPOLOGY_SYSTEM,
    "prefix_aggregate": PREFIX_AGGREGATE_TOPOLOGY_SYSTEM,
    "prefix_local": PREFIX_LOCAL_TOPOLOGY_SYSTEM,
}

ROUTER_SYSTEM = """You are an inference-only ReAct-style coordinator. No GRPO, reward model, gradient update, explicit cost penalty, or expert-call budget is used. You may autonomously stop or call any unused semantic, profile, or topology expert. Each expert is executed at most once because repeated execution would receive the same evidence. Expert observations are real tool outputs. Candidate order is random and you do not know the true next participant. If another expert could provide useful independent evidence, return JSON only as {\"action\":\"call\",\"expert\":\"semantic|profile|topology\"}. Otherwise stop and return {\"action\":\"stop\",\"top_user_ids\":[...]}. On stop return exactly {top_k} unique candidate IDs. Never invent an expert observation and do not output chain-of-thought."""

FUSION_SYSTEM = """You are an inference-only evidence-fusion coordinator using the unmodified base model. No GRPO, PPO, reward model, gradient update, or label feedback is available. Combine holistic cards with semantic, persistent-profile, and full-topology observations. Candidate order is random and the true next participant is unknown. Prefer high-confidence agreement; do not demote a strong candidate for one low-confidence conflict. Direct topology is strong, degree alone weak. Return compact JSON only: {\"action\":\"stop\",\"top\":[...],\"conf\":0.0,\"why\":\"brief fusion basis\"}. Return exactly {top_k} unique IDs; keep why under 12 words. No chain-of-thought."""


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def fill_top_k(system: str, top_k: int) -> str:
    """Replace only the declared placeholder; JSON braces remain literal."""
    return system.replace("{top_k}", str(top_k))


def grams(text: str) -> set[str]:
    normalized = re.sub(r"https?://\S+|\s+", "", str(text or "").lower())
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def containment(query: set[str], document: set[str]) -> float:
    return len(query & document) / max(1, len(query))


def ranking_metrics(ranks: list[int]) -> dict[str, float]:
    output: dict[str, float] = {}
    for cutoff in (1, 2, 3, 5, 10):
        output[f"H@{cutoff}"] = float(np.mean([rank <= cutoff for rank in ranks]))
        output[f"MAP@{cutoff}"] = float(
            np.mean([1.0 / rank if rank <= cutoff else 0.0 for rank in ranks])
        )
        output[f"NDCG@{cutoff}"] = float(
            np.mean(
                [1.0 / math.log2(rank + 1) if rank <= cutoff else 0.0 for rank in ranks]
            )
        )
    return output


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def parse_json(text: str) -> dict[str, Any]:
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            value = json.loads(text[start:end])
            return value if isinstance(value, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}


def parse_ranking(
    text: str,
    candidates: list[str],
    top_k: int,
    fill_order: list[str] | None = None,
) -> tuple[list[str], int, bool]:
    payload = parse_json(text)
    values = payload.get(
        "top", payload.get("top_user_ids", payload.get("ranked_user_ids", []))
    )
    parsed = [str(value).strip() for value in values] if isinstance(values, list) else []
    if not parsed:
        parsed = re.findall(r"user_\d+", text)
    candidate_set = set(candidates)
    result: list[str] = []
    seen: set[str] = set()
    for user in parsed:
        if user in candidate_set and user not in seen:
            result.append(user)
            seen.add(user)
    valid = len(result)
    exact = valid >= top_k
    for user in (fill_order or candidates):
        if len(result) >= top_k:
            break
        if user in candidate_set and user not in seen:
            result.append(user)
            seen.add(user)
    return result[:top_k], valid, exact


def bounded_score(value: Any, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def parse_expert_observation(
    text: str,
    candidates: list[str],
    top_k: int,
    fill_order: list[str],
) -> tuple[dict[str, Any], int, bool]:
    payload = parse_json(text)
    ranking, valid, exact = parse_ranking(text, candidates, top_k, fill_order)
    raw_scores = payload.get("scores", payload.get("candidate_scores", {}))
    scores: dict[str, float] = {}
    if isinstance(raw_scores, dict):
        for user, value in raw_scores.items():
            user = str(user)
            if user in ranking:
                scores[user] = bounded_score(value, 0.0)
    elif isinstance(raw_scores, list):
        for user, value in zip(ranking, raw_scores):
            scores[user] = bounded_score(value, 0.0)
    for position, user in enumerate(ranking):
        scores.setdefault(user, (top_k - position) / max(1, top_k))
    confidence_default = 0.50 if exact else 0.25
    observation = {
        "top_user_ids": ranking,
        "candidate_scores": scores,
        "confidence": bounded_score(
            payload.get("conf", payload.get("confidence")), confidence_default
        ),
        "evidence": compact(payload.get("why", payload.get("evidence", "")), 240),
        "confidence_source": (
            "model" if "conf" in payload or "confidence" in payload else "parser_default"
        ),
    }
    return observation, valid, exact


@dataclass
class ModelCall:
    role: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    error: str
    response_sha256: str
    media_bytes: int = 0


class EndpointPool:
    def __init__(
        self,
        ports: list[int],
        served_model: str,
        seed: int,
        load_balance: bool = False,
    ):
        self.clients = [
            OpenAI(
                api_key="EMPTY",
                base_url=f"http://127.0.0.1:{port}/v1",
                timeout=300,
                max_retries=2,
            )
            for port in ports
        ]
        self.served_model = served_model
        self.seed = seed
        self.load_balance = load_balance
        self._client_state_lock = threading.Lock()
        self._client_inflight = [0] * len(self.clients)
        self._client_cursor = 0

    def _acquire_client(self, key: int) -> tuple[int, OpenAI]:
        if not self.load_balance:
            index = key % len(self.clients)
            return index, self.clients[index]
        with self._client_state_lock:
            minimum = min(self._client_inflight)
            for offset in range(len(self.clients)):
                index = (self._client_cursor + offset) % len(self.clients)
                if self._client_inflight[index] == minimum:
                    break
            self._client_inflight[index] += 1
            self._client_cursor = (index + 1) % len(self.clients)
            return index, self.clients[index]

    def _release_client(self, index: int) -> None:
        if not self.load_balance:
            return
        with self._client_state_lock:
            self._client_inflight[index] -= 1
            if self._client_inflight[index] < 0:
                raise RuntimeError("Endpoint in-flight request count became negative")

    def check(self) -> None:
        for client in self.clients:
            available = {model.id for model in client.models.list().data}
            if self.served_model not in available:
                raise ValueError(f"{self.served_model} absent from {available}")

    def call(
        self,
        role: str,
        system: str,
        prompt: str,
        key: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        media_path: Path | None = None,
    ) -> tuple[str, ModelCall]:
        client_index, client = self._acquire_client(key)
        content: str | list[dict[str, Any]] = prompt
        media_bytes = 0
        if media_path is not None and media_path.exists():
            media_bytes = media_path.stat().st_size
            content = [
                {"type": "image_url", "image_url": {"url": media_path.resolve().as_uri()}},
                {"type": "text", "text": prompt},
            ]
        started = time.time()
        text = ""
        error = ""
        prompt_tokens = completion_tokens = 0
        try:
            response = client.chat.completions.create(
                model=self.served_model,
                temperature=temperature,
                top_p=top_p,
                seed=self.seed,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
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
            print(
                f"REQUEST_ERROR role={role} key={key} "
                f"prompt_chars={len(prompt)} max_tokens={max_tokens} error={error}",
                flush=True,
            )
        finally:
            self._release_client(client_index)
        return text, ModelCall(
            role=role,
            latency_seconds=time.time() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error=error,
            response_sha256=hashlib.sha256(text.encode()).hexdigest(),
            media_bytes=media_bytes,
        )


class InterestMemory:
    """Thread-safe per-seed user memory; it never stores labels."""

    def __init__(
        self,
        threshold: float,
        static_cache: dict[tuple[str, str], tuple[float, str, str]],
        static_cache_lock: threading.Lock,
        max_entries: int = 12,
    ):
        self.threshold = threshold
        self.max_entries = max_entries
        self.entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.lock = threading.Lock()
        self.static_cache = static_cache
        self.static_cache_lock = static_cache_lock

    def retrieve(
        self, user: str, news_id: str, topic_text: str, user_history: list[Any]
    ) -> tuple[float, str, str]:
        query = grams(topic_text)
        cache_key = (user, news_id)
        with self.static_cache_lock:
            cached = self.static_cache.get(cache_key)
        if cached is None:
            best_score, best_text, source = 0.0, "", "none"
            # The raw pre-event history is the initial persistent interest store.
            for value in list(user_history or [])[:24]:
                text = compact(value, 160)
                score = containment(query, grams(text))
                if score > best_score:
                    best_score, best_text, source = score, text, "pre_event_history"
            cached = (best_score, best_text, source)
            with self.static_cache_lock:
                self.static_cache.setdefault(cache_key, cached)
        best_score, best_text, source = cached
        with self.lock:
            dynamic = list(self.entries.get(user, []))
        for entry in dynamic:
            # Dynamic updates are consumed only for the same topic.  Processing
            # all prefixes of a topic serially then makes cache behavior exactly
            # reproducible while raw pre-event history still supports cross-topic
            # retrieval without depending on concurrent test-query order.
            if entry["news_id"] != news_id:
                continue
            score = containment(query, set(entry["grams"]))
            if score > best_score:
                best_score = score
                best_text = str(entry["summary"])
                source = "dynamic_memory"
        return best_score, best_text, source

    def update(self, users: list[str], news_id: str, topic_text: str) -> None:
        topic_grams = sorted(grams(topic_text))
        item = {
            "news_id": news_id,
            "summary": compact(topic_text, 160),
            "grams": topic_grams,
            "updated_at": time.time(),
        }
        with self.lock:
            for user in users:
                values = self.entries[user]
                values.insert(0, dict(item))
                del values[self.max_entries :]

    def size(self) -> tuple[int, int]:
        with self.lock:
            return len(self.entries), sum(map(len, self.entries.values()))


class RelationIndex:
    def __init__(self, users: dict[str, Any], cascades_path: Path):
        # users_all.social is the same full static relation exposed to legacy code.
        self.out = {
            str(user): set(map(str, (data or {}).get("social", []) or []))
            for user, data in users.items()
            if (data or {}).get("social")
        }
        incoming: dict[str, set[str]] = defaultdict(set)
        for source, targets in self.out.items():
            for target in targets:
                incoming[target].add(source)
        self.incoming = dict(incoming)
        user_cascades: dict[str, set[str]] = defaultdict(set)
        with cascades_path.open("r", errors="replace") as handle:
            for line in handle:
                parts = line.strip().split(None, 1)
                if len(parts) < 2:
                    continue
                news_id, events = parts
                # cascades.txt is encoded as
                #   news user timestamp,user timestamp,...
                # Splitting on whitespace loses every user after the first.
                # Extract every explicit user/timestamp pair instead.  Query-time
                # code still excludes current_news, so no suffix event can enter
                # a feature for its own current cascade.
                for user, _ in re.findall(r"(user_\d+)\s+(\d+)", events):
                    user_cascades[user].add(news_id)
        self.user_cascades = dict(user_cascades)

    def prefix_edges(self, observed: list[str]) -> list[str]:
        """Return every directed static edge inside the supplied strict prefix."""
        observed_set = set(observed)
        return [
            f"{source}->{target}"
            for source in observed
            for target in sorted(self.out.get(source, set()) & observed_set)
        ]

    def prefix_edge_summary(
        self, observed: list[str], preview_limit: int
    ) -> tuple[int, list[str]]:
        """Count all strict-prefix edges while retaining a bounded exact preview."""
        observed_set = set(observed)
        count = 0
        preview: list[str] = []
        for source in observed:
            targets = sorted(self.out.get(source, set()) & observed_set)
            count += len(targets)
            if len(preview) < preview_limit:
                remaining = preview_limit - len(preview)
                preview.extend(
                    f"{source}->{target}" for target in targets[:remaining]
                )
        return count, preview

    def retrieval_signals(
        self,
        candidate: str,
        observed: list[str],
        current_news: str,
        relation_scope: str,
    ) -> tuple[int, ...]:
        """Compute label-free structural retrieval signals for one candidate.

        Every candidate is evaluated against the complete supplied prefix.  The
        resulting tuple is used only to retrieve a context-sized set of detailed
        relation cards for the topology LLM.  It never reads the target label or
        any suffix event from the current cascade.
        """
        observed_set = set(observed)
        out = self.out.get(candidate, set())
        incoming = self.incoming.get(candidate, set())
        direct_out = len(out & observed_set)
        direct_in = len(incoming & observed_set)
        shared = [len(out & self.out.get(user, set())) for user in observed]
        shared_sum = sum(shared)
        shared_max = max(shared, default=0)
        if relation_scope == "prefix_local":
            return direct_out, direct_in, shared_sum, shared_max
        if relation_scope not in ("prefix_aggregate", "full_prefix"):
            raise ValueError(f"Unknown relation scope: {relation_scope}")
        candidate_cascades = self.user_cascades.get(candidate, set()) - {current_news}
        shared_cascades = 0
        linked_observed = 0
        for user in observed:
            overlap = candidate_cascades & (
                self.user_cascades.get(user, set()) - {current_news}
            )
            shared_cascades += len(overlap)
            linked_observed += bool(overlap)
        return (
            direct_out,
            direct_in,
            shared_sum,
            shared_max,
            linked_observed,
            shared_cascades,
            len(candidate_cascades),
            len(out) + len(incoming),
        )

    def card(
        self,
        candidate: str,
        observed: list[str],
        current_news: str,
        relation_scope: str,
        detail_limit: int = 12,
    ) -> str:
        observed_set = set(observed)
        out = self.out.get(candidate, set())
        incoming = self.incoming.get(candidate, set())
        direct_out_users = sorted(out & observed_set)
        direct_in_users = sorted(incoming & observed_set)
        shared_neighbors: list[int] = []
        shared_neighbor_by_user: list[tuple[int, int, str]] = []
        # Every observed user participates in the relational features.  Earlier
        # versions silently restricted two-hop and cross-cascade evidence to the
        # last 12 observed users, which contradicted the full-relation protocol.
        for user in observed:
            overlap_count = len(out & self.out.get(user, set()))
            shared_neighbors.append(overlap_count)
            if overlap_count:
                shared_neighbor_by_user.append(
                    (overlap_count, len(shared_neighbors) - 1, user)
                )
        shared_neighbor_details = [
            f"{user}:{count}"
            for count, _, user in sorted(
                shared_neighbor_by_user,
                key=lambda item: (-item[0], item[1], item[2]),
            )
        ]
        if relation_scope == "prefix_local":
            return (
                f"{candidate}|"
                f"to={len(direct_out_users)}:{','.join(direct_out_users) or '-'}|"
                f"fr={len(direct_in_users)}:{','.join(direct_in_users) or '-'}|"
                f"twohop={','.join(shared_neighbor_details) or '-'}"
            )
        if relation_scope not in ("prefix_aggregate", "full_prefix"):
            raise ValueError(f"Unknown relation scope: {relation_scope}")
        candidate_cascades = self.user_cascades.get(candidate, set()) - {current_news}
        shared_cascades = 0
        linked_observed = 0
        shared_cascade_by_user: list[str] = []
        for user in observed:
            overlap = candidate_cascades & (self.user_cascades.get(user, set()) - {current_news})
            shared_cascades += len(overlap)
            linked_observed += bool(overlap)
            if overlap and relation_scope == "full_prefix":
                shared_cascade_by_user.append(f"{user}:{len(overlap)}")
        if relation_scope == "prefix_aggregate":
            direct_out_preview = direct_out_users[:detail_limit]
            direct_in_preview = direct_in_users[:detail_limit]
            twohop_preview = shared_neighbor_details[:detail_limit]
            return (
                f"{candidate}|od={len(out)}|id={len(incoming)}|"
                f"to={len(direct_out_users)}:{','.join(direct_out_preview) or '-'}|"
                f"fr={len(direct_in_users)}:{','.join(direct_in_preview) or '-'}|"
                f"ss={sum(shared_neighbors)}|"
                f"sm={max(shared_neighbors, default=0)}|cd={len(candidate_cascades)}|"
                f"cs={shared_cascades}|cl={linked_observed}|"
                f"nz2={len(shared_neighbor_details)}|"
                f"twohop_top={','.join(twohop_preview) or '-'}"
            )
        return (
            f"{candidate}|od={len(out)}|id={len(incoming)}|"
            f"to={len(direct_out_users)}:{','.join(direct_out_users) or '-'}|"
            f"fr={len(direct_in_users)}:{','.join(direct_in_users) or '-'}|"
            f"ss={sum(shared_neighbors)}|"
            f"sm={max(shared_neighbors, default=0)}|cd={len(candidate_cascades)}|"
            f"cs={shared_cascades}|cl={linked_observed}|"
            f"twohop={','.join(shared_neighbor_details) or '-'}|"
            f"cocascade={','.join(shared_cascade_by_user) or '-'}"
        )


class GPUMemoryMonitor:
    def __init__(self, interval: float = 1.0, gpu_ids: list[int] | None = None):
        self.interval = interval
        self.gpu_ids = gpu_ids
        self.peak_per_gpu: list[int] = []
        self.utilization_sum_per_gpu: list[float] = []
        self.total_power_watts_sum = 0.0
        self.energy_joules = 0.0
        self.samples = 0
        self.last_sample_time: float | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            while not self.stop_event.is_set():
                try:
                    command = ["nvidia-smi"]
                    if self.gpu_ids is not None:
                        command.append(f"--id={','.join(map(str, self.gpu_ids))}")
                    command.extend(
                        [
                            "--query-gpu=memory.used,power.draw,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ]
                    )
                    output = subprocess.check_output(
                        command,
                        text=True,
                        timeout=10,
                    )
                    rows = [
                        [float(value.strip()) for value in line.split(",")]
                        for line in output.splitlines()
                        if line.strip()
                    ]
                    memory = [round(row[0]) for row in rows]
                    power = [row[1] for row in rows]
                    utilization = [row[2] for row in rows]
                    while len(self.peak_per_gpu) < len(memory):
                        self.peak_per_gpu.append(0)
                        self.utilization_sum_per_gpu.append(0.0)
                    self.peak_per_gpu = [
                        max(old, new) for old, new in zip(self.peak_per_gpu, memory)
                    ]
                    self.utilization_sum_per_gpu = [
                        old + new
                        for old, new in zip(self.utilization_sum_per_gpu, utilization)
                    ]
                    now = time.time()
                    total_power = sum(power)
                    if self.last_sample_time is not None:
                        self.energy_joules += total_power * (now - self.last_sample_time)
                    self.last_sample_time = now
                    self.total_power_watts_sum += total_power
                    self.samples += 1
                except Exception:
                    pass
                self.stop_event.wait(self.interval)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        return {
            "gpu_ids": self.gpu_ids,
            "peak_memory_mib_per_gpu": self.peak_per_gpu,
            "peak_memory_mib_total": sum(self.peak_per_gpu),
            "monitor_samples": self.samples,
            "mean_utilization_percent_per_gpu": [
                value / self.samples for value in self.utilization_sum_per_gpu
            ] if self.samples else [],
            "mean_total_power_watts": self.total_power_watts_sum / self.samples if self.samples else 0.0,
            "measured_energy_kwh": self.energy_joules / 3_600_000,
        }


def media_for(news_id: str, news: dict[str, Any], media_dir: Path) -> Path | None:
    name = str((news.get(news_id, {}) or {}).get("mm_eval_path", ""))
    if not name:
        original = Path(str((news.get(news_id, {}) or {}).get("mm_path", "")))
        # The preprocessing script creates one JPEG per source image/video.
        name = f"{original.stem}.jpg" if original.name else ""
    path = media_dir / name
    return path if name and path.exists() else None


def fallback_ranking(
    candidates: list[str], topic: str, users: dict[str, Any]
) -> list[str]:
    query = grams(topic)
    scored: list[tuple[float, int, str]] = []
    for user in candidates:
        data = users.get(user, {}) or {}
        text = f"{data.get('description', '')} " + " ".join(
            map(str, (data.get("history", []) or [])[:3])
        )
        semantic = containment(query, grams(text))
        degree = len(data.get("social", []) or [])
        score = semantic + 0.01 * math.log1p(degree)
        scored.append((score, degree, user))
    return [user for _, _, user in sorted(scored, reverse=True)]


def semantic_prompt(
    topic: str, observed: list[str], candidates: list[str], users: dict[str, Any]
) -> str:
    observed_cards = [
        f"{user}:{compact((users.get(user, {}) or {}).get('description', ''), 32)}"
        for user in observed[-9:]
    ]
    cards = []
    for user in candidates:
        data = users.get(user, {}) or {}
        history = data.get("history", []) or []
        recent = compact(history[0] if history else "", 40)
        cards.append(
            f"{user}|bio={compact(data.get('description', ''), 24)}|recent={recent}"
        )
    return (
        f"Topic text: {compact(topic, 360)}\n"
        f"Observed cascade context: {';'.join(observed_cards)}\n"
        "The attached frame/contact-sheet is the topic visual evidence.\n"
        "Candidates (random order):\n" + "\n".join(cards)
    )


def profile_prompt(
    topic: str,
    candidates: list[str],
    users: dict[str, Any],
    retrieved: dict[str, tuple[float, str, str]],
) -> str:
    cards = []
    for user in candidates:
        data = users.get(user, {}) or {}
        history = data.get("history", []) or []
        score, match, source = retrieved[user]
        cards.append(
            f"{user}|bio={compact(data.get('description', ''), 24)}|"
            f"recent={compact(history[0] if history else '', 32)}|"
            f"mem={source}:{score:.3f}:{compact(match, 32)}"
        )
    return (
        f"Topic text: {compact(topic, 360)}\n"
        "The attached frame/contact-sheet is the topic visual evidence.\n"
        "Candidate profiles and dynamic-interest retrieval (random order):\n"
        + "\n".join(cards)
    )


def topology_prompt(
    news_id: str,
    topic: str,
    observed: list[str],
    candidates: list[str],
    users: dict[str, Any],
    relations: RelationIndex,
    relation_scope: str,
    fill_order: list[str],
    card_limit: int = 64,
) -> str:
    if relation_scope == "prefix_aggregate" and len(observed) > 64:
        detailed_positions = list(range(min(16, len(observed))))
        detailed_positions.extend(
            range(max(16, len(observed) - 48), len(observed))
        )
    else:
        detailed_positions = list(range(len(observed)))
    prefix_context = []
    for position in detailed_positions:
        user = observed[position]
        data = users.get(user, {}) or {}
        history = data.get("history", []) or []
        prefix_context.append(
            f"{position + 1}:{user}|bio={compact(data.get('description', ''), 32)}|"
            f"public_recent={compact(history[0] if history else '', 48)}"
        )
    if relation_scope == "prefix_aggregate":
        prefix_edge_count, edge_preview = relations.prefix_edge_summary(
            observed, preview_limit=128
        )
    else:
        prefix_edges = relations.prefix_edges(observed)
        prefix_edge_count = len(prefix_edges)
        edge_preview = prefix_edges
    if relation_scope == "prefix_local":
        field_description = (
            "Available fields only: to/fr=exact directed links to/from observed; "
            "twohop=all nonzero candidate-to-prefix shared-neighbor counts. "
            "Global degree and every cross-cascade feature are unavailable.\n"
        )
        retrieval_label = "Candidate prefix-local relation retrieval"
    elif relation_scope == "prefix_aggregate":
        field_description = (
            "Fields: od/id=full out/in degree; to/fr=exact directed links to/from observed; "
            "ss/sm=shared-neighbor sum/max; cd=aggregate other-cascade degree; "
            "cs=aggregate shared-other-cascade count; cl=number of linked observed users; "
            "nz2=number of nonzero per-prefix-user shared-neighbor counts; "
            "twohop_top=strongest exact per-prefix-user counts. "
            "Full neighbor IDs and individual shared-cascade identities are unavailable.\n"
        )
        retrieval_label = "Candidate prefix-aware aggregate relation retrieval"
    elif relation_scope == "full_prefix":
        field_description = (
            "Fields: od/id=full out/in degree; to/fr=exact directed links to/from observed; "
            "ss/sm=shared-neighbor sum/max; cd=other-cascade degree; "
            "cs=shared other cascades; cl=linked observed users; "
            "twohop/cocascade=all nonzero per-prefix-user overlaps.\n"
        )
        retrieval_label = "Candidate complete prefix-relevant relation retrieval"
    else:
        raise ValueError(f"Unknown relation scope: {relation_scope}")
    fallback_position = {user: position for position, user in enumerate(fill_order)}
    relation_shortlist = sorted(
        candidates,
        key=lambda user: (
            tuple(
                -value
                for value in relations.retrieval_signals(
                    user, observed, news_id, relation_scope
                )
            ),
            fallback_position.get(user, len(candidates)),
            user,
        ),
    )[: min(card_limit, len(candidates))]
    if relation_scope == "prefix_aggregate" and len(observed) > 512:
        order_preview = observed[:64] + observed[-448:]
    else:
        order_preview = observed
    prefix_digest = hashlib.sha256(
        "\n".join(observed).encode()
    ).hexdigest()
    return (
        f"Root cascade content: {compact(topic, 360)}\n"
        "The attached frame/contact-sheet is the root multimodal content.\n"
        f"Complete strict-prefix length: {len(observed)}; "
        f"order_sha256={prefix_digest}.\n"
        f"Propagation-order preview ({len(order_preview)}/{len(observed)}; "
        "all users consumed by retrieval): {','.join(order_preview)}\n"
        f"Strict-prefix detailed public context ({len(prefix_context)}/{len(observed)}):\n"
        + "\n".join(prefix_context)
        + "\n"
        f"Directed static prefix edges: count={prefix_edge_count}; "
        f"exact_preview={','.join(edge_preview) if edge_preview else 'none'}\n"
        f"Current news {news_id} is excluded from all other-cascade features.\n"
        "No user or event after the supplied prefix may be accessed. "
        "All candidates, including the identity-hidden true target, are queried symmetrically.\n"
        + field_description
        + "Every candidate was scored symmetrically by the complete available "
        "relation index before retrieval.\n"
        + f"All candidate IDs: {','.join(candidates)}\n"
        + f"{retrieval_label} detailed cards "
        f"({len(relation_shortlist)}/{len(candidates)}):\n"
        + "\n".join(
            relations.card(user, observed, news_id, relation_scope)
            for user in relation_shortlist
        )
    )


def holistic_evidence(
    record: dict[str, Any], candidates: list[str], users: dict[str, Any], topic: str
) -> str:
    observed = list(map(str, record["history_users"]))
    observed_set = set(observed)
    observed_cards = [
        f"{user}:{compact((users.get(user, {}) or {}).get('description', ''), 20)}"
        for user in observed[-9:]
    ]
    candidate_cards = []
    for user in candidates:
        data = users.get(user, {}) or {}
        social = set(map(str, data.get("social", []) or []))
        history = data.get("history", []) or []
        candidate_cards.append(
            f"{user}|bio={compact(data.get('description', ''), 20)}|"
            f"direct={len(social & observed_set)}|degree={len(social)}|"
            f"recent={compact(history[0] if history else '', 20)}"
        )
    return (
        f"Topic text: {compact(topic, 300)}\n"
        f"Observed cascade: {';'.join(observed_cards)}\n"
        "Holistic candidate cards (random order):\n" + "\n".join(candidate_cards)
    )


def coordinator_candidate_shortlist(
    record: dict[str, Any],
    candidates: list[str],
    users: dict[str, Any],
    fallback: list[str],
    observations: dict[str, dict[str, Any]],
    limit: int = 64,
) -> list[str]:
    """Select coordinator cards using public, label-free retrieval signals.

    Experts still inspect every candidate.  This shortlist only removes the
    hundreds of duplicate long user cards from the coordinator context; all
    candidate IDs, the deterministic fallback, consensus and expert rankings
    remain present in ``coordinator_prompt``.
    """
    observed = set(map(str, record["history_users"]))
    priority: list[str] = list(fallback[:32])
    for tool in TOOLS:
        observation = observations.get(tool, {})
        priority.extend(map(str, observation.get("top_user_ids", []) or []))
    fallback_position = {user: index for index, user in enumerate(fallback)}
    structural = sorted(
        candidates,
        key=lambda user: (
            -len(
                set(map(str, (users.get(user, {}) or {}).get("social", []) or []))
                & observed
            ),
            -len((users.get(user, {}) or {}).get("social", []) or []),
            fallback_position.get(user, len(candidates)),
            user,
        ),
    )
    priority.extend(structural)
    selected: list[str] = []
    seen: set[str] = set()
    candidate_set = set(candidates)
    for user in priority:
        if user in candidate_set and user not in seen:
            selected.append(user)
            seen.add(user)
            if len(selected) >= min(limit, len(candidates)):
                break
    return selected


def coordinator_holistic_evidence(
    record: dict[str, Any],
    candidates: list[str],
    users: dict[str, Any],
    topic: str,
    fallback: list[str],
    observations: dict[str, dict[str, Any]],
    card_limit: int = 64,
) -> str:
    observed = list(map(str, record["history_users"]))
    observed_set = set(observed)
    observed_cards = [
        f"{user}:{compact((users.get(user, {}) or {}).get('description', ''), 20)}"
        for user in observed[-9:]
    ]
    shortlist = coordinator_candidate_shortlist(
        record, candidates, users, fallback, observations, card_limit
    )
    candidate_cards = []
    for user in shortlist:
        data = users.get(user, {}) or {}
        social = set(map(str, data.get("social", []) or []))
        history = data.get("history", []) or []
        candidate_cards.append(
            f"{user}|bio={compact(data.get('description', ''), 20)}|"
            f"direct={len(social & observed_set)}|degree={len(social)}|"
            f"recent={compact(history[0] if history else '', 20)}"
        )
    return (
        f"Topic text: {compact(topic, 300)}\n"
        f"Observed cascade: {';'.join(observed_cards)}\n"
        f"Label-free coordinator card shortlist ({len(shortlist)}/{len(candidates)}; "
        "fallback, executed-expert and public-structure retrieval):\n"
        + "\n".join(candidate_cards)
    )


def consensus_ranking(
    candidates: list[str],
    fallback: list[str],
    observations: dict[str, dict[str, Any]],
    prior_anchor: float = 0.20,
    rank_score_balance: float = 0.75,
    confidence_floor: float = 0.20,
) -> tuple[list[str], dict[str, float], dict[str, int]]:
    """Label-free confidence-weighted rank fusion used as an auditable anchor."""
    if prior_anchor < 0.0:
        raise ValueError("prior_anchor must be non-negative")
    if not 0.0 <= rank_score_balance <= 1.0:
        raise ValueError("rank_score_balance must be in [0, 1]")
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")
    scores = {user: 0.0 for user in candidates}
    agreement = {user: 0 for user in candidates}
    fallback_position = {user: position for position, user in enumerate(fallback)}
    for position, user in enumerate(fallback[:10]):
        scores[user] += prior_anchor * (10 - position) / 10
    for observation in observations.values():
        ranking = observation["top_user_ids"]
        confidence = max(confidence_floor, float(observation["confidence"]))
        candidate_scores = observation["candidate_scores"]
        for position, user in enumerate(ranking):
            positional = (len(ranking) - position) / max(1, len(ranking))
            scores[user] += confidence * (
                rank_score_balance * positional
                + (1.0 - rank_score_balance) * candidate_scores[user]
            )
            agreement[user] += 1
    ranking = sorted(
        candidates,
        key=lambda user: (
            -scores[user],
            -agreement[user],
            fallback_position.get(user, len(candidates)),
        ),
    )
    return ranking, scores, agreement


def coordinator_prompt(
    topic: str,
    candidates: list[str],
    fallback: list[str],
    observations: dict[str, dict[str, Any]],
    unused: list[str],
    holistic: str,
    final_only: bool = False,
) -> str:
    observation_text = (
        "\n".join(
            f"{tool}|confidence={observation['confidence']:.3f}|"
            f"ranking={','.join(observation['top_user_ids'])}|"
            f"scores={json.dumps(observation['candidate_scores'], ensure_ascii=False, separators=(',', ':'))}|"
            f"evidence={observation['evidence'] or 'not supplied'}"
            for tool, observation in observations.items()
        )
        if observations
        else "none"
    )
    consensus, consensus_scores, agreement = consensus_ranking(
        candidates, fallback, observations
    )
    consensus_text = ",".join(
        f"{user}:{consensus_scores[user]:.3f}/agree{agreement[user]}"
        for user in consensus[:10]
    )
    instruction = (
        f"Return the final ranking with exactly {min(10, len(candidates))} IDs."
        if final_only
        else "Choose the next action; stop when the available independent evidence is sufficient."
    )
    return (
        f"{holistic}\n"
        f"All candidate IDs: {','.join(candidates)}\n"
        f"Deterministic semantic fallback top-10: {','.join(fallback[:10])}\n"
        f"Label-free consensus anchor (score/agreement): {consensus_text}\n"
        f"Unused experts: {','.join(unused) if unused else 'none'}\n"
        f"Executed expert observations:\n{observation_text}\n"
        f"{instruction}"
    )


def execute_expert(
    tool: str,
    record: dict[str, Any],
    candidates: list[str],
    users: dict[str, Any],
    news: dict[str, Any],
    relations: RelationIndex,
    memory: InterestMemory,
    endpoints: EndpointPool,
    key: int,
    media_dir: Path,
    temperature: float,
    top_p: float,
    fill_order: list[str],
    relation_scope: str,
) -> tuple[dict[str, Any], list[ModelCall], dict[str, Any]]:
    news_id = str(record["news_id"])
    topic = str((news.get(news_id, {}) or {}).get("text", ""))
    observed = list(map(str, record["history_users"]))
    top_k = min(10, len(candidates))
    media = media_for(news_id, news, media_dir)
    diagnostics: dict[str, Any] = {"tool": tool}
    if tool == "semantic":
        prompt = semantic_prompt(topic, observed, candidates, users)
        text, call = endpoints.call(
            "semantic", fill_top_k(SEMANTIC_SYSTEM, top_k), prompt, key, 256,
            temperature, top_p, media,
        )
    elif tool == "profile":
        # Update memory only from users already observed in the cascade.  Earlier
        # code wrote predicted candidates into memory, creating self-reinforcing
        # pseudo-interests despite those candidates never being verified.
        memory.update(observed, news_id, topic)
        retrieved = {
            user: memory.retrieve(
                user, news_id, topic,
                (users.get(user, {}) or {}).get("history", []) or []
            )
            for user in candidates
        }
        hits = sorted(
            [(value[0], user) for user, value in retrieved.items() if value[0] >= memory.threshold],
            reverse=True,
        )
        diagnostics.update(
            {
                "memory_hit_candidates": len(hits),
                "memory_hit_rate": len(hits) / len(candidates),
                "profile_llm_skipped": len(hits) >= top_k,
            }
        )
        if len(hits) >= top_k:
            ranking = [user for _, user in hits[:top_k]]
            observation = {
                "top_user_ids": ranking,
                "candidate_scores": {user: bounded_score(score, 0.0) for score, user in hits[:top_k]},
                "confidence": bounded_score(np.mean([score for score, _ in hits[:top_k]]), 0.5),
                "evidence": "persistent-interest retrieval satisfied the complete top-k request",
                "confidence_source": "memory_retrieval",
            }
            diagnostics.update({"parsed_valid_ids": top_k, "exact_top10": True})
            return observation, [], diagnostics
        prompt = profile_prompt(topic, candidates, users, retrieved)
        text, call = endpoints.call(
            "profile", fill_top_k(PROFILE_SYSTEM, top_k), prompt, key, 256,
            temperature, top_p, media,
        )
    elif tool == "topology":
        prompt = topology_prompt(
            news_id, topic, observed, candidates, users, relations,
            relation_scope, fill_order,
        )
        text, call = endpoints.call(
            "topology", fill_top_k(TOPOLOGY_SYSTEMS[relation_scope], top_k),
            prompt, key, 256,
            temperature, top_p, media,
        )
    else:
        raise ValueError(f"Unknown tool {tool}")
    observation, valid, exact = parse_expert_observation(
        text, candidates, top_k, fill_order
    )
    diagnostics.update(
        {
            "parsed_valid_ids": valid,
            "exact_top10": exact,
            "confidence": observation["confidence"],
            "confidence_source": observation["confidence_source"],
            "evidence_present": bool(observation["evidence"]),
        }
    )
    return observation, [call], diagnostics


def evaluate_query(
    index: int,
    record: dict[str, Any],
    pool_size: int,
    seed: int,
    users: dict[str, Any],
    news: dict[str, Any],
    relations: RelationIndex,
    memory: InterestMemory,
    endpoints: EndpointPool,
    media_dir: Path,
    temperature: float,
    top_p: float,
    method: str,
    relation_scope: str,
) -> dict[str, Any]:
    started = time.time()
    positive = str(record["next_user"])
    candidates = list(map(str, record["neg_users"])) + [positive]
    random.Random(seed * 1_000_003 + pool_size * 10_007 + index).shuffle(candidates)
    news_id = str(record["news_id"])
    topic = str((news.get(news_id, {}) or {}).get("text", ""))
    fallback = fallback_ranking(candidates, topic, users)
    observations: dict[str, dict[str, Any]] = {}
    # The coordinator receives every candidate ID and every expert ranking below,
    # but it does not need a second copy of every long user card.  Keeping the
    # unbounded holistic evidence here overflowed the 131,072-token context on
    # large pools before the model could return its JSON decision.
    holistic = coordinator_holistic_evidence(
        record, candidates, users, topic, fallback, observations, card_limit=64
    )
    calls: list[ModelCall] = []
    expert_diagnostics: list[dict[str, Any]] = []
    unused = list(TOOLS)
    top_k = min(10, pool_size)
    final = fallback[:top_k]
    invalid_router_actions = 0
    fusion_diagnostics: dict[str, Any] = {}

    if method == "fixed_all":
        for step, tool in enumerate(TOOLS):
            observation, new_calls, diag = execute_expert(
                tool, record, candidates, users, news, relations, memory,
                endpoints, index * 11 + step, media_dir, temperature, top_p,
                fallback, relation_scope,
            )
            observations[tool] = observation
            calls.extend(new_calls)
            expert_diagnostics.append(diag)
            unused.remove(tool)
        # Refresh the compact coordinator evidence after expert execution so its
        # shortlist includes every expert's top candidates without duplicating
        # all candidate cards.
        holistic = coordinator_holistic_evidence(
            record, candidates, users, topic, fallback, observations, card_limit=64
        )
        consensus, consensus_scores, agreement = consensus_ranking(
            candidates, fallback, observations
        )
        prompt = coordinator_prompt(
            topic, candidates, fallback, observations, unused, holistic,
            final_only=True,
        )
        text, call = endpoints.call(
            "coordinator_stop", fill_top_k(FUSION_SYSTEM, top_k), prompt,
            index * 11 + 9, 256, temperature, top_p,
        )
        calls.append(call)
        final, valid, exact = parse_ranking(
            text, candidates, top_k, consensus
        )
        payload = parse_json(text)
        fusion_diagnostics = {
            "parsed_valid_ids": valid,
            "exact_top10": exact,
            "confidence": bounded_score(
                payload.get("conf", payload.get("confidence")),
                0.5 if exact else 0.25,
            ),
            "evidence": compact(
                payload.get("why", payload.get("evidence", "")), 240
            ),
            "consensus_top10": consensus[:top_k],
            "consensus_top1_score": consensus_scores[consensus[0]],
            "consensus_top1_agreement": agreement[consensus[0]],
        }
    elif method == "zero_cost_router":
        # There is no configured call-count budget.  The finite horizon is only
        # the three distinct experts followed by a mandatory stop decision.
        for step in range(len(TOOLS) + 1):
            prompt = coordinator_prompt(
                topic, candidates, fallback, observations, unused, holistic
            )
            text, call = endpoints.call(
                "coordinator", fill_top_k(ROUTER_SYSTEM, top_k), prompt,
                index * 11 + step, 256, temperature, top_p,
            )
            calls.append(call)
            payload = parse_json(text)
            action = str(payload.get("action", "")).lower()
            tool = str(payload.get("expert", "")).lower()
            if action == "call" and tool in unused:
                observation, new_calls, diag = execute_expert(
                    tool, record, candidates, users, news, relations, memory,
                    endpoints, index * 11 + step + 4, media_dir, temperature, top_p,
                    fallback, relation_scope,
                )
                observations[tool] = observation
                calls.extend(new_calls)
                expert_diagnostics.append(diag)
                unused.remove(tool)
                continue
            if action == "stop":
                consensus, _, _ = consensus_ranking(candidates, fallback, observations)
                final, valid, exact = parse_ranking(
                    text, candidates, top_k, consensus
                )
                fusion_diagnostics = {
                    "parsed_valid_ids": valid,
                    "exact_top10": exact,
                    "consensus_top10": consensus[:top_k],
                }
                break
            invalid_router_actions += 1
            if unused:
                # Invalid call is auditable; execute no invented observation.
                continue
            consensus, _, _ = consensus_ranking(candidates, fallback, observations)
            final, valid, exact = parse_ranking(text, candidates, top_k, consensus)
            fusion_diagnostics = {
                "parsed_valid_ids": valid,
                "exact_top10": exact,
                "consensus_top10": consensus[:top_k],
            }
            break
        else:
            final = fallback[:top_k]
    else:
        raise ValueError(f"Unknown method: {method}")

    rank = final.index(positive) + 1 if positive in final else pool_size
    return {
        "rank": rank,
        "top_user_ids": final,
        "called_experts": list(observations),
        "calls": [call.__dict__ for call in calls],
        "expert_diagnostics": expert_diagnostics,
        "expert_observations": observations,
        "fusion_diagnostics": fusion_diagnostics,
        "invalid_router_actions": invalid_router_actions,
        "latency_seconds": time.time() - started,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["rank"]) for row in rows]
    latencies = [float(row["latency_seconds"]) for row in rows]
    calls = [call for row in rows for call in row["calls"]]
    expert_counts = {
        tool: sum(tool in row["called_experts"] for row in rows) for tool in TOOLS
    }
    profile_diags = [
        diag for row in rows for diag in row["expert_diagnostics"] if diag["tool"] == "profile"
    ]
    fusion_diags = [row["fusion_diagnostics"] for row in rows if row["fusion_diagnostics"]]
    confidence_by_tool = {
        tool: [
            float(row["expert_observations"][tool]["confidence"])
            for row in rows
            if tool in row["expert_observations"]
        ]
        for tool in TOOLS
    }
    expert_calls = sum(expert_counts.values())
    return {
        "ranks": ranks,
        "ranked_top10": [row["top_user_ids"] for row in rows],
        "metrics": ranking_metrics(ranks),
        "diagnostics": {
            "records": len(rows),
            "request_errors": sum(bool(call["error"]) for call in calls),
            "invalid_router_actions": sum(row["invalid_router_actions"] for row in rows),
            "call_rate_any_expert": sum(bool(row["called_experts"]) for row in rows) / len(rows),
            "expert_calls_per_query": expert_calls / len(rows),
            "expert_call_rate": {tool: count / len(rows) for tool, count in expert_counts.items()},
            "model_requests_per_query": len(calls) / len(rows),
            "prompt_tokens_per_query": sum(call["prompt_tokens"] for call in calls) / len(rows),
            "completion_tokens_per_query": sum(call["completion_tokens"] for call in calls) / len(rows),
            "total_tokens_per_query": sum(call["prompt_tokens"] + call["completion_tokens"] for call in calls) / len(rows),
            "media_bytes_per_query": sum(call["media_bytes"] for call in calls) / len(rows),
            "latency_median_seconds": percentile(latencies, 50),
            "latency_p95_seconds": percentile(latencies, 95),
            "profile_tool_accesses": len(profile_diags),
            "profile_llm_skips": sum(bool(value.get("profile_llm_skipped")) for value in profile_diags),
            "profile_memory_hit_rate_mean": float(np.mean([value["memory_hit_rate"] for value in profile_diags])) if profile_diags else 0.0,
            "expert_confidence_mean": {
                tool: float(np.mean(values)) if values else 0.0
                for tool, values in confidence_by_tool.items()
            },
            "fusion_exact_top10_rate": (
                float(np.mean([bool(value.get("exact_top10")) for value in fusion_diags]))
                if fusion_diags else 0.0
            ),
            "fusion_consensus_top1_agreement_rate": (
                float(
                    np.mean(
                        [
                            bool(value.get("consensus_top10"))
                            and row["top_user_ids"][0] == value["consensus_top10"][0]
                            for row, value in zip(
                                [row for row in rows if row["fusion_diagnostics"]],
                                fusion_diags,
                            )
                        ]
                    )
                )
                if fusion_diags else 0.0
            ),
            "normalized_unit_tool_cost": expert_calls / (3 * len(rows)),
            "calls_by_role": {
                role: sum(call["role"] == role for call in calls)
                for role in sorted({call["role"] for call in calls})
            },
        },
        "per_query_diagnostics": [
            {
                "called_experts": row["called_experts"],
                "latency_seconds": row["latency_seconds"],
                "invalid_router_actions": row["invalid_router_actions"],
                "expert_diagnostics": row["expert_diagnostics"],
                "expert_observations": row["expert_observations"],
                "fusion_diagnostics": row["fusion_diagnostics"],
                "calls": row["calls"],
            }
            for row in rows
        ],
    }


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pool in seed_results[0]["pools"]:
        output[pool] = {"metrics": {}, "efficiency": {}}
        for metric in seed_results[0]["pools"][pool]["metrics"]:
            values = [seed["pools"][pool]["metrics"][metric] for seed in seed_results]
            output[pool]["metrics"][metric] = {
                "mean": float(statistics.mean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else None,
            }
        keys = [
            "call_rate_any_expert", "expert_calls_per_query", "model_requests_per_query",
            "prompt_tokens_per_query", "completion_tokens_per_query", "total_tokens_per_query",
            "latency_median_seconds", "latency_p95_seconds", "profile_memory_hit_rate_mean",
            "normalized_unit_tool_cost",
        ]
        for key in keys:
            values = [seed["pools"][pool]["diagnostics"][key] for seed in seed_results]
            output[pool]["efficiency"][key] = {
                "mean": float(statistics.mean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else None,
            }
        for tool in TOOLS:
            values = [seed["pools"][pool]["diagnostics"]["expert_call_rate"][tool] for seed in seed_results]
            output[pool]["efficiency"][f"{tool}_call_rate"] = {
                "mean": float(statistics.mean(values)),
                "std": float(statistics.stdev(values)) if len(values) > 1 else None,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("zero_cost_router", "fixed_all"), default="zero_cost_router")
    parser.add_argument(
        "--relation-scope",
        choices=tuple(TOPOLOGY_SYSTEMS),
        default="prefix_aggregate",
    )
    parser.add_argument("--served-model-name", default="Qwen3.5_4B")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--server-max-model-len",
        type=int,
        help="Audited max-model-len used by every backing inference server.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--pool-sizes", type=int, nargs="+", default=list(POOLS))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--interest-threshold", type=float, default=0.20)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--graphhard-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-protocol-sha256",
        required=True,
        help="Expected hash of the frozen candidate protocol report.",
    )
    parser.add_argument(
        "--fail-on-request-error",
        action="store_true",
        help="Abort instead of reporting fallback-contaminated metrics.",
    )
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    report_path = args.graphhard_dir / "graphhard_protocol_report.json"
    protocol_hash = sha256_file(report_path)
    if protocol_hash != args.expected_protocol_sha256:
        raise ValueError(f"Unexpected graphhard protocol hash: {protocol_hash}")
    users = load_pickle(args.data_dir / "users_all.pkl")
    news = load_pickle(args.data_dir / "news_all.pkl")
    pools = {
        size: load_pickle(
            args.graphhard_dir / f"{args.split}_graphhard_pools_N{size}.pkl"
        )[: args.limit]
        for size in args.pool_sizes
    }
    counts = {len(values) for values in pools.values()}
    if len(counts) != 1:
        raise ValueError(f"Unaligned pool counts: {counts}")
    print("Building full-relation inference index...", flush=True)
    relations = RelationIndex(users, args.data_dir / "cascades.txt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.method}_results.json"
    partial_path = output_path.with_suffix(".partial.json")

    monitor = GPUMemoryMonitor()
    monitor.start()
    inference_started = time.time()
    seed_results: list[dict[str, Any]] = []
    static_interest_cache: dict[tuple[str, str], tuple[float, str, str]] = {}
    static_interest_cache_lock = threading.Lock()
    for seed in args.seeds:
        endpoints = EndpointPool(args.ports, args.served_model_name, seed)
        endpoints.check()
        seed_pools: dict[str, Any] = {}
        for pool_size in args.pool_sizes:
            # Each candidate-pool evaluation is an independent condition.  Reusing
            # dynamic memory across N=20/50/100/500 would let earlier conditions
            # alter later ones and make the matrix order-dependent.
            memory = InterestMemory(
                args.interest_threshold,
                static_interest_cache,
                static_interest_cache_lock,
            )
            records = pools[pool_size]
            outputs: list[dict[str, Any] | None] = [None] * len(records)
            pool_started = time.time()
            # Prefixes of one news topic are evaluated serially so persistent
            # memory is causal and deterministic; different topics run in parallel.
            grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
            for index, record in enumerate(records):
                grouped[str(record["news_id"])].append((index, record))

            def run_group(
                values: list[tuple[int, dict[str, Any]]]
            ) -> list[tuple[int, dict[str, Any]]]:
                group_outputs = []
                for index, record in values:
                    group_outputs.append(
                        (
                            index,
                            evaluate_query(
                                index, record, pool_size, seed, users, news,
                                relations, memory, endpoints, args.media_dir,
                                args.temperature, args.top_p, args.method,
                                args.relation_scope,
                            ),
                        )
                    )
                return group_outputs

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_group, values): news_id
                    for news_id, values in grouped.items()
                }
                completed = 0
                for future in as_completed(futures):
                    group_results = future.result()
                    before = completed
                    for index, value in group_results:
                        outputs[index] = value
                        completed += 1
                    if completed // 50 != before // 50 or completed == len(records):
                        ranks = [int(value["rank"]) for value in outputs if value is not None]
                        metric = ranking_metrics(ranks)
                        print(
                            f"method={args.method} seed={seed} N={pool_size} "
                            f"{completed}/{len(records)} H@1={metric['H@1']:.4f} "
                            f"H@5={metric['H@5']:.4f} elapsed={time.time()-pool_started:.1f}s",
                            flush=True,
                        )
            rows = [value for value in outputs if value is not None]
            seed_pools[str(pool_size)] = summarize_rows(rows)
            if (
                args.fail_on_request_error
                and int(seed_pools[str(pool_size)]["diagnostics"]["request_errors"])
                != 0
            ):
                raise RuntimeError(
                    f"method={args.method} seed={seed} N={pool_size} produced "
                    f"{seed_pools[str(pool_size)]['diagnostics']['request_errors']} "
                    "request errors"
                )
            memory_users, memory_entries = memory.size()
            seed_pools[str(pool_size)]["diagnostics"].update(
                {"memory_users": memory_users, "memory_entries": memory_entries}
            )
            partial_path.write_text(json.dumps({"seeds": seed_results + [{"seed": seed, "pools": seed_pools}]}, ensure_ascii=False))
        seed_results.append({"seed": seed, "pools": seed_pools})

    gpu = monitor.stop()
    media_manifest_path = args.media_dir / "manifest.json"
    media_manifest = (
        json.loads(media_manifest_path.read_text()) if media_manifest_path.exists() else {}
    )
    media_records = media_manifest.get("records", {})
    topology_scope_description = {
        "prefix_local": "strict-prefix induced edges plus exact candidate-prefix directed links and nonzero two-hop counts only; no global degree or cross-cascade features",
        "prefix_aggregate": "strict-prefix induced edges, exact candidate-prefix links, per-prefix two-hop counts, full degree, and aggregate cross-cascade counts; no full neighbor IDs or individual shared-cascade identities",
        "full_prefix": "complete static relations for every strict-prefix user and every candidate, including the identity-hidden target candidate; exact directed prefix links and every nonzero two-hop/cross-cascade prefix overlap are exposed",
    }[args.relation_scope]
    audit = {
        "method": args.method,
        "training": "none; inference-only; no GRPO/PPO/reward-model updates",
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": sha256_file(args.checkpoint / "config.json"),
        "served_model_name": args.served_model_name,
        "ports": args.ports,
        "server_max_model_len": args.server_max_model_len,
        "seeds": args.seeds,
        "pool_sizes": args.pool_sizes,
        "split": args.split,
        "records": next(iter(counts)),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(
            (
                SEMANTIC_SYSTEM + PROFILE_SYSTEM
                + TOPOLOGY_SYSTEMS[args.relation_scope]
                + ROUTER_SYSTEM + FUSION_SYSTEM
            ).encode()
        ).hexdigest(),
        "candidate_pool": "frozen coverage-matched graph-hard protocol; unchanged",
        "graphhard_report_sha256": protocol_hash,
        "fail_on_request_error": args.fail_on_request_error,
        "multimodal_topic": "one static image or four-frame video contact-sheet per available original topic medium; visible to semantic, profile, and topology experts",
        "media_manifest_sha256": sha256_file(media_manifest_path) if media_manifest_path.exists() else None,
        "media_topics_available": sum(bool(value.get("output_exists")) for value in media_records.values()),
        "media_topics_missing": sum(not bool(value.get("output_exists")) for value in media_records.values()),
        "profile_memory": "per-seed persistent dynamic interest memory; deterministic pre-event-history retrieval; updates use observed cascade participants only; no predicted-candidate writes and no labels",
        "interest_threshold": args.interest_threshold,
        "relation_scope": args.relation_scope,
        "topology": topology_scope_description + "; current-news suffix users/events are never query nodes",
        "cascade_prefix": "record.history_users is the sole authoritative current-cascade prefix; the full raw current cascade is not consulted for prefix completion or suffix features",
        "fusion": "unmodified base-model coordinator receives holistic raw cards, expert rankings, calibrated-format scores/confidence/evidence, and a label-free consensus anchor",
        "uses_test_label_in_prompt_or_features": False,
        "normalized_unit_tool_cost": "executed expert calls / 3; all-three call cost equals 1",
        "latency": "end-to-end query wall time under declared concurrent serving load",
        "gpu": gpu,
    }
    report = {
        "audit": audit,
        "seeds": seed_results,
        "aggregate": aggregate(seed_results),
        "elapsed_seconds": time.time() - started,
        "inference_wall_seconds": time.time() - inference_started,
        "gpu_hours": (time.time() - inference_started) * len(gpu["peak_memory_mib_per_gpu"]) / 3600,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2), flush=True)
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
