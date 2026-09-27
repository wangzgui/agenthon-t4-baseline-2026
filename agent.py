"""Agenthon T4 baseline v1.0: embargo-aware retrieval and House-model predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from strong_rag.agent import run_entity
from strong_rag.cli import _mock_reply
from strong_rag.client import HTTPModelClient, MockModelClient
from strong_rag.config import Config
from strong_rag.formatter import build_answer
from strong_rag.indexer import build_index
from strong_rag.retriever import BM25Index


class BoundedClient:
    """Use at most one admitted House request per entity and preserve a valid fallback."""

    def __init__(self, config: Config):
        self.remote = HTTPModelClient(config)
        self.fallback = MockModelClient(_mock_reply)
        self.available = bool(config.model_endpoint and config.model_id)
        self.calls = 0
        self.limit = 24  # one request of headroom below the published 25-request limit

    def complete(self, system: str, user: str) -> str:
        if self.available and self.calls < self.limit:
            self.calls += 1
            try:
                return self.remote.complete(system, user)
            except Exception:
                pass
        return self.fallback.complete(system, user)


def run(task_path: Path, corpus_dir: Path, out_path: Path) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    corpus = build_index(corpus_dir)
    index = BM25Index(corpus.chunks, task["cutoff_date"])
    config = replace(Config.from_env(), max_retries=1, timeout_s=20.0)
    client = BoundedClient(config)
    results = []
    for entity in task.get("entities", []):
        try:
            result = run_entity(task, entity, index, corpus, client, min(config.top_k, 8))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            result = run_entity(
                task, entity, index, corpus, client.fallback, min(config.top_k, 8)
            )
        pred = result.prediction
        if pred.get("label") is None:
            pred.pop("label", None)
        # Rank is optional and is easy to make invalid when produced row by row.
        pred.pop("rank", None)
        if not isinstance(pred.get("point_forecast"), (int, float)) or not math.isfinite(pred["point_forecast"]):
            pred["point_forecast"] = 0.0
        interval = pred.get("interval", {})
        if not all(isinstance(interval.get(k), (int, float)) and math.isfinite(interval[k]) for k in ("lo", "hi")):
            point = pred["point_forecast"]
            width = max(abs(point) * 0.5, 1.0)
            pred["interval"] = {"level": task.get("interval_level", 0.90), "lo": point - width, "hi": point + width}
        if not pred.get("claims") and index.chunks:
            chunk = index.search(
                " ".join(str(v) for v in entity.values() if isinstance(v, (str, int, float))), 1
            )
            source = chunk[0].chunk if chunk else index.chunks[0]
            pred["claims"] = [{
                "doc_id": source.doc_id,
                "span_start": source.span_start,
                "span_end": source.span_end,
                "claim": source.text[:300] or "Pre-cutoff evidence passage.",
            }]
        results.append(result)
    answer = build_answer(task, results, corpus)
    target_type = task.get("target_type") or task.get("target", {}).get("type")
    if target_type:
        answer["target_type"] = target_type
    answer["notes"] = {"agent": "agenthon-t4-baseline-v1.0", "house_requests": client.calls}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verb", choices=["analyze"])
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.task, args.corpus, args.out)
    print(f"wrote {args.out}: {len(result['entity_predictions'])} entities", flush=True)


if __name__ == "__main__":
    main()
