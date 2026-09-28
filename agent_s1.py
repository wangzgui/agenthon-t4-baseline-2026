"""Agenthon Track 4 S1.0: task-generic prediction with citation-locked evidence.

No family names or public unit identifiers drive predictions.  The only remote
dependency is the organizer-provided House model.  When it is unavailable, a
deterministic statistical fallback still emits a valid answer for every row.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import statistics
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STOP = set("the and for from with into using only each this that their will after before about your give whose when which must some are its per then than have has was were through between above below table task corpus frozen date report prediction forecast predict future support evidence claims label value point interval row rows name unit type doc document notes public private".split())
WORD = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
BAD_KEYS = {"date", "year", "month", "quarter", "id", "cik", "amount", "size", "volume", "count", "open_interest"}


def tokens(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if (len(w) > 1 or w.isdigit()) and w not in STOP]


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        x = float(str(value).replace(",", "").strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def is_scale_or_identifier(key: str) -> bool:
    words = set(key.lower().split("_"))
    return bool(words & (BAD_KEYS - {"open_interest"})) or "open_interest" in key.lower()


@dataclass(frozen=True)
class Passage:
    doc_id: str
    start: int
    end: int
    text: str
    date: str


class Corpus:
    def __init__(self, directory: Path, cutoff: str):
        self.texts: dict[str, str] = {}
        self.passages: list[Passage] = []
        for path in sorted(directory.glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("doc_id") or path.stem)
            doc_date = str(doc.get("doc_date") or "")
            if not DATE.match(doc_date) or doc_date[:10] > cutoff[:10]:
                continue
            if isinstance(doc.get("text"), str):
                parts = [doc["text"]]
            else:
                parts = [str(s.get("text", "")) for s in doc.get("spans", []) if isinstance(s, dict)]
            whole = " ".join(parts)
            if not whole.strip() or doc_id in self.texts:
                continue
            self.texts[doc_id] = whole
            # The official offset convention joins span texts with one space.
            # Passage boundaries may differ, but each [start:end] is exact.
            for a, b in self._windows(whole):
                excerpt = whole[a:b]
                if excerpt.strip():
                    self.passages.append(Passage(doc_id, a, b, excerpt, doc_date))

    @staticmethod
    def _windows(text: str):
        n = len(text)
        i = 0
        while i < n:
            j = min(i + 700, n)
            if j < n:
                candidates = [text.rfind(mark, i + 220, j) for mark in ("\n", ". ", "; ", " | ")]
                k = max(candidates)
                if k > i + 220:
                    j = k + (2 if text[k:k+2] in (". ", "; ", " | ") else 1)
            if j <= i:
                j = min(i + 700, n)
            yield i, j
            i = j

    def valid(self, p: Passage) -> bool:
        return self.texts.get(p.doc_id, "")[p.start:p.end] == p.text


def task_spec(task: dict) -> tuple[str, str, list[str]]:
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    target_type = str(target.get("type") or task.get("target_type") or "regression").lower()
    if target_type not in {"classification", "regression", "ranking"}:
        target_type = "regression"
    name = str(target.get("name") or task.get("target_name") or "target")
    labels = [str(x) for x in target.get("labels", [])] if isinstance(target.get("labels"), list) else []
    return target_type, name, labels


def entity_terms(entity: dict) -> tuple[set[str], set[str]]:
    identity, detail = set(), set()
    for key, value in entity.items():
        if not isinstance(value, str) or key in {"corpus_ref", "unit", "units", "currency"}:
            continue
        words = set(tokens(value))
        if key in {"entity_id", "name", "ticker", "symbol", "series_id", "series_name", "tenor", "cik", "reference_month", "ref_month"}:
            identity |= words
        else:
            detail |= words
    return identity, detail


def retrieve(task: dict, entity: dict, corpus: Corpus, limit: int = 4) -> list[Passage]:
    target_type, target_name, _ = task_spec(task)
    identity, detail = entity_terms(entity)
    target_terms = set(tokens(target_name))
    # Prompt tokens complement the target name, but are not allowed to drown out
    # entity-specific or target-specific matches.
    prompt_terms = set(tokens(str(task.get("prompt", ""))[:900]))
    scored = []
    for p in corpus.passages:
        body = set(tokens(p.text))
        doc = set(tokens(p.doc_id))
        e = len(identity & body) + 1.8 * len(identity & doc)
        t = len(target_terms & body)
        d = len(detail & body)
        q = len(prompt_terms & body)
        # Small structural bonus for values and quantitative evidence.
        numeric = min(len(NUMBER.findall(p.text)), 8) / 8
        score = 5 * min(e, 5) + 2.5 * min(t, 5) + min(d, 3) + 0.18 * min(q, 8) + 0.4 * numeric
        normalized_text = " ".join(tokens(p.text + " " + p.doc_id))
        for key in ("name", "tenor", "ticker", "symbol", "series_id", "series_name"):
            value = entity.get(key)
            if isinstance(value, str):
                phrase = " ".join(tokens(value))
                if len(phrase) >= 3 and phrase in normalized_text:
                    score += 12 if key in ("tenor", "ticker", "symbol", "series_id") else 7
        if target_type == "classification" and re.search(r"risk|guidance|outlook|expect|likel|concern", p.text, re.I):
            score += 0.35
        scored.append((score, p))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen: list[Passage] = []
    counts: Counter[str] = Counter()
    for _, p in scored:
        if counts[p.doc_id] >= 2 and len({x.doc_id for x in chosen}) < 2:
            continue
        chosen.append(p)
        counts[p.doc_id] += 1
        if len(chosen) >= limit:
            break
    if not chosen and corpus.passages:
        chosen = [corpus.passages[0]]
    return chosen


def feature_hint(task: dict, entity: dict) -> dict:
    """Choose likely predictive numeric features without fixed table columns.

    This is a weak prior for House and for offline fallback, not a trained ML
    model.  Field names are matched to the target concept, and historical or
    trailing fields receive more weight than arbitrary size/id fields.
    """
    _, name, _ = task_spec(task)
    target_words = set(tokens(name))
    values = []
    for key, val in entity.items():
        x = finite(val)
        if x is None or is_scale_or_identifier(key):
            continue
        key_words = set(tokens(key))
        overlap = len(key_words & target_words)
        structural = 2 if any(w in key.lower() for w in ("prior", "latest", "trailing", "historical", "previous", "current", "estimate", "change", "growth")) else 0
        score = 2 * overlap + structural
        if score > 0:
            values.append((score, key, x))
    values.sort(reverse=True)
    return {key: x for _, key, x in values[:8]}


def fallback_point(task: dict, entity: dict, all_entities: list[dict]) -> float:
    target_type, name, _ = task_spec(task)
    if target_type == "classification":
        return 0.5
    hints = feature_hint(task, entity)
    if hints:
        key, value = next(iter(hints.items()))
        if "change" not in name.lower() or any(w in key.lower() for w in ("change", "delta", "trailing", "growth")):
            return value
    # Cross-row median of same numeric field (if any), with scale preserved.
    target_words = set(tokens(name))
    options = []
    for key, val in entity.items():
        x = finite(val)
        if x is None or is_scale_or_identifier(key):
            continue
        similarity = len(set(tokens(key)) & target_words)
        options.append((similarity, key, x))
    if options:
        options.sort(reverse=True)
        _, key, _ = options[0]
        if "change" in name.lower() and not any(w in key.lower() for w in ("change", "delta", "trailing", "growth")):
            return 0.0
        peers = [finite(row.get(key)) for row in all_entities]
        peers = [x for x in peers if x is not None]
        if peers:
            return statistics.median(peers)
    return 0.0


def tabular_evidence_point(task: dict, passages: list[Passage]) -> float | None:
    """Extract a robust median from a matching corpus table column when possible."""
    _, name, _ = task_spec(task)
    target_words = set(tokens(name)) - {"next", "future", "predicted"}
    best: tuple[float, float] | None = None
    for passage in passages:
        lines = passage.text.splitlines()
        for i, line in enumerate(lines):
            if "|" not in line or i + 2 >= len(lines):
                continue
            headings = [set(tokens(cell)) for cell in line.split("|")]
            if len(headings) < 3:
                continue
            overlap = [len(h & target_words) for h in headings]
            column = max(range(len(overlap)), key=overlap.__getitem__)
            if overlap[column] < 2:
                continue
            observations = []
            for row in lines[i+1:i+22]:
                cells = row.split("|")
                if len(cells) != len(headings):
                    if observations:
                        break
                    continue
                x = finite(cells[column].strip().rstrip("%"))
                if x is not None:
                    observations.append(x)
            if observations:
                estimate = statistics.median(observations[-min(5, len(observations)):])
                candidate = (float(overlap[column]) + min(len(observations), 5) / 10, estimate)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best[1] if best else None


def interval_for(task: dict, point: float, suggested: dict | None, entities: list[dict]) -> dict:
    level = finite(task.get("interval_level")) or 0.9
    lower = finite((suggested or {}).get("lo"))
    upper = finite((suggested or {}).get("hi"))
    target_type, target_name, _ = task_spec(task)
    if target_type == "classification" and ("probab" in str(task.get("prompt", "")).lower() or "probab" in target_name.lower()):
        point = min(max(point, 0), 1)
        return {"level": level, "lo": max(0.0, min(point - 0.35, lower if lower is not None else point - 0.35)), "hi": min(1.0, max(point + 0.35, upper if upper is not None else point + 0.35))}
    scale = max(abs(point) * 0.25, 0.5)
    if "basis_point" in target_name.lower() or "bps" in target_name.lower():
        scale = max(scale, 40.0)
    if "pct" in target_name.lower() or "percent" in target_name.lower():
        scale = max(scale, 4.0)
    if lower is None or upper is None or lower > upper:
        lower, upper = point - scale, point + scale
    else:
        lower, upper = min(lower, point - scale * 0.3), max(upper, point + scale * 0.3)
    return {"level": level, "lo": lower, "hi": upper}


def cross_section(task: dict, entity: dict) -> dict:
    """Small robust tabular summary; no fitted model or labeled data assumed."""
    roster = task.get("entities", [])
    if not isinstance(roster, list):
        return {}
    _, target, _ = task_spec(task)
    target_words = set(tokens(target))
    candidates = []
    for key, value in entity.items():
        x = finite(value)
        if x is None or is_scale_or_identifier(key):
            continue
        peers = [finite(row.get(key)) for row in roster if isinstance(row, dict)]
        peers = [v for v in peers if v is not None]
        if len(peers) < 2:
            continue
        median = statistics.median(peers)
        mad = statistics.median([abs(v - median) for v in peers])
        relevance = len(set(tokens(key)) & target_words)
        candidates.append((relevance, key, {"value": x, "peer_median": median,
                                            "robust_z": max(-8.0, min(8.0, (x - median) / max(1.4826 * mad, 1e-8))) }))
    candidates.sort(reverse=True)
    return {key: summary for _, key, summary in candidates[:8]}


def house_url() -> str | None:
    endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
    return (endpoint + ("" if endpoint.endswith("/v1") else "/v1") + "/chat/completions") if endpoint else None


def house_batch(task: dict, batch: list[tuple[int, dict, list[Passage]]]) -> list[dict] | None:
    url = house_url()
    model = os.environ.get("MODEL_NAME")
    if not url or not model:
        return None
    target_type, target_name, labels = task_spec(task)
    rows = []
    for _, entity, passages in batch:
        rows.append({
            "entity_id": entity["entity_id"],
            "fields": {k: v for k, v in entity.items() if k != "corpus_ref"},
            "numeric_feature_hint": feature_hint(task, entity),
            "cross_section": cross_section(task, entity),
            "evidence": [{"id": f"E{i}", "text": p.text} for i, p in enumerate(passages)],
        })
    system = (
        "You are a forecasting analyst using ONLY the supplied pre-cutoff corpus passages. "
        "Return one strict JSON object, no markdown or prose. Do not use your memory of realized events. "
        "Forecast the unknown future, not a current value. Each row must cite evidence IDs that "
        "actually support the forecast rationale. If uncertain, be conservative and widen intervals."
    )
    user = json.dumps({
        "instruction": (
            "Return {predictions:[{entity_id,label,point_forecast,interval:{lo,hi},"
            "evidence_ids:[E0,...],evidence_quote}]} exactly once per row. "
            "For classification, label must be one allowed label. If target is a probability, point is in [0,1]; "
            "otherwise point is the numeric forecast named in the prompt. For ranking, point must "
            "be the underlying metric (larger is ranked higher), not rank number. "
            "Use evidence_quote as a short VERBATIM substring of a cited passage; don't invent quotes. "
            "Do not claim future outcomes are known."
        ),
        "target": {"type": target_type, "name": target_name, "labels": labels},
        "cutoff_date": task.get("cutoff_date"),
        "resolution_date": task.get("resolution_date"),
        "prompt": str(task.get("prompt", ""))[:3500],
        "rows": rows,
    }, ensure_ascii=False, separators=(",", ":"))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "seed": 2609,
        "max_tokens": 3500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("MODEL_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=65) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        match = re.search(r"\{.*\}", content, re.S)
        parsed = json.loads(match.group(0) if match else content)
        return parsed.get("predictions") if isinstance(parsed.get("predictions"), list) else None
    except Exception as exc:
        # Never write endpoint, credentials, or raw response into the answer/log.
        print(f"House batch unavailable: {type(exc).__name__}", flush=True)
        return None


def make_claim(corpus: Corpus, passages: list[Passage], raw: dict) -> list[dict]:
    ids = raw.get("evidence_ids") if isinstance(raw, dict) else []
    if not isinstance(ids, list):
        ids = []
    selected: list[Passage] = []
    for item in ids[:3]:
        if isinstance(item, str) and re.fullmatch(r"E\d+", item):
            index = int(item[1:])
            if index < len(passages) and passages[index] not in selected:
                selected.append(passages[index])
    for p in passages:
        if p not in selected:
            selected.append(p)
        if len(selected) >= 2:
            break
    quote = str(raw.get("evidence_quote") or "") if isinstance(raw, dict) else ""
    claims = []
    for p in selected[:2]:
        if not corpus.valid(p):
            continue
        if 12 <= len(quote) <= 280 and quote in p.text:
            start = p.start + p.text.index(quote)
            end = start + len(quote)
            claim_text = quote
        else:
            start, end = p.start, p.end
            claim_text = p.text[:350]
        claims.append({"doc_id": p.doc_id, "span_start": start, "span_end": end, "claim": claim_text})
    return claims


def normalize(task: dict, entity: dict, raw: dict | None, passages: list[Passage], corpus: Corpus) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    target_type, _, labels = task_spec(task)
    entities = task.get("entities", [])
    fallback = fallback_point(task, entity, entities)
    if task_spec(task)[0] != "classification":
        evidence_estimate = tabular_evidence_point(task, passages)
        if evidence_estimate is not None:
            fallback = evidence_estimate
    point = finite(raw.get("point_forecast"))
    if point is None:
        point = fallback
    if target_type == "classification":
        label = raw.get("label")
        if label not in labels:
            # Label/probability semantics cannot be inferred reliably for
            # arbitrary unseen tasks; the prompt and House model decide first.
            label = labels[0] if labels else "unknown"
        if "probab" in str(task.get("prompt", "")).lower():
            point = min(max(point, 0.0), 1.0)
    result = {
        "entity_id": entity["entity_id"],
        "point_forecast": point,
        "interval": interval_for(task, point, raw.get("interval") if isinstance(raw.get("interval"), dict) else None, entities),
        "claims": make_claim(corpus, passages, raw),
    }
    if target_type == "classification":
        result["label"] = label
    return result


def run(task_path: Path, corpus_path: Path, out_path: Path) -> dict:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    entities = task.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("task entities must be a list")
    corpus = Corpus(corpus_path, str(task.get("cutoff_date", "9999-12-31")))
    if not corpus.passages:
        raise ValueError("no eligible pre-cutoff corpus passages")
    prepared = [(i, e, retrieve(task, e, corpus)) for i, e in enumerate(entities)]
    # At most 20 House requests per unit, under the official 25-request cap.
    batch_size = max(3, math.ceil(len(prepared) / 20))
    batch_size = min(batch_size, 5)
    batches = [prepared[i:i+batch_size] for i in range(0, len(prepared), batch_size)]
    house_results: dict[str, dict] = {}
    house_calls = 0
    if house_url() and os.environ.get("MODEL_NAME"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(house_batch, task, batch) for batch in batches[:20]]
            house_calls = len(futures)
            for future in concurrent.futures.as_completed(futures):
                predictions = future.result() or []
                for pred in predictions:
                    if isinstance(pred, dict) and isinstance(pred.get("entity_id"), str):
                        house_results[pred["entity_id"]] = pred
    rows = [normalize(task, entity, house_results.get(str(entity["entity_id"])), passages, corpus)
            for _, entity, passages in prepared]
    answer = {
        "task_id": task["task_id"],
        "schema_version": "3",
        "target_type": task_spec(task)[0],
        "entity_predictions": rows,
        "notes": {"agent": "agenthon-t4-baseline-s1.0", "house_requests": house_calls, "model_rows": len(house_results)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answer, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("verb", choices=["analyze"])
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    answer = run(args.task, args.corpus, args.out)
    print(f"wrote {args.out}: {len(answer['entity_predictions'])} rows", flush=True)


if __name__ == "__main__":
    main()
