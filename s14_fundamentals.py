"""Cutoff-safe financial statement features from the current unit's corpus.

No competition outcome or external financial data is loaded. This adapter is
activated by target semantics and entity document identity, never task IDs.
"""

from __future__ import annotations

import re
import statistics
from typing import Any


_EPS = re.compile(
    r"\bdiluted\s+(?:earnings?(?:\s+per\s+(?:common\s+)?share)?|"
    r"income(?:\s+from\s+continuing\s+operations)?)", re.I)
_NUM = re.compile(r"(?<![\w])\$?\(?-?(?:\d{1,2}\.\d{1,2}|\.\d{1,2})\)?")


def _float(token: str) -> float:
    return float(token.replace("$", "").replace("(", "-").replace(")", ""))


def _eps_pair(text: str) -> tuple[float, float] | None:
    compact = re.sub(r"\s+", " ", text)
    candidates: list[tuple[int, int, float, float]] = []
    for match in _EPS.finditer(compact):
        segment = compact[match.end():match.end() + 190]
        comparison = re.search(r"compared\s+(?:with|to)", segment, re.I)
        if comparison and re.search(r"\b(?:billion|million)\b", segment[:comparison.end() + 35], re.I):
            continue
        pair = None
        if comparison:
            before = _NUM.search(segment[:comparison.start()])
            after = _NUM.search(segment[comparison.end():])
            if before and after:
                pair = (_float(before.group()), _float(after.group()))
        if pair is None:
            numbers = list(_NUM.finditer(segment[:80]))
            if len(numbers) >= 2:
                pair = (_float(numbers[0].group()), _float(numbers[1].group()))
        if pair is None or not all(0.01 <= abs(value) <= 20 for value in pair):
            continue
        # Prefer an explicit prior-year comparison; a table row is secondary.
        score = 3 if comparison and "billion" not in segment[:comparison.end() + 35].lower() else 0
        score += 1 if "per share" in match.group().lower() else 0
        candidates.append((score, match.start(), pair[0], pair[1]))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], row[1]))
    return candidates[0][2], candidates[0][3]


def eps_growth_priors(task: dict, corpus: Any) -> dict[str, float]:
    target = task.get("target") or {}
    name = str(target.get("name", "")).lower()
    if target.get("type") != "regression" or not ("eps" in name and "growth" in name):
        return {}
    raw: dict[str, float] = {}
    for entity in task.get("entities", []):
        cik = str(entity.get("cik", ""))
        if not cik:
            continue
        filings = [(doc_id, text) for doc_id, text in corpus.texts.items()
                   if cik in doc_id and "10Q" in doc_id.upper()]
        for _, text in sorted(filings, reverse=True):
            pair = _eps_pair(text)
            if pair is None or abs(pair[1]) < 0.01:
                continue
            growth = 100 * (pair[0] / pair[1] - 1)
            if -100 <= growth <= 300:
                raw[str(entity["entity_id"])] = growth
                break
    if len(raw) < 3:
        return {}
    center = statistics.median(raw.values())
    # Preserve modest cross-company differences and temper one-off large EPS
    # jumps before transferring an already-reported quarter to the next one.
    return {entity_id: center + (value - center) * (0.5 if abs(value - center) > 25 else 1.0)
            for entity_id, value in raw.items()}
