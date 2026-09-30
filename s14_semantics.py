"""Task-derived numeric meaning, unit conversion, and robust priors for S1.4.

This module never dispatches on public task IDs or family names.  It reads the
target, prompt, and row fields; unsupported conversions are left unresolved.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Any


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(str(value).replace(",", "").strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def words(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+", value.lower()))


def unit_of(value: str) -> str:
    low = value.lower()
    terms = words(low)
    if "bps" in terms or "bp" in terms or "basis_point" in low or "basis point" in low:
        return "bps"
    if "pct" in terms or "percent" in terms or "percentage" in terms or "%" in low:
        return "pct"
    if "probability" in terms or "prob" in terms:
        return "probability"
    if "ratio" in terms or "multiple" in terms:
        return "ratio"
    if "usd" in terms or "dollar" in terms or "eps" in terms:
        return "usd"
    return "level"


def convert(value: float, from_unit: str, to_unit: str) -> float | None:
    if from_unit == to_unit or to_unit == "level" or from_unit == "level":
        return value if from_unit == to_unit else None
    if from_unit == "pct" and to_unit == "bps":
        return value * 100.0
    if from_unit == "bps" and to_unit == "pct":
        return value / 100.0
    return None


@dataclass(frozen=True)
class Meaning:
    kind: str  # level, change, growth, probability
    unit: str  # bps, pct, probability, ratio, usd, level
    target_name: str
    numeric_description: str


def infer(task: dict) -> Meaning:
    target = task.get("target") if isinstance(task.get("target"), dict) else {}
    name = str(target.get("name") or task.get("target_name") or "target")
    prompt = str(task.get("prompt") or "")
    target_type = str(target.get("type") or task.get("target_type") or "regression")
    entities = task.get("entities") or []
    row_units = " ".join(str(row.get(k, "")) for row in entities[:3] if isinstance(row, dict)
                         for k in ("unit", "units", "currency"))
    text = name + " " + row_units
    unit = unit_of(text)
    if unit == "level":
        unit = unit_of(prompt)
    # A class label can coexist with a numeric outcome.  Look for the quantity
    # on which the requested interval is defined, not just the label name.
    if target_type == "classification":
        if re.search(r"point[_ ]forecast\s*(?:=|of|:)\s*(?:your predicted\s*)?probability", prompt, re.I):
            return Meaning("probability", "probability", name, "event probability in [0,1]")
        if re.search(r"interval on (?:the )?abnormal return|interval on (?:the )?return", prompt, re.I):
            return Meaning("change", "pct", name, "abnormal return in percentage points")
        if re.search(r"point forecast of the revised value|point_forecast of the revised value", prompt, re.I):
            return Meaning("level", unit, name, "next revised level")
    terms = words(name)
    if ("growth" in terms or "yoy" in terms) and unit == "pct":
        kind = "growth"
    elif terms & {"change", "delta", "revision", "reaction", "difference", "shift"} or re.search(r"change in (?:its|the) .*? from .*? to ", prompt.lower()):
        kind = "change"
    elif unit == "probability":
        kind = "probability"
    else:
        kind = "level"
    description = {
        "growth": f"relative growth expressed as {unit}; base is a prior comparable level",
        "change": f"future minus starting value expressed as {unit}",
        "probability": "event probability in [0,1]",
        "level": f"forecast level expressed as {unit}",
    }[kind]
    return Meaning(kind, unit, name, description)


def base_field(entity: dict, meaning: Meaning) -> tuple[str, float, str] | None:
    if meaning.kind not in {"change", "growth"}:
        return None
    target_terms = words(meaning.target_name) - {"change", "delta", "growth", "yoy", "rank", "pct", "bps", "intermeeting", "direction"}
    candidates = []
    for key, raw in entity.items():
        value = number(raw)
        if value is None:
            continue
        terms = words(key)
        base_marker = bool(terms & {"start", "starting", "initial", "prior", "previous", "baseline", "current"})
        if meaning.kind == "growth":
            base_marker = base_marker or "prior_year" in key.lower()
        if not base_marker or terms & {"date", "month", "cik", "id", "count", "amount"}:
            continue
        field_unit = unit_of(key)
        score = 3 * len(terms & target_terms) + (4 if base_marker else 0)
        if meaning.kind == "growth" and field_unit == "usd":
            score += 2
        if meaning.kind == "change" and field_unit in {meaning.unit, "pct" if meaning.unit == "bps" else meaning.unit}:
            score += 2
        candidates.append((score, key, value, field_unit))
    if not candidates:
        return None
    _, key, value, field_unit = max(candidates)
    return key, value, field_unit


def derived_point(entity: dict, meaning: Meaning, projected_level: Any) -> float | None:
    future = number(projected_level)
    base = base_field(entity, meaning)
    if future is None or base is None:
        return None
    _, start, start_unit = base
    if meaning.kind == "change":
        if start_unit == "level":
            return None
        return convert(future - start, start_unit, meaning.unit)
    if meaning.kind == "growth" and abs(start) > 1e-10:
        return 100.0 * (future / start - 1.0)
    return None


def historical_candidate(entity: dict, meaning: Meaning) -> float | None:
    """Same-unit recent target proxy; never use a starting *level* as a change."""
    target_terms = words(meaning.target_name) - {"future", "next", "first", "forecast", "print", "rank"}
    candidates = []
    for key, raw in entity.items():
        value = number(raw)
        if value is None:
            continue
        terms = words(key)
        if terms & {"date", "year", "month", "id", "cik", "size", "amount", "volume", "count", "close", "threshold"}:
            continue
        if meaning.kind == "change" and terms & {"start", "starting", "initial", "baseline"}:
            continue
        if meaning.kind == "growth" and terms & {"prior", "previous", "baseline"} and not terms & {"growth", "change"}:
            continue
        field_unit = unit_of(key)
        if meaning.unit != "level" and field_unit != meaning.unit:
            converted = convert(value, field_unit, meaning.unit)
            if converted is None:
                continue
            value = converted
        score = 3 * len(terms & target_terms)
        score += 3 * bool(terms & {"latest", "trailing", "recent", "historical", "previous"})
        if meaning.kind == "change":
            score += 3 * bool(terms & {"change", "delta", "growth", "mom", "yoy"})
        if score >= 3:
            candidates.append((score, key, value))
    return max(candidates)[2] if candidates else None


def robust_scale(values: list[float]) -> float | None:
    if len(values) < 3:
        return None
    median = statistics.median(values)
    mad = statistics.median(abs(x - median) for x in values)
    if mad > 1e-8:
        return 1.4826 * mad
    if max(values) > min(values):
        return statistics.pstdev(values)
    return None


def horizon_days(task: dict) -> int:
    from datetime import date
    try:
        return max(0, (date.fromisoformat(str(task.get("resolution_date"))[:10]) -
                       date.fromisoformat(str(task.get("cutoff_date"))[:10])).days)
    except ValueError:
        return 30
