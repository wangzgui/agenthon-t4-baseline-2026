"""Cutoff-safe numerical priors fitted only from the supplied frozen corpus.

The predictor reads dated numeric tables, creates historical rolling-origin
examples, fits a two-feature ridge rule on older examples, and selects a rule
on later *pre-cutoff* examples.  No public task ID, realized competition label,
external inference document, or downloaded model is used.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import statistics
from dataclasses import dataclass
from typing import Any

from s14_semantics import Meaning, horizon_days, infer, number, unit_of, words


@dataclass(frozen=True)
class DatedSeries:
    doc_id: str
    dates: tuple[dt.date, ...]
    values: tuple[float, ...]
    heading: str


@dataclass(frozen=True)
class FittedPrior:
    model: str
    coefficients: tuple[float, float]
    horizon_steps: int
    validation_score: float
    validation_baseline: float
    examples: int


def _date(value: str) -> dt.date | None:
    try:
        token = value.strip()[:10]
        if re.fullmatch(r"\d{4}-\d{2}", token):
            token += "-01"
        return dt.date.fromisoformat(token)
    except ValueError:
        return None


def _series_from_text(doc_id: str, text: str, cutoff: str, meaning: Meaning) -> list[DatedSeries]:
    """Select columns with target tokens and matching units from dated tables."""
    lines = text.splitlines()
    cutoff_day = _date(cutoff)
    target = words(meaning.target_name) - {"rank", "next", "future", "forecast", "direction"}
    found: list[DatedSeries] = []
    for i, line in enumerate(lines[:-2]):
        if "|" not in line:
            continue
        heads = [cell.strip() for cell in line.split("|")]
        if len(heads) < 3 or not (words(heads[0]) & {"date", "week", "month", "period"}):
            continue
        choices = []
        for col, head in enumerate(heads[1:], 1):
            field_unit = unit_of(head)
            overlap = len(words(head) & target)
            compatible = field_unit == meaning.unit or meaning.unit == "level"
            if not compatible and field_unit == "level" and meaning.unit == "ratio" and overlap >= 2:
                compatible = True
            if not compatible and field_unit == "level" and meaning.unit == "pct" and "percent changes" in text[:1000].lower():
                compatible = True
            if compatible and (overlap >= 1 or meaning.kind == "level"):
                choices.append((overlap, col, head))
        if not choices:
            continue
        for _, col, head in choices[:20]:
            pairs: list[tuple[dt.date, float]] = []
            for row in lines[i + 1:]:
                if "|" not in row:
                    break
                cells = [cell.strip() for cell in row.split("|")]
                if len(cells) != len(heads):
                    break
                day = _date(cells[0])
                value = number(cells[col].replace("%", ""))
                if day is not None and value is not None and (cutoff_day is None or day <= cutoff_day):
                    pairs.append((day, value))
            if len(pairs) >= 8:
                unique = dict(pairs)
                ordered = sorted(unique.items())
                found.append(DatedSeries(doc_id, tuple(d for d, _ in ordered),
                                         tuple(v for _, v in ordered), head))
    return found


def collect_series(corpus: Any, cutoff: str, meaning: Meaning) -> list[DatedSeries]:
    found = []
    for doc_id, body in corpus.texts.items():
        found.extend(_series_from_text(doc_id, body, cutoff, meaning))
    return found


def _features(values: tuple[float, ...], index: int, steps: int) -> tuple[float, float]:
    recent = values[index] - values[index - steps]
    anchor = statistics.median(values[max(0, index - 3 * steps): index + 1])
    return recent, values[index] - anchor


def _examples(series: list[DatedSeries], steps: int):
    examples = []
    for s in series:
        for i in range(steps, len(s.values) - steps):
            x = _features(s.values, i, steps)
            y = s.values[i + steps] - s.values[i]
            examples.append((s.dates[i], s.dates[i + steps], s.doc_id + "|" + s.heading, x, y, s.values[i]))
    return sorted(examples, key=lambda row: row[0])


def _ridge(samples: list[tuple]) -> tuple[float, float]:
    a = sum(x[0] * x[0] for _, _, _, x, y, level in samples)
    b = sum(x[0] * x[1] for _, _, _, x, y, level in samples)
    d = sum(x[1] * x[1] for _, _, _, x, y, level in samples)
    u = sum(x[0] * y for _, _, _, x, y, level in samples)
    v = sum(x[1] * y for _, _, _, x, y, level in samples)
    penalty = max(1e-8, 0.1 * (a + d) / max(1, len(samples)))
    a += penalty
    d += penalty
    determinant = a * d - b * b
    if determinant <= 1e-12:
        return 0.0, 0.0
    return (max(-1.5, min(1.5, (u * d - v * b) / determinant)),
            max(-1.5, min(1.5, (v * a - u * b) / determinant)))


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for index in order[i:j]:
            ranks[index] = (i + j - 1) / 2
        i = j
    return ranks


def _corr(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    am, bm = statistics.mean(a), statistics.mean(b)
    numerator = sum((x - am) * (y - bm) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - am) ** 2 for x in a) * sum((y - bm) ** 2 for y in b))
    return numerator / denominator if denominator > 1e-12 else 0.0


def _forecast(model: str, x: tuple[float, float], beta: tuple[float, float]) -> float:
    if model == "momentum":
        return x[0]
    if model == "anti_momentum":
        return -x[0]
    if model == "reversion":
        return -0.5 * x[1]
    if model == "ridge":
        return beta[0] * x[0] + beta[1] * x[1]
    return 0.0


def _quality(samples: list[tuple], model: str, beta: tuple[float, float],
             ranking: bool, level_target: bool, monthly: bool) -> float:
    """Validate against the same-cohort realized-mean or rank baseline."""
    by_cohort: dict[object, list[tuple[float, float]]] = {}
    for day, _, series_id, x, y, level in samples:
        cohort = (day.year, day.month) if monthly else day
        predicted = _forecast(model, x, beta) + (level if level_target else 0.0)
        actual = y + (level if level_target else 0.0)
        by_cohort.setdefault(cohort, []).append((predicted, actual))
    scores = []
    for cohort in sorted(by_cohort):
        rows = by_cohort[cohort]
        if len(rows) < 3:
            continue
        predictions, actuals = zip(*rows)
        if ranking:
            scores.append(_corr(_rank(list(predictions)), _rank(list(actuals))))
        else:
            baseline = statistics.mean(actuals)
            baseline_mae = statistics.mean(abs(value - baseline) for value in actuals)
            if baseline_mae > 1e-9:
                model_mae = statistics.mean(abs(actual - pred) for pred, actual in rows)
                scores.append(1.0 - model_mae / baseline_mae)
    if not scores:
        return float("-inf")
    # Ranking relationships can drift around macro events; weight the recent
    # pre-cutoff cohorts while retaining all validation origins.
    if ranking and len(scores) >= 3:
        weights = [0.7 ** (len(scores) - 1 - i) for i in range(len(scores))]
        return sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
    return statistics.mean(scores)


def fit_prior(task: dict, series: list[DatedSeries]) -> FittedPrior | None:
    """Select on later historical windows, never on the competition outcome."""
    if len(series) < 1:
        return None
    intervals = [(b - a).days for s in series for a, b in zip(s.dates, s.dates[1:]) if 0 < (b - a).days < 100]
    if not intervals:
        return None
    cadence = statistics.median(intervals)
    steps = max(1, min(12, round(horizon_days(task) / cadence)))
    examples = _examples(series, steps)
    if len(examples) < 24:
        return None
    dates = sorted(set(row[0] for row in examples))
    if len(dates) < 6:
        return None
    split = dates[max(1, int(0.7 * len(dates)))]
    training = [row for row in examples if row[1] < split]
    validation = [row for row in examples if row[0] >= split]
    if len(training) < 12 or len(validation) < 6:
        return None
    beta = _ridge(training)
    ranking = str(task.get("target", {}).get("type")) == "ranking"
    level_target = infer(task).kind == "level"
    monthly = cadence > 20
    candidates = ["zero", "momentum", "anti_momentum", "reversion", "ridge"]
    scores = {name: _quality(validation, name, beta, ranking, level_target, monthly)
              for name in candidates}
    baseline = scores["zero"]
    name = max(candidates, key=lambda candidate: scores[candidate])
    # Small-sample margin: avoid activating a noisy fitted rule.
    minimum_gain = 0.05
    if name == "zero" or scores[name] <= baseline + minimum_gain:
        return None
    if name == "ridge":
        beta = _ridge(examples)
    return FittedPrior(name, beta, steps, scores[name], baseline, len(examples))


def prior_for_entity(entity: dict, series: list[DatedSeries], fit: FittedPrior,
                     meaning: Meaning) -> float | None:
    entity_id = re.sub(r"[^a-z0-9]", "", str(entity.get("entity_id", "")).lower())
    name_terms = words(str(entity.get("name", ""))) - {"future", "futures", "market", "series", "yield"}
    normalized_name = re.sub(r"[^a-z0-9]", "", str(entity.get("name", "")).lower())
    id_parts = re.findall(r"[a-z]*\d+[a-z]+", str(entity.get("entity_id", "")).lower())
    ranked = []
    for s in series:
        if len(s.values) <= fit.horizon_steps:
            continue
        normalized = re.sub(r"[^a-z0-9]", "", s.doc_id.lower())
        score = 12 if entity_id and entity_id in normalized else 0
        if not score and any(part in normalized for part in id_parts):
            score = 8
        normalized_heading = re.sub(r"[^a-z0-9]", "", s.heading.lower())
        if normalized_name and normalized_heading == normalized_name:
            score += 20
        score += 3 * len(name_terms & words(s.heading))
        score -= max(0, len(words(s.heading) - name_terms))
        score += len(name_terms & words(s.doc_id))
        ranked.append((score, s))
    if not ranked:
        return None
    score, chosen = max(ranked, key=lambda item: item[0])
    if score <= 0:
        return None
    x = _features(chosen.values, len(chosen.values) - 1, fit.horizon_steps)
    delta = _forecast(fit.model, x, fit.coefficients)
    return chosen.values[-1] + delta if meaning.kind == "level" else delta


def revision_prior(entity: dict, corpus: Any) -> tuple[float, float, int] | None:
    """Estimate the next vintage change from *earlier* reference-month revisions.

    The as-of columns and all observations come from the supplied pre-cutoff
    corpus.  The current reference month is excluded from the training deltas.
    """
    reference = str(entity.get("ref_month") or entity.get("reference_month") or "")
    identity = str(entity.get("series_id") or entity.get("entity_id") or "")
    if not reference or not identity:
        return None
    cutoff_day = _date(str(getattr(corpus, "cutoff", "9999-12-31")))
    candidate_docs = [(doc_id, body) for doc_id, body in corpus.texts.items()
                      if re.sub(r"[^a-z0-9]", "", identity.lower()) in
                      re.sub(r"[^a-z0-9]", "", doc_id.lower())]
    if not candidate_docs:
        return None
    for _, body in candidate_docs:
        lines = body.splitlines()
        for i, line in enumerate(lines[:-2]):
            heads = [c.strip().lower() for c in line.split("|")]
            if len(heads) < 4 or heads[0] not in {"reference_month", "ref_month"}:
                continue
            if not all(h.startswith("as_of_") for h in heads[1:]):
                continue
            eligible_cols = [j for j, head in enumerate(heads[1:], 1)
                             if (vintage := _date(head.removeprefix("as_of_"))) is not None
                             and (cutoff_day is None or vintage <= cutoff_day)]
            if not eligible_cols:
                continue
            rows = {}
            for raw in lines[i + 1:]:
                if "|" not in raw:
                    break
                cells = [c.strip() for c in raw.split("|")]
                if len(cells) != len(heads):
                    break
                rows[cells[0]] = [number(cells[j]) for j in eligible_cols]
            current = rows.get(reference)
            if not current:
                continue
            observed = [x for x in current if x is not None]
            stage = len(observed)
            if stage < 1:
                continue
            changes = []
            for month, values in rows.items():
                if month >= reference:
                    continue
                available = [x for x in values if x is not None]
                if len(available) > stage:
                    changes.append(available[stage] - available[stage - 1])
            nonzero = [x for x in changes if abs(x) > 1e-10]
            if len(nonzero) < 3:
                continue
            positive = sum(x > 0 for x in nonzero) / len(nonzero)
            return statistics.median(nonzero), positive, len(nonzero)
    return None
