"""Meaningful cutoff, rolling-validation, and label/point checks for S1.4."""

import datetime as dt
import json
import math
import sys
from pathlib import Path

from agent_s14 import Corpus, normalize, retrieve
from s14_numerics import collect_series, fit_prior, prior_for_entity, revision_prior
from s14_semantics import infer
from s14_fundamentals import _eps_pair, eps_growth_priors


def synthetic_history() -> None:
    cutoff = dt.date(2021, 6, 1)
    docs = {}
    entities = []
    for j in range(5):
        identifier = f"SPREAD_{j}"
        entities.append({"entity_id": identifier, "name": f"Spread {j}"})
        levels = [10.0 + j]
        for i in range(1, 30):
            levels.append(levels[-1] + 0.15 * (j + 1) + 0.02 * i)
        lines = ["date | spread_bps | unrelated", *[
            f"{(dt.date(2020, 11, 1) + dt.timedelta(days=7 * i)).isoformat()} | {value:.5f} | 99"
            for i, value in enumerate(levels)
        ], "2021-10-01 | 999999 | 99"]
        docs[identifier] = "\n".join(lines)
    class Frozen:
        texts = docs
    task = {"target": {"type": "ranking", "name": "spread_change_bps"},
            "prompt": "Predict spread change in basis points", "entities": entities,
            "cutoff_date": cutoff.isoformat(), "resolution_date": "2021-06-08"}
    meaning = infer(task)
    series = collect_series(Frozen(), task["cutoff_date"], meaning)
    assert len(series) == len(entities)
    assert all(max(s.values) < 999999 for s in series), "post-cutoff row was used"
    fitted = fit_prior(task, series)
    assert fitted and fitted.examples >= 24
    assert all(math.isfinite(prior_for_entity(row, series, fitted, meaning)) for row in entities)


def revision_and_threshold(units: Path) -> None:
    class Frozen:
        cutoff = "2020-02-15"
        texts = {"SERIES_A": """reference_month | as_of_2020-01-01 | as_of_2020-02-01 | as_of_2020-03-01
2019-09 | 10 | 12 | -999
2019-10 | 20 | 22 | -999
2019-11 | 30 | 32 | -999
2019-12 | 40 | 42 | -999
2020-01 | 50 | -- | --
"""}
    prior = revision_prior({"series_id": "SERIES_A", "ref_month": "2020-01"}, Frozen())
    assert prior == (2.0, 1.0, 4), prior

    unit = units / "t4-postearn-20240201-megacap"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    corpus = Corpus(unit / "corpus", task["cutoff_date"])
    entity = task["entities"][0]
    passages = retrieve(task, entity, corpus)
    for label, expected_sign in (("positive_reaction", 1), ("negative_reaction", -1)):
        row = normalize(task, entity, {"label": label, "point_forecast": 0.0}, passages, corpus)
        assert row["label"] == label
        assert expected_sign * row["point_forecast"] > entity["flat_threshold_abn_pct"]
    row = normalize(task, entity, {}, passages, corpus)
    threshold = entity["flat_threshold_abn_pct"]
    if row["label"] == "positive_reaction":
        assert row["point_forecast"] > threshold
    elif row["label"] == "negative_reaction":
        assert row["point_forecast"] < -threshold
    else:
        assert row["label"] == "flat" and abs(row["point_forecast"]) <= threshold


def main() -> None:
    units = Path(sys.argv[1])
    synthetic_history()
    revision_and_threshold(units)
    unit = units / "t4-eps-growth-2024Q3-banks"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    corpus = Corpus(unit / "corpus", task["cutoff_date"])
    priors = eps_growth_priors(task, corpus)
    assert len(priors) == len(task["entities"])
    assert priors["BAC"] < 0 < priors["GS"]
    for entity in task["entities"]:
        premise = retrieve(task, entity, corpus)[0]
        assert entity["cik"] in premise.doc_id
        assert _eps_pair(premise.text) is not None
    print("PASS: rolling-origin model, future exclusion, vintage revision, reaction consistency")


if __name__ == "__main__":
    main()
