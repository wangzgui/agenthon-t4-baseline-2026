"""Check target-unit arithmetic and explicit class thresholds across task shapes."""

import json
from pathlib import Path

from agent_s12 import Corpus, feature_hint, house_batch, normalize, retrieve, threshold_label
from s12_semantics import base_field, derived_point, historical_candidate, infer


def main(units: Path) -> None:
    def load(name):
        unit = units / name
        task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
        return task, Corpus(unit / "corpus", task["cutoff_date"])

    rates, rates_corpus = load("t4-fomc-curve-20220728")
    row = rates["entities"][0]
    meaning = infer(rates)
    assert (meaning.kind, meaning.unit) == ("change", "bps")
    assert base_field(row, meaning) == ("start_yield_pct", 2.68, "pct")
    assert historical_candidate(row, meaning) is None
    assert "recent_same_quantity" not in feature_hint(rates, row)
    assert abs(derived_point(row, meaning, 3.23) - 55.0) < 1e-8
    passages = retrieve(rates, row, rates_corpus)
    answer = normalize(rates, row, {"point_forecast": 0.55, "projected_level": 3.23}, passages, rates_corpus)
    assert abs(answer["point_forecast"] - 55.0) < 1e-8
    assert answer["interval"]["lo"] < -60 and answer["interval"]["hi"] > 170

    growth, _ = load("t4-eps-growth-2024Q3-banks")
    grow_meaning = infer(growth)
    assert (grow_meaning.kind, grow_meaning.unit) == ("growth", "pct")
    assert abs(derived_point(growth["entities"][0], grow_meaning, 0.99) - 10.0) < 1e-8

    cpi, _ = load("t4-cpicomp-202410-us11")
    assert historical_candidate(cpi["entities"][0], infer(cpi)) == 0.18

    event, _ = load("t4-postearn-20240201-megacap")
    assert (infer(event).kind, infer(event).unit) == ("change", "pct")
    event_row = event["entities"][0]
    assert threshold_label(event, event_row, 2.1) == "positive_reaction"
    assert threshold_label(event, event_row, -2.1) == "negative_reaction"
    assert threshold_label(event, event_row, 0.3) == "flat"

    unknown = {"target": {"type": "regression", "name": "revenue_growth_pct"},
               "prompt": "Forecast next-year revenue growth percentage.",
               "entities": [{"entity_id": "X", "prior_year_revenue_usd": 100.0}]}
    assert abs(derived_point(unknown["entities"][0], infer(unknown), 110.0) - 10.0) < 1e-8
    print("PASS: generic units, future-level arithmetic, same-unit prior, directional threshold")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]))
