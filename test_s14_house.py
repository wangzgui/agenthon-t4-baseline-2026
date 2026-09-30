"""Exercise the House request and evidence-ID rail without a paid model call."""

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent_s14 import Corpus, house_batch, house_medoid, normalize, retrieve, run


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main(units: Path) -> None:
    unit = units / "t4-EXAMPLE-eps-beat"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    corpus = Corpus(unit / "corpus", task["cutoff_date"])
    entity = task["entities"][0]
    passages = retrieve(task, entity, corpus)
    assert passages

    def fake_open(request, timeout):
        payload = json.loads(request.data)
        assert request.full_url == "http://house.test/v1/chat/completions"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["temperature"] == 0
        prompt = json.loads(payload["messages"][1]["content"])
        assert prompt["rows"][0]["evidence"][0]["text"] == passages[0].text
        reply = {"choices": [{"message": {"content": json.dumps({"predictions": [{
            "entity_id": entity["entity_id"],
            "label": task["target"]["labels"][0],
            "point_forecast": 1.57,
            "interval": {"lo": 1.4, "hi": 1.7},
            "evidence_ids": ["E0", "E999"],
            "evidence_quote": passages[0].text[:40],
        }]})}}]}
        return FakeResponse(json.dumps(reply).encode("utf-8"))

    with patch.dict(os.environ, {"MODEL_ENDPOINT": "http://house.test", "MODEL_NAME": "house"}), patch("urllib.request.urlopen", fake_open):
        prediction = house_batch(task, [(0, entity, passages)])[0]
    row = normalize(task, entity, prediction, passages, corpus)
    assert row["point_forecast"] == 1.57
    assert row["label"] == task["target"]["labels"][0]
    assert all(claim["doc_id"] in corpus.texts for claim in row["claims"])
    citation = row["claims"][0]
    assert passages[0].text[:40] in corpus.texts[citation["doc_id"]][citation["span_start"]:citation["span_end"]]

    # A six-row rate curve goes through one batch and carries the entire roster.
    rates_unit = units / "t4-fomc-curve-20220728"
    rates = json.loads((rates_unit / "task.json").read_text(encoding="utf-8"))
    rate_corpus = Corpus(rates_unit / "corpus", rates["cutoff_date"])
    rate_batch = [(i, row, retrieve(rates, row, rate_corpus)) for i, row in enumerate(rates["entities"])]

    def fake_rates(request, timeout):
        prompt = json.loads(json.loads(request.data)["messages"][1]["content"])
        assert len(prompt["rows"]) == len(prompt["roster_overview"]) == 6
        assert prompt["target"]["numeric_kind"] == "change"
        assert prompt["target"]["numeric_unit"] == "bps"
        assert prompt["rows"][0]["numeric_feature_hint"]["arithmetic_base"]["unit"] == "pct"
        predictions = [{"entity_id": row["entity_id"], "point_forecast": 0.5,
                        "projected_level": row["start_yield_pct"] + 0.5,
                        "evidence_ids": ["E0"]} for row in rates["entities"]]
        return FakeResponse(json.dumps({"choices": [{"message": {"content": json.dumps({"predictions": predictions})}}]}).encode())

    with patch.dict(os.environ, {"MODEL_ENDPOINT": "http://house.test", "MODEL_NAME": "house"}), patch("urllib.request.urlopen", fake_rates):
        rate_predictions = house_batch(rates, rate_batch)
    assert len(rate_predictions) == 6
    assert abs(normalize(rates, rates["entities"][0], rate_predictions[0], rate_batch[0][2], rate_corpus)["point_forecast"] - 50.0) < 1e-8
    forecasts = [
        [{"entity_id": "A", "point_forecast": 1.0}, {"entity_id": "B", "point_forecast": 2.0}],
        [{"entity_id": "A", "point_forecast": 9.0}, {"entity_id": "B", "point_forecast": -3.0}],
        [{"entity_id": "A", "point_forecast": 1.2}, {"entity_id": "B", "point_forecast": 2.1}],
    ]
    selected = house_medoid(forecasts, "ranking")
    assert selected in (forecasts[0], forecasts[2])
    selected = house_medoid(forecasts, "regression")
    assert [row["point_forecast"] for row in selected] == [1.2, 2.0]
    ranking_unit = units / "t4-cotpos-202411-us10"
    def fake_batch(_, batch, replicate=0, numeric_prior=None):
        return [{"entity_id": entity["entity_id"],
                 "point_forecast": float(entity["trailing_4wk_net_change_pct_oi"]) + replicate,
                 "evidence_ids": ["E0"]} for _, entity, _ in batch]
    with tempfile.TemporaryDirectory() as directory:
        with patch.dict(os.environ, {"MODEL_NAME": "house"}), \
             patch("agent_s14.house_url", return_value="http://house.test"), \
             patch("agent_s14.house_batch", side_effect=fake_batch):
            answer = run(ranking_unit / "task.json", ranking_unit / "corpus",
                         Path(directory) / "answer.json")
        assert answer["notes"]["house_requests"] == 3
        assert answer["notes"]["model_rows"] == 10
    print("PASS: House request, response, quote offsets, invalid evidence-ID rejection")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]))
