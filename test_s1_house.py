"""Exercise the House request and evidence-ID rail without a paid model call."""

import io
import json
import os
from pathlib import Path
from unittest.mock import patch

from agent_s1 import Corpus, house_batch, normalize, retrieve


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
    assert row["claims"][0]["span_end"] - row["claims"][0]["span_start"] == 40
    print("PASS: House request, response, quote offsets, invalid evidence-ID rejection")


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]))
