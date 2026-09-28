"""Contract and citation checks over the official public practice units."""

import importlib.resources as resources
import json
import sys
from pathlib import Path

import jsonschema

from agent_s1 import Corpus, run, task_spec


def main(units: Path) -> None:
    schema = json.loads((resources.files("qfbench2_common") / "schemas" / "analysis.schema.json").read_text())
    total_rows = total_citations = 0
    out = Path("test-output/s1-check.json")
    for unit in sorted(units.iterdir()):
        if not (unit / "task.json").is_file():
            continue
        task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
        answer = run(unit / "task.json", unit / "corpus", out)
        jsonschema.validate(answer, schema)
        assert answer["task_id"] == task["task_id"]
        assert answer["target_type"] == task_spec(task)[0]
        assert [p["entity_id"] for p in answer["entity_predictions"]] == [e["entity_id"] for e in task["entities"]]
        corpus = Corpus(unit / "corpus", task["cutoff_date"])
        for row in answer["entity_predictions"]:
            assert row["claims"]
            assert row["interval"]["lo"] <= row["point_forecast"] <= row["interval"]["hi"]
            if answer["target_type"] == "classification":
                assert row["label"] in task["target"]["labels"]
            for claim in row["claims"]:
                text = corpus.texts[claim["doc_id"]]
                assert 0 <= claim["span_start"] < claim["span_end"] <= len(text)
                assert text[claim["span_start"]:claim["span_end"]].strip()
                assert any(p.doc_id == claim["doc_id"] and p.start <= claim["span_start"] < claim["span_end"] <= p.end
                           for p in corpus.passages)
                total_citations += 1
            total_rows += 1
        print(f"OK {unit.name}: {len(answer['entity_predictions'])} rows")
    print(f"PASS: {total_rows} rows; {total_citations} valid pre-cutoff citations")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
