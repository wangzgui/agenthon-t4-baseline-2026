"""Local production-NLI core check without the Linux-only corpus loader.

Uses the official canonical hypothesis and pinned DeBERTa ensemble; citation
resolution and embargo are separately checked by verify_s11.py.  This measures
the no-House fallback answer unless --answer points to a captured model answer.
"""

import argparse
import json
from pathlib import Path

from agent_s11 import Corpus
from faithfulness.judge import build_ensemble_judge
from qfbench2_track_analysis.hypothesis import HypothesisSpec, canonical_hypothesis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--answer", type=Path, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=1000)
    args = parser.parse_args()
    task = json.loads((args.unit / "task.json").read_text(encoding="utf-8"))
    answer = json.loads(args.answer.read_text(encoding="utf-8"))
    corpus = Corpus(args.unit / "corpus", task["cutoff_date"])
    spec = HypothesisSpec.from_task(task, target_type=answer["target_type"], interval_level=0.9)
    judge = build_ensemble_judge(cache_dir=args.cache_dir)
    supported = 0
    for row in answer["entity_predictions"][:args.max_rows]:
        interval = row["interval"]
        hypothesis = canonical_hypothesis(
            spec, entity_id=row["entity_id"], label=row.get("label", ""),
            point_forecast=float(row["point_forecast"]), rank=row.get("rank"),
            lo=float(interval["lo"]), hi=float(interval["hi"]),
        )
        scores = []
        for claim in row["claims"]:
            premise = corpus.texts[claim["doc_id"]][claim["span_start"]:claim["span_end"]]
            scores.append(float(judge.entail(premise, hypothesis)))
        best = max(scores, default=0.0)
        supported += best > 0.5
        print(row["entity_id"], "best_nli", round(best, 4), "supported", best > 0.5, flush=True)
    print("SUPPORTED", supported, "/", min(len(answer["entity_predictions"]), args.max_rows), flush=True)


if __name__ == "__main__":
    main()
