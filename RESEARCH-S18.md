## Executive summary (read this first)
The participant-reported S1.6 score is 0.4793 and S1.7 is 0.4787. S1.8 has no official score yet. This experiment returns to S1.6 and changes evidence supplementation only. The diagnosis/control mode proves matching serialized House requests and entity predictions against S1.6 under mock replies, not deterministic live model service behavior. No resolved outcomes or task-ID dispatch were added.

## Diagnostics first
Local traces record initial and supplemented source spans, uncovered keyword sets, House finish reasons and token counts, missing first/review rows, candidate values, quote acceptance checks, statistical validation, actual fusion weights, final points/labels and interval sources. The model payload and answer predictions never contain the trace object. Diagnostics default off and never add model calls. Trace filenames use a task-ID hash; authentication and endpoint information are excluded. Snapshot output is local and ignored by Git.

## Single intervention
The initial S1.6 retrieval and statement-context insertion occur unchanged. Up to two eligible non-overlapping spans are appended if a task-derived keyword need lacks coverage. Original E IDs are stable; all evidence is rebuilt before fact extraction. Sparse matching does not claim to know which missing fact causally determines a forecast. It may miss synonyms or add a superficially relevant span. The House prediction and extraction prompts, temperature/seeds, batch allocation, fusion weights and interval rules are unchanged.

## Validation and limitations
Eleven official practice units / 78 rows pass schema and owned exact citations. Three control modes (S1.6, S1.8 diagnostics off, S1.8 diagnostics on) produce matching request payloads and predictions with supplementation disabled. Synthetic coverage checks verify useful driver addition, unchanged original evidence positions, a maximum of two additions and rejection of wrong-entity evidence. Linux CI runs scorer 5.2.2 deterministic claim rules and official container smoke. No live House score or production NLI result is claimed.

Offline corpus audit: 12 rows received 15 supplemental passages after the actual statement-context stage. This measures input changes, not improved information quality or forecast accuracy. Evidence after the task cutoff cannot enter the corpus; no external retrieval or trained embedding is added. The S1.6 signed image remains the stable scored control. Full final-predictor interval calibration and isolated Final reason generation remain separate future experiments.
