# S1.5 experiment record

Previous scores: initial flow 0.0021; S1.0 0.0692; S1.1 0.1494; S1.2 0.1621; S1.3 0.1932; S1.4 0.3530; S1.5 0.4463 (participant report, corroborated by the public board on 2026-09-30). The scorer changed before S1.4, so the S1.3-to-S1.4 score change is not a measured model-only improvement. The S1.5 combination improved the reported aggregate by 0.0933; hidden unit details and platform scorer provenance are unavailable, so individual change contributions and repeatability remain unknown.

Changes: canonical numeric aggregation/fusion; held-out statistical residual bands with honest low-sample/House-correction guards; period-labelled EPS extraction; manifest-based entity citations and official task-row citations. Preserve S1.4 modules/image as control. Small ranking rosters remain whole-roster House calls; larger batches retain full numeric roster context.

Validation: verify_s15.py checks all 11 public shapes (78 rows), new schema, ownership, spans, extractive quote bypass, probability bounds, unit/fusion equivalence, untouched-calibration selection invariant, mixed income/EPS statements, and mocked House call budgeting. Linux GitHub workflow runs the actual image and official non-rankable smoke for every public unit. Live House and hidden outcomes are not available locally; neither score uplift nor 90% future coverage is asserted.

Known limits: sparse historical cohorts, residuals belong to the statistical predictor rather than a validated House blend; financial extraction intentionally drops ambiguous pairs; return/curve units without histories use uncalibrated bands; large ranking rosters still require batches. Evidence checks validate truth of source facts, not predictive causal relevance.

## Post-score audit, 2026-09-30

Official main still equals fe313cee2865fbfbe47b65a8fcf7b830a40ea141, the validation pin. Website snapshot: Autonomous Alpha 0.5633, in3lab 0.5091, Garros Solo Lab 0.4548, zhuozhuowang 0.4463. These are Development standings, not Final outcomes.

Confirmed unit-invariance defect: fusion_weight uses an absolute regression MAE gain. With 6 selection origins, validation MAE 0.5 versus baseline 1.0 gives prior weight 0.525; multiplying both errors by 100 gives 0.600, despite identical relative skill. Repair should use a dimensionless gain and explicit evidence of model applicability.

Coverage audit (offline paths only): auction, COT and CPI have historical residuals; COT has four calibration origins, CPI two. Six of eight EPS-growth entities have accepted period-labelled statement priors. FOMC, earnings reaction and other unsupported targets have no fitted residual band. This is mechanism coverage, not evidence that any listed task has a poor platform score.

Further confirmed gaps: _unit_conflict is recorded but not acted upon; interval_source is computed but not retained; ternary classification uses the legacy wide interval for label adjustment before emitting a different new interval; retrieval is lexical and retains four passages; ranking coherence is batch-local beyond 12 rows; roster overview truncates at 120 rows and House scheduling caps at 20 batches.

Proposed next work, not implemented: dimensionless fusion and conflict handling; target/period-aware facts and dynamic retrieval; budgeted fact extraction followed by forecasting and numeric verification; expanding cutoff-safe calibration evidence; global ranking consistency and large-roster coverage. Preserve current S1.5 image and ZIP as the control. Do not optimize against hidden outcome labels or infer competitors' methods from their aggregate score.
