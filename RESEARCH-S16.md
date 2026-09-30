# S1.6 experiment record

Observed CodaBench score supplied by the participant: S1.5 = 0.4463. S1.6 = 0.4793 (participant report, 2026-10-01). Gain over S1.5: +0.0330 leaderboard points; per-unit diagnostics unavailable. The aggregate score alone cannot identify per-task gains, and pre-S1.4 scores used a different scoring regime.

## Changes and hypotheses
- Target/driver query coverage should supply more useful facts than one repeated keyword query; entity ownership is checked before retrieval.
- Extract then forecast then review should make period/unit/basis distinctions more visible. Fact quotes are verified exactly; extracted semantics are provisional and forecasts are not implied by quotations.
- Scale-invariant validation weights should behave consistently across currency/percentage changes. Sparse-history priors retain conservative fallback behavior.
- Canonical quantities and coherent whole-table ranking review should reduce arithmetic conflicts and mixing of rank scales.

## Verification
Local verification: 11 official units, 78 rows; answer schema, entity ownership and exact citation offsets. Targeted checks cover cutoff/time-split separation, EPS metric and period ambiguity, scale invariance, conversion idempotence, invented fact rejection, review anchoring, and actual House HTTP request/response parsing through mocks. These are functional checks, not live House accuracy estimates.
Official pinned source: fe313cee2865fbfbe47b65a8fcf7b830a40ea141; toolkit v2.5.1; scorer 5.2.1. CI runs the official container smoke before publication.

## Limits
No public outcome fitting or leaderboard-derived task constants were added. Historical corpus availability at the task cutoff does not prove unrevised vintage availability at every pseudo-origin. Residual bands do not provide formal coverage guarantees. House review is dependent on the draft, and exact factual citation does not establish causal support for a future numeric forecast. Very large rosters may exceed the useful model context even though row scheduling and the 25-call cap are enforced. Hidden-task gains require official evaluation.
