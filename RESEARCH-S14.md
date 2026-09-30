# Track 4 S1.4 research and experiment ledger

## What the four submissions established

| Version | Public score | What changed | Interpretation |
|---|---:|---|---|
| S1.0 | 0.0021 | Contract-safe first submission | Ten units scored; prediction and evidence quality were weak. |
| S1.1 | 0.1494 | Unit-aware targets, whole-roster context, NLI repair | FOMC failures changed from -0.27 to -0.03; structural admissibility improved. |
| S1.2 | 0.1621 | Historical numerical prior and label/point reconciliation | COT rose 0.0870 to 0.1914; no measured regression gain. |
| S1.3 | 0.1932 | Cross-sectional validation, wider use of prior, ternary direction calibration | Post-earnings rose -0.0700 to 0.3967; COT fell 0.1914 to 0.0360 despite the same effective COT code path. Regression units stayed at zero skill. |

The public score is only a noisy diagnostic, not a training label. The hidden
evaluation contains additional task categories. No resolved public outcome is
used for local model training, selection, calibration, or entity routing.

## Research used and limited conclusions

- The [M4 competition study](https://doi.org/10.1016/j.ijforecast.2019.04.014)
  found successful combinations of statistical forecasts. This motivates an
  ensemble of observed House forecasts and historical rules; it does not imply
  an M4 method will beat the Agenthon cross-entity baseline.
- [Tan et al., NeurIPS 2024](https://arxiv.org/abs/2406.16964) found no consistent
  benefit from the LLM component in several time-series forecasters. We use
  House for task interpretation and event context while keeping small numeric
  rules explicit and cutoff-safe.
- [FinQA](https://arxiv.org/abs/2109.00122) documents the challenge of
  numerical reasoning over financial reports. S1.4 retrieves entity-matched
  EPS premises and computes a transparent reported-quarter growth proxy.
- [Qlib](https://github.com/microsoft/qlib) stresses nonstationary financial
  distributions; S1.4 recency-weights cutoff-safe ranking validation cohorts.
- [StatsForecast](https://github.com/Nixtla/statsforecast) provides a model
  family with rolling-origin validation, and
  [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) documents residual
  interval calibration. These are design references, not bundled dependencies;
  S1.4 uses only Python's standard library. Interval calibration remains a
  future experiment because small, dependent cohorts do not justify coverage
  guarantees on unseen task categories.

## S1.4 changes and provenance

1. **Ranking coherence:** S1.3 split the 10-row COT roster into batches of
   6 and 4 despite giving both batches an overview. S1.4 sends a ranking roster
   of up to 12 rows in one House request so all numeric estimates share a scale.
2. **Ranking drift:** add anti-momentum as a candidate and select by a
   recency-weighted, pre-cutoff Spearman validation. On the published COT input,
   the selected rule has weighted validation rho 0.203; that is historical
   evidence, not an estimate of the hidden outcome.
3. **Forecast variability:** take up to three House samples for ranking and
   regression within the 25-request unit cap, then choose a complete observed
   ranking or a row forecast near the numeric median. Cited evidence remains
   paired with the selected original forecast.
4. **Financial statement arithmetic:** when a regression target asks for EPS
   growth, extract the latest reported diluted EPS pair from the entity's own
   pre-cutoff 10-Q. Shrink unusually large growth rates toward the roster
   median, and include the matching passage among House's evidence. The public
   bank unit yields eight nonzero, entity-specific priors; no Q3 outcomes are
   read or inferred as known.

## Limits before interpreting the next leaderboard result

Public unit score alone cannot separate predictive quality from interval
coverage or model-output variability. The live House route is unavailable in
local smoke tests. Contract and citation tests verify runtime validity, not
future score. A gain in one public family may fail to generalize to hidden
targets, so future changes should be accepted primarily through pre-cutoff
rolling tests and source-identity checks.
