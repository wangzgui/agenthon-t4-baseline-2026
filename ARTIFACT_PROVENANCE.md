# S1.5 artifact provenance

No resolved competition outcomes, answer lookup, external datasets or neural checkpoints are bundled. All local fitting, selection and residual estimation use dated tables in the current task's official, pre-cutoff corpus. Corpus publication date and observation date must both satisfy task.cutoff_date. No inference internet access is added.

## Numerical pipeline
Canonical target units are computed before House forecast aggregation and before statistical fusion. A projected level and a direct change follow the same fusion path. The numeric modules, agent and financial parser are hashed into the local-model descriptor revision.

## Fitting, selection, calibration
Historical examples use only observed endpoints present in the eligible corpus. Calibration starts at a fixed 75% historical-origin boundary. All training/selection label endpoints are strictly before that boundary. A purged older split fits coefficients; later prefix examples select among zero-change, momentum, anti-momentum, reversion and ridge. Regression selection uses MAE, not a realized-cohort mean. The local zero-change/persistence reference is not claimed to equal a private unit's declared naive answer. Refit is restricted to the prefix, excluding calibration outcomes. Selection sample size determines a heuristic prior weight; this is not learned House-blend validation.

Held-out errors are linked to the same entity-matched historical series. Counts of time origins are distinct from counts of entity rows; serial dependence prevents a formal coverage guarantee. Residual quantiles are shrunk towards explicitly uncalibrated semantic bands. The final House correction and model disagreement enlarge the band because residuals validate the statistical component only. With no residuals, the House/semantic band is explicitly uncalibrated. Calibration observations are never used to pick parameters or the model. The frozen documents establish availability by the task cutoff; they do not establish original unrevised vintages at every historical pseudo-origin.

## Financial statements
EPS proxies require an explicitly quarterly, year-over-year reported diluted-EPS comparison. Mixed metric comparisons select EPS after the matching metric label, rather than net-income billions. QoQ, accumulated-period and unlabeled pairs are rejected. Source document, exact span, unit, basis and periods are recorded. Reported-quarter growth is a weak forecast proxy, blended at 0.35 when a House forecast exists; it is not the target quarter's known result.

## Evidence
Source eligibility reads entity_ids/shared from official manifests. Source claims are exact quotes from eligible entity/shared documents. The task table is reconstructed exactly as the scorer defines it, with row-scoped offsets, and is available as an honest feature citation or fallback. No empty claims array or arbitrary different-company padding is emitted.

## House and reproducibility
The organizer's MODEL_NAME is passed unchanged to its House endpoint. The descriptor retains the registered House disclosure nvidia/nemotron-3-super-120b-a12b / rl-030326-fp8 (base training cutoff unpublished); deployment model identity is organizer controlled. There are at most 25 calls/unit, 4000 output tokens/call, temperatures 0/0.2, fixed seeds. No retries add calls. The image is standard-library Python, Linux amd64.

Official validation: toolkit v2.5.1; Track 4 commit fe313cee2865fbfbe47b65a8fcf7b830a40ea141 (scorer 5.2.1). Public examples validate interface and source integrity, not hidden-task accuracy. Sources: official unit corpus under its supplied licenses; local code MIT. No external training artifact is bundled.
