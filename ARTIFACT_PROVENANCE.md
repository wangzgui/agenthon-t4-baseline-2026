# S1.3 artifact provenance

S1.3 uses the organizer provided House model and a small local statistical rule. It does not bundle external training data, resolved public practice outcomes, neural weights, or an inference corpus.

## Runtime fitted statistical rule

- **Code:** `s13_numerics.py`; its SHA-256 is recorded as the local model revision in `submission.json`.
- **Inputs:** dated numeric tables in the current unit's organizer supplied frozen corpus, filtered to dates no later than `task.cutoff_date`.
- **Fitting:** a two feature ridge rule predicts a future change from the recent change and deviation from a trailing median. Deterministic zero, momentum and reversion rules are also candidates.
- **Selection:** older pre-cutoff rolling origin examples fit coefficients; later pre-cutoff examples select a candidate by same-cohort realized-mean MAE skill for regression or cross-sectional rank correlation for ranking. A minimum margin prevents use of a weak candidate. The selected ridge rule is then refit on all historical examples available before the unit cutoff.
- **Labels:** each historical label is a change between two observations both present in the supplied pre-cutoff table. No task resolution, development leaderboard, post-cutoff label, or answer lookup is used for fitting, selection, or calibration.
- **Calibration:** S1.3 retains a deterministic interval heuristic and uses the emitted interval to reconcile ternary threshold labels with numeric forecasts. No labeled calibration artifact is bundled.
- **License/source:** the runtime training data are part of the official unit corpus supplied under the competition terms. The statistical code is MIT licensed with this repository. No external source is fetched at evaluation time.

If the supplied corpus has too few dated observations or historical validation does not beat the zero-change baseline, the local fitted rule is not used. This also prevents a public task family from becoming an implicit hard-coded route for hidden units.

## House model

`nvidia/nemotron-3-super-120b-a12b`, revision `rl-030326-fp8`, is served through the organizer's approved endpoint. The image contains no House weights or adapter. Its base training cutoff is unpublished; S1.3 supplies only the current task and its pre-cutoff corpus passages in prompts.
