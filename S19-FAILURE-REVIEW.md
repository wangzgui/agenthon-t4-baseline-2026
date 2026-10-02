# S1.9 failed submission and S1.9.1 runtime repair

## Observed platform evidence

Submission 956646, `submission-s1.9.zip`, uploaded 2026-10-02 19:43
(page display, Asia/Shanghai), status Failed. The tooltip says:
"Submission processing failed. Diagnostic details are available to the organizers."
The Logs tab says program logs are withheld. Neither prediction nor scoring
output can be downloaded from the visible detail panel. This does not identify
the failing stage, exception, unit or infrastructure reason.

Original ZIP schema, team binding, source revision and immutable public
linux/amd64 image were rechecked successfully. The original ZIP is preserved.

## Reproduced defect, distinct from proven platform cause

A House review returning `correction_kind: ["arithmetic"]` reaches an unchecked
set membership in `s19_decisions.review_admission`. The original code raises
`TypeError: unhashable type: 'list'`. This can abort a unit. It is a demonstrated
code bug; without organizer diagnostics it is not a demonstrated explanation
of submission 956646's Failed status.

Additional related fault paths: null/malformed evidence IDs, non-string unit
objects reaching an alias dictionary, and exceptions escaping a parallel House
batch. These now decline the invalid review, use the existing numerical fallback
or isolate the failed batch, instead of aborting the full unit.

## Repair and tests

S1.9.1 changes runtime type guards and per-batch recovery only. Valid review
selection, numerical priors, fusion, interval calibration, retrieval, House
model/budget and cutoff rules stay as in S1.9. Original S1.6 remains frozen.

Tests include the pre-fix reproducer, malformed field fixtures, a full-unit HTTP
serialization/parser fault-injection test and an escaped parallel-batch fault.
These fixtures are robustness evidence, not real House forecast validation.
Linux CI repeats official schema/citation checks and public-unit smoke.

Before attributing the platform failure or expecting a successful retry, the
organizer would need to confirm the stage/reason for submission 956646. A useful
request is: please confirm whether the failure was image pull, descriptor/team
validation, agent exit/timeout, or organizer evaluation, and whether a rerun of
the original submission is appropriate. No organizer message is sent here.
