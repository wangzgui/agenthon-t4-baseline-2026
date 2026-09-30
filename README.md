# Agenthon Track 4 S1.7

A cutoff-filtered financial forecasting agent using the organizer House endpoint and standard-library statistical priors.

S1.7 adds literal-source quantities, checked observed comparisons, forecast decomposition validation, supported draft selection, and bounded Final submitted_reasons tied to the normalized answer. Existing time-split residual intervals and quarterly EPS parsing are retained. No resolved competition labels, external training datasets or checkpoints are bundled.

Build: `docker build --platform linux/amd64 -f Dockerfile.s17 -t agenthon-t4-baseline:s1.7 .`

Run: `docker run --rm -v UNIT:/input:ro -v OUT:/output agenthon-t4-baseline:s1.7 analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json`

See `ARTIFACT_PROVENANCE.md` for model/cutoff disclosure and `RESEARCH-S17.md` for validation and limitations. GitHub Actions verifies the agent and publishes its Linux amd64 image to GHCR. The submission ZIP references that immutable image and includes a signed team claim; it does not contain a Team Key.

Adapted from the official MIT Track 4 scaffold; upstream license is in `UPSTREAM_LICENSE`.
