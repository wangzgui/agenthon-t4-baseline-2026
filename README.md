## Executive summary (read this first)

S1.3 is a general-purpose Agenthon Track 4 submission. It runs the
organizer-hosted House model when available and cites passages retrieved from
the task's frozen, cutoff-filtered corpus. A lightweight numerical rule is
fitted and selected from historical observations already in that corpus when
rolling-origin validation supports it. This is a forecast prior, not a claim
that the unpublished outcome is known. Tasks without adequate historical
observations retain a deterministic contract-safe fallback.
The code adapts the MIT-licensed strong RAG scaffold from the official public
Track 4 repository, with its license copied to `UPSTREAM_LICENSE`.

Build: `docker build --platform linux/amd64 -f Dockerfile.s13 -t agenthon-t4-baseline:s1.3 .`

Run: `docker run --rm -v UNIT:/input:ro -v OUT:/output agenthon-t4-baseline:s1.3 analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json`

`ARTIFACT_PROVENANCE.md` records the statistical rule, cutoff checks and model disclosure.

The GitHub workflow validates all published practice unit shapes on Linux and
then publishes the image to GHCR. No Team Key is stored in this repository.
