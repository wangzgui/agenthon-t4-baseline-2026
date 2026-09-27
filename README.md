## Executive summary (read this first)

This is a first, general-purpose Agenthon Track 4 submission. It runs the
organizer-hosted House model when available and cites passages retrieved from
the task's frozen, cutoff-filtered corpus. The model-free fallback is for
interface checks and House failures; it is not a reliable prediction method.
The code adapts the MIT-licensed strong RAG scaffold from the official public
Track 4 repository, with its license copied to `UPSTREAM_LICENSE`.

Build: `docker build --platform linux/amd64 -t agenthon-t4-baseline:v1.0 .`

Run: `docker run --rm -v UNIT:/input:ro -v OUT:/output agenthon-t4-baseline:v1.0 analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json`

The GitHub workflow validates all published practice unit shapes on Linux and
then publishes the image to GHCR. No Team Key is stored in this repository.
