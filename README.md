## Executive summary (read this first)
S1.8 is a controlled retrieval experiment built on S1.6. The numerical and House prediction path remains the S1.6 path. A bounded second pass adds eligible corpus evidence when task-derived keyword needs are not covered. Optional local diagnostics expose retrieval, model completion and numerical decisions without modifying predictions. No outcome lookup, extra model or external inference corpus is used.

Build: `docker build --platform linux/amd64 -f Dockerfile.s18 -t agenthon-t4-baseline:s1.8 .`

Run: `docker run --rm -v UNIT:/input:ro -v OUT:/output agenthon-t4-baseline:s1.8 analyze --task /input/task.json --corpus /input/corpus --out /output/answer.json`

Local control: set `AGENTHON_SUPPLEMENT_RETRIEVAL=0`. Local trace: set `AGENTHON_DIAGNOSTICS_DIR` to your local output directory. It is off by default on the platform. `test_s18_diagnostics.py OFFICIAL_REPO` checks that the control generates identical S1.6 model requests and predictions under the same mock replies.

See `RESEARCH-S18.md` and `ARTIFACT_PROVENANCE.md`. The signed submission ZIP references the published image by immutable digest. The original S1.6 ZIP remains the scored control (0.4793). Upstream MIT license: `UPSTREAM_LICENSE`.
