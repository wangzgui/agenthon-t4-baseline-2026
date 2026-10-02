# S1.9 — conservative decision repair from frozen S1.6

## Evidence and scope

S1.6 remains the active control (reported Development score 0.4793). S1.8's
keyword-driven supplementary retrieval changed passages and could replace
original extracted facts. Without real House intermediate outputs or per-unit
scores, this is an identified mechanism of risk, not a proven attribution of
the full score loss. S1.7's outcome likewise does not isolate any component.

S1.9 starts from the byte-frozen S1.6 control, with these bounded changes:

1. Keep baseline numeric prior selection, fusion, intervals, filing parser,
   initial retrieval, request budget, seeds and House model. No supplementary
   retrieval and no external training labels are introduced.
2. Append at most three unique exact-quote House facts to the original local
   observations instead of replacing those observations. Model interpretations
   remain provisional. Internal fields are excluded from raw entity features.
3. Accept a changed review point only with a typed correction, a specific change
   explanation, and a mechanism anchored to a cited exact premise. A real quote
   alone no longer authorizes a numerical change. Ranking review is accepted
   as a whole roster or the whole first draft is retained.
4. Request bounded economic reasoning in the existing calls; build up to three
   Final reasons against the actual normalized point, not the pre-fusion value.
   Ownership, quote offsets, text bounds and sign conflicts are checked.
5. Opt-in local diagnostics record selected facts, review acceptance/fallback,
   selected forecasts and final outputs without credentials. Diagnostic logging
   is disabled by default and never enters model requests.

## What vintage evidence establishes

The locally archived primary-release study verified 91 Treasury auction values
and 99 rounded CPI first-print values against dated official releases. It did
not establish a historical byte snapshot of every source or high-precision CPI
ALFRED vintages. COT publication availability remains unverified. There are too
few independent CPI origins to use this sample for reliable fusion or interval
parameter selection. These research labels and files are excluded from the
image. No lookup of public task IDs, entities' future outcomes or leaderboard
scores occurs in inference.

The S1.6 historical fit remains a frozen-corpus statistical heuristic. Date
purging prevents observation-date overlap but does not prove historical release
availability. S1.9 corrects its prompt status to explicitly say vintage
unverified; it does not falsely describe snapshot residuals as a fully audited
real-time backtest.

## Validation and limitations

Contract fixtures/mocked HTTP responses verify serialization and failure paths,
not forecast accuracy. Linux container smoke uses the official pinned public
repository and toolkit. Offline control comparison verifies the unchanged
numeric fallback. No authorized local House endpoint is available; real-model
forecast uplift and semantic faithfulness remain unmeasured before submission.

The structured correction gate is not an NLI model: a coherent but economically
incorrect explanation can pass. Appending facts can still affect attention.
Final reasoning can fail semantic judging even with exact citations. Treat
S1.9 as a candidate, preserve S1.6, and do not promote it based on local schema
tests. Development scores alone cannot establish hidden-task generalization.

Sources already archived in the diagnostic study:
- [Track 4 training policy](https://github.com/Agenthon-2026/track4-analysis-public/blob/ede7381d8c1ba9d8c84068f9d142f5e093a33892/docs/TRAINING-POLICY.md)
- [Development runtime](https://github.com/Agenthon-2026/track4-analysis-public/blob/ede7381d8c1ba9d8c84068f9d142f5e093a33892/docs/DEVELOPMENT-RUNTIME.md)
- Local evidence: `test-output/diagnostic-study/vintage/data_report.md`.
