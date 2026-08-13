# ContextBench action protocol v2

## Why v1 was stopped

The v1 pilot was manually paused on 2026-08-13 after 15 complete paired
blocks (75 runs). Its model profiles had full native input context but no
per-action output limit. The OpenAI-compatible payload therefore omitted
`max_tokens`; llama.cpp exposed the active slots as `max_tokens = -1`.

With thinking enabled, a single agent action could consequently decode toward
the remaining 131,072- or 262,144-token context window. The six-hour timeout
was applied to each model call, not to the whole paired block. Five observed
pathological calls used about 30 of the first 35.4 aggregate run-hours. The
agent also asked models to encode complete-file replacements inside one JSON
string, while JSON validity was prompt-only rather than decode-constrained.
Those conditions explain both the hours-long generations and most truncated or
unparseable action responses.

The v1 results remain usable pilot data. They must not be pooled with v2
because the action protocol and time limit are experimental-condition changes.
The authoritative paused snapshot is:

`/var/tmp/agent-test-rig-runtime/e9ee54a/scheduler/backups/scheduler-20260813T130452Z-pilot-v1-paused.sqlite3`

It passed the scheduler's thorough SQLite integrity check when created. The
matching experiment is `contextbench_scaffold_boundaries_v1`, harness commit
`e9ee54a`, with 15 complete blocks, no leased blocks, and no failed blocks at
the archival pause.

## Frozen v2 changes

The v2 experiment is `contextbench_scaffold_boundaries_v2` and keeps:

- all 150 frozen ContextBench tasks;
- all 11 locked model profiles and full native input context;
- thinking mode and each profile's recommended sampling settings;
- the five paired scaffold conditions, three repeats, and same-host block rule;
- terminal model/scaffold failures as usable observations.

It changes only the action execution protocol:

- every model action has `max_tokens = 8192`;
- every model call has a 1,800-second deadline;
- a server `finish_reason = length` is stored as the usable terminal outcome
  `model_output_limit`, rather than becoming a parse error or infrastructure
  retry;
- the main action call uses a server-side JSON schema; auxiliary scaffold model
  calls remain unconstrained so upstream summarizers/localizers retain their
  native output contracts;
- `replace_text` performs an exact, localized edit after the file was read.
  It refuses empty, missing, or ambiguous matches. `write_file` remains for new
  files and genuinely necessary complete-file rewrites.

These are output/action bounds, not input-context restrictions. A model that
cannot produce a valid action within the frozen bound is recorded as such; its
result is not discarded.

## Launch criteria

The full v2 scheduler may start only after all of the following pass:

1. complete unit and distributed-recovery validation;
2. a clean exact harness commit and a 24,750-run/4,950-block v2 manifest;
3. the v2 launch gate, including the 8,192-token limit and 1,800-second timeout;
4. a live action-protocol smoke for both Qwen and Gemma that records a valid
   JSON action, a normal stop reason, and bounded completion tokens;
5. a separate v2 scheduler database so v1 and v2 records cannot mix.
