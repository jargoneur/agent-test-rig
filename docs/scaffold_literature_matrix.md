# Literature Matrix for Context-Management Scaffolds

**Status:** design decision v1.0, frozen on 2026-08-05  
**Scope:** repository-level issue resolution with small local coding models  
**Selection rule:** each main condition must represent a documented and commonly used context-management mechanism, be reproducible in the same agent harness, and avoid changing planning, editing, testing, or retry strategy unless that change is intrinsic to the context mechanism.

## Decision matrix

| Candidate | Primary evidence | Evidence of practical use | Context problem addressed | Isolatable in this harness? | Main confounds | Decision |
|---|---|---|---|---|---|---|
| Rule-based recent-observation truncation | SWE-agent paper; SWE-agent `LastNObservations` documentation | Used in the original SWE-agent configuration; the current implementation explicitly supports keeping the last *n* observations and eliding older outputs | Temporal growth of tool observations | Yes. It only changes the history view sent to the model | Can discard an early decisive observation; favors short trajectories | **Selected as control: `sweagent_last5`** |
| Graph-ranked repository map | Aider repository-map documentation | Aider sends a token-bounded map of files, symbols, signatures, and important definition lines; relevance is ranked over a dependency/reference graph | Global spatial orientation in a large repository | Yes. It can be added as a read-only context provider without changing tools or control loop | Extra tokens can overwhelm weaker models; static/global context may lower precision | **Selected: `aider_repomap`** |
| Hierarchical code localization | Agentless; AutoCodeRover; supporting evidence from RepoCoder | Agentless localizes file -> class/function -> edit location; AutoCodeRover uses AST-aware iterative code search; RepoCoder establishes iterative repository retrieval | Targeted spatial selection of files, symbols, and spans | Yes, if localization calls and their cost are fully logged and the resulting context is injected into the unchanged action loop | Adds model calls and may improve performance through decomposition as well as context selection | **Selected: `hierarchical_localization`** |
| Threshold-triggered LLM history summarization | Aider automatic history summarization; source-code taxonomy of 13 coding agents; Claude Code architecture analysis | Aider, OpenHands, Gemini CLI, Codex CLI, and OpenCode use scaffold-triggered LLM summarization; Claude Code uses a multi-layer pipeline ending in model-generated auto-compaction | Temporal retention after the raw trajectory becomes too large | Yes. It replaces the control condition's truncation policy while leaving repository access unchanged | Adds summarization calls; summaries may omit or fabricate facts | **Selected: `threshold_summary`** |
| Persistent cross-session memory | General agent-memory literature and interactive coding products | Project instruction files and long-term memory are common in interactive tools | Knowledge reuse across sessions/tasks | No for the main construct. It introduces learning across tasks and order effects | Data leakage, task-order dependence, contamination between benchmark instances | Rejected from main experiment |
| Planning, reflection, test-first, critic, or multi-agent delegation | Broad agent literature | Common in agent systems | Reasoning/control rather than context management | Not cleanly | Changes number of attempts, reasoning depth, roles, and compute | Rejected from main experiment |
| Full Aider, Agentless, SWE-agent, Claude Code, or OpenHands systems | System papers and implementations | Widely used complete products/frameworks | Multiple problems simultaneously | No | Prompts, tools, edit formats, retries, models, and context policies all differ | Rejected; mechanisms are adapted instead |
| Unbounded full history | mini-SWE-agent and simple loops | Exists as a minimal implementation | None | Technically yes | Overflow behavior depends on model/backend; not a defensible production control | Rejected as main control |

## Why these four

The selected set covers two independent context axes while retaining one established control:

1. **Temporal prevention:** `sweagent_last5` bounds history by eliding old observations.
2. **Global spatial orientation:** `aider_repomap` gives a compact structural overview.
3. **Targeted spatial selection:** `hierarchical_localization` selects likely files, symbols, and spans.
4. **Temporal cure:** `threshold_summary` compresses older trajectory information instead of merely dropping it.

The conditions are not claims that one product is superior to another. They are controlled operationalizations of recurring architectural mechanisms.

## Frozen operational definitions

### 1. `sweagent_last5` — control

Faithful mechanism: SWE-agent `LastNObservations(n=5)`.

- Keep the immutable system prompt, task statement, tool schema, current repository state, and model actions.
- Keep complete outputs for the five most recent observations.
- Replace older observation outputs with a fixed elision marker containing only tool type and omitted-length metadata.
- No repository map, pre-localization, retrieval index, or LLM-generated summary.
- This history policy is also the temporal base for `aider_repomap` and `hierarchical_localization`.

### 2. `aider_repomap`

Faithful mechanism: Aider-style token-bounded repository map.

- Parse Python with Tree-sitter or the standard AST where Tree-sitter is unavailable.
- Represent file paths, classes, functions, methods, signatures, imports, and critical definition lines.
- Build a directed reference/import graph and rank nodes using PageRank-style centrality plus task/chat relevance.
- Refresh ranking after files are read or edited, without an additional LLM call.
- Inject at most **2,048 tokens** of map content per model turn.
- Retain `sweagent_last5` history processing.
- Do not inject full retrieved source bodies automatically.

### 3. `hierarchical_localization`

Faithful mechanism: Agentless file -> symbol -> span localization, supported by AutoCodeRover's structure-aware search.

Before the normal action loop, the evaluated model performs:

1. **File localization:** select likely files from the repository tree and task statement.
2. **Symbol localization:** select classes/functions from skeletons of the shortlisted files.
3. **Span localization:** select concrete source ranges from the shortlisted symbols.

Controls:

- Use the same evaluated model as the action loop.
- Temperature `0`; deterministic derived seed where supported.
- Maximum **512 output tokens per localization call**.
- One schema-repair call is allowed for malformed localization output and is counted.
- Inject at most **2,048 tokens** of localized source context into every action-loop turn.
- All localization calls, tokens, latency, parser failures, and selected entities are logged.
- Retain `sweagent_last5` history processing.
- No repo map and no history summary.

This condition is a system-level context-management comparison. Because decomposition requires extra calls, token/call cost is a required outcome rather than an uncontrolled omission.

### 4. `threshold_summary`

Faithful mechanism: scaffold-triggered model summarization as used by Aider and several contemporary coding agents.

- No repository map and no pre-localization.
- Keep the immutable system prompt and task statement.
- When raw trajectory history exceeds **4,096 tokens**, summarize all but the two most recent observations.
- Use the same evaluated model, temperature `0`, maximum **1,024 summary tokens**, and a deterministic derived seed where supported.
- Summary schema must separate:
  - confirmed repository facts,
  - hypotheses,
  - relevant files/symbols,
  - edits made,
  - test commands and results,
  - unresolved questions,
  - failed approaches.
- The summary replaces the older history in the active context but the raw event log remains immutable on disk.
- Re-summarize recursively when summary plus new older history again exceeds the threshold.
- All summary calls and token use are logged.

The structured fact/hypothesis separation is a safeguard, not a claim that summarization is lossless. Summary omissions and unsupported assertions are explicit failure modes to measure.

## Common invariants

All four conditions use the same:

- model artifact, quantization, prompt template, tokenizer, backend, and backend version;
- sampling parameters and paired run seed;
- ReAct-style action loop and JSON action protocol;
- read, write, test, and finish tools;
- edit format and write restrictions;
- maximum action steps;
- test commands and success criterion;
- input context window: **16,384 tokens**;
- maximum generated tokens per action call: **2,048**;
- per-tool-observation cap and truncation rule;
- repository checkout and dependency environment.

Auxiliary localization and summarization calls are part of their scaffold and therefore logged rather than hidden or equalized away.

## Literature strength and limitations

- `sweagent_last5`, `aider_repomap`, and `hierarchical_localization` have direct coding-agent implementations and task-level evidence.
- `threshold_summary` is clearly common in deployed/open-source coding-agent architectures, but no dominant summarization design has emerged. The 2026 source-code taxonomy reports seven distinct compaction strategies across 13 agents and explicitly characterizes compaction as an active design frontier.
- Therefore, the fourth condition tests the common **family** of threshold-triggered LLM summarization, not an assertion that the chosen structured summary is the unique industry standard.

## References

1. Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models*. https://arxiv.org/abs/2210.03629
2. Yang et al. (2024), *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. https://arxiv.org/abs/2405.15793
3. SWE-agent documentation, *History processors / LastNObservations*. https://swe-agent.com/1.0/reference/history_processor_config/
4. Aider documentation, *Repository map*. https://aider.chat/docs/repomap.html
5. Zhang et al. (2024), *AutoCodeRover: Autonomous Program Improvement*. https://arxiv.org/abs/2404.05427
6. Xia et al. (2024), *Agentless: Demystifying LLM-based Software Engineering Agents*. https://arxiv.org/abs/2407.01489
7. Zhang et al. (2023), *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation*. https://arxiv.org/abs/2303.12570
8. Aider documentation, `--max-chat-history-tokens` and automatic summarization. https://aider.chat/docs/config/options.html
9. Rombaut (2026), *Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures*. https://arxiv.org/abs/2604.03515
10. Liu et al. (2026), *Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems*. https://arxiv.org/abs/2604.14228
11. *Compaction as Epistemic Failure: How Agentic LLM Tools Fabricate Confirmed Results from Killed Processes* (2026). https://arxiv.org/abs/2607.13071
