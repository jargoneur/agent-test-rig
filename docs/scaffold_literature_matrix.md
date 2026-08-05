# Literature Matrix for Context-Management Scaffolds

**Status:** design decision v1.1, frozen on 2026-08-05  
**Scope:** repository-level issue resolution with small local coding models

## Source-code rule

Benchmark and scaffold algorithms are **not reimplemented in this repository**. The study uses pinned upstream implementations and permits only thin adapters for message schemas, model endpoints, scheduler integration, and logging. Any compatibility change must be an explicit patch file with a recorded hash. Exact repositories and commits are frozen in `docs/upstream_components.lock.yml`.

This rule matters scientifically: the named conditions must be actual published implementations, not locally recreated approximations that merely resemble them.

## Decision matrix

| Condition | Upstream implementation | Literature/practice basis | Context problem | Decision |
|---|---|---|---|---|
| `sweagent_last5` | SWE-agent `LastNObservations(n=5)` in `sweagent/agent/history_processors.py` | Original SWE-agent history policy; directly documented in source and configuration | Temporal growth of tool observations | **Selected control** |
| `aider_repomap` | Aider `RepoMap` in `aider/repomap.py` | Widely used Aider repository map; Tree-sitter and graph-ranked structural context | Global repository orientation | **Selected** |
| `agentless_localization` | Agentless fault-localization pipeline in `agentless/fl/` | Agentless hierarchical file -> symbol -> location pipeline; supported conceptually by AutoCodeRover and RepoCoder | Targeted spatial context selection | **Selected** |
| `aider_chat_summary` | Aider `ChatSummary` in `aider/history.py` | Production coding-agent history compaction with threshold-triggered recursive summarization | Temporal retention after context growth | **Selected** |
| Persistent cross-task memory | No single selected implementation | Common in interactive products | Reuse across tasks | Rejected: leakage and task-order effects |
| Planning, reflection, test-first, critic, multi-agent | Many implementations | Common agent strategies | Reasoning/control | Rejected: not primarily context management |
| Complete Aider, Agentless, SWE-agent, or OpenHands systems | Full upstream systems | Established products/frameworks | Several mechanisms simultaneously | Rejected: too many confounded variables |

## Why these four

The set covers two independent context axes while retaining one established control:

1. `sweagent_last5`: discard old observation content while keeping recent observations.
2. `aider_repomap`: supply a compact global map of repository structure.
3. `agentless_localization`: select likely files, symbols, and edit locations hierarchically.
4. `aider_chat_summary`: compress old dialogue into a model-generated summary instead of dropping it.

The experiment compares mechanisms, not brands. Upstream code is called through a common harness, but its ranking, elision, localization, and summarization logic remains unchanged.

## Allowed adapters

Adapters may only:

- convert the harness event schema into the upstream component's required schema and back;
- expose the same local model through the API expected by the component;
- pass repository paths and fixed configuration values;
- capture outputs, calls, tokens, latency, exceptions, and provenance;
- enforce the experiment-wide resource limits outside the upstream algorithm.

Adapters may not:

- rewrite prompts or ranking formulas;
- replace parsers with locally invented alternatives;
- post-process outputs to improve relevance or correctness;
- add retries not present in the frozen protocol;
- silently fall back to a custom local implementation.

A component that cannot run under these restrictions is removed through a protocol amendment rather than recreated locally.

## Frozen upstream sources

### Benchmark: ContextBench

- Repository: `EuniAI/ContextBench`
- Pinned commit: `1436c28a8eb95496da4ea69ad458b9f8a8eb7d61`
- Reused directly: dataset loader, unified runner, trajectory processors, symbol/span extraction, gold-context metrics, and evaluator.
- No local rewrite of task definitions or context metrics.

### Control: SWE-agent LastNObservations

- Repository: `SWE-agent/SWE-agent`
- Pinned commit: `3ea751c087f32b16e039a2233dd6eefecef325d5`
- Source: `sweagent/agent/history_processors.py`
- Symbol: `LastNObservations`
- Frozen parameters: `n=5`, `polling=1`.

The upstream class itself determines which observation outputs are elided and the replacement text.

### Repository map: Aider RepoMap

- Repository: `Aider-AI/aider`
- Pinned commit: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`
- Source: `aider/repomap.py`
- Symbol: `RepoMap`.

The map content, parsing, graph construction, ranking, and rendering come from Aider. The adapter only supplies repository files, task/chat context, model token metadata, and the frozen token budget.

### Hierarchical localization: Agentless

- Repository: `OpenAutoCoder/Agentless`
- Pinned commit: `5ce5888b9f149beaace393957a55ea8ee46c9f71`
- Sources: `agentless/fl/FL.py`, `localize.py`, `retrieve.py`, and `Index.py`.

The original localization stages and prompts are retained. Repair generation and patch ranking from Agentless are not used, because the condition is limited to context localization.

### History compaction: Aider ChatSummary

- Repository: `Aider-AI/aider`
- Pinned commit: `5dc9490bb35f9729ef2c95d00a19ccd30c26339c`
- Source: `aider/history.py`
- Symbol: `ChatSummary`.

The upstream split, recursion, prompt, prefix, and summary replacement behavior remain unchanged. The adapter only provides the evaluated model through Aider's expected model interface.

## Common controls

All conditions use the same evaluated model artifact, quantization, backend, sampling settings, action tools, edit format, maximum action steps, repository checkout, validation tests, and outer context window. Additional calls made by Agentless localization or Aider summarization are part of those published mechanisms and are logged as cost outcomes.

## Main limitations

- Agentless localization uses extra model calls, so its effect combines context selection with decomposition. This cannot be removed without ceasing to use the upstream method; calls and tokens therefore become explicit dependent cost measures.
- Aider's repository map and summary are two components from the same mature system, but they address different spatial and temporal problems and are used independently.
- Upstream packages may assume different message schemas and model APIs. Thin adapters are unavoidable, but adapter tests must prove that inputs and outputs are passed without semantic modification.

## References

1. Yao et al. (2022), *ReAct: Synergizing Reasoning and Acting in Language Models*. https://arxiv.org/abs/2210.03629
2. Yang et al. (2024), *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. https://arxiv.org/abs/2405.15793
3. SWE-agent source, `LastNObservations`. https://github.com/SWE-agent/SWE-agent/blob/3ea751c087f32b16e039a2233dd6eefecef325d5/sweagent/agent/history_processors.py
4. Aider source, `RepoMap`. https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/repomap.py
5. Zhang et al. (2024), *AutoCodeRover: Autonomous Program Improvement*. https://arxiv.org/abs/2404.05427
6. Xia et al. (2024), *Agentless: Demystifying LLM-based Software Engineering Agents*. https://arxiv.org/abs/2407.01489
7. Zhang et al. (2023), *RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation*. https://arxiv.org/abs/2303.12570
8. Aider source, `ChatSummary`. https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/aider/history.py
9. Li et al. (2026), *ContextBench: A Benchmark for Context Retrieval in Coding Agents*. https://arxiv.org/abs/2602.05892
