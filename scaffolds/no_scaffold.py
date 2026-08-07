from __future__ import annotations

from scaffolds.base import BaseScaffold


class NoScaffold(BaseScaffold):
    """Control arm: tool protocol plus the complete, unmodified history."""

    strategy_name = "no_scaffold"
    scaffold_context = (
        "No repository localization, repository map, history truncation, or "
        "history summary is supplied. Use the ordinary repository tools to "
        "find context."
    )

    def repository_context(self, issue, tools, history, state):
        state["context_files"] = []
        state["context_spans"] = {}
        return "No precomputed repository context is supplied."

    def history_for_prompt(self, issue, tools, history, state):
        return list(history)


Scaffold = NoScaffold
