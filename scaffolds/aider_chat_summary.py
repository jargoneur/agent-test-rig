from __future__ import annotations

from harness.upstreams import import_upstream
from scaffolds.upstream_base import AiderModelShim, UpstreamScaffold


class Scaffold(UpstreamScaffold):
    strategy_name = "aider_chat_summary"
    scaffold_context = (
        "Older interaction history is compacted by Aider ChatSummary from the "
        "frozen upstream checkout when the configured history budget is exceeded."
    )

    def __init__(self, model=None, task=None, options=None):
        super().__init__(model=model, task=task, options=options)
        module = import_upstream("aider_chat_summary", "aider.history")
        model_shim = AiderModelShim(
            self,
            name=getattr(self.model, "model_name", "evaluated-model"),
        )
        self.summarizer = module.ChatSummary(
            models=[model_shim],
            max_tokens=int(self.options.get("max_history_tokens", 4096)),
        )

    def history_for_prompt(self, issue, tools, history, state):
        messages = self.history_as_messages(history)
        if not messages:
            return []
        if self.summarizer.too_big(messages):
            return self.summarizer.summarize(messages)
        return messages
