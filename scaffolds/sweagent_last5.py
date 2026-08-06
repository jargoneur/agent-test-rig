from __future__ import annotations

import json

from harness.upstreams import import_upstream
from scaffolds.upstream_base import UpstreamScaffold


class Scaffold(UpstreamScaffold):
    strategy_name = "sweagent_last5"
    scaffold_context = (
        "SWE-agent LastNObservations is applied to the action/observation history. "
        "Older environment outputs are elided by the frozen upstream implementation."
    )

    def __init__(self, model=None, task=None, options=None):
        super().__init__(model=model, task=task, options=options)
        module = import_upstream(
            "sweagent_last5",
            "sweagent.agent.history_processors",
        )
        self.processor = module.LastNObservations(
            n=int(self.options.get("n", 5)),
            polling=int(self.options.get("polling", 1)),
        )

    def history_for_prompt(self, issue, tools, history, state):
        upstream_history = [
            {
                "role": "user",
                "content": "Initial task description is provided above.",
                "message_type": "observation",
                "is_demo": False,
            }
        ]
        for item in history:
            upstream_history.append(
                {
                    "role": "assistant",
                    "content": str(
                        item.get("response")
                        or json.dumps(
                            item.get("action") or {},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                    "message_type": "action",
                    "is_demo": False,
                }
            )
            upstream_history.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        item.get("observation") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "message_type": "observation",
                    "is_demo": False,
                }
            )
        return self.processor(upstream_history)
