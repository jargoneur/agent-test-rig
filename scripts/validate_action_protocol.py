#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.model_adapter import OpenAICompatibleModel
from harness.outcomes import TerminalRunError


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live bounded JSON-action/EOG check for a llama.cpp model."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-budget-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output")
    args = parser.parse_args()

    model = OpenAICompatibleModel(
        args.model,
        base_url=args.base_url,
        options={
            "do_sample": False,
            "num_predict": args.max_tokens,
            "reasoning_budget_tokens": args.reasoning_budget_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        timeout=args.timeout,
        backend_name="llama_cpp",
    )
    prompt = (
        "This is a protocol validation, not a coding task. Return the finish "
        "action now with reason protocol_ok."
    )
    report = {
        "schema_version": 1,
        "model": args.model,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "reasoning_budget_tokens": args.reasoning_budget_tokens,
        "timeout_seconds": args.timeout,
        "ok": False,
    }
    try:
        response = model.generate_action(prompt, seed=args.seed)
        action = json.loads(response)
        metadata = dict(model.last_generation_metadata)
        usage = dict(metadata.get("usage") or {})
        completion_tokens = usage.get("completion_tokens")
        report.update(
            {
                "response": response,
                "action": action,
                "finish_reason": metadata.get("finish_reason"),
                "usage": usage,
                "ok": (
                    action.get("action") == "finish"
                    and metadata.get("finish_reason") == "stop"
                    and isinstance(completion_tokens, int)
                    and completion_tokens <= args.max_tokens
                ),
            }
        )
    except TerminalRunError as error:
        report["terminal_error"] = error.as_dict()
    except Exception as error:
        report["error"] = "%s: %s" % (error.__class__.__name__, error)

    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
