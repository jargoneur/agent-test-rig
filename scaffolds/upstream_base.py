from __future__ import annotations

import contextlib
import json
import math
import time
from typing import Any, Dict, Iterable, List, Optional

from harness.reproducibility import derive_step_seed
from scaffolds.base import BaseScaffold


class UpstreamScaffold(BaseScaffold):
    """Utilities shared by thin adapters around frozen upstream algorithms."""

    def _resource_checkpoint(self, label: str) -> None:
        policy = self.options.get("_resource_policy")
        if policy is not None:
            policy.checkpoint(label)

    def _next_auxiliary_seed(self) -> Optional[int]:
        run_seed = self.options.get("_run_seed")
        if run_seed is None:
            return None
        return derive_step_seed(
            int(run_seed),
            100_000 + len(self.auxiliary_calls) + 1,
        )

    def _record_auxiliary_call(
        self,
        purpose: str,
        prompt: str,
        response: str,
        started_at: float,
        seed: Optional[int],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = dict(getattr(self.model, "last_generation_metadata", {}) or {})
        self.auxiliary_calls.append(
            {
                "purpose": purpose,
                "seed": seed,
                "prompt_characters": len(prompt),
                "response_characters": len(response or ""),
                "elapsed_seconds": max(0.0, time.time() - started_at),
                "options": dict(options or {}),
                "generation_metadata": metadata,
            }
        )

    @contextlib.contextmanager
    def _temporary_model_options(self, overrides: Optional[Dict[str, Any]] = None):
        if self.model is None:
            raise RuntimeError("Scaffold requires a bound model")
        previous = dict(getattr(self.model, "options", {}) or {})
        if overrides:
            merged = dict(previous)
            merged.update(overrides)
            self.model.options = merged
        try:
            yield
        finally:
            self.model.options = previous

    def auxiliary_generate(
        self,
        prompt: str,
        purpose: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        seed = self._next_auxiliary_seed()
        label = "auxiliary:%s:%d" % (purpose, len(self.auxiliary_calls) + 1)
        self._resource_checkpoint(label + ":before_model_call")
        started = time.time()
        with self._temporary_model_options(options):
            response = self.model.generate(prompt, seed=seed)
        self._resource_checkpoint(label + ":after_model_call")
        self._record_auxiliary_call(
            purpose,
            prompt,
            response,
            started,
            seed=seed,
            options=options,
        )
        return response

    def auxiliary_generate_messages(
        self,
        messages,
        purpose: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        seed = self._next_auxiliary_seed()
        label = "auxiliary:%s:%d" % (purpose, len(self.auxiliary_calls) + 1)
        self._resource_checkpoint(label + ":before_model_call")
        started = time.time()
        prompt = json.dumps(messages, indent=2, ensure_ascii=False)
        with self._temporary_model_options(options):
            if hasattr(self.model, "generate_messages"):
                response = self.model.generate_messages(messages, seed=seed)
            else:
                response = self.model.generate(prompt, seed=seed)
        self._resource_checkpoint(label + ":after_model_call")
        self._record_auxiliary_call(
            purpose,
            prompt,
            response,
            started,
            seed=seed,
            options=options,
        )
        return response

    def token_count(self, value: Any) -> int:
        if self.model is not None and hasattr(self.model, "token_count"):
            return int(self.model.token_count(value))
        if isinstance(value, list):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        return max(1, int(math.ceil(len(text) / 4.0)))

    @staticmethod
    def history_as_messages(history: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for item in history:
            response = item.get("response")
            if response:
                messages.append({"role": "assistant", "content": str(response)})
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            item.get("action") or {},
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        item.get("observation") or {},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        return messages


class AiderModelShim:
    """Small interface expected by Aider RepoMap and ChatSummary."""

    def __init__(self, scaffold: UpstreamScaffold, name: str = "evaluated-model"):
        self.scaffold = scaffold
        self.name = name
        max_input = int(scaffold.options.get("max_input_tokens", 32768))
        self.info = {"max_input_tokens": max_input}

    def token_count(self, value: Any) -> int:
        return self.scaffold.token_count(value)

    def simple_send_with_retries(self, messages) -> str:
        return self.scaffold.auxiliary_generate_messages(
            messages,
            purpose="aider_chat_summary",
        )


class AiderIOShim:
    """Minimal non-interactive subset of Aider's InputOutput interface."""

    def __init__(self, encoding: str = "utf-8"):
        self.messages: List[str] = []
        self.encoding = encoding

    def _capture(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self.messages.append(str(message))

    tool_output = _capture
    tool_warning = _capture
    tool_error = _capture

    def read_text(self, filename: Any, silent: bool = False) -> Optional[str]:
        """Match the text-reading contract used by the frozen Aider RepoMap."""
        try:
            with open(str(filename), "r", encoding=self.encoding) as handle:
                return handle.read()
        except FileNotFoundError:
            if not silent:
                self.tool_error("%s: file not found error" % filename)
        except IsADirectoryError:
            if not silent:
                self.tool_error("%s: is a directory" % filename)
        except OSError as error:
            if not silent:
                self.tool_error("%s: unable to read: %s" % (filename, error))
        except UnicodeError as error:
            if not silent:
                self.tool_error("%s: %s" % (filename, error))
                self.tool_error("Use --encoding to set the unicode encoding.")
        return None
