from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Dict, List

from harness.upstreams import import_upstream
from scaffolds.upstream_base import UpstreamScaffold


class _DecoderShim:
    def __init__(
        self,
        scaffold: UpstreamScaffold,
        name: str,
        temperature: float,
        max_new_tokens: int,
    ):
        self.scaffold = scaffold
        self.name = name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def codegen(self, message: str, num_samples: int = 1, prompt_cache: bool = False):
        if num_samples != 1:
            raise ValueError("The frozen main protocol uses one Agentless sample")
        response = self.scaffold.auxiliary_generate(
            message,
            purpose="agentless_localization",
            options={
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        )
        metadata = dict(getattr(self.scaffold.model, "last_generation_metadata", {}) or {})
        usage = dict(metadata.get("usage") or {})
        return [
            {
                "response": response,
                "usage": {
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                },
            }
        ]

    def is_direct_completion(self):
        return False


class Scaffold(UpstreamScaffold):
    strategy_name = "agentless_localization"
    scaffold_context = (
        "Before the action loop, the frozen Agentless file and symbol localization "
        "pipeline selected the repository context shown below."
    )

    def __init__(self, model=None, task=None, options=None):
        super().__init__(model=model, task=task, options=options)
        self.fl_module = import_upstream(
            "agentless_localization",
            "agentless.fl.FL",
        )
        self.preprocess = import_upstream(
            "agentless_localization",
            "agentless.util.preprocess_data",
        )
        self.model_module = import_upstream(
            "agentless_localization",
            "agentless.util.model",
        )
        self.structure_module = import_upstream(
            "agentless_localization",
            "get_repo_structure.get_repo_structure",
        )
        self._localized_context: Dict[str, str] = {}

    @contextlib.contextmanager
    def _patched_model_factory(self):
        original = self.model_module.make_model

        def make_model(
            model,
            backend,
            logger,
            batch_size=1,
            max_tokens=1024,
            temperature=0.0,
        ):
            if batch_size != 1:
                raise ValueError("Agentless adapter supports batch_size=1")
            return _DecoderShim(
                self,
                name=str(model),
                temperature=float(temperature),
                max_new_tokens=int(max_tokens),
            )

        self.model_module.make_model = make_model
        try:
            yield
        finally:
            self.model_module.make_model = original

    @staticmethod
    def _normalise_path(path: str, repo_root: Path) -> str:
        candidate = repo_root / path
        if candidate.exists():
            return path
        parts = Path(path).parts
        if len(parts) > 1:
            stripped = str(Path(*parts[1:]))
            if (repo_root / stripped).exists():
                return stripped
        return path

    def _build_localized_context(self, issue, tools):
        key = str(tools.repo_path)
        if key in self._localized_context:
            return self._localized_context[key]

        structure = self.structure_module.create_structure(key)
        self.preprocess.filter_none_python(structure)
        if bool(self.options.get("exclude_tests", True)):
            self.preprocess.filter_out_test_files(structure)

        logger = logging.getLogger("agentless.%s" % self.task.get("instance_id", "task"))
        logger.addHandler(logging.NullHandler())
        fl = self.fl_module.LLMFL(
            str(self.task.get("instance_id") or self.task.get("id") or "task"),
            structure,
            issue,
            getattr(self.model, "model_name", "evaluated-model"),
            "openai",
            logger,
        )
        top_n = int(self.options.get("top_n", 5))
        context_window = int(self.options.get("context_window", 10))

        with self._patched_model_factory():
            found_files, _file_artifact, _file_traj = fl.localize(top_n=top_n)
            found_files = list(found_files or [])[:top_n]
            found_locs = {}
            if found_files:
                found_locs, _loc_artifact, _loc_traj = (
                    fl.localize_function_from_compressed_files(
                        found_files,
                        temperature=0.0,
                        keep_old_order=True,
                    )
                )

        snippets: List[str] = []
        file_contents = self.preprocess.get_repo_files(structure, found_files) if found_files else {}
        for upstream_path in found_files:
            display_path = self._normalise_path(upstream_path, tools.repo_path)
            content = file_contents.get(upstream_path, "")
            locs = found_locs.get(upstream_path, []) if isinstance(found_locs, dict) else []
            intervals = []
            if locs:
                _raw, intervals = self.preprocess.transfer_arb_locs_to_locs(
                    locs,
                    structure,
                    upstream_path,
                    context_window=context_window,
                    loc_interval=True,
                    file_content=content,
                )
            rendered = self.preprocess.line_wrap_content(
                content,
                context_intervals=intervals or None,
                no_line_number=False,
                sticky_scroll=True,
            )
            snippets.append("### %s\n%s" % (display_path, rendered))

        if not snippets:
            context = "Agentless returned no valid localization for this task."
        else:
            context = "\n\n".join(snippets)
        self._localized_context[key] = context
        return context

    def init_state(self, issue, tools):
        state = super().init_state(issue, tools)
        context = self._build_localized_context(issue, tools)
        state["agentless_context_characters"] = len(context)
        state["auxiliary_calls"] = list(self.auxiliary_calls)
        return state

    def repository_context(self, issue, tools, history, state):
        return self._build_localized_context(issue, tools)
