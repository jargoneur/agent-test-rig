from __future__ import annotations

import re

from harness.upstreams import import_upstream
from scaffolds.upstream_base import AiderIOShim, AiderModelShim, UpstreamScaffold


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_FILE = re.compile(r"(?:^|\s)([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)")


class Scaffold(UpstreamScaffold):
    strategy_name = "aider_repomap"
    scaffold_context = (
        "The repository index below is the unmodified map produced by Aider RepoMap "
        "from the frozen upstream checkout."
    )

    def __init__(self, model=None, task=None, options=None):
        super().__init__(model=model, task=task, options=options)
        module = import_upstream("aider_repomap", "aider.repomap")
        self.repo_map_class = module.RepoMap
        self._instances = {}

    def _get_instance(self, tools):
        key = str(tools.repo_path)
        instance = self._instances.get(key)
        if instance is None:
            io = AiderIOShim()
            instance = self.repo_map_class(
                map_tokens=int(self.options.get("map_tokens", 2048)),
                root=key,
                main_model=AiderModelShim(
                    self,
                    name=getattr(self.model, "model_name", "evaluated-model"),
                ),
                io=io,
                verbose=False,
                max_context_window=int(
                    self.options.get("max_context_window", 32768)
                ),
                map_mul_no_files=int(self.options.get("map_mul_no_files", 1)),
                refresh="always",
            )
            self._instances[key] = instance
        return instance

    def repository_context(self, issue, tools, history, state):
        relative_files = tools.list_files()
        files = [str((tools.repo_path / name).resolve()) for name in relative_files]
        mentioned_fnames = set(_FILE.findall(issue))
        mentioned_idents = set(_IDENTIFIER.findall(issue))
        for item in history:
            action = item.get("action") or {}
            path = action.get("path")
            if path:
                mentioned_fnames.add(str(path))
        repo_map = self._get_instance(tools).get_repo_map(
            chat_files=[],
            other_files=files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            force_refresh=bool(state.get("files_written")),
        )
        if not repo_map:
            raise RuntimeError("Frozen Aider RepoMap returned no repository map")

        normalized_map = repo_map.replace("\\", "/")
        exposed = [
            path
            for path in relative_files
            if path in normalized_map
            or path.rsplit("/", 1)[-1] in normalized_map
        ]
        state["context_files"] = exposed
        state["context_spans"] = {}
        return repo_map
