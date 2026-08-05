from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _append_span(spans: Dict[str, List[Dict[str, int]]], path: str, start: int, end: int):
    value = {"start": int(start), "end": int(end)}
    existing = spans.setdefault(path, [])
    if value not in existing:
        existing.append(value)


def _merge_spans(values: List[Dict[str, int]]) -> List[Dict[str, int]]:
    intervals = sorted((int(item["start"]), int(item["end"])) for item in values)
    merged: List[Tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            previous = merged[-1]
            merged[-1] = (previous[0], max(previous[1], end))
    return [{"start": start, "end": end} for start, end in merged]


def build_contextbench_trajectory(
    task: Dict[str, Any],
    result: Dict[str, Any],
    model_patch: str,
    job: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    steps = []
    final_files: List[str] = []
    final_spans: Dict[str, List[Dict[str, int]]] = {}

    for item in result.get("history", []):
        observation = item.get("observation") or {}
        step_files: List[str] = []
        step_spans: Dict[str, List[Dict[str, int]]] = {}
        observation_type = observation.get("type")

        if observation_type == "file_content":
            path = str(observation["path"])
            step_files.append(path)
            _append_span(
                step_spans,
                path,
                int(observation.get("start_line") or 1),
                int(observation.get("end_line") or observation.get("total_lines") or 1),
            )
        elif observation_type == "search_result":
            for match in observation.get("results") or []:
                path = str(match.get("path") or "")
                line = int(match.get("line") or 1)
                if not path:
                    continue
                if path not in step_files:
                    step_files.append(path)
                _append_span(step_spans, path, line, line)

        if step_files or step_spans:
            steps.append(
                {
                    "files": step_files,
                    "spans": step_spans,
                    "symbols": {},
                }
            )
            for path in step_files:
                if path not in final_files:
                    final_files.append(path)
            for path, spans in step_spans.items():
                for span in spans:
                    _append_span(
                        final_spans,
                        path,
                        span["start"],
                        span["end"],
                    )

    final_spans = {
        path: _merge_spans(spans)
        for path, spans in sorted(final_spans.items())
    }
    job = dict(job or {})
    model = dict(job.get("model") or {})
    return {
        "instance_id": task["instance_id"],
        "original_inst_id": task.get("original_inst_id", ""),
        "traj_data": {
            "pred_steps": steps,
            "pred_files": final_files,
            "pred_spans": final_spans,
            "pred_symbols": {},
        },
        "model_patch": model_patch,
        "agent_rig": {
            "task_id": task.get("id"),
            "bench": task.get("bench"),
            "repo": task.get("repo"),
            "base_commit": task.get("base_commit"),
            "run_id": job.get("run_id"),
            "experiment_id": job.get("experiment_id"),
            "model_id": model.get("id"),
            "scaffold": job.get("scaffold"),
            "repeat": job.get("repeat"),
            "run_seed": job.get("run_seed"),
        },
    }
