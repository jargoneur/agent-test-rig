from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(record: Dict[str, Any], source: Path) -> Dict[str, Any]:
    job = record["job"]
    result = record["result"]
    model = job["model"]
    return {
        "run_id": record["run_id"],
        "experiment_id": job.get("experiment_id"),
        "wave": job.get("wave"),
        "task": job["task_id"],
        "model_id": model.get("id"),
        "model_name": model.get("name"),
        "backend": model.get("backend"),
        "quantization": model.get("quantization"),
        "scaffold": job["scaffold"],
        "repeat": job["repeat"],
        "run_seed": job["run_seed"],
        "final_tests_passed": result.get("final_tests_passed"),
        "termination_reason": result.get("termination_reason"),
        "steps": result.get("steps"),
        "parse_errors": result.get("parse_errors"),
        "execution_errors": result.get("execution_errors"),
        "target_file_read": result.get("target_file_read"),
        "target_file_written": result.get("target_file_written"),
        "worker_id": (record.get("worker") or {}).get("worker_id"),
        "source_file": str(source),
        "source_sha256": sha256_file(source),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output", default="merged_results.csv")
    args = parser.parse_args()

    by_id: Dict[str, Dict[str, Any]] = {}
    for root_value in args.roots:
        root = Path(root_value)
        candidates = root.rglob("run_*.json") if root.is_dir() else [root]
        for path in candidates:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if record.get("status") != "completed" or not record.get("run_id"):
                continue
            row = flatten(record, path)
            previous = by_id.get(row["run_id"])
            if previous and previous["source_sha256"] != row["source_sha256"]:
                raise RuntimeError(
                    "Conflicting completed results for %s" % row["run_id"]
                )
            by_id[row["run_id"]] = row

    rows: List[Dict[str, Any]] = sorted(
        by_id.values(),
        key=lambda row: (
            str(row["experiment_id"]),
            int(row["wave"] or 0),
            str(row["model_id"]),
            str(row["task"]),
            int(row["repeat"]),
            str(row["scaffold"]),
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed result files found")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checksum = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        "%s  %s\n" % (checksum, output.name),
        encoding="utf-8",
    )
    print("Merged runs:", len(rows))
    print("Output:", output)
    print("SHA-256:", checksum)


if __name__ == "__main__":
    main()
