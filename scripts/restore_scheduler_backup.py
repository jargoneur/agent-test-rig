#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from harness.distributed_scheduler_store import DistributedSchedulerStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore a validated scheduler SQLite backup while the scheduler is stopped."
    )
    parser.add_argument("--backup", required=True)
    parser.add_argument("--database", default="scheduler/contextbench.sqlite3")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Allow replacement of an existing scheduler database.",
    )
    args = parser.parse_args()

    database = Path(args.database).expanduser().resolve()
    if database.exists() and not args.replace_existing:
        raise SystemExit(
            "Destination database already exists. Stop the scheduler and pass "
            "--replace-existing after preserving the current file."
        )

    result = DistributedSchedulerStore.restore_database(
        args.backup,
        str(database),
    )
    store = DistributedSchedulerStore(str(database))
    result["integrity"] = store.integrity_check(thorough=True)
    result["status"] = store.status()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
