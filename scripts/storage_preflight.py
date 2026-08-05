from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]


def directory_size(path: Path) -> Optional[int]:
    if not path.exists():
        return 0
    try:
        output = subprocess.check_output(
            ["du", "-sx", "--block-size=1", str(path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(output.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def quota_output() -> Optional[str]:
    try:
        result = subprocess.run(
            ["quota", "-s"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return round(value / GIB, 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report and enforce the project storage envelope."
    )
    parser.add_argument("--path", default=str(ROOT))
    parser.add_argument(
        "--soft-limit-gib",
        type=float,
        help="Administrative usage ceiling for the measured home directory.",
    )
    parser.add_argument(
        "--minimum-filesystem-free-gib",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--required-additional-gib",
        type=float,
        default=0.0,
        help="Estimated temporary space needed by the next operation.",
    )
    args = parser.parse_args()

    project = Path(args.path).expanduser().resolve()
    home = Path.home().resolve()
    filesystem = shutil.disk_usage(project)
    project_bytes = directory_size(project)
    home_bytes = directory_size(home)
    required = max(0.0, args.required_additional_gib)
    minimum_free = max(0.0, args.minimum_filesystem_free_gib)

    failures = []
    filesystem_free_gib = filesystem.free / GIB
    if filesystem_free_gib < required + minimum_free:
        failures.append(
            "filesystem free space %.3f GiB is below required %.3f GiB plus %.3f GiB reserve"
            % (filesystem_free_gib, required, minimum_free)
        )

    if args.soft_limit_gib is not None and home_bytes is not None:
        projected_home_gib = home_bytes / GIB + required
        if projected_home_gib > args.soft_limit_gib:
            failures.append(
                "projected home usage %.3f GiB exceeds administrative soft limit %.3f GiB"
                % (projected_home_gib, args.soft_limit_gib)
            )

    report: Dict[str, Any] = {
        "project_path": str(project),
        "home_path": str(home),
        "project_usage_gib": gib(project_bytes),
        "home_usage_gib": gib(home_bytes),
        "filesystem_total_gib": gib(filesystem.total),
        "filesystem_used_gib": gib(filesystem.used),
        "filesystem_free_gib": gib(filesystem.free),
        "required_additional_gib": required,
        "minimum_filesystem_free_gib": minimum_free,
        "soft_limit_gib": args.soft_limit_gib,
        "quota_command_output": quota_output(),
        "ok": not failures,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
