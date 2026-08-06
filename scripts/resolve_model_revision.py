#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a Hugging Face model ref to its exact commit SHA."
    )
    parser.add_argument("model_id")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    info = HfApi(token=args.token).model_info(
        repo_id=args.model_id,
        revision=args.revision,
        token=args.token,
        files_metadata=False,
    )
    result = {
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": str(info.sha),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
