#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import sys
from typing import Dict


REQUIRED_MODULES = (
    "torch",
    "numpy",
    "sentencepiece",
    "transformers",
    "gguf",
    "google.protobuf",
)


def main() -> None:
    versions: Dict[str, str] = {}
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            missing.append("%s: %s" % (module_name, error))
            continue
        root_name = module_name.split(".", 1)[0]
        root_module = importlib.import_module(root_name)
        versions[module_name] = str(getattr(root_module, "__version__", "unknown"))

    if missing:
        raise RuntimeError(
            "llama.cpp converter environment is incomplete:\n- "
            + "\n- ".join(missing)
            + "\nInstall the pinned converter requirements with:\n"
            + "uv pip install --python .venv/bin/python -r "
            + ".upstreams/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
        )

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "converter_environment": "ok",
                "modules": versions,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
