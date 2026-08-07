#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP = ROOT / ".upstreams" / "llama.cpp"
REQUIRED_ARCHITECTURES = (
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
)
REQUIRED_MODULES = (
    "torch",
    "numpy",
    "sentencepiece",
    "transformers",
    "gguf",
    "google.protobuf",
)


def main() -> None:
    # Mirror convert_hf_to_gguf.py: its pinned gguf-py must win over any
    # separately installed gguf package.
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    sys.path.insert(0, str(LLAMA_CPP))

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

    from conversion import get_model_class

    architecture_handlers = {}
    unsupported = []
    for architecture in REQUIRED_ARCHITECTURES:
        try:
            model_class = get_model_class(architecture)
            architecture_handlers[architecture] = model_class.__name__
        except Exception as error:
            unsupported.append("%s: %s" % (architecture, error))
    if unsupported:
        raise RuntimeError(
            "Pinned llama.cpp converter lacks a selected architecture:\n- "
            + "\n- ".join(unsupported)
        )

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "converter_environment": "ok",
                "architectures": architecture_handlers,
                "modules": versions,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
