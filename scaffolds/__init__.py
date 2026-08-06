from __future__ import annotations

from importlib import import_module


def load_scaffold(name: str, model=None, task=None, options=None):
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid scaffold name: {name}")

    module = import_module(f"scaffolds.{name}")

    if not hasattr(module, "Scaffold"):
        raise ValueError(
            f"Scaffold module 'scaffolds.{name}' must define a class or alias called Scaffold"
        )

    scaffold = module.Scaffold(model=model, task=task, options=options)
    if hasattr(scaffold, "bind"):
        scaffold.bind(model=model, task=task, options=options)
    return scaffold
