from importlib import import_module


def load_scaffold(name: str):
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid scaffold name: {name}")

    module = import_module(f"scaffolds.{name}")

    if not hasattr(module, "Scaffold"):
        raise ValueError(
            f"Scaffold module 'scaffolds.{name}' must define a class or alias called Scaffold"
        )

    return module.Scaffold()