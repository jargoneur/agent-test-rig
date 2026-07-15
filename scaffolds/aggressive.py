from scaffolds.base import BaseScaffold


class AggressiveScaffold(BaseScaffold):
    strategy_name = "inspect likely source file quickly, edit early"
    scaffold_context = (
        "Act efficiently. Inspect the most likely source file early. "
        "Avoid unnecessary test runs before reading code."
    )


Scaffold = AggressiveScaffold
