from scaffolds.base import BaseScaffold


class TestFirstScaffold(BaseScaffold):
    strategy_name = "run tests before editing"
    scaffold_context = (
        "Before editing files, run the tests first to observe the failure. "
        "Then inspect the relevant source file, fix the bug, and run tests again."
    )


Scaffold = TestFirstScaffold
