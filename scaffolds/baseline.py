from scaffolds.base import BaseScaffold


class BaselineScaffold(BaseScaffold):
    strategy_name = "baseline"
    scaffold_context = "No additional scaffold-specific guidance."


Scaffold = BaselineScaffold
