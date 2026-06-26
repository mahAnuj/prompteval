"""prompteval init — bootstrap an evals/ folder in the user's project.

The public surface is `bootstrap()` (importable for testing + scripting) and
the `init` CLI subcommand wired up in `prompteval.cli`. Templates live in
`prompteval.init.templates` and are read at runtime via importlib.resources,
so the package wheel ships them as ordinary data.
"""

from prompteval.init.bootstrap import BootstrapResult, bootstrap

__all__ = ["BootstrapResult", "bootstrap"]
