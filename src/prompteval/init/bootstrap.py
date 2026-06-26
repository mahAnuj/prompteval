"""Copy the init-template tree into a target folder.

Templates live in `prompteval.init.templates` as resource files. We read them
via `importlib.resources` so the same code works whether installed from a
wheel, run from a source checkout, or zipped — no `__file__` path hacks.

Only one filename rewrite: source `env.example` becomes target `.env.example`.
Dotfiles can be excluded by some packaging tools; storing the source un-dotted
keeps the package contents predictable, and the rename is a one-liner.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a `bootstrap()` call. Returned for testing + CLI reporting."""

    target: Path
    files_written: int


def bootstrap(target: Path, force: bool = False) -> BootstrapResult:
    """Copy the init templates into `target`.

    Refuses to write into an existing non-empty target unless `force=True`.
    Creates the target (and any intermediate parents) when it doesn't exist.

    Source-side `env.example` is written as target-side `.env.example`.
    All other files are copied with their source name.
    """
    if target.exists() and any(target.iterdir()) and not force:
        raise FileExistsError(
            f"{target} already exists and is not empty. "
            f"Pass --force to overwrite, or --dir to pick a different name."
        )

    target.mkdir(parents=True, exist_ok=True)

    files_written = 0
    for rel_path, content in _iter_templates():
        out_rel = ".env.example" if rel_path == "env.example" else rel_path
        out_path = target / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        files_written += 1

    return BootstrapResult(target=target, files_written=files_written)


def _iter_templates() -> list[tuple[str, str]]:
    """Walk the templates package, yielding (relative_path, contents).

    Skips the `__init__.py` marker file. Recurses into subdirs (e.g. `prompts/`).
    Returns a list (not a generator) so callers can count without exhausting.
    """
    root = files("prompteval.init.templates")
    out: list[tuple[str, str]] = []
    _walk(root, "", out)
    return out


def _walk(node: Traversable, prefix: str, out: list[tuple[str, str]]) -> None:
    for child in node.iterdir():
        name = child.name
        # Skip Python package machinery — the `__init__.py` marker, the
        # compiled `__pycache__/` (which `importlib.resources` exposes
        # alongside source), and any stray .pyc.
        if name == "__init__.py" or name == "__pycache__" or name.endswith(".pyc"):
            continue
        rel = f"{prefix}{name}"
        if child.is_file():
            out.append((rel, child.read_text()))
        elif child.is_dir():
            _walk(child, f"{rel}/", out)
