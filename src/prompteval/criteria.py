"""Your own judging criteria, one file per task, in a `criteria/` folder (#24).

A criteria file says, in your words, what makes one answer better than another
for one job: `criteria/prompt-create-kundli.md` for the kundli prompt,
`criteria/support-bot.md` for the support bot. Every judge reads it:

- `compare --pairwise` / `sweep --pairwise` judge the two answers against it,
  in place of the built-in checklist;
- `llm_judge(rubric=criteria_rubric("prompt-create-kundli"), text=...)` scores
  one reply against it, so a per-reply scorer and the pairwise judge apply the
  same rules.

Which file a comparison uses, first match wins: `--criteria NAME`, then the
`criteria=` the eval set (saved on the run), then `criteria/<eval name>.md`,
else the built-in checklist. The report always says which.

Paths are relative to where you run prompteval, like `.prompteval/runs`.
"""

from __future__ import annotations

from pathlib import Path

CRITERIA_DIR = Path("criteria")
DRAFT_MARKER = "# PROMPTEVAL-DRAFT"
_SUFFIXES = ("", ".md", ".txt")

TEMPLATE = """# Criteria: {name}

<!--
How a judge should decide which of two answers is better for this prompt.
Write one rule per line, most important first: when two answers differ on an
early rule, that decides it. Be concrete enough that two people would agree
on each line. prompteval adds the ground rules itself (use only what you are
given, length is not quality), so you don't need to repeat them.
This comment is shown to the judge too; delete it once you've written yours.
-->

1. <the rule that matters most, e.g. "Uses only facts given in the input; invents nothing">
2. <the next one>
3. <...>
"""


def resolve_criteria(ref: str, base: Path | None = None) -> Path:
    """The criteria file `ref` names: a path, or a name inside `criteria/`."""
    root = base or Path.cwd()
    direct = root / ref
    if direct.is_file():
        return direct
    folder = root / CRITERIA_DIR
    for suffix in _SUFFIXES:
        candidate = folder / f"{ref}{suffix}"
        if candidate.is_file():
            return candidate
    known = sorted(p.name for p in folder.glob("*") if p.is_file()) if folder.is_dir() else []
    hint = f" Files in {CRITERIA_DIR}/: {', '.join(known)}." if known else ""
    raise FileNotFoundError(
        f"No criteria file {ref!r} (looked for {ref} and {CRITERIA_DIR}/{ref}[.md|.txt]).{hint} "
        f"Create one with `prompteval criteria new {ref}`."
    )


def read_criteria(path: Path) -> tuple[str, bool]:
    """(the text a judge sees, whether it is still an unreviewed draft)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    draft = any(line.startswith(DRAFT_MARKER) for line in lines)
    text = "\n".join(line for line in lines if not line.startswith(DRAFT_MARKER)).strip()
    if not text:
        raise ValueError(f"{path} is empty: write the criteria a judge should apply")
    return text, draft


def load_criteria(ref: str, base: Path | None = None) -> str:
    """The text of criteria file `ref` (a name in `criteria/`, or a path)."""
    return read_criteria(resolve_criteria(ref, base))[0]


def criteria_rubric(ref: str, base: Path | None = None) -> str:
    """A rubric for `llm_judge` that scores ONE reply against criteria file `ref`."""
    return (
        "Score the reply against these criteria, which are in priority order:\n\n"
        f"{load_criteria(ref, base)}\n\n"
        "1 means it meets every criterion; 0 means it fails the first. Failing an "
        "earlier criterion costs more than failing a later one. Reply with only a "
        "number between 0 and 1."
    )


def find_for_run(
    explicit: str | None, saved: str | None, eval_name: str, base: Path | None = None
) -> Path | None:
    """The criteria file a comparison should use, or None for the built-in checklist.

    An explicit or saved reference that cannot be found is an error (a typo
    must not silently fall back); the `criteria/<eval name>.md` convention is
    only used when it exists.
    """
    if explicit:
        return resolve_criteria(explicit, base)
    if saved:
        return resolve_criteria(saved, base)
    folder = (base or Path.cwd()) / CRITERIA_DIR
    for suffix in (".md", ".txt"):
        candidate = folder / f"{eval_name}{suffix}"
        if eval_name and candidate.is_file():
            return candidate
    return None


def display(path: Path, base: Path | None = None) -> str:
    """The path as the user would type it, relative to where prompteval runs."""
    try:
        return str(path.resolve().relative_to((base or Path.cwd()).resolve()))
    except ValueError:
        return str(path)
