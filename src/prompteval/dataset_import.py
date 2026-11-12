"""Build dataset.jsonl from real inputs: a CSV or JSONL export of logs (#21).

The most representative examples are the ones your feature already receives.
This turns an export (support inbox, analytics table, request log) into
prompteval's format:

- `input_column` becomes each example's `input`; `--map col=expected.key` and
  `--map col=vars.name` carry other columns across.
- Exact duplicates (same input after trimming) are dropped, then a seeded
  random sample is taken, so the same file and seed always give the same set.
- Every row is marked `"source": "real"`, so reports can tell real examples
  from AI-drafted ones.
- `redact` masks e-mail addresses and phone numbers *before* anything is
  written. Nothing is sent anywhere during import.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 7+ digits with common separators, optionally with a leading +country code.
PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)")
REDACTORS = {"emails": (EMAIL, "[email]"), "phones": (PHONE, "[phone]")}


@dataclass(frozen=True)
class ImportResult:
    read: int
    duplicates: int
    empty: int
    written: int


def redact(text: str, kinds: tuple[str, ...]) -> str:
    for kind in kinds:
        pattern, mask = REDACTORS[kind]
        text = pattern.sub(mask, text)
    return text


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows from a CSV (header row required) or JSONL file."""
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected an object per line")
            rows.append(obj)
        return rows
    return list(csv.DictReader(text.splitlines()))


def parse_maps(specs: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """`col=expected.key` / `col=vars.name` -> (column, section, key)."""
    out = []
    for spec in specs:
        col, sep, target = spec.partition("=")
        section, dot, key = target.partition(".")
        if not sep or not dot or section not in {"expected", "vars"} or not col or not key:
            raise ValueError(f"Invalid --map {spec!r}: use column=expected.key or column=vars.name")
        out.append((col.strip(), section, key.strip()))
    return out


def build_examples(
    rows: list[dict[str, Any]],
    *,
    input_column: str,
    id_column: str | None = None,
    maps: list[tuple[str, str, str]] | None = None,
    sample: int | None = None,
    seed: int = 1,
    redact_kinds: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], ImportResult]:
    if rows and input_column not in rows[0]:
        raise ValueError(
            f"No column {input_column!r}. Columns: {', '.join(map(str, rows[0].keys()))}"
        )
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates = empty = 0
    for n, row in enumerate(rows, start=1):
        raw = str(row.get(input_column) or "").strip()
        if not raw:
            empty += 1
            continue
        if raw in seen:
            duplicates += 1
            continue
        seen.add(raw)
        example: dict[str, Any] = {
            "id": str(row.get(id_column) or f"row-{n}") if id_column else f"row-{n}",
            "input": redact(raw, redact_kinds),
            "source": "real",
        }
        for col, section, key in maps or []:
            value = row.get(col)
            if value not in (None, ""):
                value = redact(value, redact_kinds) if isinstance(value, str) else value
                example.setdefault(section, {})[key] = value
        kept.append(example)
    if sample is not None and sample < len(kept):
        # Seeded, and kept in file order, so the same file and seed give the same set.
        chosen = sorted(random.Random(seed).sample(range(len(kept)), sample))
        kept = [kept[i] for i in chosen]
    return kept, ImportResult(read=len(rows), duplicates=duplicates, empty=empty, written=len(kept))


def write_dataset(examples: list[dict[str, Any]], out: Path, force: bool = False) -> None:
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8"
    )
