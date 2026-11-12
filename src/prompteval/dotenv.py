"""Load API keys from a `.env` file, so `prompteval init`'s instructions work.

`init` writes `.env.example` and tells the user to copy it to `.env` and add
their key. Nothing used to read that file, so `run` failed with an
authentication error unless the key was also exported in the shell.

Read at CLI start: `.env` in the current directory, then `evals/.env` (where
`init` puts the example). A variable already set in the environment always
wins, so a key exported in the shell or set as a CI secret is never replaced.
Deliberately tiny (KEY=value lines, optional quotes, `#` comments, an optional
`export `), to avoid a dependency for one job.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_LOCATIONS = (Path(".env"), Path("evals") / ".env")


def parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()  # trailing comment
        out[key] = value
    return out


def load_env_files(paths: tuple[Path, ...] = DEFAULT_LOCATIONS) -> list[str]:
    """Set variables from the first files found, never overriding the environment.

    Returns the names that were set (values are never logged).
    """
    loaded: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for key, value in parse_env(text).items():
            if key not in os.environ and value and not value.endswith("..."):
                os.environ[key] = value
                loaded.append(key)
    return loaded
