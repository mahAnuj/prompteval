"""Run models through the local Claude Code CLI: no API key needed (`--base-url claude-code`).

For people who have Claude Code installed and signed in, on a subscription or
an API account. prompteval calls `claude -p` on *their* machine, with *their*
login: nothing is shared, and calls count against their own plan.

How each call is made, and why:

- Your prompt is the system prompt (`--system-prompt`), replacing Claude Code's
  own ~36k-token one, and tools are off (`--tools ""`): measured, that is the
  difference between $0.0865 and $0.00027 for the same two-word reply.
- It runs in a fresh, empty temporary folder. Claude Code reads CLAUDE.md
  from the folder it starts in (and its parents), and measured, a CLAUDE.md
  saying "always answer BANANA" overrode the eval prompt. A user-level
  ~/.claude/CLAUDE.md still applies, so keep eval machines free of one.
- The prompt goes in on stdin, not the command line.
- Cost is the figure Claude Code reports per call (API list prices, including
  cache writes, which token-based estimates miss). On a subscription it is
  usage against the plan, not money spent.
- Claude Code has no temperature setting, so answers sample at its default;
  `--repeats` helps.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any

CLAUDE_CODE = "claude-code"
DEFAULT_SYSTEM = "You are a helpful assistant."
JSON_INSTRUCTION = "Respond with a single JSON object and nothing else: no prose, no code fences."
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class ClaudeCodeError(RuntimeError):
    """The CLI is missing, failed, or returned an error."""


class ClaudeCodeClient:
    """Duck-types the part of the OpenAI client prompteval uses:
    `client.chat.completions.create(model=..., messages=[...])`."""

    def __init__(self, executable: str = "claude", timeout_s: float = 300.0) -> None:
        self.executable = executable
        self.timeout_s = timeout_s
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        **_ignored: Any,  # temperature, max_tokens, seed: the CLI has no such settings
    ) -> Any:
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in messages if m.get("role") != "system")
        wants_json = bool(response_format and response_format.get("type") == "json_object")
        if wants_json:
            system = f"{system}\n\n{JSON_INSTRUCTION}" if system else JSON_INSTRUCTION
        exe = shutil.which(self.executable)
        if exe is None:
            raise ClaudeCodeError(
                "Claude Code ('claude') is not installed or not on PATH. Install it and sign in, "
                "or use an API key with another --base-url."
            )
        cmd = [
            exe, "-p",
            "--system-prompt", system or DEFAULT_SYSTEM,
            "--model", model,
            "--tools", "",
            "--no-session-persistence",
            "--output-format", "json",
        ]  # fmt: skip
        with tempfile.TemporaryDirectory(prefix="prompteval-cc-") as cwd:
            try:
                proc = subprocess.run(
                    cmd, input=user, cwd=cwd, capture_output=True, text=True, timeout=self.timeout_s
                )
            except subprocess.TimeoutExpired as err:
                raise ClaudeCodeError(
                    f"Claude Code did not answer within {self.timeout_s:.0f}s"
                ) from err
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as err:
            tail = (proc.stderr or proc.stdout or "").strip()[-300:]
            raise ClaudeCodeError(f"Claude Code failed (exit {proc.returncode}): {tail}") from err
        if data.get("is_error") or proc.returncode != 0:
            raise ClaudeCodeError(f"Claude Code returned an error: {str(data.get('result'))[:300]}")
        text = str(data.get("result") or "")
        if wants_json:
            m = _FENCE.match(text)
            text = m.group(1) if m else text
        usage = data.get("usage") or {}
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        prompt_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + cache_read
        )
        cost = data.get("total_cost_usd")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=int(usage.get("output_tokens") or 0),
                prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read),
                completion_tokens_details=None,
            ),
            reported_cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
        )
