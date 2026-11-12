"""#21: build dataset.jsonl from a CSV or JSONL export of real inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from prompteval.cli import main
from prompteval.dataset_import import build_examples, parse_maps, read_rows, redact
from prompteval.eval.runner import _load_dataset

CSV = (
    "ticket,message,team,expected_answer\n"
    "T1,My order arrived broken,returns,refund\n"
    "T2,Where is my parcel? Call me on +44 20 7946 0958,shipping,tracking\n"
    "T3,My order arrived broken,returns,refund\n"
    "T4,,returns,\n"
    "T5,Email me at jane.doe@example.com please,billing,invoice\n"
)


def test_redaction_masks_emails_and_phones() -> None:
    text = "mail a.b@example.co.uk or ring +1 (415) 555-0100, order 12345"
    assert redact(text, ("emails", "phones")) == "mail [email] or ring [phone], order 12345"


def test_build_dedupes_skips_empty_maps_columns_and_marks_real(tmp_path: Path) -> None:
    f = tmp_path / "logs.csv"
    f.write_text(CSV)
    examples, stats = build_examples(
        read_rows(f),
        input_column="message",
        id_column="ticket",
        maps=parse_maps(("team=vars.team", "expected_answer=expected.reference")),
        redact_kinds=("emails", "phones"),
    )
    assert (stats.read, stats.duplicates, stats.empty, stats.written) == (5, 1, 1, 3)
    assert [e["id"] for e in examples] == ["T1", "T2", "T5"]
    assert examples[0] == {
        "id": "T1",
        "input": "My order arrived broken",
        "source": "real",
        "vars": {"team": "returns"},
        "expected": {"reference": "refund"},
    }
    assert "[phone]" in examples[1]["input"] and "[email]" in examples[2]["input"]


def test_sampling_is_reproducible_and_keeps_file_order() -> None:
    rows = [{"m": f"message {i}"} for i in range(100)]
    a, _ = build_examples(rows, input_column="m", sample=10, seed=7)
    b, _ = build_examples(rows, input_column="m", sample=10, seed=7)
    c, _ = build_examples(rows, input_column="m", sample=10, seed=8)
    assert a == b and a != c and len(a) == 10
    assert [e["id"] for e in a] == sorted((e["id"] for e in a), key=lambda s: int(s.split("-")[1]))


def test_jsonl_input_and_bad_inputs(tmp_path: Path) -> None:
    f = tmp_path / "logs.jsonl"
    f.write_text('{"q": "hello"}\n\n{"q": "bye"}\n')
    examples, _ = build_examples(read_rows(f), input_column="q")
    assert [e["input"] for e in examples] == ["hello", "bye"]
    with pytest.raises(ValueError, match="No column 'nope'"):
        build_examples(read_rows(f), input_column="nope")
    with pytest.raises(ValueError, match="Invalid --map"):
        parse_maps(("team=other.x",))


def test_cli_import_writes_a_loadable_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs.csv").write_text(CSV)
    args = [
        "dataset",
        "import",
        "logs.csv",
        "--input-column",
        "message",
        "--redact",
        "emails,phones",
    ]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "wrote 3 real examples" in result.output
    loaded = _load_dataset(tmp_path / "evals" / "dataset.jsonl")
    assert [e.source for e in loaded] == ["real", "real", "real"]
    again = CliRunner().invoke(main, args)
    assert again.exit_code != 0 and "already exists" in again.output  # never overwrites
    assert CliRunner().invoke(main, [*args, "--force"]).exit_code == 0
    bad = CliRunner().invoke(main, [*args[:5], "--redact", "names"])
    assert bad.exit_code != 0 and "Unknown --redact" in bad.output


def test_synthetic_examples_are_flagged_in_compare() -> None:
    from dataclasses import replace

    from prompteval.compare import compute_comparison, render_text
    from tests.test_compare_decisions import _run

    a = _run("v1", {"q": [1.0] * 4}, [1] * 4)
    b = _run("v2", {"q": [1.0] * 4}, [1] * 4)
    b = replace(
        b, examples=[replace(e, source="synthetic") for e in b.examples[:3]] + b.examples[3:]
    )
    assert "3 of 4 examples are synthetic" in render_text(compute_comparison(a, b))
    json.dumps(b.examples[0].source)
