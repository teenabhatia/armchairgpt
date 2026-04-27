import json
from pathlib import Path

import pytest

from eval.run_eval import main as eval_main

pytestmark = pytest.mark.agent_eval


def test_offline_eval_json_suite_passes(capsys):
    """Full eval/cases.jsonl in offline mode (no network); asserts summary schema."""
    cases_path = Path(__file__).resolve().parents[1] / "eval" / "cases.jsonl"
    rc = eval_main(["--mode", "offline", "--cases", str(cases_path), "--json"])
    assert rc == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert set(data.keys()) >= {"mode", "total", "passed", "failed", "failures"}
    assert data["mode"] == "offline"
    assert data["passed"] == data["total"]
    assert data["failed"] == 0
