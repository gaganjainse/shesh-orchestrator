"""Offline tests for LLM-backed agents (with a fake model client)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesha_orchestrator.llm import LLMAgents, _extract_json  # noqa: E402


def fake_client(response: str):
    def _c(model: str, prompt: str) -> str:
        return response
    return _c


def test_extract_json_from_fence():
    text = 'Here:\n```json\n{"steps": []}\n```'
    assert _extract_json(text) == {"steps": []}


def test_extract_json_from_plain():
    assert _extract_json(_json := 'noise {"a": 1} trailing') == {"a": 1}


def test_extract_json_none():
    assert _extract_json("no json here") is None


def test_planner_parses_valid_steps():
    agents = LLMAgents(
        client=fake_client('{"steps": [{"role": "coder", "instruction": "do x"}]}'),
        model_for_role=lambda r: "m",
    )
    out = agents.planner("goal", {})
    assert out["steps"][0]["instruction"] == "do x"
    assert out["steps"][0]["role"] == "coder"


def test_planner_falls_back_on_bad_json():
    agents = LLMAgents(client=fake_client("not json at all"))
    out = agents.planner("goal", {})
    # fallback has 3 steps
    assert len(out["steps"]) == 3
    assert out["steps"][0]["role"] == "researcher"


def test_agent_returns_output():
    agents = LLMAgents(client=fake_client("done"), model_for_role=lambda r: "m")
    result = agents.agent("coder")("fix bug", {})
    assert result["ok"] is True
    assert result["output"] == "done"
    assert result["role"] == "coder"


def test_agent_handles_client_error():
    def boom(model, prompt):
        raise OSError("offline")
    agents = LLMAgents(client=boom)
    result = agents.agent("coder")("x", {})
    assert result["ok"] is False
    assert "offline" in result["error"]


def test_critic_parses_verdict():
    agents = LLMAgents(
        client=fake_client('{"approved": false, "notes": "needs tests"}'),
        model_for_role=lambda r: "m",
    )
    out = agents.critic("goal", {"steps": [], "prior": []})
    assert out["approved"] is False
    assert out["notes"] == "needs tests"


def test_critic_falls_back_to_approve():
    agents = LLMAgents(client=fake_client("garbage"))
    out = agents.critic("goal", {})
    assert out["approved"] is True
