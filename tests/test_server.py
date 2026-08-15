"""Smoke tests for the orchestrator MCP server (offline — no Ollama needed).

These verify the two things a public MCP surface needs: every tool is
*registered*, and each tool body is *callable* and returns its documented
shape. The guard wrapper (GuardedMCP.tool) runs on every direct call, so
invoking the module-level functions also exercises the policy seam.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_orchestrator import server  # noqa: E402

EXPECTED_TOOLS = {
    "execute",
    "list_roles",
    "post_message",
    "llm_status",
    "start_session",
    "get_session",
    "list_sessions",
    "cancel_session",
    "reset_state",
}


def _registered() -> set[str]:
    return {t.name for t in asyncio.run(server.mcp.list_tools())}


def test_all_tools_registered():
    missing = EXPECTED_TOOLS - _registered()
    assert not missing, f"unregistered tools: {sorted(missing)}"


def test_execute_offline_stubs_returns_schema():
    res = server.execute("summarize the plan", use_llm=False)
    assert set(res) == {"ok", "used_llm", "steps", "stopped_reason"}
    assert res["used_llm"] is False
    assert res["ok"] is True          # stub planner/critic approve
    assert len(res["steps"]) >= 1
    for s in res["steps"]:
        assert set(s) == {"role", "status", "result"}
        assert s["status"] == "done"


def test_execute_budget_caps_turns():
    res = server.execute("loop", use_llm=False, max_turns=1)
    assert res["ok"] is False
    assert "budget" in res["stopped_reason"]
    assert sum(s["status"] == "done" for s in res["steps"]) <= 1


def test_list_roles_returns_schema():
    roles = server.list_roles()
    assert isinstance(roles, list) and roles
    for r in roles:
        assert set(r) == {"name", "model", "tools", "description"}


def test_post_message_and_llm_status():
    assert server.post_message("coordinator", "hi") == {"ok": True, "recipient": "coordinator"}
    st = server.llm_status()
    assert set(st) >= {"llm_available"}
    # Offline sandbox: no Ollama -> stubs. Either value is acceptable here.
    assert isinstance(st["llm_available"], bool)


def test_session_lifecycle_and_reset():
    server.reset_state()
    started = server.start_session("do a thing")
    sid = started["id"]
    assert started["status"] in {"running", "done"}
    listed = server.list_sessions()
    assert any(s["id"] == sid for s in listed)
    got = server.get_session(sid)
    assert got["id"] == sid
    server.cancel_session(sid)
    # reset clears everything
    assert server.reset_state() == {"ok": True}
    assert server.list_sessions() == []


def test_reset_state_clears_bus_and_agents():
    server.reset_state()
    server.post_message("coordinator", "leftover")
    server.execute("x", use_llm=False)
    server.reset_state()
    # after reset, no sessions, no cached agents, and a fresh (empty) bus
    assert server._sessions is None
    assert server._agents == {}
    assert len(server._bus._queues) == 0
