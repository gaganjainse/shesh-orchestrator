"""MCP server exposing the orchestrator.

In production, handlers are LLM-backed agents via Ollama (selected by
shesh-mind routing). If Ollama is unreachable, deterministic stubs keep
the server usable for tests and offline operation. The LLM client and model
router are injected/lazily constructed so tests can replace them.
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

try:
    from shesh_audit.mcp_guard import GuardedMCP as _MCP
except ImportError:  # audit not installed; fall back to plain FastMCP
    _MCP = FastMCP

from .agents import Agent, Budget
from .bus import Message, MessageBus
from .llm import LLMAgents
from .orchestrator import Orchestrator, make_agent
from .sessions import SessionManager
from .stubs import always_approve, default_planner, echo_agent

mcp = _MCP("shesh-orchestrator")

_bus = MessageBus()
_agents: dict[str, Agent] = {}

# Lazy LLM bundle; None until first use (so tests stay fast/offline).
_llm: LLMAgents | None = None


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _get_llm() -> LLMAgents:
    """Construct LLM agents on demand, or fall back to stubs if Ollama is down."""
    global _llm
    if _llm is not None:
        return _llm

    try:
        from shesh_mind.client import OllamaClient, http_transport
        from shesh_mind.router import ModelRouter

        client = OllamaClient(http_transport(_ollama_url()))
        if not client.list_models():
            raise RuntimeError("no models available")
        router = ModelRouter()

        def model_for(role: str) -> str:
            from shesh_mind.router import Role
            return router.select(Role(role)).model

        _llm = LLMAgents(client=client.generate, model_for_role=model_for)
    except Exception:  # noqa: BLE001 - offline/import failure -> stubs
        _llm = _StubAgents()
    return _llm


class _StubAgents:
    """Stand-in that uses deterministic stubs when no LLM is available."""

    def planner(self, goal: str, ctx: dict) -> dict:
        return default_planner(goal, ctx)

    def agent(self, role: str):
        return echo_agent

    def critic(self, goal: str, ctx: dict) -> dict:
        return always_approve(goal, ctx)


def _ensure_agents() -> dict[str, Agent]:
    if not _agents:
        llm = _get_llm()
        for name in ("planner", "coder", "researcher", "critic", "coordinator", "vision"):
            handler = llm.agent(name) if hasattr(llm, "agent") else echo_agent
            _agents[name] = make_agent(name, handler)
    return _agents


@mcp.tool()
def execute(goal: str, max_turns: int = 12, max_tokens: int = 20_000,
            use_llm: bool = True) -> dict:
    """Run a goal through the multi-agent orchestrator."""
    llm = _get_llm() if use_llm else _StubAgents()
    agents = _ensure_agents()
    orch = Orchestrator(
        agents, bus=_bus, budget=Budget(max_turns=max_turns, max_tokens=max_tokens),
    )
    result = orch.execute(goal, planner=llm.planner, critic=llm.critic)
    return {
        "ok": result.ok,
        "used_llm": not isinstance(llm, _StubAgents),
        "steps": [{"role": s.role, "status": s.status, "result": s.result}
                  for s in result.steps],
        "stopped_reason": result.stopped_reason,
    }


@mcp.tool()
def list_roles() -> list[dict]:
    from .roles import ROLES
    return [{"name": r.name, "model": r.model, "tools": list(r.tools),
             "description": r.description} for r in ROLES.values()]


@mcp.tool()
def post_message(recipient: str, content: str) -> dict:
    """Send an A2A message to a role (e.g., for coordinator direction)."""
    _bus.post(Message(sender="user", recipient=recipient, content=content))
    return {"ok": True, "recipient": recipient}


@mcp.tool()
def llm_status() -> dict:
    """Report whether LLM-backed agents are live or falling back to stubs."""
    try:
        llm = _get_llm()
        return {"llm_available": not isinstance(llm, _StubAgents),
                "ollama_url": _ollama_url()}
    except Exception as e:  # noqa: BLE001
        return {"llm_available": False, "error": str(e)}



_sessions: SessionManager | None = None


def _get_sessions() -> SessionManager:
    global _sessions
    if _sessions is None:
        _sessions = SessionManager(_ensure_agents())
    return _sessions


@mcp.tool()
def start_session(goal: str) -> dict:
    """Start a background agent session for a goal; returns its id for polling."""
    state = _get_sessions().start(goal)
    return state.to_dict()


@mcp.tool()
def get_session(session_id: str) -> dict:
    """Return status/trace/result of a background session."""
    state = _get_sessions().get(session_id)
    return state.to_dict() if state else {"error": "not found"}


@mcp.tool()
def list_sessions() -> list[dict]:
    """List all sessions."""
    return [s.to_dict() for s in _get_sessions().list()]


@mcp.tool()
def cancel_session(session_id: str) -> dict:
    """Cancel a running session."""
    ok = _get_sessions().cancel(session_id)
    return {"ok": ok}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
