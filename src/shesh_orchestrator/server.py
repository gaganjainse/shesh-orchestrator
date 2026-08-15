"""MCP server exposing the orchestrator.

In production, handlers are LLM-backed agents via Ollama (selected by
shesh-mind routing). If Ollama is unreachable, deterministic stubs keep
the server usable for tests and offline operation. The LLM client and model
router are injected/lazily constructed so tests can replace them.

State model: the server is a long-lived, *single-user* process. `_bus`,
`_agents`, `_llm` and `_sessions` are intentionally shared across calls (the
bus is the A2A medium, sessions are meant to be re-attachable). Call
``reset_state()`` for a clean slate — e.g. between test suites or when a
client wants fresh token accounting and an empty bus.
"""
from __future__ import annotations

import os

from shesh_audit.mcp_guard import GuardedMCP as _MCP

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


class NoUsableModelError(RuntimeError):
    """Ollama answered but serves no models; callers fall back to stub agents."""

    def __init__(self) -> None:
        super().__init__("no models available")


def _require_models(client) -> None:
    """Raise NoUsableModelError when the server has no models (kept out of
    the caller's try body so the try only wraps calls, not control flow)."""
    if not client.list_models():
        raise NoUsableModelError()


def _get_llm() -> LLMAgents:
    """Construct LLM agents on demand, or fall back to stubs if Ollama is down."""
    global _llm
    if _llm is not None:
        return _llm

    try:
        from shesh_mind.client import OllamaClient, http_transport
        from shesh_mind.router import ModelRouter

        client = OllamaClient(http_transport(_ollama_url()))
        _require_models(client)
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


def _build_agents(llm) -> dict[str, Agent]:
    """Build a fresh agent set from the given LLM bundle.

    Fresh per call so handler identity and token counters never leak across
    execute() invocations.
    """
    out: dict[str, Agent] = {}
    for name in ("planner", "coder", "researcher", "critic", "coordinator", "vision"):
        handler = llm.agent(name) if hasattr(llm, "agent") else echo_agent
        out[name] = make_agent(name, handler)
    return out


def _ensure_agents() -> dict[str, Agent]:
    """Lazily build (and cache) the session agent set."""
    global _agents
    if not _agents:
        _agents = _build_agents(_get_llm())
    return _agents


@mcp.tool()
def execute(goal: str, max_turns: int = 12, max_tokens: int = 20_000,
            use_llm: bool = True) -> dict:
    """Run a goal through the multi-agent orchestrator.

    ``use_llm=False`` forces deterministic stubs for the planner, critic AND
    agents, so the choice is consistent end to end (offline/testing).
    """
    llm = _get_llm() if use_llm else _StubAgents()
    agents = _build_agents(llm)
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


@mcp.tool()
def reset_state() -> dict:
    """Drop all process-local state: agents, LLM bundle, message bus, sessions.

    The server is a long-lived process; this gives a client (or a test suite)
    a clean slate — empty bus, fresh agents, and no residual sessions. Token
    accounting and the message bus otherwise accumulate across calls by design.
    """
    global _agents, _llm, _bus, _sessions
    _agents = {}
    _llm = None
    _bus = MessageBus()
    _sessions = None
    return {"ok": True}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
