"""MCP server exposing the orchestrator.

Handlers default to deterministic offline stubs so the server is testable and
safe out of the box; real LLM-backed agents are injected at runtime.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .agents import Agent, Budget
from .bus import Message, MessageBus
from .orchestrator import Orchestrator, make_agent

mcp = FastMCP("sesha-orchestrator")

_bus = MessageBus()


def _stub_planner(goal: str, ctx: dict) -> dict:
    return {"steps": [
        {"role": "researcher", "instruction": f"Gather context for: {goal}"},
        {"role": "coder", "instruction": f"Execute: {goal}"},
        {"role": "critic", "instruction": "Verify the result"},
    ]}


def _echo(prompt: str, ctx: dict) -> dict:
    return {"ok": True, "echo": prompt[:200]}


def _always_approve(goal: str, ctx: dict) -> dict:
    return {"approved": True, "notes": "stub critic approves"}


_agents: dict[str, Agent] = {}


def _ensure_agents() -> dict[str, Agent]:
    if not _agents:
        for name in ("planner", "coder", "researcher", "critic", "coordinator"):
            _agents[name] = make_agent(name, _echo)
    return _agents


@mcp.tool()
def execute(goal: str, max_turns: int = 12, max_tokens: int = 20_000) -> dict:
    """Run a goal through the multi-agent orchestrator (planner→roles→critic)."""
    orch = Orchestrator(
        _ensure_agents(), bus=_bus,
        budget=Budget(max_turns=max_turns, max_tokens=max_tokens),
    )
    result = orch.execute(goal, planner=_stub_planner, critic=_always_approve)
    return {
        "ok": result.ok,
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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
