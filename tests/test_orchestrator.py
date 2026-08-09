"""Offline tests for the multi-agent orchestrator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesha_orchestrator.agents import Budget  # noqa: E402
from shesha_orchestrator.bus import Message, MessageBus  # noqa: E402
from shesha_orchestrator.orchestrator import (  # noqa: E402
    Orchestrator,
    make_agent,
)
from shesha_orchestrator.roles import ROLES, role  # noqa: E402


def _agents(**handlers):
    out = {}
    for name in set(["planner", "coder", "researcher", "critic", "coordinator"]) | set(handlers):
        def _h(prompt, ctx, _n=name):
            return handlers.get(_n, lambda p, c: {"ok": True, "by": _n})(prompt, ctx)
        out[name] = make_agent(name, _h)
    return out


def _planner(steps):
    return lambda goal, ctx: {"steps": steps}


def _critic(approved=True, notes=""):
    return lambda goal, ctx: {"approved": approved, "notes": notes}


def test_roles_exist():
    assert {"coordinator", "planner", "coder", "researcher", "vision", "critic"} <= set(ROLES)
    assert role("coder").model == "code"


def test_executes_planned_steps_in_order():
    seen = []

    def coder(p, c):
        seen.append(p)
        return {"did": p}

    agents = _agents(coder=coder)
    orch = Orchestrator(agents)
    plan = [
        {"role": "researcher", "instruction": "look up X"},
        {"role": "coder", "instruction": "implement X"},
    ]
    res = orch.execute("do X", planner=_planner(plan))
    assert res.ok
    assert [s.role for s in res.steps] == ["researcher", "coder"]
    assert "implement X" in seen


def test_budget_stops_runaway():
    agents = _agents()
    orch = Orchestrator(agents, budget=Budget(max_turns=1, max_tokens=10**9))
    plan = [{"role": "coder", "instruction": f"step {i}"} for i in range(5)]
    res = orch.execute("loop", planner=_planner(plan))
    assert not res.ok
    assert "budget" in res.stopped_reason
    assert sum(s.status == "done" for s in res.steps) <= 1


def test_step_failure_is_caught():
    def boom(p, c):
        raise RuntimeError("nope")
    agents = _agents(coder=boom)
    orch = Orchestrator(agents)
    res = orch.execute("x", planner=_planner([{"role": "coder", "instruction": "go"}]))
    assert not res.ok
    assert res.steps[0].status == "failed"
    assert "nope" in res.stopped_reason


def test_critic_can_reject():
    agents = _agents()
    orch = Orchestrator(agents)
    res = orch.execute("x", planner=_planner([{"role": "coder", "instruction": "go"}]),
                       critic=_critic(approved=False, notes="needs tests"))
    assert not res.ok
    assert res.stopped_reason == "needs tests"


def test_message_bus_delivery():
    bus = MessageBus()
    bus.post(Message(sender="a", recipient="coder", content="hi"))
    bus.post(Message(sender="b", recipient="coder", content="there"))
    msgs = bus.receive("coder")
    assert [m.content for m in msgs] == ["hi", "there"]
    assert bus.receive("coder") == []  # consumed


def test_make_agent_has_role():
    a = make_agent("vision", lambda p, c: {})
    assert a.role.name == "vision"
    assert a.id.startswith("vision-")


def test_parse_steps_accepts_strings():
    orc = Orchestrator(_agents())
    steps = orc._parse_steps({"steps": ["just do it"]})
    assert steps[0].role == "coder" and steps[0].instruction == "just do it"


def test_audit_log_records_events():
    events = []

    class Audit:
        def record(self, e):
            events.append(e)

    agents = _agents()
    orch = Orchestrator(agents, audit=Audit())
    orch.execute("x", planner=_planner([{"role": "coder", "instruction": "go"}]))
    kinds = [e["event"] for e in events]
    assert "plan" in kinds
    assert any("done" in str(e) for e in events)
