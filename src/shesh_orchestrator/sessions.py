"""Persistent, reattachable agent sessions.

A Session runs an Orchestrator execution in a background thread and keeps
its state (goal, trace, status, result) so a client can disconnect and
re-connect later. This is the local, single-user analogue of Prime Agent's
detach/reattach, built on top of the in-process Orchestrator and the A2A bus.

Sessions are kept in an in-memory registry by default; a future version can
persist them to ~/.local/state/shesha/orchestrator/.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from collections.abc import Callable

from .agents import Agent, Budget
from .bus import MessageBus
from .orchestrator import ExecutionResult, Orchestrator
from .traces import get_recorder
from .stubs import always_approve, default_planner


@dataclass
class SessionState:
    id: str
    goal: str
    status: str = "pending"   # pending | running | done | failed | cancelled
    created: str = ""
    updated: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "goal": self.goal, "status": self.status,
            "created": self.created, "updated": self.updated,
            "result": self.result, "trace": self.trace, "error": self.error,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionManager:
    """Owns long-running sessions and their background threads."""

    def __init__(self, agents: dict[str, Agent], bus: MessageBus | None = None,
                 budget: Budget | None = None,
                 planner: Callable | None = None,
                 critic: Callable | None = None) -> None:
        self.agents = agents
        self.bus = bus or MessageBus()
        self.budget = budget or Budget()
        self.planner = planner or default_planner
        self.critic = critic or always_approve
        self._sessions: dict[str, SessionState] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: set[str] = set()
        self._lock = threading.Lock()

    def start(self, goal: str) -> SessionState:
        sid = uuid.uuid4().hex[:12]
        state = SessionState(id=sid, goal=goal, created=_now(), updated=_now())
        with self._lock:
            self._sessions[sid] = state
        thread = threading.Thread(target=self._run, args=(sid,), daemon=True)
        self._threads[sid] = thread
        state.status = "running"
        thread.start()
        return state

    def _run(self, sid: str) -> None:
        state = self._sessions[sid]
        recorder = get_recorder()
        try:
            with recorder.trace(f'session:{sid}', goal=state.goal) as span:
                state_span = span
                orch = Orchestrator(self.agents, bus=self.bus, budget=self.budget)

            result: ExecutionResult = self._execute_with_cancel(sid, orch, 
                state.goal, self.planner, self.critic, context=None)
            with self._lock:
                state.trace = result.trace
                state_span.set_attribute('steps', len(result.steps))
                state.result = {
                    "ok": result.ok,
                    "steps": [{"role": s.role, "status": s.status} for s in result.steps],
                    "stopped_reason": result.stopped_reason,
                }
                if result.stopped_reason == "cancelled" or sid in self._cancel:
                    state.status = "cancelled"
                else:
                    state.status = "done" if result.ok else "failed"
                    if not result.ok:
                        state.error = result.stopped_reason or state.error
                state.updated = _now()
        except Exception as e:  # noqa: BLE001
            with self._lock:
                state.status = "failed"
                state.error = str(e)
                state.updated = _now()

    def _execute_with_cancel(self, sid, orch, goal, planner, critic, context):
        """Run an Orchestrator.execute but abort early if cancelled."""
        ctx = dict(context or {})
        plan_result = planner(goal, ctx)
        from .orchestrator import Step
        steps = [Step(role=item.get("role", "coder"),
                      instruction=item.get("instruction", ""))
                 for item in plan_result.get("steps", [])]
        trace = []
        for step in steps:
            if sid in self._cancel:
                return ExecutionResult(goal, steps, ok=False,
                                       stopped_reason="cancelled", trace=trace)
            agent = orch._agent_for(step.role)
            if sid in self._cancel:
                return ExecutionResult(goal, steps, ok=False,
                                       stopped_reason="cancelled", trace=trace)
            if not orch.budget.allow(agent):
                if sid in self._cancel:
                    return ExecutionResult(goal, steps, ok=False,
                                           stopped_reason="cancelled", trace=trace)
                return ExecutionResult(goal, steps, ok=False,
                                       stopped_reason="budget exhausted", trace=trace)
            try:
                step.result = agent.run(step.instruction, ctx)
                step.status = "done"
                orch.budget.record(agent)
            except Exception as e:  # noqa: BLE001
                step.status = "failed"
                step.result = {"error": str(e)}
                return ExecutionResult(goal, steps, ok=False,
                                       stopped_reason=str(e), trace=trace)
            trace.append({"role": step.role, "instruction": step.instruction,
                          "result": step.result})
            ctx["prior"] = trace
        return ExecutionResult(goal, steps, ok=True, trace=trace)

    def get(self, sid: str) -> SessionState | None:
        return self._sessions.get(sid)

    def list(self) -> list[SessionState]:
        return list(self._sessions.values())

    def cancel(self, sid: str) -> bool:
        if sid not in self._sessions:
            return False
        self._cancel.add(sid)
        state = self._sessions[sid]
        if state.status == "running":
            state.status = "cancelled"
            state.updated = _now()
        return True

    def is_done(self, sid: str) -> bool:
        state = self.get(sid)
        return bool(state and state.status in {"done", "failed", "cancelled"})
