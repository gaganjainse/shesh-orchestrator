"""The RLM-style orchestrator.

`execute(goal)`:
  1. asks the planner for a step list,
  2. runs each step by delegating to the right role (spawning a child agent),
  3. collects results, optionally has the critic review,
  4. records everything for the audit log,
  5. stops when the goal is complete or a budget is exhausted.

Handlers are injected so no LLM/network is needed for tests or safe defaults.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .agents import Agent, Budget, Handler
from .bus import Message, MessageBus
from .roles import Role, role


@dataclass
class Step:
    role: str
    instruction: str
    result: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"   # pending | done | failed | skipped


@dataclass
class ExecutionResult:
    goal: str
    steps: list[Step]
    ok: bool
    stopped_reason: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)


# A planner turns (goal, ctx) into a list of {role, instruction}.
Planner = Handler
# A critic reviews (goal, steps) -> {"approved": bool, "notes": str}.
Critic = Handler


class Orchestrator:
    def __init__(
        self,
        agents: dict[str, Agent],
        bus: MessageBus | None = None,
        budget: Budget | None = None,
        planner_role: str = "planner",
        critic_role: str = "critic",
        audit: Any = None,        # anything with .record(event_dict)
    ) -> None:
        self.agents = agents
        self.bus = bus or MessageBus()
        self.budget = budget or Budget()
        self.planner_role = planner_role
        self.critic_role = critic_role
        self.audit = audit

    def _log(self, event: str, **payload: Any) -> None:
        if self.audit is not None:
            self.audit.record({"event": event, **payload})

    def execute(
        self,
        goal: str,
        planner: Planner,
        critic: Critic | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        ctx = dict(context or {})
        ctx["goal"] = goal
        ctx["bus"] = self.bus

        # 1. Plan
        plan_result = planner(goal, ctx)
        steps = self._parse_steps(plan_result)
        self._log("plan", goal=goal, steps=[s.instruction for s in steps])

        trace: list[dict[str, Any]] = []
        # 2. Execute steps
        for step in steps:
            if not self.budget.allow(self.agents.get(step.role, next(iter(self.agents.values())))):
                return ExecutionResult(goal, steps, ok=False,
                                      stopped_reason="budget exhausted", trace=trace)
            try:
                agent = self._agent_for(step.role)
                step.result = agent.run(step.instruction, ctx)
                step.status = "done"
                self.budget.record(agent)
                self._log("step_done", role=step.role)
                self.bus.post(Message(
                    sender=agent.id, recipient="coordinator",
                    kind="event", content=f"{step.role} done",
                    correlation_id=ctx.get("goal_id")))
            except Exception as e:  # noqa: BLE001 - turn failures into results
                step.status = "failed"
                step.result = {"error": str(e)}
                self._log("step_failed", role=step.role, error=str(e))
                return ExecutionResult(goal, steps, ok=False,
                                      stopped_reason=f"{step.role} failed: {e}", trace=trace)
            trace.append({"role": step.role, "instruction": step.instruction,
                          "result": step.result})
            ctx["prior"] = trace

        # 3. Critic review (optional)
        ok = True
        reason = ""
        if critic is not None and self.critic_role in self.agents:
            review = critic(goal, {"steps": trace, **ctx})
            ok = bool(review.get("approved", True))
            reason = review.get("notes", "")
            self._log("review", approved=ok, notes=reason)

        return ExecutionResult(goal, steps, ok=ok, stopped_reason=reason, trace=trace)

    def _agent_for(self, role_name: str) -> Agent:
        if role_name not in self.agents:
            # Fall back to a sensible default role rather than crashing.
            role_name = "coordinator"
        return self.agents[role_name]

    @staticmethod
    def _parse_steps(plan_result: dict) -> list[Step]:
        raw = plan_result.get("steps") if isinstance(plan_result, dict) else None
        if not raw:
            return [Step(role="researcher", instruction=str(plan_result))]
        steps: list[Step] = []
        for item in raw:
            if isinstance(item, str):
                steps.append(Step(role="coder", instruction=item))
            else:
                steps.append(Step(
                    role=item.get("role", "coder"),
                    instruction=item.get("instruction", ""),
                ))
        return [s for s in steps if s.instruction]


def make_agent(role_name: str, handler: Handler) -> Agent:
    r: Role = role(role_name)
    return Agent(id=f"{role_name}-{uuid.uuid4().hex[:6]}", role=r, handler=handler)
