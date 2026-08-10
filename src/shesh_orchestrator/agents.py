"""Agent abstraction.

An agent is a role + a handler that takes a task and context and returns a
result. In production the handler calls an LLM; in tests it is a plain
function, which is what makes the orchestrator deterministic and testable.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .roles import Role

# A handler receives (prompt, context) and returns a result dict.
Handler = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass
class Agent:
    id: str
    role: Role
    handler: Handler
    in_tokens: int = 0
    out_tokens: int = 0

    def run(self, prompt: str, ctx: dict[str, Any]) -> dict[str, Any]:
        result = self.handler(prompt, ctx)
        # Rough accounting so the budget can be enforced.
        self.in_tokens += len(prompt) // 4
        self.out_tokens += len(str(result)) // 4
        return result


@dataclass
class Budget:
    max_turns: int = 12
    max_tokens: int = 20_000
    max_seconds: int = 300
    used_turns: int = 0
    used_tokens: int = 0

    def allow(self, agent: Agent) -> bool:
        return (
            self.used_turns < self.max_turns
            and (self.used_tokens + agent.in_tokens + agent.out_tokens) < self.max_tokens
        )

    def record(self, agent: Agent) -> None:
        self.used_turns += 1
        self.used_tokens += agent.in_tokens + agent.out_tokens
