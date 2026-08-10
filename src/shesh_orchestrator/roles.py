"""Agent roles and their tool/policy allow-lists."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    name: str
    model: str                 # logical model key; the router maps to a real model
    tools: tuple[str, ...]     # MCP tool names this role may call
    description: str = ""
    system_prompt: str = ""


# Roles map to the 6 GB-safe models; larger SheshOS models can be substituted.
ROLES: dict[str, Role] = {
    "coordinator": Role(
        name="coordinator", model="primary",
        tools=("spawn", "delegate", "audit", "recall", "assemble_context"),
        description="Routes tasks, spawns subagents, enforces policy and budget.",
    ),
    "planner": Role(
        name="planner", model="primary",
        tools=("recall", "assemble_context", "search", "fetch"),
        description="Breaks goals into ordered steps and identifies risks.",
    ),
    "coder": Role(
        name="coder", model="code",
        tools=("read_file", "write_file", "run_tests", "shell", "git"),
        description="Edits code, runs tests, produces diffs (ACP-facing).",
    ),
    "researcher": Role(
        name="researcher", model="primary",
        tools=("search", "fetch", "recall", "note_fact"),
        description="Web/doc research with citations.",
    ),
    "vision": Role(
        name="vision", model="vision",
        tools=("screenshot", "describe_image"),
        description="Interprets screenshots and GUI state.",
    ),
    "critic": Role(
        name="critic", model="primary",
        tools=("recall", "run_tests", "assemble_context"),
        description="Reviews outputs, gates promotion, runs evals.",
    ),
}


def role(name: str) -> Role:
    if name not in ROLES:
        raise KeyError(f"unknown role {name!r}; choose from {sorted(ROLES)}")
    return ROLES[name]
