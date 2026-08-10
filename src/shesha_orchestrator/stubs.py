"""Deterministic offline fallbacks used when no LLM is available."""
from __future__ import annotations


def default_planner(goal: str, ctx: dict) -> dict:
    return {"steps": [
        {"role": "researcher", "instruction": f"Gather context for: {goal}"},
        {"role": "coder", "instruction": f"Execute: {goal}"},
        {"role": "critic", "instruction": "Verify the result"},
    ]}


def echo_agent(prompt: str, ctx: dict) -> dict:
    return {"ok": True, "echo": prompt[:200]}


def always_approve(goal: str, ctx: dict) -> dict:
    return {"approved": True, "notes": "stub critic approves"}
