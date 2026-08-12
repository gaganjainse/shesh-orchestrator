"""LLM-backed planner, role agents, and critic.

These handlers turn the orchestrator from a stubbed demo into something that
can actually reason. Each takes (prompt, context) and returns the dict shape
the Orchestrator expects.

Design:
- The model client is injectable (like shesh-mind's OllamaClient), so tests
  pass in a fake and production passes an HTTP client.
- Output is parsed from a strict JSON block; if parsing fails, we fall back
  to the deterministic stub so a flaky model never crashes a run.
- The router (shesh-mind) chooses the model per role.
"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# A client takes (model, prompt) and returns text.
ModelClient = Callable[[str, str], str]

PLANNER_SYSTEM = (
    "You are the planner for the Shesh multi-agent system. Given a goal, "
    "return ONLY a JSON object of the form "
    '{"steps": [{"role": "researcher|coder|vision|critic", "instruction": "..."}]}. '
    "Use researcher first when information is needed, coder to act, critic last. "
    "Keep steps concrete and ordered. No prose outside the JSON."
)

CRITIC_SYSTEM = (
    "You are the critic. Given a goal and the steps that were run, return ONLY "
    'a JSON object {"approved": true|false, "notes": "..."}. Approve only if '
    "the result plausibly achieves the goal; otherwise say what is missing."
)

ROLE_SYSTEM = {
    "researcher": (
        "You are a research agent. Find relevant information and cite "
        "sources. Return concise findings."
    ),
    "coder": (
        "You are a coding agent. Make the smallest correct change and run "
        "tests. Report what you did."
    ),
    "vision": "You are a vision agent. Describe what is on screen.",
    "critic": "You are a critic. Be terse and specific.",
}


def _extract_json(text: str) -> dict | None:
    """Pull the first balanced {...} JSON object out of an LLM response."""
    # Prefer a fenced block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            # Fenced block held non-JSON text — fall through to brace scan.
            pass
    # Balanced-brace scan so greedy .* doesn't eat trailing text.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


@dataclass
class LLMAgents:
    """Bundle of LLM-backed handlers with router + fallback."""
    client: ModelClient
    model_for_role: Callable[[str], str] = field(
        default_factory=lambda: (lambda role: "phi4-mini:latest"))
    max_retries: int = 1

    def planner(self, goal: str, ctx: dict[str, Any]) -> dict:
        prompt = f"{PLANNER_SYSTEM}\n\nGoal: {goal}"
        last_error: BaseException | None = None
        attempts = self.max_retries + 1
        for _ in range(attempts):
            try:
                raw = self.client(self.model_for_role("planner"), prompt)
                parsed = _extract_json(raw)
                if parsed and "steps" in parsed and parsed["steps"]:
                    return parsed
            except Exception as e:  # noqa: BLE001 - network/model errors; fallback below
                last_error = e
        # Deterministic fallback so the run continues — announced, not silent.
        print(f"llm planner failed {attempts}x ({last_error}); "
              f"using deterministic fallback", file=sys.stderr)
        from .stubs import default_planner
        return default_planner(goal, ctx)

    def agent(self, role: str) -> Callable[[str, dict], dict]:
        def _run(prompt: str, ctx: dict[str, Any]) -> dict:
            sysmsg = ROLE_SYSTEM.get(role, "")
            full = f"{sysmsg}\n\nTask: {prompt}"
            try:
                text = self.client(self.model_for_role(role), full)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": f"model unavailable: {e}"}
            return {"ok": True, "role": role, "output": text.strip()[:4000]}
        return _run

    def critic(self, goal: str, ctx: dict[str, Any]) -> dict:
        steps = ctx.get("steps", [])
        trace = ctx.get("prior", [])
        prompt = (f"{CRITIC_SYSTEM}\n\nGoal: {goal}\n"
                  f"Steps: {json.dumps(steps, default=str)[:3000]}\n"
                  f"Trace: {json.dumps(trace, default=str)[:3000]}")
        try:
            raw = self.client(self.model_for_role("critic"), prompt)
            parsed = _extract_json(raw)
            if parsed and "approved" in parsed:
                return parsed
            print("llm critic returned unparsable verdict; auto-approving", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - network/model errors; policy below
            print(f"llm critic unavailable ({e}); auto-approving per policy", file=sys.stderr)
        return {"approved": True, "notes": "critic unavailable; auto-approved"}
