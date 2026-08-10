# shesh-orchestrator

**multi-agent RLM runtime** — Coordinator/planner/coder/researcher/vision/critic over an A2A bus.

- Layer: Mind (Mind)
- License: GPL-3.0
- Part of: [Shesh ecosystem](https://github.com/gaganjainse/shesh-ecosystem)

---
**Multi-agent RLM runtime for Shesh.** A coordinator decomposes goals into
steps and routes them to role-based child agents (planner, coder, researcher,
vision, critic), with an A2A-lite message bus and turn/token/time budgets.

- License: GPL-3.0
- Layer: Mind
- Part of: [Shesh ecosystem](https://github.com/gaganjainse/shesh-ecosystem)

## Design

- **Roles** carry their model key and tool allow-list; the Brain (policy) still
  gates every tool call.
- **Handlers are injected** — production uses LLM-backed agents, tests use plain
  functions. No network or model is required to run or test the runtime.
- **Bounded autonomy:** `Budget` caps turns/tokens; failures become results, not
  crashes; the critic can reject output.
- **A2A bus:** agents post messages to roles; a real transport can replace the
  in-process bus later without changing callers.

## Tools (MCP)

- `execute(goal, max_turns, max_tokens)` — plan→delegate→review
- `list_roles()`
- `post_message(role, content)` — inter-agent messaging

## Develop

```bash
uv sync --extra dev
uv run pytest -q        # 9 offline tests
uv run ruff check .
uv run shesh-orchestrator-mcp
```