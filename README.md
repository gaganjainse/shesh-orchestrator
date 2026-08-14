# 🎼 shesh-orchestrator

> **Multi-agent RLM runtime for Shesh.** A coordinator decomposes goals into steps
> and routes them to role-based child agents (planner, coder, researcher, vision,
> critic) over an A2A-lite bus, with turn/token/time budgets.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python) ![License](https://img.shields.io/badge/License-GPL--3.0--or--later-blue) ![Tests](https://img.shields.io/badge/Tests-28-success) ![CI](https://github.com/gaganjainse/shesh-orchestrator/actions/workflows/ci.yml/badge.svg)

- **License:** GPL-3.0-or-later
- **Owner:** Gagan Jain ([@gaganjainse](https://github.com/gaganjainse))
- **Layer:** Mind (multi-agent)
- **Part of:** [shesh-ecosystem](https://github.com/gaganjainse/shesh-ecosystem)

---

## Why this repo exists

Single-model agents hit a wall on compound goals. This runtime splits work across
specialist roles while the Brain (policy) still gates every tool call.

---

## Quick start

```bash
uv sync --extra dev
uv run pytest -q        # 28 tests
uv run ruff check .
```

## Design

- **Roles** carry their model key and tool allow-list; the Brain gates every call.
- **Handlers are injected** — production uses LLM-backed agents, tests use plain
  functions (no network or model needed to test).
- **Bounded autonomy:** `Budget` caps turns/tokens; failures become results, not
  crashes; the critic can reject output.
- **A2A bus:** agents post messages to roles; a real transport can replace the
  in-process bus without changing callers.

## Tools (MCP)

- `execute(goal, max_turns, max_tokens)` — plan → delegate → review
- `list_roles()` · `post_message(role, content)`


> **Reproducible install:** `uv.lock` pins the full dependency tree. Install with
> `uv sync --frozen` (or `uv pip install -r <(uv export --frozen)`) for a locked build.

## Status

Component CI is green (reusable ecosystem pipeline). Security posture and
vulnerability reporting: [SECURITY.md](SECURITY.md).

## Documentation index

- **Part of:** [shesh-ecosystem](https://github.com/gaganjainse/shesh-ecosystem)
- **Compiled reading:** [shesh-docs](https://github.com/gaganjainse/shesh-docs)

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
