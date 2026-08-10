"""shesh-orchestrator: multi-agent RLM runtime.

Implements the Recursive Language Model pattern (from Prime Agent) at small,
local-first scale:

- A **coordinator** decomposes a goal into steps and spawns role-based child
  agents (planner, coder, researcher, vision, critic).
- Child agents are processes/handlers that return results; they communicate
  over an in-process bus (A2A-style messages).
- Every action is policy-gated and audited: models propose, the Brain disposes.
- Execution is bounded by turn/token/time budgets so autonomy can't run away.

The LLM/provider is injected, so the runtime is fully testable without a model.
"""
from __future__ import annotations

__version__ = "0.1.0"
