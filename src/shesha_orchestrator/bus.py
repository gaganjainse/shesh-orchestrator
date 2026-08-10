"""A tiny local agent-to-agent message bus (A2A-lite).

Synchronous and in-process by design: for our single-user laptop, an
event loop over Unix sockets is premature. Agents post messages addressed
to a role/agent id; recipients read their inbox. This is the seam where a
real A2A transport could be plugged in later without changing callers.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class Message:
    sender: str
    recipient: str            # role name or agent id
    content: str
    kind: str = "message"     # message | request | reply | event
    ts: float = field(default_factory=time.time)
    correlation_id: str | None = None


class MessageBus:
    def __init__(self, maxlen: int = 1000) -> None:
        self._queues: dict[str, deque[Message]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._all: deque[Message] = deque(maxlen=maxlen)

    def post(self, msg: Message) -> None:
        self._queues[msg.recipient].append(msg)
        self._all.append(msg)

    def receive(self, recipient: str) -> list[Message]:
        q = self._queues[recipient]
        out = list(q)
        q.clear()
        return out

    def history(self, limit: int = 50) -> list[Message]:
        return list(self._all)[-limit:]
