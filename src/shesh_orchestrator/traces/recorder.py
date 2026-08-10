"""JSONL span recorder compatible with local inspection and future OTel export."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_TRACE_DIR = Path.home() / ".local" / "share" / "shesha" / "traces"


class Span:
    def __init__(self, recorder: "TraceRecorder", name: str, attributes: dict) -> None:
        self.recorder = recorder
        self.name = name
        self.attributes = attributes
        self.id = uuid.uuid4().hex[:16]
        self.start = time.time()
        self.end = 0.0
        self.status = "ok"
        self.events: list[dict] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attrs) -> None:
        self.events.append({"name": name, "ts": time.time(), **attrs})

    def fail(self, reason: str) -> None:
        self.status = "error"
        self.attributes["error"] = reason

    def finish(self) -> dict:
        self.end = time.time()
        record = {
            "id": self.id,
            "name": self.name,
            "start": self.start,
            "duration_ms": round((self.end - self.start) * 1000, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }
        self.recorder._write(record)
        return record


class TraceRecorder:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (DEFAULT_TRACE_DIR / "traces.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def span(self, name: str, **attributes) -> Span:
        return Span(self, name, dict(attributes))

    @contextmanager
    def trace(self, name: str, **attributes):
        s = self.span(name, **attributes)
        try:
            yield s
        except Exception as e:  # noqa: BLE001
            s.fail(str(e))
            raise
        finally:
            s.finish()

    def _write(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def recent(self, limit: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


_DEFAULT: TraceRecorder | None = None


def get_recorder() -> TraceRecorder:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TraceRecorder()
    return _DEFAULT
