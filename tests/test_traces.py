"""Offline tests for the local trace recorder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_orchestrator.traces import TraceRecorder  # noqa: E402


def test_span_records_to_jsonl(tmp_path):
    rec = TraceRecorder(tmp_path / "traces.jsonl")
    with rec.trace("plan", goal="x") as s:
        s.set_attribute("model", "phi4")
        s.add_event("step", n=1)
    lines = (tmp_path / "traces.jsonl").read_text().splitlines()
    assert len(lines) == 1
    import json
    r = json.loads(lines[0])
    assert r["name"] == "plan" and r["status"] == "ok"
    assert r["attributes"]["model"] == "phi4"
    assert r["duration_ms"] >= 0


def test_span_failure(tmp_path):
    rec = TraceRecorder(tmp_path / "t.jsonl")
    try:
        with rec.trace("boom"):
            raise RuntimeError("nope")
    except RuntimeError:
        pass
    import json
    r = json.loads((tmp_path / "t.jsonl").read_text())
    assert r["status"] == "error" and "nope" in r["attributes"]["error"]


def test_recent(tmp_path):
    rec = TraceRecorder(tmp_path / "t.jsonl")
    rec.span("a").finish()
    rec.span("b").finish()
    assert len(rec.recent()) == 2
