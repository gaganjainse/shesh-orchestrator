"""Offline tests for persistent sessions."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesha_orchestrator.orchestrator import make_agent  # noqa: E402
from shesha_orchestrator.sessions import SessionManager  # noqa: E402


def _agents():
    return {n: make_agent(n, lambda p, c: {"ok": True, "by": n})
            for n in ("researcher", "coder", "critic", "coordinator")}


def test_start_runs_to_completion():
    mgr = SessionManager(_agents())
    s = mgr.start("do a thing")
    # wait for the background thread
    for _ in range(50):
        if mgr.is_done(s.id):
            break
        time.sleep(0.02)
    assert s.status == "done"
    assert s.result["ok"] is True
    assert len(s.trace) >= 1


def test_cancel_stops_session():
    import threading
    agents = _agents()
    # A slow agent gives the main thread time to cancel before step 2.
    def slow(p, c):
        time.sleep(0.1)
        return {"ok": True}
    agents["coder"] = make_agent("coder", slow)
    mgr = SessionManager(agents)
    mgr.planner = lambda g, c: {"steps": [{"role": "coder", "instruction": f"step {i}"} for i in range(20)]}
    s = mgr.start("long task")
    mgr.cancel(s.id)
    for _ in range(100):
        if mgr.is_done(s.id):
            break
        time.sleep(0.02)
    assert s.status == "cancelled"


def test_list_and_get():
    mgr = SessionManager(_agents())
    s = mgr.start("x")
    assert mgr.get(s.id) is s
    assert s in mgr.list()


def test_failure_recorded():
    def boom(p, c):
        raise RuntimeError("kaboom")
    agents = _agents()
    agents["coder"] = make_agent("coder", boom)
    mgr = SessionManager(agents)
    s = mgr.start("x")
    for _ in range(50):
        if mgr.is_done(s.id):
            break
        time.sleep(0.02)
    assert s.status == "failed"
    assert "kaboom" in s.error
