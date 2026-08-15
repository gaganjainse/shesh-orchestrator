"""Regressions for defects found in the 2026-08-15 fleet audit.

Each test fails against the code as it was before the fix.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_orchestrator import a2a  # noqa: E402
from shesh_orchestrator.sessions import SessionManager  # noqa: E402

# ── BUG-1: the broker socket path was a non-f string ────────────────────────

def test_socket_path_has_no_literal_braces():
    """The default socket was "/run/user/{os.getuid()}/shesh-a2a.sock".

    Missing the f prefix meant os.getuid() never ran and the path could never
    be created, breaking the whole A2A transport at startup.
    """
    p = str(a2a.default_socket())
    assert "{" not in p and "}" not in p, f"unresolved placeholder in {p}"
    assert "os.getuid" not in p


def test_socket_path_resolves_to_a_real_runtime_dir(monkeypatch):
    monkeypatch.delenv("SHESH_A2A_SOCKET", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/4242")
    assert a2a.default_socket() == Path("/run/user/4242/shesh-a2a.sock")


def test_socket_path_honours_explicit_override(monkeypatch):
    monkeypatch.setenv("SHESH_A2A_SOCKET", "/tmp/custom-a2a.sock")
    assert a2a.default_socket() == Path("/tmp/custom-a2a.sock")


def test_socket_path_is_evaluated_lazily(monkeypatch):
    """Resolution must happen at call time, not import time."""
    monkeypatch.setenv("SHESH_A2A_SOCKET", "/tmp/first.sock")
    first = a2a.default_socket()
    monkeypatch.setenv("SHESH_A2A_SOCKET", "/tmp/second.sock")
    assert a2a.default_socket() != first


def test_socket_path_falls_back_without_xdg_runtime_dir(monkeypatch):
    monkeypatch.delenv("SHESH_A2A_SOCKET", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert a2a.default_socket() == Path(f"/run/user/{os.getuid()}/shesh-a2a.sock")


# ── ARCH-2 / RED-6: the session path silently skipped critic review ─────────

class _Agent:
    def __init__(self, role):
        self.role = role
        self.in_tokens = 0
        self.out_tokens = 0

    def run(self, instruction, ctx):
        self.in_tokens += 1
        self.out_tokens += 1
        return {"ok": True, "role": self.role}


def _plan(goal, ctx):
    return {"steps": [{"role": "coder", "instruction": "do the thing"}]}


def test_session_runs_the_critic():
    """A session and a direct execute() must agree for the same goal.

    The session path accepted a critic argument and never called it, so a
    critic that rejects had no effect on a session run.
    """
    called = {}

    def critic(goal, ctx):
        called["yes"] = True
        return {"approved": True, "notes": "ok"}

    agents = {"coder": _Agent("coder"), "critic": _Agent("critic")}
    mgr = SessionManager(agents, planner=_plan, critic=critic)
    sid = mgr.start("build something").id
    _wait(mgr, sid)

    assert called.get("yes"), "critic was never invoked by the session path"


def test_session_honours_a_rejecting_critic():
    def reject(goal, ctx):
        return {"approved": False, "notes": "not good enough"}

    agents = {"coder": _Agent("coder"), "critic": _Agent("critic")}
    mgr = SessionManager(agents, planner=_plan, critic=reject)
    sid = mgr.start("build something").id
    state = _wait(mgr, sid)

    assert state.result is not None
    assert state.result["ok"] is False
    # The reason carries the critic's own notes, which is more useful than a
    # fixed string when diagnosing why a session stopped.
    assert state.result["stopped_reason"] == "not good enough"


class SessionTimeout(AssertionError):
    """A session did not reach a terminal state in time."""

    def __init__(self, sid: str, timeout: float) -> None:
        super().__init__(f"session {sid} did not finish within {timeout}s")


def _wait(mgr, sid, timeout=5.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = mgr.get(sid)
        if st and st.status in {"done", "failed", "cancelled"}:
            return st
        time.sleep(0.02)
    raise SessionTimeout(sid, timeout)
