"""Offline tests for the Unix-socket A2A broker."""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shesh_orchestrator import a2a  # noqa: E402


@pytest.fixture()
def broker(tmp_path):
    sock = tmp_path / "a2a.sock"
    srv = a2a.serve(sock)
    yield srv, sock
    srv.shutdown()


def _connect(sock_path: Path, role: str):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall((f'{{"sender":"{role}","recipient":"*","content":"hello","kind":"event"}}\n').encode())
    return s


def _read_until(s, needle, timeout=2.0):
    s.settimeout(timeout)
    buf = b""
    end = time.time() + timeout
    while time.time() < end:
        try:
            chunk = s.recv(4096)
        except TimeoutError:
            break
        if not chunk:
            break
        buf += chunk
        if needle.encode() in buf:
            for line in buf.split(b"\n"):
                if needle.encode() in line:
                    return line.decode()
    return None


def test_broker_starts_and_creates_socket(broker):
    _, sock = broker
    assert sock.exists()
    assert oct(sock.stat().st_mode)[-3:] == "600"


def test_broadcast_reaches_all(broker):
    _, sock = broker
    a = _connect(sock, "a")
    b = _connect(sock, "b")
    time.sleep(0.1)
    a2a.send(sock, {"sender": "x", "recipient": "*", "content": "ping", "kind": "event"})
    assert "ping" in (_read_until(a, "ping") or "")
    assert "ping" in (_read_until(b, "ping") or "")
    a.close()
    b.close()


def test_send_invalid_json_ignored(broker):
    _, sock = broker
    s = _connect(sock, "c")
    s.sendall(b"not json\n")
    a2a.send(sock, {"sender": "x", "recipient": "*", "content": "ok", "kind": "event"})
    assert "ok" in (_read_until(s, "ok") or "")
    s.close()
