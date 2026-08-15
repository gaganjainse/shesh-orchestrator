"""Unix-socket A2A transport for agents.

The in-process MessageBus is great for tests and single-process runs, but the
coordinator and child agents may run as separate processes (especially the LLM
handlers). This module provides a tiny line-delimited JSON transport over a
Unix domain socket, plus a bus implementation that speaks it.

Protocol:
  - Clients connect to SOCKET_PATH and send one JSON message per line.
  - Each message is: {"sender":..., "recipient":..., "content":..., "kind":...}
  - The broker fans messages out to connected clients whose role matches
    the recipient (and broadcasts "event" kind messages to all).

This is intentionally simple: no auth beyond filesystem permissions on the
socket (created 0600), no persistence. It is the seam where a real A2A
implementation (or remote transport) can be substituted.
"""
from __future__ import annotations

import contextlib
import json
import os
import socketserver
import threading
from pathlib import Path
from typing import Any


def default_socket() -> Path:
    """Resolve the broker socket path.

    Evaluated at call time, not import time, so a changed UID or
    SHESH_A2A_SOCKET is respected by tests and by re-exec.
    """
    override = os.environ.get("SHESH_A2A_SOCKET")
    if override:
        return Path(override)
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime) / "shesh-a2a.sock"


# Backwards-compatible module attribute; prefer default_socket().
DEFAULT_SOCKET = default_socket()

# F-05: 100 KB per line — oversized frames are dropped before parsing.
MAX_MESSAGE_BYTES = 100_000


class _Broker(socketserver.ThreadingUnixStreamServer):
    """Fan-out broker: keeps connected clients and routes messages."""

    daemon_threads = True

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        super().__init__(str(socket_path), _Handler)
        os.chmod(socket_path, 0o600)
        self._lock = threading.Lock()
        self._clients: list[_Handler] = []

    def broadcast(self, message: dict[str, Any], exclude=None) -> None:
        line = (json.dumps(message) + "\n").encode()
        with self._lock:
            dead = []
            for c in self._clients:
                if c is exclude:
                    continue
                try:
                    c.wfile.write(line)
                    c.wfile.flush()
                except (BrokenPipeError, OSError):
                    dead.append(c)
            for c in dead:
                if c in self._clients:
                    self._clients.remove(c)

    def register(self, client: _Handler) -> None:
        with self._lock:
            self._clients.append(client)

    def unregister(self, client: _Handler) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    def shutdown(self) -> None:
        super().shutdown()
        # socketserver.shutdown() only stops serve_forever; without
        # server_close() the listening FD leaks (ResourceWarning).
        self.server_close()
        with contextlib.suppress(OSError):
            self.socket_path.unlink()


class _Handler(socketserver.StreamRequestHandler):
    server: _Broker

    def handle(self) -> None:
        self.server.register(self)
        self.role: str | None = None
        try:
            for raw in self.rfile:
                # Hard size cap: refuse to even parse an oversized line.
                if len(raw) > MAX_MESSAGE_BYTES:
                    continue
                try:
                    msg = json.loads(raw.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                # F-05: reject malformed messages — a message must declare a
                # sane sender (its first message binds the routing role) and
                # carry a content field.
                sender = msg.get("sender")
                if not isinstance(sender, str) or not sender or \
                        any(c in sender for c in ("/", "\\")) or \
                        any(ord(c) < 32 for c in sender):
                    continue
                if self.role is None:
                    self.role = sender
                if "content" not in msg:
                    continue
                recipient = msg.get("recipient", "*")
                # Broadcast events and wildcards; route to matching roles; skip sender.
                if recipient in ("*", "event") or msg.get("kind") == "event":
                    self.server.broadcast(msg, exclude=self)
                else:
                    self._route(recipient, msg, exclude=self)
        finally:
            self.server.unregister(self)

    def _route(self, recipient: str, msg: dict[str, Any], exclude=None) -> None:
        line = (json.dumps(msg) + "\n").encode()
        with self.server._lock:
            dead = []
            for c in self.server._clients:
                if c is exclude:
                    continue
                if getattr(c, "role", None) == recipient:
                    try:
                        c.wfile.write(line)
                        c.wfile.flush()
                    except (BrokenPipeError, OSError):
                        # Client vanished mid-broadcast. Prune it instead of
                        # writing to a dead socket on every future message.
                        dead.append(c)
            for c in dead:
                self.server._clients.remove(c)


def serve(socket_path: Path | None = None) -> _Broker:
    """Start the broker in a background daemon thread. Returns the server."""
    broker = _Broker(socket_path or default_socket())
    thread = threading.Thread(target=broker.serve_forever, daemon=True)
    thread.start()
    return broker


def send(socket_path: Path, message: dict[str, Any]) -> None:
    """Send a single message (connect, write, close)."""
    import socket as _socket

    with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
        s.connect(str(socket_path))
        s.sendall((json.dumps(message) + "\n").encode())


def listen(role: str, socket_path: Path | None = None,
           on_message=None) -> None:
    """Block and call on_message(msg) for each line addressed to this role."""
    import socket as _socket

    socket_path = socket_path or default_socket()
    with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
        s.connect(str(socket_path))
        # Identify self by sending a hello with our role.
        s.sendall((json.dumps({"sender": role, "recipient": "*",
                                "content": f"{role} connected",
                                "kind": "event"}) + "\n").encode())
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                if on_message:
                    on_message(msg)
