"""Live mode: the connector process talks to the RS-StatChem window the user has open.

The visible app listens on a local socket only while the user has switched "Allow Claude to use
this window" on (off by default). Both sides run as the same user on the same computer: the
socket is limited to that user and every request carries a random token from a file only the
user can read. Nothing is reachable over the network.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

from rsstatchem.paths import user_data_dir

SETTING = "connector/live"


def socket_name() -> str:
    key = hashlib.sha256(str(user_data_dir().resolve()).encode()).hexdigest()[:16]
    return f"rsstatchem-{key}"


def token_file() -> Path:
    return user_data_dir() / "connector.token"


def write_token() -> str:
    token = secrets.token_hex(32)
    path = token_file()
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as f:
        f.write(token)
    return token


def encode(parts: list) -> list[dict]:
    return [{"text": p} if isinstance(p, str) else
            {"png": base64.b64encode(p.png).decode("ascii"), "title": p.title} for p in parts]


def decode(items: list[dict]) -> list:
    from rsstatchem.mcp.engine import Picture

    return [i["text"] if "text" in i else Picture(base64.b64decode(i["png"]), i.get("title", ""))
            for i in items]


class LiveServer(QObject):
    """In the visible app (Qt thread): answers the connector's requests."""

    _answer = Signal(object, bytes)
    used = Signal(str)                   # name of the tool being run (for the status text)

    def __init__(self, ctx) -> None:
        super().__init__()
        self.ctx = ctx
        self.engine = None
        self._token = ""
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._connected)
        self._buffers: dict[QLocalSocket, bytearray] = {}
        self._run_lock = threading.Lock()
        self._answer.connect(self._send)

    @property
    def listening(self) -> bool:
        return self._server.isListening()

    def start(self) -> bool:
        if self.listening:
            return True
        probe = QLocalSocket()
        probe.connectToServer(socket_name())
        if probe.waitForConnected(200):      # another RS-StatChem is already connected
            probe.abort()
            return False
        QLocalServer.removeServer(socket_name())
        if not self._server.listen(socket_name()):
            return False
        self._token = write_token()
        if self.engine is None:
            from rsstatchem.mcp.engine import Engine

            self.engine = Engine(self.ctx, live=True)
        return True

    def stop(self) -> None:
        self._server.close()
        for socket in list(self._buffers):
            socket.abort()
        self._buffers.clear()
        token_file().unlink(missing_ok=True)
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def _connected(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            self._buffers[socket] = bytearray()
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._gone(s))

    def _gone(self, socket) -> None:
        self._buffers.pop(socket, None)
        socket.deleteLater()

    def _read(self, socket) -> None:
        buffer = self._buffers.get(socket)
        if buffer is None:
            return
        buffer += bytes(socket.readAll().data())
        while b"\n" in buffer:
            line, _, rest = bytes(buffer).partition(b"\n")
            buffer[:] = rest
            self._handle(socket, line)

    def _handle(self, socket, line: bytes) -> None:
        try:
            request = json.loads(line)
            ok = hmac.compare_digest(str(request.get("token", "")), self._token)
        except (ValueError, TypeError, AttributeError):
            ok = False
        if not ok or not self._token:
            socket.abort()
            return
        engine, name = self.engine, str(request.get("name"))
        from rsstatchem.mcp.engine import all_tools

        if name == "ping":
            self._send(socket, json.dumps({"parts": []}).encode() + b"\n")
            return
        if engine is None or name not in {t.name for t in all_tools()}:
            self._send(socket, json.dumps({"error": f"Unknown tool '{name}'."}).encode() + b"\n")
            return
        self.used.emit(name)

        def work() -> None:
            with self._run_lock:
                try:
                    reply = {"parts": encode(engine.call(name, request.get("args") or {}))}
                except Exception as exc:  # noqa: BLE001 - reported to the connector
                    reply = {"error": str(exc.args[0] if isinstance(exc, KeyError) and exc.args
                                          else exc), "kind": type(exc).__name__}
            try:
                self._answer.emit(socket, json.dumps(reply, ensure_ascii=False).encode() + b"\n")
            except RuntimeError:
                pass                      # the app is closing

        threading.Thread(target=work, name="connector", daemon=True).start()

    def _send(self, socket, data: bytes) -> None:
        if socket in self._buffers:
            socket.write(data)
            socket.flush()


class LiveError(ValueError):
    """An error reported by a tool running in the visible app."""


class LiveClient:
    """In the connector process: forwards tool calls to the visible app when it listens."""

    def __init__(self) -> None:
        self._socket: QLocalSocket | None = None
        self._thread: int | None = None

    def _connect(self) -> QLocalSocket | None:
        if not token_file().exists():
            return None
        socket = QLocalSocket()
        socket.connectToServer(socket_name())
        return socket if socket.waitForConnected(1000) else None

    def available(self) -> bool:
        socket = self._connect()
        if socket is None:
            return False
        socket.abort()
        return True

    def call(self, name: str, args: dict, timeout: float = 900) -> list | None:
        """The tool's result, or None when no visible app is listening."""
        socket = self._connect()
        if socket is None:
            return None
        try:
            token = token_file().read_text(encoding="ascii").strip()
        except OSError:
            return None
        request = {"token": token, "name": name, "args": args}
        socket.write(json.dumps(request, ensure_ascii=False).encode() + b"\n")
        socket.waitForBytesWritten(5000)
        data = bytearray()
        end = time.monotonic() + timeout
        while b"\n" not in data:
            # short waits: Qt's blocking calls keep Python's interpreter lock
            if socket.bytesAvailable() or socket.waitForReadyRead(5):
                data += bytes(socket.readAll().data())
                continue
            if socket.state() != QLocalSocket.LocalSocketState.ConnectedState:
                if b"\n" in data:
                    break
                raise LiveError("RS-StatChem closed the connection. Check whether the window is "
                                "still open and the connector is still allowed there.")
            time.sleep(0.02)
            if time.monotonic() > end:
                raise LiveError("RS-StatChem did not answer in time.")
        socket.abort()
        reply = json.loads(bytes(data).partition(b"\n")[0])
        if "error" in reply:
            raise LiveError(reply["error"])
        return decode(reply["parts"])
