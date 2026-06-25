"""channel: a unix-domain socket transport for the wire protocol.

A WiredAgent talks over any object with recv()/sendall(); a connected unix
socket is exactly that. This module produces such a socket on both ends - a
server that listens and accepts connections, and a client connect(). The
returned connected socket IS the channel, handed straight to a WiredAgent or
driven by a client with cai.wire.

Deliberately not wired into Agent/WiredAgent yet - a caller pairs them:

  server = UnixSocketServer(path)
  channel = server.accept()              # WiredAgent(agent, channel).serve()
  ...
  channel = connect(path)                # the client end; drive with cai.wire

Minimal: blocking accept/connect, one socket per connection, no TLS or
abstract-namespace addresses. Serving threads and a registry are later layers."""
from __future__ import annotations

import os
import socket


class UnixSocketServer:
    """a listening unix-domain socket. accept() blocks for a client and returns
    the connected socket (a channel); close() shuts the listener and unlinks the
    socket file so the path is free to reuse."""

    def __init__(self, path, *, backlog=8):
        self.path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # clear a stale socket file left by a previous (crashed) server, else
        # bind() fails with EADDRINUSE on a path nobody is listening on.
        try:
            os.unlink(path)
        except OSError:
            pass
        self._sock.bind(path)
        self._sock.listen(backlog)

    def accept(self):
        """block until a client connects; return the connected socket."""
        conn, _addr = self._sock.accept()
        return conn

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def connect(path):
    """connect to a unix-domain socket at path; return the connected socket (the
    client end of a channel, driven with cai.wire.send / a Decoder)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    return sock
