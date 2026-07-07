"""agents registry: the common folder where every UnixWiredAgent binds its
unix socket - ~/.config/cai/agents/<agent_name>.sock.

the socket file IS the registry entry: a served agent creates it on bind and
the UnixSocketServer unlinks it on close (deterministic cleanup); a crashed
agent leaves a stale socket, which connect() reaps the moment any caller
finds it refusing connections. no metadata lives on disk - the file name
carries the agent's name and nothing else. kept free of heavy imports so it
stays cheap to pull in.

AgentsRegistry is a static class (no instances): it owns the folder, not any
one agent, so its methods read like AgentsRegistry.sock_path(name)."""
from __future__ import annotations

import os

from cai import config


class AgentsRegistry:
    """manages ~/.config/cai/agents/ - the one place UnixWiredAgents put their
    sockets. all methods are static; the class is a namespace, not a value."""

    @staticmethod
    def dir():
        return os.path.join(config.config_dir(), "agents")

    @staticmethod
    def ensure_dir():
        os.makedirs(AgentsRegistry.dir(), exist_ok=True)

    @staticmethod
    def sock_path(name):
        return os.path.join(AgentsRegistry.dir(), f"{name}.sock")

    @staticmethod
    def list_names():
        """names of agents with a socket file present (live or stale)."""
        try:
            entries = os.listdir(AgentsRegistry.dir())
        except OSError:
            return []
        names = []
        for entry in entries:
            if not entry.endswith(".sock"):
                continue
            names.append(entry[:-len(".sock")])
        return names

    @staticmethod
    def connect(name):
        """connect to a served agent's socket - the one way callers should
        reach a registry agent. a refused connection is the definitive stale
        signal (the file exists, nobody listens: a crash leftover), so it is
        reaped here before the error propagates - every probe doubles as the
        registry's garbage collection. other failures (already gone, transient)
        raise untouched."""
        from cai.channel import connect
        try:
            return connect(AgentsRegistry.sock_path(name))
        except ConnectionRefusedError:
            AgentsRegistry.reap(name)
            raise

    @staticmethod
    def reap(name):
        """unlink a stale socket left by a crashed agent. clean shutdowns unlink
        their own socket, so this is only the crash safety-net."""
        try:
            os.unlink(AgentsRegistry.sock_path(name))
        except OSError:
            pass
