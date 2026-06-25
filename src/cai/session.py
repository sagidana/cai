"""session: the on-disk conversation format behind Agent.save / Agent.load.

A ".flow" file is a small JSON document - the same shape the reference frontend
uses - holding a conversation plus the settings needed to resume it:

    {
      "version": 3,
      "messages": [ {"role": "system", ...}, {"role": "user", ...}, ... ],
      "settings": {
        "system_prompt_base": <str|null>,   # the user-supplied base prompt
        "skills":           [<name>, ...],
        "selected_tools":   [<name>, ...],   # names only; callables don't serialise
        "model":            <str|null>,
        "reasoning_effort": <str|null>,
        "temperature":      <float|null>,
        "max_steps":        <int|null>
      }
    }

The leading system message is the *composed* prompt (base + skills) for a
portable, human-readable record; on load it is dropped and the prompt is
re-derived from system_prompt_base + skills, so the base stays the single source
of truth. Files live in ~/.config/cai/sessions/ named by the agent's id
('<name>.flow', the reference's '<run_id>.flow' convention) - but any path may be
given.

SessionsRegistry is a static class (no instances): it owns the sessions folder
and the flow format, so its methods read like SessionsRegistry.session_path(name)
- the same shape as AgentsRegistry.
"""
import os
import json
import logging
from datetime import datetime

from cai import config


log = logging.getLogger("cai")


class SessionsRegistry:
    """manages ~/.config/cai/sessions/ and the .flow format. all methods are
    static; the class is a namespace, not a value."""

    FLOW_VERSION = 3

    @staticmethod
    def sessions_dir():
        return os.path.join(config.config_dir(), "sessions")

    @staticmethod
    def session_path(name):
        """the flow path for an agent, named by its id - '<name>.flow' under the
        sessions dir (the reference's '<run_id>.flow' convention, no timestamp).
        a stable per-agent file, so re-saving the same agent overwrites its own."""
        return os.path.join(SessionsRegistry.sessions_dir(), f"{name}.flow")

    @staticmethod
    def has_real_messages(messages):
        """True when messages holds anything worth saving (a non-system turn)."""
        for message in messages:
            if message.get("role") != "system":
                return True
        return False

    @staticmethod
    def flow_payload(messages,
                     system_prompt,
                     system_prompt_base,
                     skills,
                     selected_tools,
                     model,
                     reasoning_effort=None,
                     temperature=None,
                     max_steps=None):
        """build the flow JSON payload. system_prompt (the composed prompt) is
        prepended as a leading system message when set; system_prompt_base is
        what actually round-trips."""
        full = list(messages)
        if system_prompt:
            lead = {}
            lead["role"] = "system"
            lead["content"] = system_prompt
            full = [lead] + full

        settings = {}
        settings["system_prompt_base"] = system_prompt_base
        settings["skills"] = list(skills)
        settings["selected_tools"] = list(selected_tools)
        settings["model"] = model
        settings["reasoning_effort"] = reasoning_effort
        settings["temperature"] = temperature
        settings["max_steps"] = max_steps

        payload = {}
        payload["version"] = SessionsRegistry.FLOW_VERSION
        payload["messages"] = full
        payload["settings"] = settings
        return payload

    @staticmethod
    def write_flow(path, payload):
        """write payload to path atomically (tmp + replace), creating the dir."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def read_flow(path):
        """parse a .flow file into its payload dict."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def list_sessions():
        """saved .flow paths under the sessions dir, newest first."""
        try:
            names = os.listdir(SessionsRegistry.sessions_dir())
        except OSError:
            return []
        paths = []
        for name in names:
            if not name.endswith(".flow"): continue
            paths.append(os.path.join(SessionsRegistry.sessions_dir(), name))
        paths.sort(key=os.path.getmtime, reverse=True)
        return paths

    @staticmethod
    def session_label(path):
        """one-line picker label: '<date>  (<n> msgs)  <first user snippet>'."""
        try:
            stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        except OSError:
            stamp = os.path.basename(path)
        try:
            payload = SessionsRegistry.read_flow(path)
        except (OSError, ValueError):
            return stamp

        convo = []
        for message in (payload.get("messages") or []):
            if message.get("role") == "system": continue
            convo.append(message)

        first_user = ""
        for message in convo:
            if message.get("role") != "user": continue
            first_user = str(message.get("content", ""))
            break

        snippet = " ".join(first_user.split())[:60]
        return f"{stamp}  ({len(convo)} msgs)  {snippet}".rstrip()
