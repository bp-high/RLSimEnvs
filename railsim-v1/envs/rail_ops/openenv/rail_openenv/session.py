"""A client session on a RailSimEnv server (OpenEnv's MCP client over its WebSocket), for scripts and agents.

    env = EnvSession("http://127.0.0.1:8000")          # or LocalSession(pack): the same tools, in this process
    meta = env.reset("gnlw-2025-03-13-0703-real-0")      # metadata["instructions"] is the system prompt
    text, done, reward, meta = env.call("submit_plan", {"plan": [...]})
"""

from __future__ import annotations

import json

from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient


class EnvSession:
    """One WebSocket session (one episode at a time)."""

    def __init__(self, base_url: str, timeout_s: float = 300.0):
        self.client = MCPToolClient(base_url.rstrip("/"), message_timeout_s=timeout_s).sync()
        self._tools = None

    def reset(self, task_id: str | None = None, **kw) -> dict:
        if task_id is not None:
            kw["task_id"] = task_id
        return dict(self.client.reset(**kw).observation.metadata or {})

    @property
    def tools(self):
        if self._tools is None:
            self._tools = self.client.list_tools()
        return self._tools

    def call(self, name: str, arguments: dict) -> tuple[str, bool, float | None, dict]:
        """(tool result text, done, reward, observation metadata)."""
        step = self.client.step(CallToolAction(tool_name=name, arguments=arguments))
        obs = step.observation
        err = getattr(obs, "error", None)
        res = getattr(obs, "result", None)
        text = None
        if isinstance(res, dict):
            blocks = res.get("content") or []
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text") or None
            if text is None and res.get("data") is not None:
                text = res["data"] if isinstance(res["data"], str) else json.dumps(res["data"])
        elif res is not None:
            data = getattr(res, "data", None)
            text = data if isinstance(data, str) else json.dumps(data) if data is not None else str(res)
        if err:
            msg = err.get("message") if isinstance(err, dict) else getattr(err, "message", str(err))
            text = json.dumps({"error": str(msg)})
        reward = step.reward if step.reward is not None else obs.reward
        return text or "", bool(step.done or obs.done), reward, dict(obs.metadata or {})

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


def _result_text(res) -> str | None:
    if res is None:
        return None
    if isinstance(res, dict):
        blocks = res.get("content") or []
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if text:
            return text
        data = res.get("data")
        return data if isinstance(data, str) else (json.dumps(data) if data is not None else None)
    content = getattr(res, "content", None)
    if content:
        text = "\n".join(getattr(b, "text", "") for b in content)
        if text:
            return text
    data = getattr(res, "data", None)
    return data if isinstance(data, str) else (json.dumps(data) if data is not None else str(res))


class LocalSession:
    """The environment in this process (no server, no WebSocket): the same tools, results and rubric. For tests,
    pilots and quick experiments."""

    def __init__(self, pack=None):
        from openenv.core.env_server.mcp_types import ListToolsAction

        from .environment import RailOpsEnvironment

        self.env = RailOpsEnvironment(pack)
        self._list = ListToolsAction
        self._tools = None

    def reset(self, task_id: str | None = None, **kw) -> dict:
        if task_id is not None:
            kw["task_id"] = task_id
        return dict(self.env.reset(**kw).metadata or {})

    @property
    def tools(self):
        if self._tools is None:
            self._tools = self.env.step(self._list()).tools
        return self._tools

    def call(self, name: str, arguments: dict) -> tuple[str, bool, float | None, dict]:
        obs = self.env.step(CallToolAction(tool_name=name, arguments=arguments))
        err = getattr(obs, "error", None)
        text = _result_text(getattr(obs, "result", None))
        if err:
            msg = err.get("message") if isinstance(err, dict) else getattr(err, "message", str(err))
            text = json.dumps({"error": str(msg)})
        return text or "", bool(obs.done), obs.reward, dict(obs.metadata or {})

    def close(self):
        pass
