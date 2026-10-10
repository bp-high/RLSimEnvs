"""The Claude tool loop (`rail_openenv.agent`) against a real server, with a scripted client in place of the API."""

import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

TASKS = Path(__file__).resolve().parents[2] / "tasks"
if not os.environ.get("RAIL_TASKS_DIR"):
    os.environ["RAIL_TASKS_DIR"] = ":".join(str(TASKS / p) for p in ("rail-v1-eval", "rail-v1-train")
                                            if (TASKS / p / "manifest.json").is_file())

from anthropic.types.beta import BetaMessage  # noqa: E402

from rail_core import load_pack  # noqa: E402
from rail_openenv.agent import echo_content, run_episode  # noqa: E402
from rail_openenv.server import create_server  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("RAIL_TASKS_DIR"), reason="no task pack built")
PACK = load_pack() if os.environ.get("RAIL_TASKS_DIR") else None


def _msg(content, stop_reason="tool_use", model="claude-sonnet-5-5"):
    return BetaMessage.model_validate({
        "id": "msg_x", "type": "message", "role": "assistant", "model": model, "stop_reason": stop_reason,
        "stop_sequence": None, "content": content,
        "usage": {"input_tokens": 10, "output_tokens": 100, "cache_read_input_tokens": 1000,
                  "cache_creation_input_tokens": 200}})


def _tool(name, args, i):
    return {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": args}


class ScriptedClient:
    """Stands in for `anthropic.Anthropic`: each `beta.messages.stream` call returns the next scripted message."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kw):
        self.requests.append({**kw, "history": [(m["role"], id(m["content"])) for m in kw["messages"]]})
        msg = self.script.pop(0)
        return _Ctx(msg)


class _Ctx:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


@pytest.fixture(scope="module")
def base_url():
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_server(PACK), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(timeout=10)


def test_scripted_episode_scores_the_optimum(base_url):
    task = PACK.tasks[0]
    plan = task.reference["optimal_plan"]
    client = ScriptedClient([
        _msg([{"type": "thinking", "thinking": "draft", "signature": "sig"}, _tool("check_plan", {"plan": plan}, 1)]),
        _msg([{"type": "text", "text": "Looks clean."}], stop_reason="end_turn"),   # no tool call: nudged once
        _msg([_tool("submit_plan", {"plan": plan}, 2)]),
    ])
    rec = run_episode(base_url, task.task_id, model="claude-sonnet-5-5", client=client)
    assert rec["end_reason"] == "submitted" and rec["reward"] == 1.0 and rec["turns"] == 3
    assert [s["tool"] for s in rec["steps"]] == ["check_plan", "submit_plan"]
    # caching: an explicit breakpoint on the rules, automatic caching for the history
    for req in client.requests:
        assert req["system"][-1]["cache_control"] == {"type": "ephemeral"}
        assert req["cache_control"] == {"type": "ephemeral"}
        assert all(t["eager_input_streaming"] for t in req["tools"])
    # append-only: every request's history starts with the previous request's history
    for a, b in zip(client.requests, client.requests[1:]):
        assert b["history"][: len(a["history"])] == a["history"]
    assert rec["usage"] == {"input_tokens": 30, "cache_creation_input_tokens": 600, "cache_read_input_tokens": 3000,
                            "output_tokens": 300}
    assert rec["served_by"] == ["claude-sonnet-5-5"] and len(rec["turn_log"]) == 3


def test_invalid_tool_input_is_refused_before_it_runs(base_url):
    task = PACK.tasks[0]
    client = ScriptedClient([
        _msg([_tool("check_plan", {"plan": [1, 2]}, 1)]),
        _msg([_tool("submit_plan", {"plan": task.reference["optimal_plan"]}, 2)]),
    ])
    rec = run_episode(base_url, task.task_id, client=client)
    assert [s["tool"] for s in rec["steps"]] == ["submit_plan"]   # the bad check never reached the server
    first_result = rec["messages"][2]["content"][0]
    assert first_result["is_error"] and "every plan entry" in first_result["content"]
    assert rec["reward"] == 1.0


def test_echo_content_drops_declined_blocks_before_a_fallback():
    m = _msg([{"type": "thinking", "thinking": "a", "signature": "s"}, {"type": "text", "text": "partial"},
              _tool("check_plan", {"plan": []}, 1),
              {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-5"},
               "trigger": {"type": "refusal", "category": None}},
              {"type": "text", "text": "continued"}, _tool("submit_plan", {"plan": []}, 2)])
    kept = echo_content(m.content)
    assert [b.type for b in kept] == ["text", "fallback", "text", "tool_use"]
    assert kept[-1].name == "submit_plan"
    plain = _msg([{"type": "thinking", "thinking": "a", "signature": "s"}, _tool("check_plan", {"plan": []}, 1)])
    assert echo_content(plain.content) == list(plain.content)
