"""The OpenAI-compatible tool loop (`rail_openenv.agent_openai`) in process, with scripted streams in place of the API."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

TASKS = Path(__file__).resolve().parents[2] / "tasks"
if not os.environ.get("RAIL_TASKS_DIR"):
    os.environ["RAIL_TASKS_DIR"] = ":".join(str(TASKS / p) for p in ("rail-v1-eval", "rail-v1-train")
                                            if (TASKS / p / "manifest.json").is_file())

from openai.types.chat import ChatCompletionChunk  # noqa: E402

from rail_core import load_pack  # noqa: E402
from rail_openenv.agent_openai import run_episode  # noqa: E402
from rail_openenv.session import LocalSession  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("RAIL_TASKS_DIR"), reason="no task pack built")
PACK = load_pack() if os.environ.get("RAIL_TASKS_DIR") else None
MODEL = "zai-org/GLM-5.3-Flash:novita"


def _chunk(delta=None, finish=None, usage=None):
    choices = [] if delta is None and finish is None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    return ChatCompletionChunk.model_validate({"id": "c", "object": "chat.completion.chunk", "created": 0,
                                               "model": "zai-org/glm-5.3-flash", "choices": choices,
                                               "usage": usage})


def _turn(reasoning="", text="", call=None, i=0, finish=None):
    """One streamed assistant turn: reasoning and text deltas, a tool call whose arguments arrive in pieces, usage."""
    out = [_chunk({"role": "assistant", "reasoning_content": reasoning})] if reasoning else []
    if text:
        out.append(_chunk({"content": text}))
    if call:
        name, args = call
        body = json.dumps(args) if not isinstance(args, str) else args
        half = len(body) // 2
        out.append(_chunk({"tool_calls": [{"index": 0, "id": f"call_{i}", "type": "function",
                                           "function": {"name": name, "arguments": body[:half]}}]}))
        out.append(_chunk({"tool_calls": [{"index": 0, "function": {"arguments": body[half:]}}]}))
    out.append(_chunk(finish=finish or ("tool_calls" if call else "stop")))
    out.append(_chunk(usage={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
                             "prompt_tokens_details": {"cached_tokens": 400}}))
    return out


class ScriptedClient:
    """Stands in for `openai.OpenAI`: each `chat.completions.create(stream=True)` yields the next scripted turn."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        assert kw["stream"] and kw["stream_options"] == {"include_usage": True}
        self.requests.append({**kw, "messages": [dict(m) for m in kw["messages"]]})
        return _play(self.script.pop(0))


def _play(turn):
    for item in turn:
        if isinstance(item, Exception):   # the connection drops partway through the stream
            raise item
        yield item


def test_scripted_episode_scores_the_optimum():
    task = PACK.tasks[0]
    plan = task.reference["optimal_plan"]
    client = ScriptedClient([
        _turn("draft first", "Checking.", ("check_plan", {"plan": plan}), 1),
        _turn("looks clean", "Done thinking."),                                     # no tool call: nudged once
        _turn("submit", "", ("submit_plan", {"plan": plan}), 2),
    ])
    rec = run_episode(None, task.task_id, model=f"hf:{MODEL}", client=client, session=LocalSession(PACK))
    assert rec["end_reason"] == "submitted" and rec["reward"] == 1.0 and rec["turns"] == 3
    assert [s["tool"] for s in rec["steps"]] == ["check_plan", "submit_plan"]
    assert rec["steps"][0]["arguments"] == {"plan": plan}                     # streamed argument pieces joined
    req = client.requests[-1]
    assert req["model"] == MODEL and "reasoning_effort" not in req
    assert req["tools"][0]["type"] == "function"
    roles = [m["role"] for m in req["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user", "assistant", "user"]
    first = req["messages"][2]
    assert first["reasoning_content"] == "draft first" and first["tool_calls"][0]["id"] == "call_1"
    assert req["messages"][3]["tool_call_id"] == "call_1"
    # append-only: every request's history starts with the previous request's history
    for a, b in zip(client.requests, client.requests[1:]):
        assert b["messages"][: len(a["messages"])] == a["messages"]
    assert rec["usage"] == {"input_tokens": 1800, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1200,
                            "output_tokens": 300}
    assert rec["served_by"] == ["zai-org/glm-5.3-flash"] and len(rec["turn_log"]) == 3


def test_a_turn_cut_at_max_tokens_is_named_in_the_nudge():
    task = PACK.tasks[0]
    client = ScriptedClient([
        _turn("thinking and thinking", "", None, finish="length"),
        _turn("", "", ("submit_plan", {"plan": task.reference["optimal_plan"]}), 1),
    ])
    rec = run_episode(None, task.task_id, model=f"hf:{MODEL}", max_tokens=500, client=client,
                      session=LocalSession(PACK))
    nudge = client.requests[1]["messages"][-1]
    assert nudge["role"] == "user" and nudge["content"].startswith("Your output hit the 500-token limit")
    assert rec["turn_log"][0]["stop_reason"] == "length" and rec["reward"] == 1.0


def test_bad_tool_arguments_are_refused_before_they_run():
    task = PACK.tasks[0]
    client = ScriptedClient([
        _turn("", "", ("check_plan", '{"plan": [{"train": '), 1),                   # not JSON
        _turn("", "", ("check_plan", {"plan": [1, 2]}), 2),                       # fails validate
        _turn("", "", ("submit_plan", {"plan": task.reference["optimal_plan"]}), 3),
    ])
    rec = run_episode(None, task.task_id, model=f"hf:{MODEL}", effort="high", client=client,
                      session=LocalSession(PACK))
    assert [s["tool"] for s in rec["steps"]] == ["submit_plan"]   # neither bad check reached the environment
    tool_msgs = [m for m in rec["messages"] if m["role"] == "tool"]
    assert "not valid JSON" in tool_msgs[0]["content"] and "every plan entry" in tool_msgs[1]["content"]
    assert client.requests[0]["reasoning_effort"] == "high"
    assert rec["reward"] == 1.0 and rec["effort"] == "high"


def test_a_dropped_stream_is_retried_from_the_same_history(monkeypatch):
    import httpx
    from openai import APIConnectionError

    import rail_openenv.agent_openai as ao

    monkeypatch.setattr(ao.time, "sleep", lambda s: None)
    task = PACK.tasks[0]
    drop = APIConnectionError(request=httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions"))
    submit = _turn("", "", ("submit_plan", {"plan": task.reference["optimal_plan"]}), 1)
    client = ScriptedClient([submit[:2] + [drop], submit])
    rec = run_episode(None, task.task_id, model=f"hf:{MODEL}", client=client, session=LocalSession(PACK))
    assert rec["reward"] == 1.0 and rec["errors"] == [] and len(rec["retries"]) == 1 and rec["turns"] == 1
    assert client.requests[0]["messages"] == client.requests[1]["messages"]


def test_a_rate_limit_is_waited_out(monkeypatch):
    import httpx
    from openai import RateLimitError

    import rail_openenv.agent_openai as ao

    waits = []
    monkeypatch.setattr(ao.time, "sleep", waits.append)
    task = PACK.tasks[0]
    req = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    limited = RateLimitError("Too many requests in a short window", body=None,
                             response=httpx.Response(429, request=req))
    submit = _turn("", "", ("submit_plan", {"plan": task.reference["optimal_plan"]}), 1)
    client = ScriptedClient([[limited]] * 5 + [submit])
    rec = run_episode(None, task.task_id, model=f"hf:{MODEL}", client=client, session=LocalSession(PACK))
    assert rec["reward"] == 1.0 and len(rec["retries"]) == 5 and waits == [5, 10, 20, 40, 60]


def test_a_prompt_over_the_context_window_ends_the_episode():
    import httpx
    from openai import BadRequestError

    task = PACK.tasks[0]
    req = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    over = BadRequestError("This model's maximum context length is 262144 tokens", body=None,
                           response=httpx.Response(400, request=req))
    client = ScriptedClient([_turn("", "", ("check_plan", {"plan": []}), 1), [over]])
    rec = run_episode(None, task.task_id, model=f"hf:{MODEL}", client=client, session=LocalSession(PACK))
    assert rec["end_reason"] == "context_length" and rec["reward"] == 0.0 and rec["errors"] == []
    assert "262144" in rec["end_detail"] and [s["tool"] for s in rec["steps"]] == ["check_plan"]
