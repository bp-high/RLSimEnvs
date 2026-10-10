"""An open model plays one RailSimEnv episode through an OpenAI-compatible chat-completions API (Hugging Face's
inference router by default), with the same episode rules as the Claude harness in `agent.py`.

Same as `agent.run_episode`: the task's rules are the system prompt, its situation the first user message (`OPENING`),
`max_turns` turns, each tool result says how many turns are left, the last turn asks for a submit, two turns in a row
without a tool call end the episode, and an episode without `submit_plan` scores 0. Tool inputs go through the same
`validate` before anything runs. The rollout record has the same fields, so `run_eval.py` and `report.py` read both.

Differences forced by the API: requests are streamed and the deltas assembled here (a long reasoning turn over a plain
request can outlive the router's connection); the model's reasoning (`reasoning_content`, GLM's interleaved thinking)
goes back with each assistant message so tool-calling turns keep their thinking; tool calls whose arguments are not
JSON are answered with an error instead of being run. Token usage is recorded per turn, cached prompt tokens apart
where a provider reports them.

A model spec is `hf:<model id>[:<provider>]`, for example `hf:zai-org/GLM-5.3-Flash:novita`. The key is `HF_TOKEN`,
else `HUGGING_FACE_API_KEY`; `HF_BILL_TO` names the organisation billed (`X-HF-Bill-To`).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import APIConnectionError, BadRequestError, InternalServerError, RateLimitError

try:
    from .agent import OPENING, validate
    from .session import EnvSession
except ImportError:  # flat layout
    from agent import OPENING, validate
    from session import EnvSession

HF_ROUTER = "https://router.huggingface.co/v1"


def is_openai_spec(model: str) -> bool:
    return model.startswith("hf:")


def make_openai_client(max_retries: int = 8, bill_to: str | None = None, timeout_s: float = 1800.0):
    """A client on the Hugging Face router keyed from HF_TOKEN, else HUGGING_FACE_API_KEY."""
    from openai import OpenAI

    key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_API_KEY")
    if not key:
        raise RuntimeError("set HF_TOKEN (or HUGGING_FACE_API_KEY) for the Hugging Face router")
    bill_to = bill_to or os.environ.get("HF_BILL_TO")
    headers = {"X-HF-Bill-To": bill_to} if bill_to else None
    return OpenAI(base_url=HF_ROUTER, api_key=key, default_headers=headers, max_retries=max_retries,
                  timeout=timeout_s)


def openai_tools(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                              "parameters": t.input_schema or {"type": "object", "properties": {}}}}
            for t in mcp_tools]


def _stream(client, **kw) -> dict:
    """One streamed completion assembled into {content, reasoning, tool_calls, finish_reason, model, usage}."""
    content, reasoning, calls = [], [], {}
    finish, model, usage = None, None, None
    for chunk in client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **kw):
        model = chunk.model or model
        if chunk.usage is not None:
            usage = chunk.usage
        for ch in chunk.choices or []:
            d = ch.delta
            if d is not None:
                if d.content:
                    content.append(d.content)
                r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                if r:
                    reasoning.append(r)
                for tc in d.tool_calls or []:
                    c = calls.setdefault(tc.index if tc.index is not None else len(calls),
                                         {"id": None, "name": "", "arguments": []})
                    if tc.id:
                        c["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            c["name"] += tc.function.name
                        if tc.function.arguments:
                            c["arguments"].append(tc.function.arguments)
            if ch.finish_reason:
                finish = ch.finish_reason
    tool_calls = [{"id": c["id"] or f"call_{i}", "name": c["name"], "arguments": "".join(c["arguments"])}
                  for i, c in sorted(calls.items())]
    return {"content": "".join(content), "reasoning": "".join(reasoning), "tool_calls": tool_calls,
            "finish_reason": finish, "model": model, "usage": usage}


def _stream_retrying(client, record: dict, turn: int, attempts: int = 8, **kw) -> dict:
    """`_stream`, run again from the same history when the connection drops or the server fails mid-stream (the
    client's own retries cover only the start of a request). The waits (5, 10, 20, 40, then 60 s; about 4.5 minutes
    in all) outlast the router's circuit breaker, which pauses a failing provider with a 503, and a provider's dynamic
    rate limit (429), which the client's own short retries do not. Recovered drops are
    kept in `record["retries"]`, not `errors`, so a resume does not rerun the episode for them."""
    for attempt in range(1, attempts + 1):
        try:
            return _stream(client, **kw)
        except (APIConnectionError, InternalServerError, RateLimitError) as e:
            if attempt == attempts:
                raise
            record["retries"].append(f"turn {turn} attempt {attempt}: {type(e).__name__}: {e}"[:500])
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def _context_overflow(e: Exception) -> bool:
    """A provider's refusal of a prompt longer than its context window (plus the max_tokens asked for)."""
    msg = str(e).lower()
    return "context" in msg and any(w in msg for w in ("length", "window", "too long", "exceed", "maximum"))


def _usage(u) -> tuple[int, int, int]:
    """(uncached prompt tokens, cached prompt tokens, completion tokens)."""
    if u is None:
        return 0, 0, 0
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) or 0) if details is not None else 0
    return (u.prompt_tokens or 0) - cached, cached, u.completion_tokens or 0


def run_episode(base_url: str | None, task_id: str, model: str, effort: str | None = None, max_turns: int = 12,
                max_tokens: int = 64000, wall_clock_s: float | None = None, client=None, session=None,
                progress=None) -> dict:
    """Play one episode with `model` (`hf:<model id>[:<provider>]`); returns a rollout record shaped like
    `agent.run_episode`'s. `effort`, when not None or "default", is sent as `reasoning_effort`. `progress`, if given,
    is called with a one-line summary after every turn."""
    client = client or make_openai_client()
    api_model = model.removeprefix("hf:")
    env = session if session is not None else EnvSession(base_url)
    t0 = time.time()
    meta = env.reset(task_id)
    tools = openai_tools(env.tools)
    messages: list[dict] = [{"role": "system", "content": meta["instructions"]},
                            {"role": "user", "content": OPENING.format(situation=meta["situation"])}]
    record = {"task_id": task_id, "model": model, "effort": effort or "default", "max_tokens": max_tokens,
              "steps": [], "errors": [], "retries": [],
              "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                        "output_tokens": 0},
              "served_by": [], "turn_log": [], "reward": 0.0, "end_reason": None, "final": None, "turns": 0}
    extra = {"reasoning_effort": effort} if effort and effort != "default" else {}
    done, idle = False, 0
    try:
        turn = 0
        while turn < max_turns:
            if wall_clock_s is not None and time.time() - t0 > wall_clock_s:
                record["end_reason"] = "wall_clock"
                break
            turn += 1
            left = max_turns - turn
            t_turn = time.time()
            try:
                resp = _stream_retrying(client, record, turn, model=api_model, messages=messages, tools=tools,
                                        max_tokens=max_tokens, **extra)
            except BadRequestError as e:   # the history outgrew the provider's context: the episode ends here
                if not _context_overflow(e):
                    raise
                record["end_reason"] = "context_length"
                record["end_detail"] = f"turn {turn}: {e}"[:500]
                break
            record["turns"] = turn
            inp, cached, out = _usage(resp["usage"])
            record["usage"]["input_tokens"] += inp
            record["usage"]["cache_read_input_tokens"] += cached
            record["usage"]["output_tokens"] += out
            served = resp["model"] or api_model
            if served not in record["served_by"]:
                record["served_by"].append(served)
            record["turn_log"].append({"turn": turn, "model": served, "stop_reason": resp["finish_reason"],
                                       "seconds": round(time.time() - t_turn, 1), "input_tokens": inp,
                                       "cache_read": cached, "cache_write": 0, "output_tokens": out,
                                       "reasoning_chars": len(resp["reasoning"])})
            if progress is not None:
                progress(f"{task_id} turn {turn}: {time.time() - t_turn:.0f}s in={inp + cached} out={out} "
                         f"finish={resp['finish_reason']} calls={[c['name'] for c in resp['tool_calls']]}")
            msg: dict[str, Any] = {"role": "assistant", "content": resp["content"] or None}
            if resp["reasoning"]:
                msg["reasoning_content"] = resp["reasoning"]
            if resp["tool_calls"]:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}}
                                     for c in resp["tool_calls"]]
            messages.append(msg)
            if not resp["tool_calls"]:
                idle += 1
                if idle >= 2:
                    record["end_reason"] = "no_tool_call"
                    break
                cut = (f"Your output hit the {max_tokens:,}-token limit before you called a tool. Reason less per "
                       f"turn and let check_plan do the simulating. ") if resp["finish_reason"] == "length" else ""
                messages.append({"role": "user", "content": f"{cut}Call a tool: check_plan to test a draft, or "
                                                            f"submit_plan with your final plan. {left} turns left."})
                continue
            idle = 0
            results = []
            for c in resp["tool_calls"]:
                if resp["finish_reason"] == "length":
                    err = "your output hit max_tokens before this call was complete; send it again, more compactly"
                else:
                    try:
                        args = json.loads(c["arguments"]) if c["arguments"].strip() else {}
                        err = validate(c["name"], args)
                    except json.JSONDecodeError as e:
                        args, err = None, f"tool arguments are not valid JSON ({e}); send the call again"
                    err = err or ("the episode is over" if done else None)
                if err:
                    results.append({"role": "tool", "tool_call_id": c["id"], "content": f"Error: {err}"})
                    continue
                text, step_done, reward, step_meta = env.call(c["name"], args)
                record["steps"].append({"turn": turn, "tool": c["name"], "arguments": args, "result": text})
                results.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if step_done:
                    done = True
                    record["reward"] = float(reward or 0.0)
                    record["final"] = step_meta
                    record["end_reason"] = "submitted" if c["name"] == "submit_plan" else step_meta.get("end_reason")
            messages.extend(results)
            if done:
                break
            note = f"{left} turns left." + (" This is your last turn: call submit_plan now." if left == 1 else "")
            messages.append({"role": "user", "content": note})
        else:
            record["end_reason"] = record["end_reason"] or "max_turns"
    finally:
        env.close()
    record["seconds"] = round(time.time() - t0, 1)
    record["messages"] = messages
    return record
