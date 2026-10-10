"""Claude plays one RailSimEnv episode through the environment's MCP tools (Anthropic Messages API, manual tool loop).

Every tool call goes through the environment (a server over OpenEnv's WebSocket session, or `LocalSession` in this
process), so the reward recorded is the environment's rubric. One episode: the system prompt is the task's rules, the first user message its situation; the model gets
`max_turns` turns (a turn may hold several tool calls); each tool result says how many turns are left; the last turn
asks it to submit. Two turns in a row without a tool call end the episode. An episode without `submit_plan` scores 0.

The history is append-only (each response's full content, thinking blocks included, goes back unchanged), which is
what thinking-block validation on current Claude models requires. The one exception is a mid-output refusal fallback:
the blocks the declined model produced before the `fallback` marker are dropped (`echo_content`), as the API asks.

Caching: the system prompt (the task's rules) carries an explicit breakpoint, and top-level automatic caching moves a
second one to the end of the history on every request, so each turn re-reads everything before it from the cache.

The key is `ANTHROPIC_API_KEY`, or `LLM_API_KEY` when that is unset (`make_client`).
"""

from __future__ import annotations

import os
import time
from typing import Any

try:
    from .session import EnvSession
except ImportError:  # flat layout
    from session import EnvSession

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"   # fallbacks="default": a declined request is re-run server-side
THINKING = {"type": "adaptive", "display": "summarized"}   # summaries make transcripts readable
OPENING = ("{situation}\n\nMake the new dispatching plan. Use check_plan to test drafts, then call submit_plan with your "
           "final plan.")


def anthropic_tools(mcp_tools) -> list[dict]:
    """MCP tool listings -> Anthropic tool definitions. Requests are streamed, so tool inputs stream eagerly too
    (a 100-turn plan is a large argument); `validate` checks them before anything runs."""
    return [{"name": t.name, "description": t.description or "",
             "input_schema": t.input_schema or {"type": "object", "properties": {}},
             "eager_input_streaming": True} for t in mcp_tools]


def validate(name: str, args: Any) -> str | None:
    """A reason the tool input cannot be run, or None. Eagerly streamed inputs are not validated by the API."""
    if not isinstance(args, dict):
        return "tool arguments must be a JSON object"
    if name in ("check_plan", "submit_plan"):
        plan = args.get("plan")
        if not isinstance(plan, (list, str)):
            return 'missing "plan": a list of objects like {"train": "37413", "to": "HDG"} ([] changes nothing)'
        if isinstance(plan, list) and not all(isinstance(e, dict) for e in plan):
            return 'every plan entry must be an object like {"train": "37413", "hold": {"HDG": "07:31"}}'
    elif name != "get_situation":
        return f"unknown tool {name!r}: use get_situation, check_plan or submit_plan"
    return None


def make_client(max_retries: int = 4):
    """An Anthropic client keyed from ANTHROPIC_API_KEY, else LLM_API_KEY, else the SDK's own credential lookup."""
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
    kw = {"api_key": key} if key else {}
    return anthropic.Anthropic(max_retries=max_retries, **kw)


def echo_content(content) -> list:
    """The response content to append to the history. After a mid-output refusal fallback, the declined model's
    thinking, tool calls and other model-internal blocks before the last `fallback` marker are omitted; its text and
    everything after the marker go back unchanged."""
    marks = [i for i, b in enumerate(content) if b.type == "fallback"]
    if not marks:
        return list(content)
    return [b for i, b in enumerate(content) if i >= marks[-1] or b.type == "text"]


def _dump(content) -> list[dict]:
    return [b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in content]


def run_episode(base_url: str | None, task_id: str, model: str = DEFAULT_MODEL, effort: str = "high", max_turns: int = 12,
                max_tokens: int = 64000, wall_clock_s: float | None = None, client=None, session=None) -> dict:
    """Play one episode; returns the rollout record (messages, every tool call with the server's answer, the
    final grade, token usage with cache reads and writes, and per-turn timings)."""
    client = client or make_client()
    env = session if session is not None else EnvSession(base_url)
    t0 = time.time()
    meta = env.reset(task_id)
    system = [{"type": "text", "text": meta["instructions"], "cache_control": {"type": "ephemeral"}}]
    tools = anthropic_tools(env.tools)
    messages: list[dict] = [{"role": "user", "content": OPENING.format(situation=meta["situation"])}]
    record = {"task_id": task_id, "model": model, "effort": effort, "max_tokens": max_tokens, "steps": [],
              "errors": [], "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                                      "output_tokens": 0},
              "served_by": [], "turn_log": [], "reward": 0.0, "end_reason": None, "final": None, "turns": 0}
    done, idle, parse_retries = False, 0, 0
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
                with client.beta.messages.stream(model=model, max_tokens=max_tokens, system=system, tools=tools,
                                                 messages=messages, thinking=THINKING,
                                                 output_config={"effort": effort}, cache_control={"type": "ephemeral"},
                                                 betas=[FALLBACK_BETA], fallbacks="default") as stream:
                    resp = stream.get_final_message()
            except ValueError as e:   # tool-input JSON the SDK could not parse at all: nothing to append, retry
                record["errors"].append(f"turn {turn}: unparseable tool input ({e})")
                parse_retries += 1
                if parse_retries > 2:
                    record["end_reason"] = "unparseable_tool_input"
                    break
                turn -= 1
                continue
            record["turns"] = turn
            u = resp.usage
            for k in record["usage"]:
                record["usage"][k] += getattr(u, k, None) or 0
            if resp.model not in record["served_by"]:
                record["served_by"].append(resp.model)
            record["turn_log"].append({"turn": turn, "model": resp.model, "stop_reason": resp.stop_reason,
                                       "seconds": round(time.time() - t_turn, 1), "input_tokens": u.input_tokens,
                                       "cache_read": u.cache_read_input_tokens or 0,
                                       "cache_write": u.cache_creation_input_tokens or 0,
                                       "output_tokens": u.output_tokens})
            content = echo_content(resp.content)
            messages.append({"role": "assistant", "content": content})
            if resp.stop_reason == "refusal":
                record["end_reason"] = "refusal"
                record["errors"].append(f"turn {turn}: refusal {getattr(resp, 'stop_details', None)}")
                break
            calls = [b for b in content if b.type == "tool_use"]
            if not calls:
                idle += 1
                if idle >= 2:
                    record["end_reason"] = "no_tool_call"
                    break
                messages.append({"role": "user", "content": f"Call a tool: check_plan to test a draft, or submit_plan "
                                                            f"with your final plan. {left} turns left."})
                continue
            idle = 0
            results = []
            for b in calls:
                if resp.stop_reason == "max_tokens":
                    err = "your output hit max_tokens before this call was complete; send it again, more compactly"
                else:
                    err = validate(b.name, b.input) or ("the episode is over" if done else None)
                if err:
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": err, "is_error": True})
                    continue
                text, step_done, reward, step_meta = env.call(b.name, b.input)
                record["steps"].append({"turn": turn, "tool": b.name, "arguments": b.input, "result": text})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": text})
                if step_done:
                    done = True
                    record["reward"] = float(reward or 0.0)
                    record["final"] = step_meta
                    record["end_reason"] = "submitted" if b.name == "submit_plan" else step_meta.get("end_reason")
            if done:
                messages.append({"role": "user", "content": results})
                break
            note = f"{left} turns left." + (" This is your last turn: call submit_plan now." if left == 1 else "")
            messages.append({"role": "user", "content": results + [{"type": "text", "text": note}]})
        else:
            record["end_reason"] = record["end_reason"] or "max_turns"
    finally:
        env.close()
    record["seconds"] = round(time.time() - t0, 1)
    record["messages"] = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
                           else _dump(m["content"])} for m in messages]
    return record
