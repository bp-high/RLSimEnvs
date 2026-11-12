"""A Tinker-hosted model (Inkling 2 Small by default) plays one RailSimEnv episode through Tinker's native sampling
API, with the episode rules of the other harnesses (`agent.py`, `agent_openai.py`). The same environment class is what
the RL loop trains on (`railsim-v1/train/train_rl.py`), so the policy is evaluated in exactly the setting it was
trained in: the same prompt tokens, sampling settings and budgets.

Same as the other harnesses: the task's rules are the system prompt, its situation the first user message, `max_turns`
turns, every tool result is followed by how many turns are left, the last turn asks for a submit, two turns in a row
without a tool call end the episode, an episode without `submit_plan` scores 0, and tool inputs go through `validate`
before anything runs. The rollout record has the same fields, so `run_eval.py`, `report.py` and the explorer read it.

Differences forced by the API:

- **Tokens, not messages.** The conversation is rendered with the model's own renderer (`tml_v0` for Inkling), sampled
  with `SamplingClient.sample`, and parsed back. Each turn's prompt is the previous prompt plus the tokens the model
  sampled plus the new messages, token for token: RL needs that (one training sequence per episode, see
  `tinker_cookbook.rl.data_processing.trajectory_to_data`), and it keeps Tinker's prefix cache warm.
- **A context window is part of the rules.** Inkling 2 Small has 65,536 tokens on Tinker (262,144 as
  `...:peft:262144`, at twice the training rate). A check result is 3,000 to 10,000 tokens, so 12 turns do not fit:
  the note after every turn also says how much context is left, and when another check would no longer fit beside
  the final answer the next turn is announced as the last (`RESULT_RESERVE`). A prompt that leaves no room at all ends
  the episode as `context_length`, as in `agent_openai`.
- **A turn that cannot be parsed** (cut at the token limit, or a tool call that is not valid JSON) runs no tool. Its
  tokens stay in the context as the model wrote them (a cut turn is closed with the end-of-turn tokens) and the model
  is told what happened.
- Reasoning effort is the renderer's float (`EFFORT`); the per-turn output limit defaults to 8,192 tokens.

A model spec is `tinker:<base model>[#<checkpoint>]`, for example `tinker:thinkingmachines/Inkling-2-Small` (the base
model) or `tinker:thinkingmachines/Inkling-2-Small#rl-a` (the last sampler checkpoint of the training run
`results/train/rl-a`; `#rl-a:000008` names a step, `#tinker://...` a sampler path). The key is `TINKER_API_KEY`.
Needs `tinker` and `tinker-cookbook` (not installed by default: see `railsim-v1/train/README.md`).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import Message, ToolSpec, get_renderer
from tinker_cookbook.rl.types import Env, StepResult, Trajectory, Transition
from tinker_cookbook.tokenizer_utils import get_tokenizer

try:
    from .agent import OPENING, validate
    from .session import LocalSession
except ImportError:  # flat layout
    from agent import OPENING, validate
    from session import LocalSession

DEFAULT_MODEL = "thinkingmachines/Inkling-2-Small"
EFFORT = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "default": 0.9}
MAX_TOKENS = 8192            # per turn; the cookbook's Inkling recipes use the same
MIN_TURN_TOKENS = 1024       # less room than this for a turn ends the episode (context_length)
RESULT_RESERVE = 10000       # tokens kept free for one more check result (the largest in the packs is 10,644)
TRAIN_DIR = Path(__file__).resolve().parents[4] / "results" / "train"   # tolerant of shallow installs: only read
_SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def is_tinker_spec(model: str) -> bool:
    return model.startswith("tinker:")


def effort_value(effort: str | float | None) -> float:
    """A reasoning effort as the renderer's float in [0, 1): a preset name or a number."""
    if effort is None:
        return EFFORT["default"]
    if isinstance(effort, str) and effort in EFFORT:
        return EFFORT[effort]
    value = float(effort)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"effort must be one of {sorted(EFFORT)} or a number in [0, 1), got {effort!r}")
    return value


def parse_model(model: str) -> tuple[str, str | None]:
    """`tinker:<base model>[#<checkpoint>]` -> (base model, checkpoint reference or None)."""
    base, _, ref = model.removeprefix("tinker:").partition("#")
    return base, ref or None


def resolve_checkpoint(ref: str, train_dir: Path | None = None) -> str:
    """A checkpoint reference -> a `tinker://` sampler path. `<run>` is the last sampler checkpoint recorded in
    `results/train/<run>/checkpoints.jsonl`, `<run>:<name>` the one saved under that name."""
    if ref.startswith("tinker://"):
        return ref
    run, _, name = ref.partition(":")
    path = (train_dir or TRAIN_DIR) / run / "checkpoints.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no training run {run!r}: {path} does not exist")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("sampler_path") and (not name or r.get("name") == name)]
    if not rows:
        raise ValueError(f"{path} has no sampler checkpoint" + (f" named {name!r}" if name else ""))
    return rows[-1]["sampler_path"]


def context_window(base_model: str) -> int:
    """The model's context on Tinker: the `:peft:<n>` suffix names the long-context variants."""
    m = re.search(r":peft:(\d+)", base_model)
    return int(m.group(1)) if m else 65536


def make_renderer(base_model: str):
    from tinker_cookbook import model_info

    return get_renderer(model_info.get_recommended_renderer_name(base_model), get_tokenizer(base_model))


def plain_text(message: Message) -> str:
    """A parsed message's text and reasoning as one string (for transcripts and for turns that did not parse)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        parts.append(part.get("thinking") or part.get("text") or "")
    return "\n".join(p for p in parts if p)


class RailEnv(Env):
    """One RailSimEnv episode as a token-level environment: `initial_observation`, then `step(sampled tokens)` until
    the episode is done. It keeps the rollout record (`self.record`) as the other harnesses do."""

    def __init__(self, session, task_id: str, renderer, *, model: str = DEFAULT_MODEL, effort: str | float = "default",
                 max_turns: int = 12, max_tokens: int = MAX_TOKENS, context: int = 65536):
        self.env = session
        self.renderer = renderer
        self.effort = effort_value(effort)
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.context = context
        self.stop = renderer.get_stop_sequences()
        meta = session.reset(task_id)
        tools = [ToolSpec(name=t.name, description=t.description or "",
                          parameters=t.input_schema or {"type": "object", "properties": {}}) for t in session.tools]
        self.messages: list[Message] = renderer.create_conversation_prefix_with_tools(
            tools, system_prompt=meta["instructions"])
        budget = (f"\n\nYour budget: {max_turns} turns and a context window of {context:,} tokens that your reasoning "
                  f"and every tool result fill up. Each turn can use at most {max_tokens:,} tokens.")
        self.messages.append(Message(role="user", content=OPENING.format(situation=meta["situation"]) + budget))
        self.record: dict[str, Any] = {
            "task_id": task_id, "model": model, "effort": effort if isinstance(effort, str) else str(effort),
            "max_tokens": max_tokens, "context": context, "steps": [], "errors": [], "retries": [],
            "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                      "output_tokens": 0},
            "served_by": [model], "turn_log": [], "reward": 0.0, "end_reason": None, "final": None, "turns": 0}
        self.turn = 0
        self.idle = 0
        self.done = False
        self.ob: tinker.ModelInput | None = None
        # the two tokens that end a model turn (end of message, end of sampling), read off a rendered turn
        self.end_message, self.end_turn = self._render(
            [Message(role="user", content="."), Message(role="assistant", content=".")])[-2:]

    # ------------------------------------------------------------ observations

    def _render(self, messages: list[Message]) -> list[int]:
        return self.renderer.build_generation_prompt(messages, effort=self.effort).to_ints()

    def _count(self, text: str) -> int:
        return len(self.renderer.tokenizer.encode(text))

    def turn_budget(self, ob: tinker.ModelInput) -> int:
        """Output tokens the coming turn may use: the per-turn limit, or what the context still holds."""
        return max(1, min(self.max_tokens, self.context - ob.length))

    async def initial_observation(self):
        self.ob = tinker.ModelInput.from_ints(self._render(self.messages))
        return self.ob, self.stop

    def _next_observation(self, action: list[int], n_before: int) -> tinker.ModelInput:
        """The next prompt: the previous prompt, the tokens the model sampled, and the new messages' tokens. The
        model reads back exactly what it wrote (re-rendered from its parsed form, a tool call can differ in spacing
        and a cut turn has no parsed form at all), and every prompt extends the one before, which is what makes an
        episode one training sequence. A turn cut at the token limit is first closed as the model would have."""
        full = self._render(self.messages)
        seen = self.ob.to_ints() + list(action)
        if full[: len(seen)] == seen:
            return tinker.ModelInput.from_ints(full)
        if not action or action[-1] != self.end_turn:
            between = not action or action[-1] == self.end_message       # cut between two of its messages
            seen += ([] if between else [self.end_message]) + [self.end_turn]
        boundary = len(self._render(self.messages[: n_before + 1]))       # the history through the model's turn
        return tinker.ModelInput.from_ints(seen + full[boundary:])

    # ------------------------------------------------------------ one turn

    def _finish(self, reason: str) -> None:
        self.done = True
        self.record["end_reason"] = self.record["end_reason"] or reason

    def _result(self, reward: float = 0.0) -> StepResult:
        empty = tinker.ModelInput.from_ints([])
        return StepResult(reward=reward, episode_done=True, next_observation=empty, next_stop_condition=self.stop,
                          metrics={"turns": self.turn, "submitted": float(self.record["end_reason"] == "submitted")})

    async def step(self, action: list[int], *, extra=None) -> StepResult:
        assert self.ob is not None and not self.done, "step() after the episode ended, or before initial_observation()"
        self.turn += 1
        self.record["turns"] = self.turn
        left = self.max_turns - self.turn
        cut = (extra or {}).get("stop_reason") == "length"
        message, termination = self.renderer.parse_response(list(action))
        clean = termination.is_clean and not cut
        n_before = len(self.messages)
        calls = list(message.get("tool_calls") or []) if clean else []
        new_tokens = 0     # an estimate of what this turn's replies add to the context
        if clean:
            self.messages.append(message)
        else:   # the transcript keeps what the model wrote as text; nothing in it is run
            raw = self.renderer.tokenizer.decode(list(action))
            text = plain_text(message) if not isinstance(message.get("content"), str) else _SPECIAL.sub(" ", raw)
            self.messages.append(Message(role="assistant", content=text.strip() or "(no output)"))
        if not calls:
            tried = (not clean and "invoke_tool" in self.renderer.tokenizer.decode(list(action))) or bool(
                message.get("unparsed_tool_calls"))
            if not (tried and not cut):
                self.idle += 1
            if self.idle >= 2 or left <= 0:
                self._finish("no_tool_call" if self.idle >= 2 else "max_turns")
                return self._result()
            if cut:
                why = (f"Your output hit the {self.max_tokens:,}-token limit before you called a tool. Reason less "
                       f"per turn and let check_plan do the simulating. ")
            elif tried:
                why = "Your tool call could not be read: its arguments must be one valid JSON object. "
            else:
                why = ""
            nudge = f"{why}Call a tool: check_plan to test a draft, or submit_plan with your final plan."
        else:
            self.idle = 0
            reward = 0.0
            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments) if call.function.arguments.strip() else {}
                    err = validate(name, args)
                except json.JSONDecodeError as e:
                    args, err = None, f"tool arguments are not valid JSON ({e}); send the call again"
                err = err or ("the episode is over" if self.record["end_reason"] else None)
                if err:
                    text = f"Error: {err}"
                else:
                    text, step_done, reward_now, meta = self.env.call(name, args)
                    self.record["steps"].append({"turn": self.turn, "tool": name, "arguments": args, "result": text})
                    if step_done:
                        reward = float(reward_now or 0.0)
                        self.record["reward"] = reward
                        self.record["final"] = meta
                        self.record["end_reason"] = "submitted" if name == "submit_plan" else meta.get("end_reason")
                tool_message = Message(role="tool", content=text, name=name)
                if call.id:
                    tool_message["tool_call_id"] = call.id
                self.messages.append(tool_message)
                new_tokens += self._count(text) + 16
            if self.record["end_reason"]:
                self.done = True
                return self._result(reward)
            if left <= 0:
                self._finish("max_turns")
                return self._result()
            nudge = ""
        # What is left: turns, and room in the context. Another check needs this turn's output, its result and the
        # final turn's output; when that no longer fits, the coming turn is the last.
        room = self.context - (self.ob.length + len(action) + new_tokens + 64)
        tight = room < 2 * self.max_tokens + RESULT_RESERVE
        if tight and left > 1:
            left = 1
            self.max_turns = self.turn + 1
        note = (f"{nudge} " if nudge else "") + f"{left} turns left, about {max(0, room) // 1000}k tokens of context left."
        if left == 1:
            note += (" The context window is almost full." if tight else "") + \
                    " This is your last turn: call submit_plan now."
        self.messages.append(Message(role="user", content=note))
        self.ob = self._next_observation(list(action), n_before)
        if self.context - self.ob.length < MIN_TURN_TOKENS:
            self._finish("context_length")
            return self._result()
        return StepResult(reward=0.0, episode_done=False, next_observation=self.ob, next_stop_condition=self.stop)

    # ------------------------------------------------------------ the record

    def log_sample(self, ob: tinker.ModelInput, ac: TokensWithLogprobs, seconds: float) -> None:
        cached = min(ob.length, ac.prompt_cache_hit_tokens or 0)
        usage = self.record["usage"]
        usage["input_tokens"] += ob.length - cached
        usage["cache_read_input_tokens"] += cached
        usage["output_tokens"] += len(ac.tokens)
        self.record["turn_log"].append({"turn": self.turn + 1, "model": self.record["model"],
                                        "stop_reason": ac.stop_reason, "seconds": round(seconds, 1),
                                        "input_tokens": ob.length - cached, "cache_read": cached, "cache_write": 0,
                                        "output_tokens": len(ac.tokens)})

    def transcript(self) -> list[dict]:
        out = []
        for m in self.messages:
            row: dict[str, Any] = {"role": m["role"], "content": m.get("content")}
            if m.get("tool_calls"):
                row["tool_calls"] = [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                                     for c in m["tool_calls"]]
            if m.get("name"):
                row["name"] = m["name"]
            out.append(row)
        return out


RETRYABLE = (tinker.APIConnectionError, tinker.APITimeoutError, tinker.InternalServerError, tinker.RateLimitError)


async def play(env: RailEnv, policy, *, wall_clock_s: float | None = None, progress=None,
               attempts: int = 5) -> Trajectory:
    """Run `env` to the end with `policy` (a token completer). Returns the cookbook trajectory (what RL trains on);
    the rollout record is `env.record`. A sampling request that fails for the service's reasons is sent again
    (`record["retries"]`); anything else raises."""
    t0 = time.time()
    record = env.record
    ob, stop = await env.initial_observation()
    transitions: list[Transition] = []
    final_ob = ob
    try:
        while True:
            if wall_clock_s is not None and time.time() - t0 > wall_clock_s:
                record["end_reason"] = "wall_clock"
                break
            t_turn = time.time()
            for attempt in range(1, attempts + 1):
                try:
                    ac = await policy(ob, stop, max_tokens=env.turn_budget(ob))
                    break
                except RETRYABLE as e:
                    if attempt == attempts:
                        raise
                    record["retries"].append(f"turn {env.turn + 1} attempt {attempt}: {type(e).__name__}: {e}"[:500])
                    await asyncio.sleep(min(60, 5 * 2 ** (attempt - 1)))
            env.log_sample(ob, ac, time.time() - t_turn)
            result = await env.step(ac.tokens, extra={"stop_reason": ac.stop_reason})
            transitions.append(Transition(ob=ob, ac=ac, reward=result.reward, episode_done=result.episode_done,
                                          metrics=result.metrics))
            if progress is not None:
                log = record["turn_log"][-1]
                progress(f"{record['task_id']} turn {env.turn}: {log['seconds']:.0f}s in={ob.length} "
                         f"out={log['output_tokens']} stop={ac.stop_reason} "
                         f"calls={[s['tool'] for s in record['steps'] if s['turn'] == env.turn]}")
            final_ob = result.next_observation
            if result.episode_done:
                break
            ob, stop = result.next_observation, result.next_stop_condition
    finally:
        env.env.close()
        record["seconds"] = round(time.time() - t0, 1)
        record["messages"] = env.transcript()
    return Trajectory(transitions=transitions, final_ob=final_ob, stop_reason=record["end_reason"])


# ---------------------------------------------------------------- the eval harness (run_eval.py)

class TinkerClient:
    """What `run_episode` needs for one or more `tinker:` specs: the service client, and a sampling client and a
    renderer per model. Sampling clients are made on first use and kept."""

    def __init__(self, service_client=None, train_dir: Path | None = None):
        self.service = service_client or tinker.ServiceClient()
        self.train_dir = train_dir
        self._samplers: dict[str, Any] = {}
        self._renderers: dict[str, Any] = {}
        self._lock = threading.Lock()

    def sampler(self, model: str):
        with self._lock:
            if model not in self._samplers:
                base, ref = parse_model(model)
                if ref:
                    self._samplers[model] = self.service.create_sampling_client(
                        model_path=resolve_checkpoint(ref, self.train_dir)).result()
                else:
                    self._samplers[model] = self.service.create_sampling_client(base_model=base).result()
            return self._samplers[model]

    def renderer(self, model: str):
        base, _ = parse_model(model)
        with self._lock:
            if base not in self._renderers:
                self._renderers[base] = make_renderer(base)
            return self._renderers[base]


def make_tinker_client(**kw) -> TinkerClient:
    return TinkerClient(**kw)


_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_LOCK = threading.Lock()


def _loop() -> asyncio.AbstractEventLoop:
    """One event loop in a daemon thread for every episode `run_episode` plays: the eval runs episodes from a
    thread pool, and the sampling client's futures should all live on one loop."""
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            threading.Thread(target=_LOOP.run_forever, daemon=True, name="tinker-episodes").start()
        return _LOOP


def run_episode(base_url: str | None, task_id: str, model: str = f"tinker:{DEFAULT_MODEL}", effort: str | None = None,
                max_turns: int = 12, max_tokens: int | None = None, wall_clock_s: float | None = None, client=None,
                session=None, progress=None, temperature: float = 1.0, context: int | None = None) -> dict:
    """Play one episode with `model` (`tinker:<base model>[#<checkpoint>]`); returns a rollout record shaped like
    the other harnesses'. `base_url` is unused: the environment runs in this process (`LocalSession`), with the same
    tools, results and rubric as the server, because an episode is also what the RL loop steps through."""
    client = client or make_tinker_client()
    base, _ = parse_model(model)
    env = RailEnv(session if session is not None else LocalSession(), task_id, client.renderer(model), model=model,
                  effort=effort or "default", max_turns=max_turns, max_tokens=max_tokens or MAX_TOKENS,
                  context=context or context_window(base))
    policy = TinkerTokenCompleter(client.sampler(model), max_tokens=env.max_tokens, temperature=temperature,
                                  context_window=env.context)
    future = asyncio.run_coroutine_threadsafe(play(env, policy, wall_clock_s=wall_clock_s, progress=progress), _loop())
    future.result()
    return env.record
