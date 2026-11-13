"""The Tinker harness and the RL loop's pieces, with a scripted policy in place of the sampling client: no API call.

Skipped when `tinker` and `tinker-cookbook` are not installed (they are optional: see railsim-v1/train/README.md).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("tinker_cookbook")

TASKS = Path(__file__).resolve().parents[2] / "tasks"
if not os.environ.get("RAIL_TASKS_DIR"):
    os.environ["RAIL_TASKS_DIR"] = ":".join(str(TASKS / p) for p in ("rail-v1-eval", "rail-v1-train")
                                            if (TASKS / p / "manifest.json").is_file())

from tinker_cookbook.completers import TokensWithLogprobs  # noqa: E402
from tinker_cookbook.renderers import Message  # noqa: E402
from tinker_cookbook.renderers.base import ToolCall  # noqa: E402
from tinker_cookbook.rl.data_processing import trajectory_to_data  # noqa: E402
from tinker_cookbook.utils.cost_estimation import TokenCostRates, TokenUsage  # noqa: E402

from rail_core import load_pack  # noqa: E402
from rail_openenv import agent_tinker as at  # noqa: E402
from rail_openenv.session import LocalSession  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("RAIL_TASKS_DIR"), reason="no task pack built")

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "train"))
import train_rl  # noqa: E402


@pytest.fixture(scope="module")
def pack():
    return load_pack()


@pytest.fixture(scope="module")
def renderer():
    return at.make_renderer(at.DEFAULT_MODEL)


def call(name, args):
    """A model turn: a little reasoning, then one tool call."""
    return Message(role="assistant", content=[{"type": "thinking", "thinking": "Hold the late train at the loop."}],
                   tool_calls=[ToolCall(id=f"functions.{name}:0", function=ToolCall.FunctionBody(
                       name=name, arguments=json.dumps(args, separators=(",", ":"))))])


def tokens_of(env, message):
    """The tokens the model samples when it writes `message` after the current prompt."""
    before = env._render(env.messages)
    after = env._render(env.messages + [message])
    assert after[: len(before)] == before
    return after[len(before):]


class Scripted:
    """A token completer that plays a script: messages (sampled cleanly), or (tokens, stop_reason) pairs."""

    def __init__(self, env, script):
        self.env, self.script, self.budgets = env, list(script), []

    async def __call__(self, ob, stop, *, max_tokens=None):
        item = self.script.pop(0)
        tokens, reason = item if isinstance(item, tuple) else (tokens_of(self.env, item), "stop")
        self.budgets.append((ob.length, max_tokens))
        return TokensWithLogprobs(tokens=list(tokens), maybe_logprobs=[-0.5] * len(tokens), stop_reason=reason,
                                  sequence_id=None, prompt_cache_hit_tokens=min(ob.length, 100))


def play(pack, renderer, task, script, **kw):
    env = at.RailEnv(LocalSession(pack), task.task_id, renderer, **kw)
    policy = Scripted(env, script(env) if callable(script) else script)
    trajectory = asyncio.run(at.play(env, policy))
    return env, policy, trajectory


def test_episode_is_one_training_sequence(pack, renderer):
    task = pack.tasks[0]
    optimum = task.reference["optimal_plan"]
    env, policy, trajectory = play(pack, renderer, task, [
        call("check_plan", {"plan": []}), call("check_plan", {"plan": optimum}), call("submit_plan", {"plan": optimum})])
    rec = env.record
    assert rec["reward"] == pytest.approx(1.0) and rec["end_reason"] == "submitted" and rec["turns"] == 3
    assert [s["tool"] for s in rec["steps"]] == ["check_plan", "check_plan", "submit_plan"]
    assert rec["final"]["grade"]["feasible"] and rec["usage"]["cache_read_input_tokens"] == 300
    assert rec["usage"]["output_tokens"] == sum(len(t.ac.tokens) for t in trajectory.transitions)
    # every prompt is the one before, the sampled tokens, and what came back: so RL trains on one sequence
    for a, b in zip(trajectory.transitions, trajectory.transitions[1:]):
        seen = a.ob.to_ints() + a.ac.tokens
        assert b.ob.to_ints()[: len(seen)] == seen
    data = trajectory_to_data(trajectory, 0.25)
    last = trajectory.transitions[-1]
    assert len(data) == 1 and data[0].model_input.length == last.ob.length + len(last.ac.tokens) - 1
    assert sum(t.reward for t in trajectory.transitions) == pytest.approx(1.0)
    # the notes say what is left of both budgets, and the first message states them
    notes = [m["content"] for m in rec["messages"] if m["role"] == "user"]
    assert "12 turns" in notes[0] and "65,536" in notes[0]
    assert notes[1].startswith("11 turns left, about ") and "context left" in notes[1]
    assert all(m.get("name") for m in rec["messages"] if m["role"] == "tool")


def test_a_cut_turn_runs_nothing_and_two_end_the_episode(pack, renderer):
    task = pack.tasks[0]

    def script(env):
        whole = tokens_of(env, call("submit_plan", {"plan": task.reference["optimal_plan"]}))
        return [(whole[:-6], "length"), (whole[:-6], "length")]   # cut inside the tool call, twice

    env, _, trajectory = play(pack, renderer, task, script, max_tokens=4000)
    rec = env.record
    assert rec["reward"] == 0.0 and rec["end_reason"] == "no_tool_call" and rec["turns"] == 2 and rec["steps"] == []
    nudge = [m["content"] for m in rec["messages"] if m["role"] == "user"][1]
    assert "4,000-token limit" in nudge and "11 turns left" in nudge
    assert "<|" not in rec["messages"][-1]["content"]          # the transcript keeps the cut turn as plain text
    assert len(trajectory.transitions) == 2 and trajectory.transitions[-1].episode_done
    # the context keeps the tokens the model wrote, closed as a turn, so the episode is still one training sequence
    first, second = trajectory.transitions
    seen = first.ob.to_ints() + first.ac.tokens
    assert second.ob.to_ints()[: len(seen) + 2] == seen + [env.end_message, env.end_turn]
    assert env.renderer.tokenizer.decode(second.ob.to_ints()[len(seen) + 2:]).startswith("<|message_user|>")
    assert len(trajectory_to_data(trajectory, -0.1)) == 1


def test_a_tool_call_that_is_not_json_is_named_and_can_be_retried(pack, renderer):
    task = pack.tasks[0]
    optimum = task.reference["optimal_plan"]

    def script(env):
        good = tokens_of(env, call("check_plan", {"plan": []}))
        text = env.renderer.tokenizer.decode(good).replace('"plan":[]', '"plan":[{"train":')
        return [(env.renderer.tokenizer.encode(text), "stop"), call("submit_plan", {"plan": optimum})]

    env, _, trajectory = play(pack, renderer, task, script)
    rec = env.record
    assert rec["reward"] == pytest.approx(1.0) and rec["turns"] == 2
    assert [s["tool"] for s in rec["steps"]] == ["submit_plan"]           # the broken call ran nothing
    assert len(trajectory_to_data(trajectory, 0.5)) == 1                   # and the episode is one sequence
    assert "valid JSON" in [m["content"] for m in rec["messages"] if m["role"] == "user"][1]


def test_bad_arguments_are_refused_before_they_run(pack, renderer):
    task = pack.tasks[0]
    env, _, _ = play(pack, renderer, task, [call("check_plan", {"plans": []}), call("fly", {}),
                                            call("submit_plan", {"plan": []})])
    tool = [m["content"] for m in env.record["messages"] if m["role"] == "tool"]
    assert tool[0].startswith('Error: missing "plan"') and tool[1].startswith("Error: unknown tool")
    assert [s["tool"] for s in env.record["steps"]] == ["submit_plan"] and env.record["end_reason"] == "submitted"


def test_a_full_context_makes_the_next_turn_the_last(pack, renderer):
    task = pack.tasks[0]
    first = at.RailEnv(LocalSession(pack), task.task_id, renderer)
    prompt = asyncio.run(first.initial_observation())[0].length
    # room for the opening, one check and its result, but not for a second check beside the final answer
    context = prompt + 2 * 2000 + at.RESULT_RESERVE + 1000
    env, policy, _ = play(pack, renderer, task, [call("check_plan", {"plan": []}), call("check_plan", {"plan": []})],
                          max_tokens=2000, context=context)
    rec = env.record
    notes = [m["content"] for m in rec["messages"] if m["role"] == "user"]
    assert notes[1].startswith("1 turns left") and "almost full" in notes[1] and "call submit_plan now" in notes[1]
    assert rec["end_reason"] == "max_turns" and rec["turns"] == 2 and rec["reward"] == 0.0
    assert all(length + budget <= context for length, budget in policy.budgets)
    # no room for a turn at all: the episode ends there
    env, _, _ = play(pack, renderer, task, [call("get_situation", {})], context=prompt + 1500)
    assert env.record["end_reason"] == "context_length" and env.record["turns"] == 1


def test_specs_and_checkpoints(tmp_path):
    assert at.parse_model("tinker:thinkingmachines/Inkling-2-Small") == ("thinkingmachines/Inkling-2-Small", None)
    assert at.parse_model("tinker:org/M:peft:262144#rl-a:000006") == ("org/M:peft:262144", "rl-a:000006")
    assert at.context_window("org/M:peft:262144") == 262144 and at.context_window("org/M") == 65536
    assert at.effort_value("high") == 0.9 and at.effort_value("0.35") == 0.35 and at.effort_value(None) == 0.9
    with pytest.raises(ValueError):
        at.effort_value("1.5")
    run = tmp_path / "rl-a"
    run.mkdir()
    rows = [{"name": "000003", "batch": 3, "state_path": "tinker://r/weights/000003",
             "sampler_path": "tinker://r/sampler_weights/000003"},
            {"name": "000006", "batch": 6, "state_path": "tinker://r/weights/000006"},        # sampler deleted
            {"name": "000009-final", "batch": 9, "sampler_path": "tinker://r/sampler_weights/000009-final"}]
    (run / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert at.resolve_checkpoint("rl-a", tmp_path) == "tinker://r/sampler_weights/000009-final"
    assert at.resolve_checkpoint("rl-a:000003", tmp_path) == "tinker://r/sampler_weights/000003"
    assert at.resolve_checkpoint("tinker://x/sampler_weights/y") == "tinker://x/sampler_weights/y"
    with pytest.raises(ValueError):
        at.resolve_checkpoint("rl-a:000006", tmp_path)
    with pytest.raises(FileNotFoundError):
        at.resolve_checkpoint("nope", tmp_path)


def test_training_batches_and_held_out_tasks(pack):
    train, val = train_rl.split_tasks(pack, 3, seed=0)
    assert len(val) == 3 and not {t.task_id for t in val} & {t.task_id for t in train}
    assert all(t.split == "train" for t in train + val)
    assert [t.task_id for t in train_rl.split_tasks(pack, 3, seed=0)[1]] == [t.task_id for t in val]
    n = len(train)
    seen = [t.task_id for step in range(n) for t in train_rl.batch_tasks(train, step, 2, seed=0)]
    assert sorted(seen[:n]) == sorted(t.task_id for t in train)        # one pass plays every task once
    assert seen[:n] != seen[n:]                                         # and the next pass is reshuffled
    assert [t.task_id for t in train_rl.batch_tasks(train, 5, 2, seed=0)] == seen[10:12]   # a function of the step


def test_budget_stops_before_the_step_that_would_pass_it():
    rates = TokenCostRates(prefill_usd_per_million=2.0, cached_prefill_usd_per_million=1.0,
                           sample_usd_per_million=4.0, train_usd_per_million=10.0)
    guess = TokenUsage(sampled=100_000, trained=500_000)                 # 5.4 at these made-up rates
    budget = train_rl.Budget(rates, max_credits=12.0)
    assert budget.refuses(guess) is None
    budget.add(TokenUsage(sampled=50_000, prefill_cached=1_000_000, trained=400_000))       # 5.2
    budget.add(TokenUsage(sampled=10_000), a_step=False)                 # validation: counted, not a step's size
    assert budget.credits() == pytest.approx(5.24) and budget.refuses(guess) is None       # 5.24 + 1.1 * 5.2 < 12
    budget.add(TokenUsage(sampled=50_000, trained=100_000))              # 1.2: a smaller step
    assert "credits" in budget.refuses(guess)                            # 6.44 + 1.1 * 5.2 > 12: the largest counts
    tokens = train_rl.Budget(None, max_train_tokens=900_000)
    assert tokens.refuses(guess) is None and tokens.credits() == 0.0
    tokens.add(TokenUsage(trained=450_000))
    assert tokens.refuses(guess) is None
    tokens.add(TokenUsage(trained=100_000))
    assert "train tokens" in tokens.refuses(guess)


def test_prune_keeps_the_newest_state_and_the_best_sampler(tmp_path):
    rows = [{"name": f"{i:06d}", "batch": i, "val_reward": v, "state_path": f"tinker://r/weights/{i}",
             "sampler_path": f"tinker://r/sampler_weights/{i}"}
            for i, v in ((3, None), (6, 0.31), (9, None), (12, 0.22), (15, None))]
    (tmp_path / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    class Rest:
        deleted = []

        async def delete_checkpoint_from_tinker_path(self, path):
            self.deleted.append(path)

    asyncio.run(train_rl.prune(Rest(), tmp_path, keep_samplers=2))
    kept = train_rl.load_rows(tmp_path / "checkpoints.jsonl")
    assert [r["name"] for r in kept if r.get("state_path")] == ["000015"]
    assert [r["name"] for r in kept if r.get("sampler_path")] == ["000006", "000012", "000015"]   # best, and newest two
    assert len(Rest.deleted) == 4 + 2 and kept[0]["deleted"] == ["tinker://r/weights/3", "tinker://r/sampler_weights/3"]
    asyncio.run(train_rl.prune(Rest(), tmp_path, keep_samplers=2))                               # nothing more to do
    assert len(Rest.deleted) == 6
    asyncio.run(train_rl.prune(Rest(), tmp_path, keep_samplers=1))                               # room for a save
    kept = train_rl.load_rows(tmp_path / "checkpoints.jsonl")
    assert [r["name"] for r in kept if r.get("sampler_path")] == ["000006", "000015"] and len(Rest.deleted) == 7
    assert [r["name"] for r in kept if r.get("state_path")] == ["000015"]                        # still resumable


def test_finished_episodes_survive_a_restart(pack, renderer, tmp_path):
    """An episode cached on disk comes back as it was, without the old session's sampling ids, marked as reused."""
    task = pack.tasks[0]
    env, _, trajectory = play(pack, renderer, task, [call("submit_plan", {"plan": task.reference["optimal_plan"]})])
    for t in trajectory.transitions:
        t.ac.sequence_id = "seq-from-the-old-session"
    row = train_rl.episode_row(env.record, task, 7, "train")
    path = train_rl.cache_dir(tmp_path, 7, "train") / f"{task.task_id}-3.pkl"
    train_rl.save_cached(path, (trajectory, row, train_rl.episode_usage(env.record)))
    again, row2, usage = train_rl.load_cached(path)
    assert row2["reused"] and row2["reward"] == row["reward"] == pytest.approx(1.0)
    assert usage == train_rl.episode_usage(env.record)
    assert [t.ac.tokens for t in again.transitions] == [t.ac.tokens for t in trajectory.transitions]
    assert all(t.ac.sequence_id is None for t in again.transitions)
    assert len(trajectory_to_data(again, 0.5)) == 1
    path.write_bytes(b"cut off mid-write")
    assert train_rl.load_cached(path) is None                # a broken file is sampled again, not trusted
