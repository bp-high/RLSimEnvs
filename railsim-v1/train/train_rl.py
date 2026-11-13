"""Reinforcement learning on RailSimEnv's train split through the Tinker API (Inkling 2 Small by default).

    cd railsim-v1
    envs/rail_ops/.venv/bin/python train/train_rl.py --run rl-a --max-credits 250

One step: draw `--groups` train tasks, play each `--group-size` times with the current weights (the episode of
`rail_openenv.agent_tinker`, the one the eval plays), centre the rewards within each task's group, and take one
policy-gradient step on every token the model sampled (`importance_sampling`, the cookbook's on-policy loss). An
episode is one training sequence: each turn's prompt extends the one before, token for token. Groups whose rewards are
all equal carry no signal and are not trained on.

Reasoning effort is part of the prompt, so train at the effort the model will be evaluated at: `--effort high`, the
model's default, here and in the eval.

Tasks: the train pack only. `--val-tasks` of them (seeded) are held out and played once every `--val-every` steps, so
there is a learning curve that never touches the eval split. Before and after, run the eval itself:

    envs/rail_ops/.venv/bin/python eval/run_eval.py --run <date>-inkling-2-small-base-eval45 \
        --models tinker:thinkingmachines/Inkling-2-Small@high --split eval --concurrency 45
    envs/rail_ops/.venv/bin/python eval/run_eval.py --run <date>-inkling-2-small-rl-a-eval45 \
        --models "tinker:thinkingmachines/Inkling-2-Small#rl-a@high" --split eval --concurrency 45

Spending: every step counts the tokens it sampled, prefilled (cached and not) and trained on, and `--max-credits`
stops the run before a step that would pass it, at the rates Tinker publishes (fetched when the run starts, never
stored here). `--max-train-tokens` is the same guard in tokens. What is written to disk is tokens and time.

Files, in `results/train/<run>/`: `config.json`, `metrics.jsonl` (a row per step and per validation),
`episodes.jsonl` (a row per episode), `checkpoints.jsonl` (the `tinker://` paths; the eval's `#<run>` reads it),
`rollouts/` (a few full transcripts per step, git-ignored). The run resumes from its last saved state when started
again with the same `--run`.

Checkpoints are large and a free Tinker plan holds 100 GB: at rank 32 a training state of this model is 50 GB and
a sampler checkpoint 17 GB (measured), so two states never fit side by side. Hence rank 8 by default (a quarter of
that), and only the newest training state is kept, and of the sampler checkpoints (saved at validations and at the
end) the newest `--keep-samplers` and the best on validation. The final ones have no expiry; the others expire after
`--ttl-days`.

Needs `tinker` and `tinker-cookbook` (see README.md here) and `TINKER_API_KEY`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils.cost_estimation import TokenUsage, fetch_model_pricing

from rail_core import load_pack
from rail_openenv import agent_tinker as at
from rail_openenv.session import LocalSession

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("sampled", "prefill_cached", "prefill_uncached", "trained")


def split_tasks(pack, n_val: int, seed: int) -> tuple[list, list]:
    """(training tasks, held-out validation tasks), both from the train split; the same for a given seed."""
    tasks = sorted((t for t in pack.tasks if t.split == "train"), key=lambda t: t.task_id)
    val_ids = set(random.Random(seed).sample([t.task_id for t in tasks], n_val)) if n_val else set()
    return [t for t in tasks if t.task_id not in val_ids], [t for t in tasks if t.task_id in val_ids]


def batch_tasks(tasks: list, step: int, groups: int, seed: int) -> list:
    """The tasks of one step: the training tasks in a seeded order, reshuffled every pass, `groups` at a time.
    A function of the step alone, so a resumed run draws what the interrupted one would have."""
    out = []
    i = step * groups
    while len(out) < groups:
        epoch, pos = divmod(i, len(tasks))
        order = list(tasks)
        random.Random(f"{seed}/{epoch}").shuffle(order)
        out.append(order[pos])
        i += 1
    return out


class Budget:
    """Tokens used so far, and whether another step fits under the caps."""

    def __init__(self, rates=None, max_credits: float | None = None, max_train_tokens: int | None = None,
                 used: TokenUsage = TokenUsage()):
        self.rates = rates
        self.max_credits = max_credits
        self.max_train_tokens = max_train_tokens
        self.used = used
        self.largest_step = TokenUsage()

    def add(self, usage: TokenUsage, a_step: bool = True) -> None:
        self.used += usage
        size = lambda u: (self.credits(u), u.trained, u.sampled)   # noqa: E731
        if a_step and size(usage) >= size(self.largest_step):
            self.largest_step = usage

    def credits(self, usage: TokenUsage | None = None) -> float:
        """The usage at Tinker's published rates (0.0 when they could not be fetched)."""
        return self.rates.estimate_usd(self.used if usage is None else usage) if self.rates else 0.0

    def refuses(self, estimate: TokenUsage) -> str | None:
        """Why the next step, assumed as large as the largest so far (or `estimate` before the first), does not
        fit; None when it does."""
        step = self.largest_step if self.largest_step != TokenUsage() else estimate
        if self.max_credits is not None and self.credits() + 1.1 * self.credits(step) > self.max_credits:
            return (f"credits: {self.credits():.2f} used and a step takes about {self.credits(step):.2f}; "
                    f"the cap is {self.max_credits:.2f}")
        if self.max_train_tokens is not None and self.used.trained + step.trained > self.max_train_tokens:
            return f"train tokens: {self.used.trained:,} used; the cap is {self.max_train_tokens:,}"
        return None

    def line(self) -> str:
        u = self.used
        credits = f"; about {self.credits():.2f} credits" if self.rates else ""
        return (f"{u.sampled:,} sampled, {u.prefill_uncached:,} + {u.prefill_cached:,} cached prefilled, "
                f"{u.trained:,} trained{credits}")


def episode_usage(record: dict) -> TokenUsage:
    u = record["usage"]
    return TokenUsage(sampled=u["output_tokens"], prefill_cached=u["cache_read_input_tokens"],
                      prefill_uncached=u["input_tokens"])


def episode_row(record: dict, task, step: int, phase: str) -> dict:
    g = (record.get("final") or {}).get("grade") or {}
    u = record["usage"]
    return {"step": step, "phase": phase, "task_id": record["task_id"], "difficulty": task.difficulty,
            "reward": record["reward"] or 0.0, "end_reason": record["end_reason"],
            "feasible": bool(g.get("feasible")), "cost": g.get("cost"), "optimal_cost": task.reference["optimal_cost"],
            "turns": record["turns"], "checks": sum(1 for s in record["steps"] if s["tool"] == "check_plan"),
            "cut_turns": sum(1 for t in record["turn_log"] if t["stop_reason"] == "length"),
            "seconds": record.get("seconds"), "output_tokens": u["output_tokens"],
            "input_tokens": u["input_tokens"] + u["cache_read_input_tokens"],
            "cache_read_tokens": u["cache_read_input_tokens"], "retries": len(record.get("retries") or []),
            "error": (record.get("errors") or [None])[0]}


def summarize(rows: list[dict]) -> dict:
    """What a batch of episodes did, for the log."""
    if not rows:
        return {"episodes": 0}
    n = len(rows)
    out = {"episodes": n, "reward_mean": round(statistics.mean(r["reward"] for r in rows), 4),
           "submitted": round(sum(r["end_reason"] == "submitted" for r in rows) / n, 3),
           "feasible": round(sum(r["feasible"] for r in rows) / n, 3),
           "optimal": sum(1 for r in rows if r["feasible"] and r["cost"] == r["optimal_cost"]),
           "turns_mean": round(statistics.mean(r["turns"] for r in rows), 2),
           "checks_mean": round(statistics.mean(r["checks"] for r in rows), 2),
           "cut_turns": round(sum(r["cut_turns"] for r in rows) / max(1, sum(r["turns"] for r in rows)), 3),
           "output_tokens_mean": round(statistics.mean(r["output_tokens"] for r in rows)),
           "seconds_median": statistics.median(r["seconds"] or 0 for r in rows),
           "end_reasons": dict(Counter(r["end_reason"] for r in rows))}
    for tier in sorted({r["difficulty"] for r in rows}):
        out[f"reward_{tier}"] = round(statistics.mean(r["reward"] for r in rows if r["difficulty"] == tier), 4)
    return out


class Runner:
    def __init__(self, args, pack, run_dir: Path):
        self.args = args
        self.pack = pack
        self.run_dir = run_dir
        self.renderer = at.make_renderer(args.model)
        self.context = at.context_window(args.model)
        self.sem = asyncio.Semaphore(args.concurrency)
        self.label = f"tinker:{args.model}"

    async def episode(self, task, sampler, step: int, phase: str, keep: int) -> tuple:
        """Play one episode: (trajectory or None, episode row, token usage)."""
        a = self.args
        async with self.sem:
            env = at.RailEnv(LocalSession(self.pack), task.task_id, self.renderer, model=self.label, effort=a.effort,
                             max_turns=a.max_turns, max_tokens=a.max_tokens, context=self.context)
            policy = TinkerTokenCompleter(sampler, max_tokens=a.max_tokens, temperature=a.temperature,
                                          context_window=self.context)
            try:
                trajectory = await at.play(env, policy, wall_clock_s=a.wall_clock)
            except Exception as e:   # a failed episode is dropped from its group; what it sampled is still paid for
                trajectory = None
                env.record["errors"].append(f"{type(e).__name__}: {e}"[:500])
                env.record["end_reason"] = "error"
        record = env.record
        if record["end_reason"] == "wall_clock":
            trajectory = None
        if keep:
            path = self.run_dir / "rollouts" / f"step-{step:04d}-{phase}" / f"{task.task_id}-{keep}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str))
        return trajectory, episode_row(record, task, step, phase), episode_usage(record)

    async def collect(self, tasks: list, sampler, step: int, phase: str, group_size: int):
        """Play every task `group_size` times, all at once: (per-task lists of trajectories, episode rows, token
        usage)."""
        keep = self.args.keep_transcripts
        jobs = [self.episode(t, sampler, step, phase, (i * group_size + j + 1) if i * group_size + j < keep else 0)
                for i, t in enumerate(tasks) for j in range(group_size)]
        results = await asyncio.gather(*jobs)
        usage = TokenUsage()
        for _, _, u in results:
            usage += u
        groups = [[r[0] for r in results[i * group_size:(i + 1) * group_size]] for i in range(len(tasks))]
        return groups, [r[1] for r in results], usage


def append(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


async def prune(rest, run_dir: Path, keep_samplers: int) -> None:
    """Delete what is no longer needed on Tinker: every training state but the newest, and sampler checkpoints
    other than the newest `keep_samplers` and the best on validation. Deleted paths are recorded, so they are not
    asked for again and the eval's `#<run>` never resolves to one. Called before a save too (with one sampler
    fewer), so the new checkpoint has room."""
    path = run_dir / "checkpoints.jsonl"
    rows = load_rows(path)
    states = [r for r in rows if r.get("state_path")]
    samplers = [r for r in rows if r.get("sampler_path")]
    scored = [r for r in samplers if r.get("val_reward") is not None]
    best = max(scored, key=lambda r: (r["val_reward"], r["batch"]))["name"] if scored else None
    keep = {r["name"] for r in samplers[len(samplers) - keep_samplers:]} | {best} if keep_samplers > 0 else {best}
    changed = False
    for r in rows:
        drop = []
        if r.get("state_path") and r is not states[-1]:
            drop.append("state_path")
        if r.get("sampler_path") and r["name"] not in keep:
            drop.append("sampler_path")
        for key in drop:
            try:
                await rest.delete_checkpoint_from_tinker_path(r[key])
            except Exception as e:   # already expired or deleted: forget it all the same
                print(f"  could not delete {r[key]}: {type(e).__name__}: {e}", flush=True)
            r.setdefault("deleted", []).append(r.pop(key))
            changed = True
    if changed:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))


async def train(args) -> None:
    run_dir = ROOT / "results" / "train" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    pack = load_pack()
    train_tasks, val_tasks = split_tasks(pack, args.val_tasks, args.seed)
    config = {k: v for k, v in vars(args).items() if k not in ("max_credits", "max_train_tokens")}
    config.update(train_tasks=len(train_tasks), val_task_ids=[t.task_id for t in val_tasks],
                  context=at.context_window(args.model), renderer="tml_v0" if "Inkling" in args.model else None)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1) + "\n")

    service = tinker.ServiceClient()
    rest = service.create_rest_client()
    resume = checkpoint_utils.get_last_checkpoint(str(run_dir))
    if resume:
        training = await service.create_training_client_from_state_with_optimizer(resume.state_path)
        start = resume.batch
        used = TokenUsage(**{k: (resume.get("tokens") or {}).get(k, 0) for k in KINDS})
        print(f"resuming {args.run} at step {start} from {resume.state_path}", flush=True)
    else:
        training = await service.create_lora_training_client(base_model=args.model, rank=args.lora_rank)
        start, used = 0, TokenUsage()
    budget = Budget(fetch_model_pricing(args.model), args.max_credits, args.max_train_tokens, used)
    if args.max_credits is not None and budget.rates is None:
        raise SystemExit("--max-credits needs Tinker's published rates and they could not be fetched; "
                         "use --max-train-tokens instead")
    runner = Runner(args, pack, run_dir)
    per_step = args.groups * args.group_size
    first_guess = TokenUsage(sampled=22_000 * per_step, prefill_uncached=130_000 * per_step,
                             trained=runner.context * per_step)
    print(f"{args.run}: {len(train_tasks)} training tasks, {len(val_tasks)} held out; {args.groups} x "
          f"{args.group_size} episodes a step, up to {args.steps} steps; used so far: {budget.line()}", flush=True)

    async def save(step: int, val_reward: float | None, kind: str, final: bool = False) -> None:
        """`state` right after each training step (what a resume starts from), `sampler` after a validation (what
        the eval reads, with its validation reward), `both` at the end."""
        name = f"{step:06d}" + ("-final" if final else "")
        state = {"batch": step, "tokens": {k: getattr(budget.used, k) for k in KINDS}, "val_reward": val_reward,
                 "final": final}
        if kind != "state":
            await prune(rest, run_dir, args.keep_samplers - 1)
        await checkpoint_utils.save_checkpoint_async(
            training, name=name, log_path=str(run_dir), loop_state=state, kind=kind,
            ttl_seconds=None if final else int(args.ttl_days * 86400))
        await prune(rest, run_dir, args.keep_samplers)

    async def validate(sampler, step: int) -> float | None:
        if not val_tasks:
            return None
        t0 = time.time()
        _, rows, usage = await runner.collect(val_tasks, sampler, step, "val", 1)
        budget.add(usage, a_step=False)
        for r in rows:
            append(run_dir / "episodes.jsonl", r)
        s = summarize(rows)
        append(run_dir / "metrics.jsonl", {"step": step, "phase": "val", **s, "seconds": round(time.time() - t0),
                                           "tokens": {k: getattr(usage, k) for k in KINDS}})
        print(f"[val  {step:3d}] reward {s['reward_mean']:.3f}  submitted {s['submitted']:.0%}  feasible "
              f"{s['feasible']:.0%}  turns {s['turns_mean']}  cut turns {s['cut_turns']:.0%}", flush=True)
        return s["reward_mean"]

    sampler = await training.save_weights_and_get_sampling_client()
    step, stopped = start, None
    validated = {r["step"] for r in load_rows(run_dir / "metrics.jsonl") if r.get("phase") == "val"}
    for step in range(start, args.steps):
        val_reward = None
        if args.val_every and step % args.val_every == 0 and step not in validated:
            val_reward = await validate(sampler, step)
            if step > 0:
                await save(step, val_reward, "sampler")
        stopped = budget.refuses(first_guess)
        if stopped:
            print(f"stopping before step {step}: {stopped}", flush=True)
            break
        t0 = time.time()
        tasks = batch_tasks(train_tasks, step, args.groups, args.seed)
        groups, rows, usage = await runner.collect(tasks, sampler, step, "train", args.group_size)
        t_sample = time.time() - t0
        for r in rows:
            append(run_dir / "episodes.jsonl", r)
        kept = []
        for trajectories in groups:
            trajectories = [t for t in trajectories if t is not None and t.transitions]
            rewards = [sum(tr.reward for tr in t.transitions) for t in trajectories]
            if len(trajectories) >= 2 and len(set(rewards)) > 1:
                kept.append(TrajectoryGroup(trajectories, [0.0] * len(trajectories), [{} for _ in trajectories]))
        metrics: dict = {}
        t1 = time.time()
        if kept:
            data, _ = assemble_training_data(kept, compute_advantages(kept))
            usage += TokenUsage(trained=sum(d.model_input.length for d in data))
            await train_step(data, training, args.lr, args.substeps, args.loss, metrics=metrics)
            sampler = await training.save_weights_and_get_sampling_client()
        budget.add(usage)
        if (step + 1) % args.save_every == 0:   # before anything else: a restart from here loses nothing trained
            await save(step + 1, None, "state")
        s = summarize(rows)
        append(run_dir / "metrics.jsonl", {
            "step": step, "phase": "train", **s, "groups_trained": len(kept), "groups_dropped": len(groups) - len(kept),
            "sequences": len(data) if kept else 0, "lr": args.lr, "seconds_sampling": round(t_sample),
            "seconds_training": round(time.time() - t1), "tokens": {k: getattr(usage, k) for k in KINDS},
            "tokens_total": {k: getattr(budget.used, k) for k in KINDS},
            "optim": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}})
        print(f"[step {step:3d}] reward {s['reward_mean']:.3f}  submitted {s['submitted']:.0%}  feasible "
              f"{s['feasible']:.0%}  turns {s['turns_mean']}  cut turns {s['cut_turns']:.0%}  trained "
              f"{len(kept)}/{len(groups)} groups  {t_sample:.0f}s + {time.time() - t1:.0f}s  | {budget.line()}",
              flush=True)
        step += 1
    done = step if stopped else args.steps
    if done > start:
        val_reward = await validate(sampler, done) if args.val_every and done not in validated else None
        await save(done, val_reward, "both", final=True)
        print(f"saved the final checkpoint at step {done}; used: {budget.line()}", flush=True)
        print(f'evaluate it with --models "tinker:{args.model}#{args.run}@{args.effort}"', flush=True)
    else:
        print("nothing to do: the run had already reached its last step or its budget", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/train/<run>/; an existing run resumes")
    ap.add_argument("--model", default=at.DEFAULT_MODEL, help="the Tinker id of the base model")
    ap.add_argument("--effort", default="high", help=f"reasoning effort: one of {sorted(at.EFFORT)} or a number in [0, 1)")
    ap.add_argument("--groups", type=int, default=16, help="tasks per step")
    ap.add_argument("--group-size", type=int, default=4, help="episodes per task; rewards are centred within a task")
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--lr", type=float, default=1e-4, help="run rl-a: 4e-5 moved nothing in six steps, 1e-4 did")
    ap.add_argument("--lora-rank", type=int, default=8, help="checkpoint size grows with it: see above")
    ap.add_argument("--loss", default="importance_sampling", choices=["importance_sampling", "ppo", "cispo"])
    ap.add_argument("--substeps", type=int, default=1, help="optimizer steps per batch")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=at.MAX_TOKENS, help="output tokens per turn")
    ap.add_argument("--wall-clock", type=float, default=1800.0, help="seconds before an episode is dropped")
    ap.add_argument("--concurrency", type=int, default=64, help="episodes in flight")
    ap.add_argument("--val-tasks", type=int, default=16, help="train-split tasks held out for validation")
    ap.add_argument("--val-every", type=int, default=6, help="steps between validations (0: none)")
    ap.add_argument("--save-every", type=int, default=1, help="steps between saved states (a resume starts there)")
    ap.add_argument("--keep-samplers", type=int, default=2, help="sampler checkpoints kept besides the best")
    ap.add_argument("--ttl-days", type=float, default=7.0, help="expiry of every checkpoint but the final ones")
    ap.add_argument("--keep-transcripts", type=int, default=4, help="full transcripts written per step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-credits", type=float, default=None, help="stop before a step that would pass this")
    ap.add_argument("--max-train-tokens", type=int, default=None, help="stop before a step that would pass this")
    args = ap.parse_args()
    at.effort_value(args.effort)
    if args.max_credits is None and args.max_train_tokens is None:
        ap.error("set a budget: --max-credits or --max-train-tokens")
    asyncio.run(train(args))


if __name__ == "__main__":
    main()
