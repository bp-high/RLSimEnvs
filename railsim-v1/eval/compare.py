"""Two runs on the same tasks, task by task: a model before and after training, or two models.

    python eval/compare.py <run before> <run after>     # reads results/rollouts/<run>/index.json, prints markdown

The difference is paired: each task's reward after minus its reward before, with a 95 % bootstrap interval over
tasks. With several rollouts of a task in a run, their mean is the task's reward. Pairing removes the spread between
easy and hard tasks, which is most of the spread in a single run's mean.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(run: str) -> dict[str, list[dict]]:
    eps = json.loads((ROOT / "results" / "rollouts" / run / "index.json").read_text())["episodes"]
    by = defaultdict(list)
    for e in eps:
        by[e["task_id"]].append(e)
    return by


def interval(diffs: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = sorted(diffs)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summary(by: dict[str, list[dict]], tasks: list[str]) -> dict:
    eps = [e for t in tasks for e in by[t]]
    n = len(eps)
    return {"episodes": n, "mean": statistics.mean(statistics.mean(e["reward"] for e in by[t]) for t in tasks),
            "submitted": sum(e["submitted"] for e in eps) / n, "feasible": sum(e["feasible"] for e in eps) / n,
            "optimal": sum(1 for e in eps if e["feasible"] and e["cost"] == e["optimal_cost"]) / n,
            "turns": statistics.median(e["agent_turns"] for e in eps),
            "output_tokens": sum(e["output_tokens"] for e in eps) / n}


def compare(before: str, after: str) -> str:
    a, b = load(before), load(after)
    tasks = sorted(set(a) & set(b))
    if not tasks:
        raise SystemExit("the two runs share no task")
    sa, sb = summary(a, tasks), summary(b, tasks)
    reward = lambda by, t: statistics.mean(e["reward"] for e in by[t])   # noqa: E731
    diffs = [reward(b, t) - reward(a, t) for t in tasks]
    lo, hi = interval(diffs)
    out = [f"## {before} -> {after}", "",
           f"{len(tasks)} tasks in both runs. Mean reward {sa['mean']:.3f} -> **{sb['mean']:.3f}**: "
           f"{statistics.mean(diffs):+.3f} per task (95% interval {lo:+.3f} to {hi:+.3f}, paired bootstrap over tasks); "
           f"better on {sum(d > 1e-9 for d in diffs)}, worse on {sum(d < -1e-9 for d in diffs)}, "
           f"the same on {sum(abs(d) <= 1e-9 for d in diffs)}.", "",
           "| | before | after |", "|---|---:|---:|"]
    for label, key, fmt in (("episodes", "episodes", "{:.0f}"), ("mean reward", "mean", "{:.3f}"),
                            ("submitted a plan", "submitted", "{:.0%}"), ("legal plan", "feasible", "{:.0%}"),
                            ("at the proven optimum", "optimal", "{:.0%}"), ("median turns", "turns", "{:.0f}"),
                            ("output tokens per episode", "output_tokens", "{:,.0f}")):
        out.append(f"| {label} | {fmt.format(sa[key])} | {fmt.format(sb[key])} |")
    tiers = sorted({a[t][0]["difficulty"] for t in tasks})
    out += ["", "| tier | tasks | before | after | difference |", "|---|---:|---:|---:|---:|"]
    for tier in tiers:
        ts = [t for t in tasks if a[t][0]["difficulty"] == tier]
        ra, rb = statistics.mean(reward(a, t) for t in ts), statistics.mean(reward(b, t) for t in ts)
        out.append(f"| {tier} | {len(ts)} | {ra:.3f} | {rb:.3f} | {rb - ra:+.3f} |")
    out += ["", "| task | tier | before | after | difference |", "|---|---|---:|---:|---:|"]
    for t in sorted(tasks, key=lambda t: reward(b, t) - reward(a, t)):
        out.append(f"| {t} | {a[t][0]['difficulty']} | {reward(a, t):.3f} | {reward(b, t):.3f} | "
                   f"{reward(b, t) - reward(a, t):+.3f} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(compare(sys.argv[1], sys.argv[2]))
