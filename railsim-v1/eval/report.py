"""Markdown board for a run: mean reward with a 95 % bootstrap CI, per tier, feasibility, how often the plan matched
the optimum or beat the greedy re-plan, episodes that ended without a submit, and the tokens and time it took.

    python eval/report.py <run>          # reads results/rollouts/<run>/index.json, prints the board
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("real", "blockage", "loop", "compound")


def ci(xs, n=2000, seed=0):
    xs = sorted(xs)                          # the same interval whatever order the episodes finished in
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def board(eps: list[dict]) -> str:
    by = defaultdict(list)
    for e in eps:
        by[e["model"]].append(e)
    rows = sorted(by.items(), key=lambda kv: -statistics.mean(e["reward"] for e in kv[1]))
    tiers = [t for t in TIERS if any(e["difficulty"] == t for e in eps)]
    out = ["| Model | n | Mean reward (95% CI) | " + " | ".join(tiers)
           + " | Feasible | Optimal | Beat greedy | Beat the real dispatchers (real tier) | Ended without submit | Median turns |",
           "|---|---:|---|" + "---:|" * len(tiers) + "---:|---:|---:|---:|---:|---:|"]
    for m, es in rows:
        r = [e["reward"] for e in es]
        lo, hi = ci(r)
        per = []
        for t in tiers:
            xs = [e["reward"] for e in es if e["difficulty"] == t]
            per.append(f"{statistics.mean(xs):.3f}" if xs else "-")
        feas = sum(e["feasible"] for e in es)
        opt = sum(1 for e in es if e["feasible"] and e["cost"] == e["optimal_cost"])
        beat = sum(1 for e in es if e["feasible"] and e["cost"] is not None
                   and (e.get("greedy_cost") is None or e["cost"] < e["greedy_cost"]))
        real = [e for e in es if e["difficulty"] == "real" and e.get("real_cost") is not None]   # seeded
        # disruptions never happened on their day, so the record is only comparable on the real tier
        beat_real = sum(1 for e in real if e["feasible"] and e["cost"] < e["real_cost"])
        out.append(f"| {m} | {len(es)} | **{statistics.mean(r):.3f}** ({lo:.3f}-{hi:.3f}) | " + " | ".join(per)
                   + f" | {100 * feas / len(es):.0f}% | {100 * opt / len(es):.0f}% | {100 * beat / len(es):.0f}% | "
                   f"{100 * beat_real / max(1, len(real)):.0f}% | "
                   f"{sum(not e['submitted'] for e in es)} | {statistics.median(e['agent_turns'] for e in es):.0f} |")
    out += ["", "| Model | input tokens | of which cache reads | output tokens | median seconds |",
            "|---|---:|---:|---:|---:|"]
    for m, es in rows:
        ti = sum(e["input_tokens"] for e in es)
        cr = sum(e["cache_read_tokens"] for e in es)
        out.append(f"| {m} | {ti:,} | {100 * cr / max(1, ti):.0f}% | {sum(e['output_tokens'] for e in es):,} | "
                   f"{statistics.median(e['seconds'] for e in es):.0f} |")
    return "\n".join(out)


def main(run: str):
    eps = json.loads((ROOT / "results" / "rollouts" / run / "index.json").read_text())["episodes"]
    print(board(eps))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "board")


def baselines(tasks) -> str:
    """The lazy policies and the record's own outcome on the same tasks, from the stored references."""
    tiers = [t for t in TIERS if any(x.difficulty == t for x in tasks)]
    rows = [("CP-SAT optimum (the reference)", lambda t: 1.0, lambda t: True),
            ("greedy dispatcher, priority to fast trains", lambda t: t.reference["greedy_fast_reward"],
             lambda t: t.reference["greedy_fast_feasible"]),
            ("greedy dispatcher, first come first served", lambda t: t.reference["greedy_reward"],
             lambda t: t.reference["greedy_feasible"]),
            ("naive: change nothing", lambda t: t.reference["naive_reward"], lambda t: t.reference["naive_feasible"])]
    out = ["| policy | mean reward | " + " | ".join(tiers) + " | feasible |", "|---|---:|" + "---:|" * len(tiers) + "---:|"]
    for name, f, feas in rows:
        per = []
        for tier in tiers:
            xs = [f(t) for t in tasks if t.difficulty == tier]
            per.append(f"{statistics.mean(xs):.3f}")
        out.append(f"| {name} | **{statistics.mean(f(t) for t in tasks):.3f}** | " + " | ".join(per)
                   + f" | {100 * sum(bool(feas(t)) for t in tasks) / len(tasks):.0f}% |")
    real = [t for t in tasks if t.difficulty == "real" and "reward_if_legal" in (t.reference.get("real") or {})]
    if real:
        out.append(f"| the real dispatchers (record, real tier only, scored as if legal) | "
                   f"**{statistics.mean(t.reference['real']['reward_if_legal'] for t in real):.3f}** | "
                   + " | ".join(f"{statistics.mean(t.reference['real']['reward_if_legal'] for t in real):.3f}"
                                if tier == "real" else "-" for tier in tiers) + " | - |")
    return "\n".join(out)
