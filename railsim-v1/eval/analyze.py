"""After a run: how the hardness prediction held up, and where the plans lost cost against the optimum.

    python eval/analyze.py <run>        # reads results/rollouts/<run>/index.json and the rollouts, prints markdown

The prediction (`reference.predicted_reward`, set by rail_core.select before any eval episode ran) is compared with
the actual reward per task: a post-hoc check of the selection, never an input to it.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from rail_core import evaluate, load_pack, parse_plan  # noqa: E402
from rail_core.select import _spearman  # noqa: E402


def main(run: str):
    run_dir = ROOT / "results" / "rollouts" / run
    eps = json.loads((run_dir / "index.json").read_text())["episodes"]
    pack = load_pack()
    rows = []
    for e in eps:
        t = pack.get(e["task_id"])
        rec_path = next(run_dir.glob(f"*/{e['task_id']}.json"))
        rec = json.loads(rec_path.read_text())
        sub = [s for s in rec["steps"] if s["tool"] == "submit_plan"]
        parts = None
        if sub:
            plan, problems = parse_plan(t, sub[-1]["arguments"]["plan"])
            r = evaluate(t, plan)
            if r.feasible and not problems:
                o, _ = parse_plan(t, t.reference["optimal_plan"])
                ro = evaluate(t, o)
                parts = {"delay": r.delay_cost - ro.delay_cost, "unserved": r.cancel_cost - ro.cancel_cost,
                         "handed_on": r.carry_cost - ro.carry_cost, "cancelled": r.cancelled - ro.cancelled,
                         "short": r.short_turned - ro.short_turned}
        rows.append((e, t, parts))
    pred = [t.reference.get("predicted_reward") for _, t, _ in rows]
    act = [e["reward"] for e, _, _ in rows]
    have = [(p, a) for p, a in zip(pred, act) if p is not None]
    print(f"## {run}\n")
    print(f"- episodes {len(rows)}, mean reward {statistics.mean(act):.3f}, median {statistics.median(act):.3f}")
    if have:
        print(f"- predicted mean {statistics.mean(p for p, _ in have):.3f}; Spearman(predicted, actual) "
              f"{_spearman([p for p, _ in have], [a for _, a in have]):.2f}")
    print(f"- reward < 0.7 on {sum(a < 0.7 for a in act)}/{len(act)}; at the optimum on "
          f"{sum(1 for e, _, _ in rows if e['feasible'] and e['cost'] == e['optimal_cost'])}")
    rel = [(e["cost"] - e["optimal_cost"]) / max(1, e["optimal_cost"]) for e, _, _ in rows if e["feasible"]]
    if rel:
        print(f"- feasible plans cost {100 * statistics.median(rel):.0f}% more than the optimum (median), "
              f"{100 * statistics.mean(rel):.0f}% (mean)")
    gaps = [p for _, _, p in rows if p]
    if gaps:
        print("- where the extra cost went (sum over feasible plans, plan minus optimum): "
              + ", ".join(f"{k} {sum(g[k] for g in gaps):+d}" for k in ("delay", "unserved", "handed_on"))
              + f"; cancellations {sum(g['cancelled'] for g in gaps):+d}, short-turns {sum(g['short'] for g in gaps):+d}")
    real = [(e, t) for e, t, _ in rows if t.difficulty == "real" and e["feasible"]]
    if real:
        print(f"- real tier: cheaper than the real dispatchers' outcome on {sum(e['cost'] < t.reference['real']['cost'] for e, t in real)}"
              f"/{len(real)} feasible plans")
    print("\n| task | tier | runs | predicted | reward | cost | optimum | floor |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for e, t, _ in sorted(rows, key=lambda x: x[0]["reward"]):
        p = t.reference.get("predicted_reward")
        print(f"| {e['task_id']} | {t.difficulty} | {len(t.runs)} | {p if p is None else f'{p:.2f}'} | "
              f"{e['reward']:.3f} | {e['cost']} | {e['optimal_cost']} | {e['floor_cost']} |")


if __name__ == "__main__":
    main(sys.argv[1])
