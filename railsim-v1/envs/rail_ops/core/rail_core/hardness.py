"""Hardness features of a task: model-independent properties that predict how badly an agent will do on it.

They come from the task and its reference only (no language model), so an eval set can be chosen by them without
ever looking at a model's results on the eval tasks themselves (see `select`).

    runs, sets            problem size
    conflicts             rule breaks if nothing is changed (single track, headway, platforms, closures)
    decisions             holds + short-turns + cancellations in the proven optimum: how much must change
    turn_points           distinct points where the optimum short-turns trains
    avoidable             optimum - floor: what good sequencing saves over the train sets planned alone
    greedy_gap            how far the best greedy dispatcher is from the optimum, in reward
    closed_min, slow_min, track_min   minutes of each kind of disruption inside the window
    single_closed         a closure on single track (no way round it)
    mid_line              a closure with turn points on both sides (trains must be turned at both ends of it)
"""

from __future__ import annotations

from collections import Counter

from .check import evaluate, network
from .model import Task, plan_from_list


def features(task: Task) -> dict:
    net = network(task)
    ref = task.reference
    naive = evaluate(task, {})
    rules = Counter(v["rule"] for v in naive.violations)
    opt = ref.get("optimal_plan") or []
    holds = sum(len(e.get("hold") or {}) for e in opt)
    shorts = sum(1 for e in opt if ("to" in e or "from" in e) and not e.get("cancel"))
    cancels = sum(1 for e in opt if e.get("cancel"))
    turn_pts = {e.get("to") for e in opt if e.get("to")} | {e.get("from") for e in opt if e.get("from")}
    clip = lambda c: max(0, min(c.end, task.end) - max(c.start, task.start))
    single_closed = mid_line = False
    for c in task.closures:
        if c.kind != "closed":
            continue
        lo, hi = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
        if any(net.sections[k].single for k in range(lo, hi)):
            single_closed = True
        west = any(p.turn for p in net.points[:lo + 1])
        east = any(p.turn for p in net.points[hi:])
        if lo > 0 and hi < len(net.points) - 1 and west and east:
            mid_line = True
    greedy = max(ref.get("greedy_reward", 0.0), ref.get("greedy_fast_reward", 0.0))
    return {
        "runs": len(task.runs), "sets": len(task.units), "stops": sum(r.n_stops for r in task.runs),
        "conflicts": len(naive.violations), "single_track": rules.get("single_track", 0),
        "headway": rules.get("headway", 0), "platforms": rules.get("platforms", 0), "closed": rules.get("closed", 0),
        "decisions": holds + shorts + cancels, "holds": holds, "short_turns": shorts, "cancels": cancels,
        "turn_points": len(turn_pts), "avoidable": ref.get("optimal_cost", 0) - ref.get("floor_cost", 0),
        "optimal": ref.get("optimal_cost", 0), "greedy_gap": round(1.0 - greedy, 4),
        "closed_min": sum(clip(c) for c in task.closures if c.kind == "closed"),
        "slow_min": sum(clip(c) for c in task.closures if c.kind == "slow"),
        "track_min": sum(clip(c) for c in task.closures if c.kind == "track"),
        "single_closed": single_closed, "mid_line": mid_line, "late_sets": sum("is held for" in n for n in task.notices),
    }
