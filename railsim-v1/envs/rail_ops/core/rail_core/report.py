"""What check_plan shows the agent: every train's simulated times, its delay and lost stops, every rule broken, the
cost. Never a score and never anything from the reference."""

from __future__ import annotations

from .check import evaluate, network
from .model import Plan, Task, hhmm

MAX_VIOLATIONS = 60


def _times(task: Task, res) -> str:
    r = task.run(res.train)
    net = network(task)
    parts = []
    first = r.fixed_to if r.started else res.first
    for i in range(first, res.last + 1):
        code = r.route[i]
        if net.by_code[code].kind == "junction":
            continue
        a, d = res.arr.get(i), res.dep.get(i)
        if a is None and d is None:
            continue
        if a is None:                                   # where it starts
            s = f"{code} dep {hhmm(d)}"
        elif d is None or d == a:
            s = f"{code} {hhmm(a)}"
        else:
            s = f"{code} {hhmm(a)}-{hhmm(d)}"
        if r.stops[i] and r.arr[i] is not None and a is not None and i > res.first:
            late = a - r.arr[i]
            if late > 0:
                s += f" (+{late})"
        elif not r.stops[i] and res.stopped.get(i) and i not in (res.first, res.last):
            s += " (unscheduled stop)"
        parts.append(s)
    return " | ".join(parts)


def plan_report(task: Task, plan: Plan, problems: list[str] | None = None) -> dict:
    res = evaluate(task, plan)
    problems = list(problems or [])
    feasible = res.feasible and not problems
    trains = []
    for x in sorted(res.runs, key=lambda x: (task.run(x.train).fixed_arr if task.run(x.train).started
                                             else task.run(x.train).dep[0])):
        r = task.run(x.train)
        if x.cancelled:
            what = "cancelled"
        else:
            a, b = r.route[x.first], r.route[x.last]
            what = f"{a}->{b}" + ("" if (x.first == 0 and x.last == len(r.route) - 1) else " (short)")
        row = {"train": x.train, "set": x.unit, "runs": what, "delay": x.delay, "unserved_stops": x.unserved}
        if not x.cancelled:
            row["times"] = _times(task, x)
        if x.problems:
            row["breaks"] = x.problems
        trains.append(row)
    out = {
        "feasible": feasible,
        "cost": res.delay_cost + res.cancel_cost + res.carry_cost if feasible else None,
        "cost_parts": {"delay": res.delay_cost, "unserved_stops": res.cancel_cost, "handed_on": res.carry_cost},
        "violations_total": len(res.violations),
        "violations": res.violations[:MAX_VIOLATIONS],
        "trains": trains,
    }
    if not feasible:
        out["note"] = ("infeasible: fix every violation (cost_parts are what the plan would cost if it were legal)"
                       if res.violations or problems else "")
    if problems:
        out["entry_problems"] = problems
    return out
