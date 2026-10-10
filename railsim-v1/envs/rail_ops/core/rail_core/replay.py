"""What the map replay animates: every run's simulated times under a plan, the plan's orders, the cost as it accrues,
the rule breaks and the moments worth marking on the timeline. Built from the same `simulate` and `evaluate` the
grader uses, so the picture is the grade. For people (the `/twin/` viewer); no MCP tool returns it.

Cost events add up to the grader's cost: a stop's delay when the train arrives there, 30 for a stop not served at
the minute it was due, and each train set's lateness handed on to the next window at the window's end.
"""

from __future__ import annotations

from .check import evaluate, network, simulate
from .model import Order, Plan, Task, hhmm
from .reward import score


def _due(r, j: int) -> int | None:
    return r.arr[j] if r.arr[j] is not None else r.dep[j]


def replay(task: Task, plan: Plan, name: str = "", parse_problems: list[str] | None = None) -> dict:
    net = network(task)
    res = evaluate(task, plan)
    _, _, _, _, carry = simulate(task, plan)
    problems = list(parse_problems or [])
    feasible = res.feasible and not problems
    clean = sum(1 for x in res.runs if not x.problems) / max(1, len(res.runs) + len(problems))
    optimum, floor = int(task.reference.get("optimal_cost", 0)), int(task.reference.get("floor_cost", 0))
    reward, _ = score(res.cost if feasible else None, feasible, clean, optimum, floor, task.rule("gap_k"))
    cancel_cost = task.rule("cancel_cost")

    runs, costs, events = [], [], []
    for x in res.runs:
        r = task.run(x.train)
        o = plan.get(x.train) or Order()
        n = len(r.route)
        runs.append({
            "train": x.train, "unit": x.unit, "kind": r.kind, "dir": "E" if net.direction(r.origin, r.dest) > 0
            else "W", "route": r.route, "stops": r.stops, "sched_arr": r.arr, "sched_dep": r.dep,
            "cancelled": x.cancelled, "first": x.first, "last": x.last,
            "arr": [x.arr.get(i) for i in range(n)], "dep": [x.dep.get(i) for i in range(n)],
            "stopped": [bool(x.stopped.get(i)) for i in range(n)],
            "started": r.started, "fixed_to": r.fixed_to, "fixed_from": r.fixed_from, "stuck": r.stuck,
            "delay": x.delay, "unserved": x.unserved, "problems": x.problems,
            "order": o.to_json(x.train) if (o.cancel or o.start or o.end or o.hold) else None})
        # cost as it accrues (the same terms as check.evaluate)
        if x.cancelled:
            for j in range(n):
                if r.stops[j] and _due(r, j) is not None:
                    costs.append([_due(r, j), cancel_cost, "unserved", x.train, r.route[j]])
            events.append({"t": r.dep[0] if r.dep[0] is not None else task.start, "kind": "cancel",
                           "train": x.train, "text": f"{x.train} cancelled ({r.origin} {hhmm(r.dep[0])} to {r.dest})"})
            continue
        s, e = x.first, x.last
        k = r.fixed_to if r.started else s
        for j in range(k if r.started else s + 1, e + 1):
            if r.stops[j] and r.arr[j] is not None and x.arr.get(j) is not None and x.arr[j] >= task.start:
                late = max(0, x.arr[j] - r.arr[j])
                if late:
                    costs.append([x.arr[j], late, "delay", x.train, r.route[j]])
        for j in range(n):
            if r.stops[j] and not (s <= j <= e) and _due(r, j) is not None:
                costs.append([_due(r, j), cancel_cost, "unserved", x.train, r.route[j]])
        if e < n - 1:
            events.append({"t": x.arr.get(e), "kind": "turn", "train": x.train,
                           "text": f"{x.train} turns back at {r.route[e]} (not to {r.dest})"})
        if s > 0 and not r.started:
            events.append({"t": x.dep.get(s), "kind": "turn", "train": x.train,
                           "text": f"{x.train} starts at {r.route[s]} (not {r.origin})"})
        for code, minute in sorted(o.hold.items(), key=lambda kv: kv[1]):
            i = r.route.index(code) if code in r.route else None
            if i is not None and x.dep.get(i) is not None:
                events.append({"t": x.arr.get(i, minute), "kind": "hold", "train": x.train,
                               "text": f"{x.train} held at {code} until {hhmm(minute)}"})
    for unit, c in carry.items():
        if c:
            costs.append([task.end, c, "handed_on", unit, ""])
    for c in task.closures:
        span = "-".join(dict.fromkeys([c.where[0], c.where[-1]]))
        what = {"closed": "closed", "slow": f"slow (+{c.extra} min)", "track": "one platform track out of use"}
        events.append({"t": c.start, "kind": "closure", "text": f"{span} {what.get(c.kind, c.kind)}: {c.cause} "
                       f"({hhmm(c.start)}-{hhmm(c.end)})"})
        events.append({"t": c.end, "kind": "reopen", "text": f"{span} back to normal"})
    for v in res.violations:
        if v.get("time"):
            h, m = v["time"].split(":")
            events.append({"t": int(h) * 60 + int(m), "kind": "violation", "rule": v["rule"],
                           "trains": v["trains"], "where": v["where"], "text": v["detail"]})
    events.sort(key=lambda ev: (ev["t"] if ev["t"] is not None else task.start, ev["kind"]))
    costs.sort(key=lambda c: c[0])
    total = sum(c[1] for c in costs)
    assert total == res.delay_cost + res.cancel_cost + res.carry_cost, "cost events must add up to the grader's cost"
    return {
        "task_id": task.task_id, "day": task.day, "start": task.start, "end": task.end,
        "difficulty": task.difficulty, "plan": name, "feasible": feasible, "reward": round(reward, 4),
        "cost": res.cost if feasible else None, "cost_if_legal": total,
        "cost_parts": {"delay": res.delay_cost, "unserved": res.cancel_cost, "handed_on": res.carry_cost},
        "optimal_cost": optimum, "floor_cost": floor,
        "cancelled": res.cancelled, "short_turned": res.short_turned,
        "violations": res.violations[:80], "parse_problems": problems,
        "points": [{"code": p.code, "name": p.name, "km": p.km, "kind": p.kind, "tracks": p.tracks,
                    "directional": p.directional, "turn": p.turn} for p in net.points],
        "sections": [{"a": s.a, "b": s.b, "single": s.single} for s in net.sections],
        "closures": [{"kind": c.kind, "where": c.where, "start": c.start, "end": c.end, "extra": c.extra,
                      "cause": c.cause, "real": c.real} for c in task.closures],
        "units": [{"id": u.id, "at": u.at, "ready": u.ready, "runs": u.runs, "after": u.after} for u in task.units],
        "runs": runs, "costs": costs, "events": events, "notices": task.notices,
        "rules": {k: task.rule(k) for k in ("h_follow", "h_opp", "turn_min", "stop_loss", "cancel_cost",
                                           "max_delay")},
    }
