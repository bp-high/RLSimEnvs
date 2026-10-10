"""The grader against an independent reference, hostile plans, and the optimum's claims.

`ref_evaluate` re-implements the physics and every rule from the task text alone, on a minute grid, without importing
rail_core.check (keep it that way): if the two disagree on any perturbed plan, one of them is wrong.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from rail_core import Order, PlanError, grade, parse_plan
from rail_core import evaluate as check_evaluate
from rail_core.model import Task
from rail_core.network import Network

BIG = 10**9


# ---------------------------------------------------------------- the independent reference

def ref_evaluate(task: Task, plan: dict) -> tuple[bool, int, set]:
    """(feasible, cost, rules broken) computed independently of rail_core.check."""
    net = Network.from_dict(task.network)
    R = task.rules
    L, turn = R["stop_loss"], R["turn_min"]
    runs = {r.id: r for r in task.runs}
    broken: set[str] = set()
    secs = []          # (section, dir, enter, exit, train, history)
    stands = []        # (station, dir, first minute, last minute)
    delay = unserved = carry = 0

    def extra(sec, t):
        best = 0
        for c in task.closures:
            if c.kind == "slow":
                a, b = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
                if a <= sec < b and c.start <= t < c.end:
                    best = max(best, c.extra)
        return best

    def limited(code):
        p = net.by_code[code]
        return p.kind != "junction" and p.tracks is not None

    for u in task.units:
        where, when = u.at, u.ready
        standing_since = task.start if (u.at and limited(u.at)) else None
        for rid in u.runs:
            r = runs[rid]
            o = plan.get(rid) or Order()
            n = len(r.route)
            dirn = 1 if net.index[r.route[-1]] > net.index[r.route[0]] else -1
            moving = r.fixed_arr is not None
            if o.cancel and moving:
                broken.add("continuity")
            if o.cancel and not moving:
                unserved += sum(r.stops)
                continue
            first = r.route.index(o.start) if o.start else 0
            last = r.route.index(o.end) if o.end else n - 1
            if o.start and o.start != r.route[0] and (moving or not net.by_code[o.start].turn):
                broken.add("continuity")
                first = 0
            if o.end and o.end != r.route[-1] and not net.by_code[o.end].turn:
                broken.add("continuity")
            if last <= first:
                broken.add("continuity")
                last = n - 1
            if moving and last < r.fixed_to:
                broken.add("continuity")
                last = n - 1
            k = r.fixed_to if moving else first
            if not moving and r.route[first] != where:
                broken.add("continuity")
            for code in o.hold:
                j = r.route.index(code)
                if not (k <= j < last) or net.by_code[code].kind == "junction":
                    broken.add("continuity")
            arr: dict[int, int] = {}
            dep: dict[int, int] = {}
            if moving:
                arr[k] = r.fixed_arr
                if r.fixed_from is not None:
                    secs.append([net.section(r.route[k - 1], r.route[k]), dirn, r.fixed_from, r.fixed_arr, rid, True])
            for j in range(k, last):
                code = r.route[j]
                h = o.hold.get(code, -BIG) if first <= j < last else -BIG
                speedup = 0
                if j == k and not moving:
                    go = max(when if when is not None else -BIG, r.dep[j], h)
                    if not r.stops[j]:
                        speedup = L
                    if standing_since is not None and limited(code):
                        stands.append((code, dirn, standing_since, go))
                elif r.stops[j]:
                    go = max(arr[j], r.dep[j], h)
                elif net.by_code[code].kind != "junction" and h > arr[j]:
                    arr[j] += L
                    if secs and secs[-1][4] == rid:
                        secs[-1][3] = arr[j]
                    go = max(arr[j], h)
                    speedup = L
                else:
                    go = arr[j]
                if moving and j == k:
                    go = max(go, task.start)
                dep[j] = go
                if (j != k or moving) and limited(code):
                    stands.append((code, dirn, arr[j], go))
                sec = net.section(code, r.route[j + 1])
                t = go + r.seg[j] + speedup + extra(sec, go)
                if r.arr[j + 1] is not None and t < r.arr[j + 1]:
                    t = r.arr[j + 1]
                if j + 1 == last and not r.stops[last]:
                    t += L
                arr[j + 1] = t
                secs.append([sec, dirn, go, t, rid, False])
            for j in range(k if moving else first + 1, last + 1):
                if r.stops[j] and r.arr[j] is not None and arr[j] >= task.start:
                    late = arr[j] - r.arr[j]
                    delay += max(0, late)
                    if late > R["max_delay"]:
                        broken.add("max_delay")
            unserved += sum(1 for j in range(n) if r.stops[j] and not first <= j <= last)
            where, when = r.route[last], arr[last] + turn
            standing_since = arr[last] if limited(r.route[last]) else None
        if standing_since is not None and where is not None:
            stands.append((where, 0, standing_since, task.end + 2000))
        if u.after:
            if where != u.after["station"]:
                broken.add("continuity")
            elif when is not None:
                carry += max(0, when - u.after["dep"]) * u.after["stops"]

    # sections on a minute grid: [enter, exit)
    hf, ho = R["h_follow"], R["h_opp"]
    for x in range(len(secs)):
        for y in range(len(secs)):
            if x >= y:
                continue
            a, b = secs[x], secs[y]
            if a[0] != b[0] or a[4] == b[4]:
                continue
            if a[1] == b[1]:
                first_, second = (a, b) if (a[2], a[3]) <= (b[2], b[3]) else (b, a)
                if second[2] - first_[2] < hf or second[3] - first_[3] < hf:
                    broken.add("headway")
            elif net.sections[a[0]].single:
                grid_a = set(range(a[2], a[3] + ho))
                grid_b = set(range(b[2], b[3] + ho))
                if grid_a & set(range(b[2], b[3])) or grid_b & set(range(a[2], a[3])):
                    broken.add("single_track")
    for c in task.closures:
        if c.kind != "closed":
            continue
        a_, b_ = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
        for s in secs:
            if a_ <= s[0] < b_ and not s[5] and set(range(s[2], s[3])) & set(range(c.start, c.end)):
                broken.add("closed")
    # stations on a minute grid: [first, last] inclusive
    count: dict[tuple, int] = {}
    for code, d, t0, t1 in stands:
        p = net.by_code[code]
        for m in range(t0, min(t1, task.end + 1500) + 1):
            for key in ([(code, d, m)] if p.directional else [(code, 0, m)]):
                count[key] = count.get(key, 0) + 1
    for (code, d, m), n in count.items():
        p = net.by_code[code]
        cap = 1 if p.directional else p.tracks
        cap -= sum(1 for c in task.closures if c.kind == "track" and c.where[0] == code and c.start <= m < c.end)
        if n > cap:
            broken.add("platforms")
    cost = delay + R["cancel_cost"] * unserved + carry
    return not broken, cost, broken


# ---------------------------------------------------------------- perturbed plans

def perturb(task: Task, base: dict, rng: random.Random) -> dict:
    net = Network.from_dict(task.network)
    plan = {k: Order(v.cancel, v.start, v.end, dict(v.hold)) for k, v in base.items()}
    runs = task.runs
    for _ in range(rng.randint(1, 4)):
        r = rng.choice(runs)
        o = plan.setdefault(r.id, Order())
        what = rng.random()
        k = r.fixed_to if r.fixed_arr is not None else 0
        stations = [c for c in r.route[k:-1] if net.by_code[c].kind != "junction"]
        if what < 0.45 and stations:
            code = rng.choice(stations)
            base_t = o.hold.get(code) or (r.dep[r.route.index(code)] or task.start)
            o.hold[code] = max(0, base_t + rng.choice((-3, -1, 1, 2, 3, 5, 8, 15, 30)))
        elif what < 0.6 and o.hold:
            o.hold.pop(rng.choice(list(o.hold)))
        elif what < 0.75:
            turns = [c for c in r.route[1:-1] if net.by_code[c].turn]
            if turns:
                o.end = rng.choice(turns)
        elif what < 0.85:
            turns = [c for c in r.route[1:-1] if net.by_code[c].turn]
            if turns:
                o.start = rng.choice(turns)
        elif what < 0.93:
            o.cancel = not o.cancel
        else:
            plan.pop(r.id, None)
    return plan


def test_reference_checker_agrees(sample):
    rng = random.Random(7)
    checked = feasible = 0
    for task in sample:
        opt, _ = parse_plan(task, task.reference["optimal_plan"])
        for base in (opt, {}):
            for _ in range(60):
                plan = perturb(task, base, rng)
                ok, cost, broken = ref_evaluate(task, plan)
                res = check_evaluate(task, plan)
                assert res.feasible == ok, (task.task_id, ok, broken, res.violations[:3], plan)
                if ok:
                    assert res.cost == cost, (task.task_id, res.cost, cost)
                    feasible += 1
                else:
                    assert broken == {v["rule"] for v in res.violations}, (task.task_id, broken, res.violations[:4])
                checked += 1
    assert checked >= 100 and feasible >= 5


def test_optimum_holds(sample):
    rng = random.Random(3)
    for task in sample:
        ref = task.reference
        opt, problems = parse_plan(task, ref["optimal_plan"])
        assert not problems
        ok, cost, _ = ref_evaluate(task, opt)
        assert ok and cost == ref["optimal_cost"]
        assert ref["floor_cost"] <= ref["lower_bound"] <= ref["optimal_cost"]
        assert grade(task, opt).reward == 1.0
        for _ in range(80):
            plan = perturb(task, opt, rng)
            ok, cost, _ = ref_evaluate(task, plan)
            if ok:
                assert cost >= ref["lower_bound"], (task.task_id, cost, ref["lower_bound"])
                assert cost >= ref["floor_cost"]


HOSTILE = [
    None, 0, 1.5, True, "", "nonsense", "[", "{}", "[]", [], {}, [None], [1, 2], ["x"], [[]],
    [{"train": None}], [{"train": True}], [{"train": "99999"}], [{"train": "37413", "hold": "07:00"}],
    [{"train": "37413", "hold": {"XX": "07:00"}}], [{"train": "37413", "hold": {"HDG": "25:99"}}],
    [{"train": "37413", "hold": {"HDG": float("nan")}}], [{"train": "37413", "hold": {"HDG": float("inf")}}],
    [{"train": "37413", "hold": {"HDG": 10**30}}], [{"train": "37413", "hold": {"HDG": -5}}],
    [{"train": "37413", "to": "ZZ"}], [{"train": "37413", "from": 3}], [{"train": "37413", "cancel": "yes"}],
    "here is my plan: ```json\n[{\"train\": \"0\"}]\n```",
]


def test_hostile_plans_never_crash_or_pass(sample):
    for task in sample:
        rid = task.runs[0].id
        cases = list(HOSTILE) + [[{"train": rid, "hold": {task.runs[0].route[0]: "07:00"}}] * 50,
                                 [{"train": rid}, {"train": rid}], json.dumps([{"train": rid, "bogus": 1}])]
        for raw in cases:
            try:
                plan, problems = parse_plan(task, raw)
            except PlanError:
                continue
            g = grade(task, plan, problems)
            assert 0.0 <= g.reward <= 1.0 and not math.isnan(g.reward)
            if problems:
                assert g.reward < 0.2


def test_empty_plan_is_infeasible_and_low(pack):
    for task in pack.tasks[:20]:
        g = grade(task, {})
        assert g.reward < 0.2 or task.reference["naive_reward"] <= 0.6


@pytest.mark.parametrize("field", ["naive_reward", "greedy_reward", "greedy_fast_reward"])
def test_lazy_policies_stay_low(pack, field):
    for task in pack.tasks:
        assert task.reference[field] <= (0.6 if field == "naive_reward" else 0.85), (task.task_id, field)
