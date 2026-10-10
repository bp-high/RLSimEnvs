"""The physics, rules and generator on tasks rebuilt from the 2025 record (no solver needed except where noted)."""

from __future__ import annotations

import pytest

from rail_core import Closure, Order, evaluate, parse_plan, simulate
from rail_core.data import days, load_day
from rail_core.generate import build_task
from rail_core.model import RULES, Task, Unit, hhmm, parse_time
from rail_core.network import load_network

CALM_DAY = "2025-03-12"


@pytest.fixture(scope="module")
def calm():
    t = build_task(CALM_DAY, 7 * 60, 9 * 60 + 30, [], task_id="calm")
    assert t is not None
    return t


@pytest.fixture(scope="module")
def blocked():
    s = 7 * 60 + 3
    t = build_task("2025-03-13", s, s + 150, [Closure("closed", ["LW", "HDG"], s, s + 54, cause="broken down train")])
    assert t is not None
    return t


def test_network_matches_the_timetable():
    """Every clean 2025 weekday runs conflict-free as scheduled under the track layout and headways."""
    net = load_network()
    clean = 0
    for date in days()[:60]:
        day = load_day(date)
        units = [Unit(day.runs[ch[0]].unit, ch, day.runs[ch[0]].origin, None) for ch in day.chains]
        task = Task("tt", "x", "GN-LW", date, 0, 2000, net.to_dict(), list(day.runs.values()), units, [], [], [],
                    "tt", {}, dict(RULES))
        clean += evaluate(task, {}).feasible
    assert clean >= 50


def test_calm_window_runs_as_scheduled(calm):
    res = evaluate(calm, {})
    assert res.feasible and res.cancel_cost == 0
    sim, *_ = simulate(calm, {})
    for r in calm.runs:
        x = sim[r.id]
        for i, t in x.arr.items():
            if r.arr[i] is not None and t >= calm.start:
                assert t >= r.arr[i], "never ahead of the timetable"


def test_unscheduled_stop_costs_two_minutes(calm):
    fast = next(r for r in calm.runs if r.kind == "fast" and not r.started and "HDG" in r.route
                and not r.stops[r.idx("HDG")])
    sim, *_ = simulate(calm, {})
    i = fast.idx("HDG")
    pass_t = sim[fast.id].arr[i]
    held, *_ = simulate(calm, {fast.id: Order(hold={"HDG": pass_t + 1})})
    x = held[fast.id]
    assert x.arr[i] == pass_t + 1 and x.dep[i] == pass_t + 1 and x.stopped[i]
    assert x.arr[i + 1] >= sim[fast.id].arr[i + 1] + 1


def test_short_turn_continuity(blocked):
    r = next(r for r in blocked.runs if not r.started and r.origin == "GN" and "HDG" in r.route)
    u = next(u for u in blocked.units if r.id in u.runs)
    k = u.runs.index(r.id)
    nxt = u.runs[k + 1]
    ok = evaluate(blocked, {r.id: Order(end="HDG"), nxt: Order(start="HDG")})
    assert not any(v["rule"] == "continuity" and r.id in v["trains"] for v in ok.violations)
    bad = evaluate(blocked, {r.id: Order(end="HDG")})
    assert any(v["rule"] == "continuity" for v in bad.violations)
    not_turn = evaluate(blocked, {r.id: Order(end="DWE")})
    assert any("can only be turned" in v["detail"] for v in not_turn.violations)


def test_closure_and_running_trains(blocked):
    res = evaluate(blocked, {})
    assert any(v["rule"] == "closed" for v in res.violations)
    moving = [r for r in blocked.runs if r.started]
    assert moving
    r = moving[0]
    res = evaluate(blocked, {r.id: Order(cancel=True)})
    assert any("cannot be cancelled" in v["detail"] for v in res.violations)


def test_plan_parsing():
    assert parse_time("07:31") == 451 and parse_time("24:10") == 1450 and parse_time(451) == 451
    for bad in ("7.31", "25:99", True, float("nan"), "99:00"):
        with pytest.raises(ValueError):
            parse_time(bad)
    t = build_task(CALM_DAY, 7 * 60, 8 * 60, [], task_id="p")
    rid = t.runs[0].id
    plan, problems = parse_plan(t, f'```json\n[{{"train": "{rid}", "hold": {{"{t.runs[0].route[0]}": "07:31"}}}}]\n```')
    assert not problems and plan[rid].hold
    plan, problems = parse_plan(t, {rid: {"cancel": True}})
    assert plan[rid].cancel
    _, problems = parse_plan(t, [{"train": rid}, {"train": rid}, {"train": "nope"}])
    assert len(problems) == 2


def test_solver_proves_a_small_window():
    pytest.importorskip("ortools")
    from rail_core.solve import floor_cost, optimal_plan

    s = 7 * 60 + 3
    t = build_task("2025-03-13", s, s + 90, [Closure("closed", ["LW", "HDG"], s, s + 40, cause="broken down train")])
    sol = optimal_plan(t, time_limit=60)
    assert sol.optimal and sol.cost == sol.bound
    assert floor_cost(t) <= sol.cost
    res = evaluate(t, sol.plan)
    assert res.feasible and res.cost == sol.cost
    print(hhmm(t.start), sol.cost)
