"""The physics and the grader: a plan in, every train's times, every rule it breaks and its cost out. Pure Python and
deterministic; nothing here needs a solver.

Physics (`simulate`). Each train set works its runs in order. A run leaves its first point at the latest of: when its
train is ready there, its scheduled departure (or passing minute), and the plan's hold. It then runs at its minimum
running times, but never reaches a point before its timetable minute there (drivers run to the timetable; a late
train makes up time). At a scheduled stop it leaves at the latest of arriving, its scheduled departure and any hold; at a
point where it has no stop it passes, unless held past its passing minute: then it stops, losing `stop_loss` minutes
braking and `stop_loss` accelerating. Junctions cannot hold a train. After a run its train is ready again `turn_min`
minutes after arriving, at the point where the run ended.

Rules (`evaluate`), each broken one is a violation:
    single track    two trains going opposite ways may not be in a single-track section at once; the second enters
                    at least h_opp minutes after the first has left
    headway         trains going the same way keep h_follow minutes apart at both ends of every section, so nothing
                    overtakes between points
    platforms       no more trains at a station than it has tracks (minus any out of use); at a directional station
                    (double track) one per direction
    closed          no train in a closed section while it is closed (a train already stuck inside waits there)
    continuity      a run starts where its train is; short-turns only at turn points; a running train cannot be
                    cancelled or turned behind itself; every train ends the window where its next run leaves from
    max delay       no stop reached more than max_delay minutes late

Cost, in stop-minutes (lower is better): the arrival delay at every scheduled stop a train serves, plus cancel_cost
for every scheduled stop it does not serve, plus, for each train set, its lateness for its first run after the
window times that run's number of stops (delay handed on to the next window).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .model import Order, Plan, Run, Task, hhmm
from .network import Network

NEG = -10**9


@dataclass
class RunResult:
    train: str
    unit: str
    cancelled: bool = False
    first: int | None = None              # route index of its first served point
    last: int | None = None               # route index of its last served point
    arr: dict[int, int] = field(default_factory=dict)
    dep: dict[int, int] = field(default_factory=dict)
    stopped: dict[int, bool] = field(default_factory=dict)
    delay: int = 0
    unserved: int = 0
    max_late: int = 0
    cost: int = 0
    problems: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    feasible: bool
    cost: int | None
    delay_cost: int
    cancel_cost: int
    carry_cost: int
    cancelled: int
    short_turned: int
    runs: list[RunResult]
    violations: list[dict]


@dataclass
class _Occ:      # a train in a section
    sec: int
    dir: int
    enter: int
    exit: int
    train: str
    history: bool = False


@dataclass
class _Pres:     # a train standing at (or passing) a station, minutes inclusive
    point: str
    dir: int
    t0: int
    t1: int
    train: str


def network(task: Task) -> Network:
    n = task.__dict__.get("_net")
    if n is None:
        n = task.__dict__["_net"] = Network.from_dict(task.network)
    return n


def run_dir(net: Network, r: Run) -> int:
    return net.direction(r.origin, r.dest)


def served_part(task: Task, r: Run, o: Order) -> tuple[int, int, list[str]]:
    """(first, last, problems) for the part of the route the plan runs."""
    net = network(task)
    probs = []
    s = r.idx(o.start) if o.start else 0
    e = r.idx(o.end) if o.end else len(r.route) - 1
    if o.start and o.start != r.origin:
        if r.started:
            probs.append(f"{r.id} is already running from {r.origin}: it cannot start at {o.start}")
            s = 0
        elif not net.by_code[o.start].turn:
            probs.append(f"{r.id} cannot start at {o.start}: trains can only be turned at "
                         + ", ".join(p.code for p in net.points if p.turn))
    if o.end and o.end != r.dest and not net.by_code[o.end].turn:
        probs.append(f"{r.id} cannot end at {o.end}: trains can only be turned at "
                     + ", ".join(p.code for p in net.points if p.turn))
    if e <= s:
        probs.append(f"{r.id}: 'to' {r.route[e]} must come after where it starts ({r.route[s]})")
        e = len(r.route) - 1
    if r.started and e < r.fixed_to:
        probs.append(f"{r.id} is already past {r.route[e]}: it can end at {r.route[r.fixed_to]} at the earliest")
        e = len(r.route) - 1
    return s, e, probs


def simulate(task: Task, plan: Plan):
    """Times for every run under the plan. Returns (results by train, structural violations, section occupancies,
    station presences, carry cost per unit)."""
    net = network(task)
    loss, turn = task.rule("stop_loss"), task.rule("turn_min")
    runs = task.run_map
    results: dict[str, RunResult] = {}
    viol: list[dict] = []
    occ: list[_Occ] = []
    pres: list[_Pres] = []
    carry: dict[str, int] = {}
    slow = [c for c in task.closures if c.kind == "slow"]

    def slow_extra(sec: int, t: int) -> int:
        x = 0
        for c in slow:
            lo, hi = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
            if lo <= sec < hi and c.start <= t < c.end:
                x = max(x, c.extra)
        return x

    def finite(code: str) -> bool:
        p = net.by_code[code]
        return p.kind != "junction" and p.tracks is not None

    for u in task.units:
        pos, ready = u.at, u.ready
        wait_from: int | None = task.start if (u.at and finite(u.at)) else None   # standing at a station at start
        for rid in u.runs:
            r = runs[rid]
            o = plan.get(rid) or Order()
            res = RunResult(train=rid, unit=u.id)
            results[rid] = res
            d = run_dir(net, r)
            if o.cancel:
                if r.started:
                    res.problems.append(f"{rid} is already running: it cannot be cancelled (short-turn it instead)")
                    o = Order(end=o.end, hold=o.hold)
                else:
                    res.cancelled = True
                    res.unserved = r.n_stops
                    continue
            s, e, probs = served_part(task, r, o)
            res.problems += probs
            k = r.fixed_to if r.started else s
            if not r.started and r.route[s] != pos:
                res.problems.append(f"{rid} would start at {r.route[s]}, but its train ({u.id}) is at {pos}")
            for code in o.hold:
                i = r.idx(code)
                if i < k or i >= e:
                    res.problems.append(f"{rid}: hold at {code} is outside the part it runs "
                                        f"({r.route[k]} to {r.route[e]}, holds only where it leaves from)")
                elif net.by_code[code].kind == "junction":
                    res.problems.append(f"{rid}: trains cannot be held at the junction {code}")
            res.first, res.last = s, e
            arr, dep, stopped = res.arr, res.dep, res.stopped
            last_occ = None
            if r.started:
                arr[k] = r.fixed_arr
                if r.fixed_from is not None:
                    last_occ = _Occ(net.section(r.route[k - 1], r.route[k]), d, r.fixed_from, r.fixed_arr, rid, True)
                    occ.append(last_occ)
            i = k
            while True:
                code = r.route[i]
                pt = net.by_code[code]
                h = o.hold.get(code) if s <= i < e else None
                accel = 0
                if i == e:
                    stopped[i] = True
                    break
                if i == k and not r.started:                 # where the run starts
                    t = max(ready if ready is not None else NEG, r.dep[i], h if h is not None else NEG)
                    stopped[i] = True
                    accel = 0 if r.stops[i] else loss
                    if wait_from is not None and finite(code):
                        pres.append(_Pres(code, d, wait_from, t, rid))
                elif r.stops[i]:
                    t = max(arr[i], r.dep[i], h if h is not None else NEG)
                    stopped[i] = True
                elif pt.kind != "junction" and h is not None and h > arr[i]:
                    arr[i] += loss                            # an unscheduled stop: it brakes ...
                    if last_occ is not None:
                        last_occ.exit = arr[i]                # ... still in the section it is leaving
                    t = max(arr[i], h)
                    stopped[i] = True
                    accel = loss                              # ... and accelerates again
                else:
                    t = arr[i]
                    stopped[i] = False
                if r.started and i == k:
                    t = max(t, task.start)                    # standing there when the task starts
                dep[i] = t
                if (i != k or r.started) and finite(code):
                    pres.append(_Pres(code, d, arr[i], t, rid))
                sec = net.section(code, r.route[i + 1])
                nxt = t + r.seg[i] + accel + slow_extra(sec, t)
                if r.arr[i + 1] is not None:
                    nxt = max(nxt, r.arr[i + 1])              # never ahead of the timetable
                if i + 1 == e and not r.stops[e]:
                    nxt += loss                               # brakes to end where it has no scheduled stop
                arr[i + 1] = nxt
                last_occ = _Occ(sec, d, t, nxt, rid)
                occ.append(last_occ)
                i += 1
            # cost
            for j in range(k if r.started else s + 1, e + 1):
                if r.stops[j] and r.arr[j] is not None and arr[j] >= task.start:
                    late = arr[j] - r.arr[j]
                    res.delay += max(0, late)
                    res.max_late = max(res.max_late, late)
            res.unserved = sum(1 for j, st in enumerate(r.stops) if st and not (s <= j <= e))
            pos, ready = r.route[e], arr[e] + turn
            wait_from = arr[e] if finite(r.route[e]) else None
        if wait_from is not None and pos is not None:
            pres.append(_Pres(pos, 0, wait_from, task.end + 24 * 60, u.id))
        if u.after:
            if pos != u.after["station"]:
                viol.append({"rule": "continuity", "trains": [u.after["run"]], "where": pos,
                             "detail": f"train set {u.id} ends the window at {pos}, but its next run "
                                       f"{u.after['run']} leaves {u.after['station']} at {hhmm(u.after['dep'])}"})
            elif ready is not None:
                carry[u.id] = max(0, ready - u.after["dep"]) * int(u.after["stops"])
    return results, viol, occ, pres, carry


def evaluate(task: Task, plan: Plan) -> PlanResult:
    net = network(task)
    results, viol, occ, pres, carry = simulate(task, plan)
    hf, ho, maxd = task.rule("h_follow"), task.rule("h_opp"), task.rule("max_delay")
    bad: dict[str, set] = defaultdict(set)

    def add(rule: str, trains: list[str], where: str, time: int | None, detail: str):
        viol.append({"rule": rule, "trains": trains, "where": where, "time": hhmm(time), "detail": detail})
        for t in trains:
            if t in results:
                bad[t].add(rule)

    for rid, res in results.items():
        for p in res.problems:
            add("continuity", [rid], "", None, p)
        if res.max_late > maxd:
            add("max_delay", [rid], "", None, f"{rid} reaches a stop {res.max_late} minutes late (limit {maxd}): "
                                               f"short-turn or cancel it")

    # sections: headways and single track
    by_sec: dict[int, list[_Occ]] = defaultdict(list)
    for o in occ:
        by_sec[o.sec].append(o)
    for sec, lst in by_sec.items():
        s = net.sections[sec]
        name = f"{s.a}-{s.b}"
        lst.sort(key=lambda o: (o.enter, o.exit))
        for i, a in enumerate(lst):
            for b in lst[i + 1:]:
                if b.enter > a.exit + max(hf, ho) and b.enter > a.enter + hf:
                    break
                if a.train == b.train:
                    continue
                if a.dir == b.dir:
                    if b.enter - a.enter < hf or b.exit - a.exit < hf:
                        what = "overtakes" if b.exit < a.exit else "follows too closely"
                        add("headway", [a.train, b.train], name, b.enter,
                            f"{b.train} {what} {a.train} in {name}: enter {hhmm(a.enter)}/{hhmm(b.enter)}, "
                            f"leave {hhmm(a.exit)}/{hhmm(b.exit)} (same direction: {hf} min apart at both ends, "
                            f"no overtaking between points)")
                elif s.single and not (a.exit + ho <= b.enter or b.exit + ho <= a.enter):
                    add("single_track", [a.train, b.train], name, b.enter,
                        f"{a.train} ({hhmm(a.enter)}-{hhmm(a.exit)}) and {b.train} ({hhmm(b.enter)}-{hhmm(b.exit)}) "
                        f"meet head-on in single-track {name}")
    # closures
    for c in task.closures:
        if c.kind != "closed":
            continue
        lo, hi = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
        for o in occ:
            if lo <= o.sec < hi and not o.history and o.enter < c.end and o.exit > c.start:
                s = net.sections[o.sec]
                add("closed", [o.train], f"{s.a}-{s.b}", o.enter,
                    f"{o.train} is in {s.a}-{s.b} ({hhmm(o.enter)}-{hhmm(o.exit)}) while it is closed "
                    f"{hhmm(c.start)}-{hhmm(c.end)} ({c.cause})")
    # platforms
    out_of_use: dict[str, list] = defaultdict(list)
    for c in task.closures:
        if c.kind == "track":
            out_of_use[c.where[0]].append((c.start, c.end))
    by_pt: dict[str, list[_Pres]] = defaultdict(list)
    for p in pres:
        by_pt[p.point].append(p)
    for code, lst in by_pt.items():
        pt = net.by_code[code]
        groups = [[p for p in lst if p.dir == d] for d in (1, -1)] if pt.directional else [lst]
        for g in groups:
            cap = 1 if pt.directional else pt.tracks
            events = sorted({p.t0 for p in g} | {a for a, _ in out_of_use.get(code, [])})
            reported = set()
            for t in events:
                here = [p for p in g if p.t0 <= t <= p.t1]
                c_t = cap - sum(1 for a, b in out_of_use.get(code, []) if a <= t < b)
                if len(here) > c_t:
                    key = tuple(sorted(p.train for p in here))
                    if key in reported:
                        continue
                    reported.add(key)
                    tracks = ("one track each way" if pt.directional else f"{c_t} track" + ("s" if c_t != 1 else ""))
                    add("platforms", list(key), code, t,
                        f"{len(here)} trains at {code} at {hhmm(t)} but it has {tracks}: "
                        + ", ".join(f"{p.train} {hhmm(p.t0)}-{hhmm(p.t1) if p.t1 < task.end + 1440 else 'end'}"
                                    for p in here))

    delay = sum(r.delay for r in results.values())
    cancel = task.rule("cancel_cost") * sum(r.unserved for r in results.values())
    carry_cost = sum(carry.values())
    for rid, res in results.items():
        res.cost = res.delay + task.rule("cancel_cost") * res.unserved
        res.problems = sorted(bad.get(rid, set()))
    feasible = not viol
    n_cxl = sum(1 for r in results.values() if r.cancelled)
    n_st = sum(1 for rid, r in results.items()
               if not r.cancelled and (r.first != 0 or r.last != len(task.run(rid).route) - 1))
    return PlanResult(feasible, delay + cancel + carry_cost if feasible else None, delay, cancel, carry_cost, n_cxl,
                      n_st, list(results.values()), viol)


def timetable(task: Task, res: RunResult) -> list[dict]:
    """The served points of one run with arrival and departure (passing) minutes, for display."""
    r = task.run(res.train)
    net = network(task)
    out = []
    if res.cancelled:
        return out
    for i in sorted(set(res.arr) | set(res.dep)):
        code = r.route[i]
        if net.by_code[code].kind == "junction":
            continue
        out.append({"at": code, "arr": hhmm(res.arr.get(i)), "dep": hhmm(res.dep.get(i)),
                    "stop": bool(res.stopped.get(i)), "sched": hhmm(r.arr[i] if r.stops[i] and r.arr[i] is not None
                                                                   else r.dep[i] if r.stops[i] else None),
                    "late": (res.arr[i] - r.arr[i]) if (r.stops[i] and r.arr[i] is not None and i in res.arr) else None})
    return out
