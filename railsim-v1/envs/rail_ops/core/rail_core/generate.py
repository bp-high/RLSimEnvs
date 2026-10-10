"""A task from a real day: the line as it really stood at `start`, every run that leaves before `end`, the closures.

The state at `start` comes from the 2025 record:
    a train running then is placed where its actual departure from its last stop puts it (passing points by the
    minimum running times); it keeps its real delay
    a train set standing at a terminal is ready `turn_min` after its last run's actual arrival
Everything after `start` is decided by the plan; the record's later times are kept only to score what the real
dispatchers did (`reference.real`).

A window is refused (`None`) when it cannot be rebuilt honestly: a run before `start` was cancelled or cut short in
the record (its train is then not where the schedule says), or the plain timetable already conflicts inside it.
"""

from __future__ import annotations

from .check import evaluate, network
from .data import Day, Service, load_day
from .model import RULES, Closure, Order, Run, Task, Unit, hhmm
from .network import Network, load_network

# causes after which the train involved is itself out of service (it blocks the section until the line reopens)
FAILURES = ("broken down train", "collision", "defective train", "a collision", "an accident")


def _actual(svc: Service, code: str) -> tuple[int | None, int | None, bool]:
    """(actual arrival, actual departure, cancelled) at a stop of the record."""
    for st in svc.stops:
        if st.code == code:
            return st.aa, st.adep, (st.arr_cxl or st.sa is None) and (st.dep_cxl or st.sd is None)
    return None, None, True


def _is_failure(net: Network, r: Run, served: list[str], closed: list[Closure]) -> bool:
    """The run that broke down (or collided): cut short by a failure closure, last served next to the closed span."""
    for c in closed:
        if not any(f in c.cause.lower() for f in FAILURES):
            continue
        lo, hi = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
        last = served[-1] if served else r.origin
        if lo - 1 <= net.index[last] <= hi + 1:
            return True
    return False


def _served_in_record(svc: Service) -> list[str]:
    return [st.code for st in svc.stops if not ((st.arr_cxl or st.sa is None) and (st.dep_cxl or st.sd is None))]


def _place(net: Network, r: Run, svc: Service, start: int, closed: list[Closure]):
    """Where a run is at `start`, from the record: None (it was cut short before `start`: the window cannot be
    rebuilt), ("done", actual arrival), ("future", None), or (fixed_to, fixed_arr, fixed_from, stuck)."""
    last_t = last_i = None
    stop_at = len(r.route) - 1                   # the first stop it had not reached by `start`
    for i, code in enumerate(r.route):
        if not r.stops[i]:
            continue
        aa, ad, cxl = _actual(svc, code)
        t_sched = r.arr[i] if r.arr[i] is not None else r.dep[i]
        if cxl:
            if t_sched < start:
                return None
            stop_at = i
            break
        if i > 0 and (aa is None or aa > start):
            stop_at = i
            break
        if i == len(r.route) - 1:
            return ("done", aa)
        if ad is None or ad > start:
            return ("future", None) if i == 0 else (i, aa, None, False)
        last_t, last_i = ad, i
    if last_i is None:
        return ("future", None)
    t = last_t
    for i in range(last_i, stop_at):
        nxt = t + r.seg[i]
        if r.arr[i + 1] is not None:
            nxt = max(nxt, r.arr[i + 1])
        sec = net.section(r.route[i], r.route[i + 1])
        for c in closed:
            lo, hi = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
            if lo <= sec < hi and c.start <= start < c.end and t < c.start:
                return (i + 1, max(start, c.end + max(0, nxt - c.start)), t, True)   # caught in the closed section
        if nxt >= start or i + 1 == stop_at:
            return (i + 1, max(start, nxt), t, False)
        t = nxt
    return ("done", t)


def build_task(date: str, start: int, end: int, closures: list[Closure] | None = None, *, task_id: str = "",
               split: str = "train", difficulty: str = "standard", notices: list[str] | None = None,
               disruptions: list[dict] | None = None, rules: dict | None = None,
               lookback: int = 150, why: list | None = None) -> Task | None:
    why = why if why is not None else []
    net = load_network()
    day = load_day(date)
    rules = {**RULES, **(rules or {})}
    turn = rules["turn_min"]
    closures = list(closures or [])
    closed = [c for c in closures if c.kind == "closed"]
    runs: list[Run] = []
    units: list[Unit] = []
    notes = list(notices or [])
    real_runs: dict[str, dict] = {}
    failed = None                                     # (unit, run, last stop): the train that broke down
    for chain in day.chains:
        u_runs: list[Run] = []
        at, ready, after = None, None, None
        last_hist: Service | None = None
        for rid in chain:
            r0 = day.runs[rid]
            svc = day.services[rid]
            sched_dep = r0.dep[0]
            if sched_dep < start - lookback:          # long before the window: history
                last_hist = svc
                at, ready = r0.dest, None
                aa, _, cxl = _actual(svc, r0.dest)
                if aa is not None and not cxl:
                    ready = aa + turn
                continue
            if last_hist is not None:
                if last_hist.full_cxl or last_hist.part_cxl:
                    why.append(f"{last_hist.train} (before the window) was cut short: its train is not where the "
                               f"schedule says")
                    return None
                last_hist = None
            if sched_dep >= end:
                after = {"run": rid, "station": r0.origin, "dep": sched_dep, "stops": r0.n_stops}
                break
            if svc.full_cxl or svc.part_cxl:
                served = _served_in_record(svc)
                cut_before = [c for c in r0.route if r0.stops[r0.route.index(c)] and c not in served]
                early = [c for c in cut_before if (r0.arr[r0.idx(c)] or r0.dep[r0.idx(c)]) < start]
                if early:
                    if failed is None and _is_failure(net, r0, served, closed):
                        failed = (r0.unit, rid, served[-1] if served else r0.origin)
                        u_runs = []
                        break
                    why.append(f"{rid} was cut short before the window ({', '.join(early)})")
                    return None
            r = Run(**{**r0.__dict__})
            r.route, r.stops, r.arr, r.dep, r.seg = list(r0.route), list(r0.stops), list(r0.arr), list(r0.dep), list(r0.seg)
            placed = _place(net, r, svc, start, closed)
            if placed is None:
                why.append(f"{rid}: a stop before the window was cancelled")
                return None
            if isinstance(placed, tuple) and placed[0] == "done":
                at, ready = r.dest, max(start, placed[1] + turn)
                continue
            if isinstance(placed, tuple) and placed[0] == "future":
                u_runs.append(r)
            elif isinstance(placed, tuple):
                fixed_to, fixed_arr, fixed_from, stuck = placed
                if u_runs:                            # a later run of the same set cannot be running already
                    why.append(f"{rid} is running but an earlier run of its set {r.unit} has not")
                    return None
                r.fixed_to, r.fixed_arr, r.fixed_from, r.stuck = fixed_to, fixed_arr, fixed_from, stuck
                u_runs.append(r)
                at, ready = None, None
            real_runs[rid] = _real(r, svc, start)
        if not u_runs or (failed and u_runs[0].unit == failed[0]):
            continue
        if u_runs[0].started:
            at, ready = None, None
        else:
            if at is None:
                at = u_runs[0].origin
            ready = start if ready is None else max(start, ready)
        units.append(Unit(id=u_runs[0].unit, runs=[r.id for r in u_runs], at=at, ready=ready, after=after))
        runs += u_runs
    if not runs:
        why.append("no runs")
        return None
    if failed:
        c = closed[0]
        notes.insert(0, f"Train {failed[1]} broke down after {failed[2]} and is stuck in {c.where[0]}-{c.where[-1]}; "
                        f"its train set is out of service for the rest of the window, so its later runs are already "
                        f"cancelled and are not yours to plan.")
    task = Task(task_id=task_id or f"{date}-{hhmm(start).replace(':', '')}", split=split, line="GN-LW", day=date,
                start=start, end=end, network=net.to_dict(), runs=runs, units=units, closures=closures, notices=notes,
                disruptions=list(disruptions or []), difficulty=difficulty, reference={}, rules=rules)
    # the plain timetable (no delays, no closures) must be conflict-free in the window
    calm = Task.from_dict({**task.to_dict(), "closures": []})
    for r in calm.runs:
        r.fixed_to, r.fixed_arr, r.fixed_from, r.stuck = 0, None, None, False
    for u in calm.units:
        first = calm.run(u.runs[0])
        u.at, u.ready = first.origin, None
    calm_res = evaluate(calm, {})
    if not calm_res.feasible:
        why.append("the plain timetable conflicts in the window: " + calm_res.violations[0]["detail"][:160])
        return None
    task.reference["real"] = _real_cost(task, real_runs)
    return task


def _real(r: Run, svc: Service, start: int) -> dict:
    """What really happened to a run: actual arrivals at the stops it served, and the stops it did not."""
    arr, unserved = {}, 0
    for i, code in enumerate(r.route):
        if not r.stops[i]:
            continue
        aa, ad, cxl = _actual(svc, code)
        if cxl:
            unserved += 1
        elif i > 0 and aa is not None:
            arr[code] = aa
    return {"arr": arr, "unserved": unserved, "cancelled": svc.full_cxl, "short": svc.part_cxl and not svc.full_cxl}


def _real_cost(task: Task, real_runs: dict[str, dict]) -> dict:
    """The record's outcome scored with the task's cost (delays at served stops reached after `start`, cancel cost
    for unserved stops). It is what Arriva and ProRail did with their information; it need not obey this model."""
    delay = unserved = cxl = short = 0
    for r in task.runs:
        x = real_runs.get(r.id)
        if x is None:
            continue
        for i, code in enumerate(r.route):
            if r.stops[i] and code in x["arr"] and x["arr"][code] >= task.start and r.arr[i] is not None:
                delay += max(0, x["arr"][code] - r.arr[i])
        unserved += x["unserved"]
        cxl += bool(x["cancelled"])
        short += bool(x["short"])
    return {"cost": delay + task.rule("cancel_cost") * unserved, "delay": delay, "unserved_stops": unserved,
            "cancelled": cxl, "short_turned": short}
