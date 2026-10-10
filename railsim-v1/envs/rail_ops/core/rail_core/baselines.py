"""Cheap dispatching policies: the lazy bounds a task must stay hard against, and the solver's starting point.

    naive     change nothing: every train runs as scheduled and as early as it can (usually breaks the rules)
    greedy    first come, first served: trains are planned one at a time in the order they are ready to leave; each
              takes the earliest times that clear every train already planned (holding at its scheduled stops), and
              a train whose path is closed for more than `turn_after` minutes is short-turned at the last turn point
              before the closure, its train starting back from there. This is how a dispatcher works by rule of
              thumb; it never cancels and never makes a train wait to let a later one through.
    greedy_fast   the same, but fast trains (Sneltrein) are planned before stopping trains that are ready at the
              same time (within 10 minutes): priority to the fast service
"""

from __future__ import annotations

import heapq
from collections import defaultdict

from .check import evaluate, network, run_dir
from .model import Order, Plan, Run, Task

MAXWAIT = 240
STEP_BUDGET = 60_000    # departure times tried per run before the greedy gives up on it


class _Book:
    """What is already planned: section occupancies and station presences."""

    def __init__(self, task: Task):
        self.task = task
        self.net = network(task)
        self.sec: dict[int, list] = defaultdict(list)       # (dir, enter, exit)
        self.pres: dict[str, list] = defaultdict(list)      # (dir, t0, t1)
        self.hf, self.ho = task.rule("h_follow"), task.rule("h_opp")
        self.closed = [c for c in task.closures if c.kind == "closed"]
        self.tracks_out = [c for c in task.closures if c.kind == "track"]
        self.slow = [c for c in task.closures if c.kind == "slow"]

    def section_ok(self, sec: int, d: int, t0: int, t1: int, history: bool = False) -> bool:
        s = self.net.sections[sec]
        for dd, a, b in self.sec[sec]:
            if dd == d:
                if (t0 >= a and (t0 - a < self.hf or t1 - b < self.hf)) or \
                   (t0 < a and (a - t0 < self.hf or b - t1 < self.hf)):
                    return False
            elif s.single and not (b + self.ho <= t0 or t1 + self.ho <= a):
                return False
        if not history:
            for c in self.closed:
                lo, hi = sorted((self.net.index[c.where[0]], self.net.index[c.where[-1]]))
                if lo <= sec < hi and t0 < c.end and t1 > c.start:
                    return False
        return True

    def station_ok(self, code: str, d: int, t0: int, t1: int) -> bool:
        p = self.net.by_code[code]
        if p.tracks is None or p.kind == "junction":
            return True
        others = [(dd, a, b) for dd, a, b in self.pres[code] if not (b < t0 or a > t1)]
        if p.directional:
            return not any(dd in (d, 0) for dd, _, _ in others)
        # peak count over [t0, t1] including this train
        times = sorted({t0} | {a for _, a, _ in others if t0 <= a <= t1} |
                       {c.start for c in self.tracks_out if c.where[0] == code and t0 <= c.start <= t1})
        for t in times:
            n = 1 + sum(1 for _, a, b in others if a <= t <= b)
            cap = p.tracks - sum(1 for c in self.tracks_out if c.where[0] == code and c.start <= t < c.end)
            if n > cap:
                return False
        return True

    def slow_extra(self, sec: int, t: int) -> int:
        x = 0
        for c in self.slow:
            lo, hi = sorted((self.net.index[c.where[0]], self.net.index[c.where[-1]]))
            if lo <= sec < hi and c.start <= t < c.end:
                x = max(x, c.extra)
        return x


def _closure_wait(book: _Book, r: Run, i: int, t: int) -> int:
    """Minutes a train entering route[i] -> route[i+1] at t would wait for a closure."""
    sec = book.net.section(r.route[i], r.route[i + 1])
    w = 0
    for c in book.closed:
        lo, hi = sorted((book.net.index[c.where[0]], book.net.index[c.where[-1]]))
        if lo <= sec < hi and c.start <= t + r.seg[i] and t < c.end:
            w = max(w, c.end - t)
    return w


def _path(book: _Book, r: Run, first: int, last: int, a0: int | None, ready: int, wait_from: int | None):
    """Earliest conflict-free times for one run over route[first..last] given what is booked: (arr, dep, stopped)
    or None. a0: the fixed arrival at `first` of a run already moving (else the run starts at `first`, where its
    train has stood since `wait_from` if that is a station with few tracks)."""
    net, task = book.net, book.task
    L = task.rule("stop_loss")
    d = run_dir(net, r)
    fail: set = set()
    steps = [0]
    arr: dict[int, int] = {}
    dep: dict[int, int] = {}
    stopped: dict[int, bool] = {}

    def go(i: int, a: int) -> bool:
        if i == last:
            arr[i], stopped[i] = a, True
            return True
        if (i, a) in fail:
            return False
        code = r.route[i]
        p = net.by_code[code]
        starting = i == first and a0 is None
        if starting:
            d0 = max(ready, r.dep[i])
        elif r.stops[i]:
            d0 = max(a, r.dep[i])
        else:
            d0 = a
        if i == first and a0 is not None:
            d0 = max(d0, task.start)
        sec = net.section(code, r.route[i + 1])
        for t in range(d0, d0 + MAXWAIT):
            steps[0] += 1
            if steps[0] > STEP_BUDGET:
                return False
            if starting:
                a_here, acc, halt, t0 = None, (0 if r.stops[i] else L), True, wait_from
            elif r.stops[i] or (i == first and a0 is not None and t > a and False):
                a_here, acc, halt, t0 = a, 0, True, a
            elif t == a:
                a_here, acc, halt, t0 = a, 0, False, a
            else:
                break                           # greedy only waits where the train has a scheduled stop
            if t0 is not None and not book.station_ok(code, d, t0, t):
                if halt:
                    break                       # a longer wait here only makes it worse
                continue
            nxt = t + r.seg[i] + acc + book.slow_extra(sec, t)
            if r.arr[i + 1] is not None:
                nxt = max(nxt, r.arr[i + 1])
            if i + 1 == last and not r.stops[last]:
                nxt += L
            if not book.section_ok(sec, d, t, nxt):
                continue
            if a_here is not None:
                arr[i] = a_here
            dep[i], stopped[i] = t, halt
            if go(i + 1, nxt):
                return True
        fail.add((i, a))
        return False

    if not go(first, a0 if a0 is not None else 0):
        return None
    return arr, dep, stopped


def greedy(task: Task, fast_first: bool = False, turn_after: int = 20) -> Plan:
    book = _Book(task)
    net = book.net
    runs = task.run_map
    turn = task.rule("turn_min")
    plan: Plan = {}
    for r in task.runs:               # trains between points when the task starts
        if r.started and r.fixed_from is not None:
            k = r.fixed_to
            book.sec[net.section(r.route[k - 1], r.route[k])].append((run_dir(net, r), r.fixed_from, r.fixed_arr))
    state = {}
    for u in task.units:
        finite = u.at is not None and net.by_code[u.at].tracks is not None
        state[u.id] = {"k": 0, "ready": u.ready, "wait_from": task.start if finite else None, "start_at": None,
                       "park": None}
        if finite:
            stay = (0, task.start, task.end + 1440)
            book.pres[u.at].append(stay)
            state[u.id]["park"] = (u.at, stay)
    heap = []
    units = {u.id: u for u in task.units}

    def push(u):
        s = state[u.id]
        if s["k"] >= len(u.runs):
            return
        r = runs[u.runs[s["k"]]]
        t = r.fixed_arr if r.started else max(s["ready"] or 0, r.dep[0])
        pri = (t // 10, 0 if r.kind == "fast" else 1, t) if fast_first else (t, 0, 0)
        heapq.heappush(heap, (pri, u.id))

    for u in task.units:
        push(u)
    while heap:
        _, uid = heapq.heappop(heap)
        u = units[uid]
        s = state[uid]
        r = runs[u.runs[s["k"]]]
        if s["park"]:                 # its provisional stay is replaced by the real one below
            code, stay = s["park"]
            book.pres[code].remove(stay)
            s["park"] = None
        has_next = s["k"] + 1 < len(u.runs)
        first = r.fixed_to if r.started else (r.idx(s["start_at"]) if s["start_at"] else 0)
        last = len(r.route) - 1
        if has_next:                  # short-turn before a long closure, if its train has a run to come back on
            nxt = runs[u.runs[s["k"] + 1]]
            t = r.fixed_arr if r.started else max(s["ready"] or 0, r.dep[first])
            for i in range(first, last):
                if i > first and r.stops[i]:
                    t = max(t, r.dep[i])
                if _closure_wait(book, r, i, t) > turn_after:
                    lo = first if r.started else first + 1
                    # only where the next run can start (some runs work part of the line, e.g. Groningen - Zuidhorn)
                    cands = [j for j in range(lo, i + 1) if net.by_code[r.route[j]].turn
                             and r.route[j] in nxt.route[:-1]]
                    if cands:
                        last = cands[-1]
                    break
                t += r.seg[i]
        a0 = r.fixed_arr if r.started else None
        wait_from = None if r.started else s["wait_from"]
        res = _path(book, r, first, last, a0, s["ready"] or 0, wait_from)
        if res is None and last != len(r.route) - 1:
            last = len(r.route) - 1
            res = _path(book, r, first, last, a0, s["ready"] or 0, wait_from)
        if res is None:
            plan[r.id] = Order(start=r.route[first] if (not r.started and first > 0) else None)
            s["k"] += 1
            s["ready"], s["wait_from"], s["start_at"] = None, None, None
            push(u)
            continue
        arr, dep, stopped = res
        d = run_dir(net, r)
        if wait_from is not None:
            book.pres[r.route[first]].append((d, wait_from, dep[first]))
        for i in range(first, last):
            book.sec[net.section(r.route[i], r.route[i + 1])].append((d, dep[i], arr[i + 1]))
            if (i > first or r.started) and net.by_code[r.route[i]].tracks is not None:
                book.pres[r.route[i]].append((d, arr[i], dep[i]))
        plan[r.id] = Order(start=r.route[first] if (not r.started and first > 0) else None,
                           end=r.route[last] if last < len(r.route) - 1 else None,
                           hold={r.route[i]: dep[i] for i in range(first, last)
                                 if net.by_code[r.route[i]].kind != "junction"})
        end_code = r.route[last]
        s["k"] += 1
        s["ready"] = arr[last] + turn
        s["start_at"] = end_code if last < len(r.route) - 1 else None
        p_end = net.by_code[end_code]
        if p_end.tracks is not None and p_end.kind != "junction":
            s["wait_from"] = arr[last]
            stay = (0, arr[last], task.end + 1440)
            book.pres[end_code].append(stay)
            s["park"] = (end_code, stay)
        else:
            s["wait_from"] = None
        push(u)
    from .solve import _tidy
    return _tidy(task, plan)


def naive(task: Task) -> Plan:
    return {}


POLICIES = {"naive": naive, "greedy": greedy, "greedy_fast": lambda t: greedy(t, fast_first=True)}
