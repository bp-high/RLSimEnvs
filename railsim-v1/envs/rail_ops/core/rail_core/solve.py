"""The reference plan: CP-SAT over the same physics and rules as `check`, proven optimal (or within a stated gap).

The model mirrors `check.simulate` exactly:
    one option per run (run in full, short-turned / started at a turn point, or cancelled)
    arrival and departure minutes at every point it may serve, with the same recurrences: minimum running times,
    never ahead of the timetable, no early departures at scheduled stops, `stop_loss` for an unscheduled stop (a Boolean per passing point)
    each train set's position and readiness carried run to run (a cancelled run leaves both unchanged)
    single-track and headway rules as pairwise disjunctions (pruned where the time windows cannot meet), platform
    tracks as cumulatives, closures as disjunctions, max delay as a domain bound
The solution is turned into a plan of holds, short-turns and cancellations and re-checked with `check.evaluate`: its
cost must equal the solver's objective.

`floor`: the same model without any interaction between train sets (no single-track, headway or platform rules).
Every real plan is feasible there, so its optimum is a lower bound: the cost the closures force on each train set
alone. The reward measures the gap against optimum - floor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .check import evaluate, network, run_dir
from .model import Order, Plan, Run, Task


@dataclass
class Solution:
    plan: Plan | None
    cost: int | None
    bound: int
    optimal: bool
    status: str
    seconds: float


def _options(task: Task, r: Run) -> list[tuple[int, int] | None]:
    net = network(task)
    n = len(r.route)
    turn = [i for i in range(1, n - 1) if net.by_code[r.route[i]].turn]
    if r.started:
        ends = [n - 1] + [i for i in turn if i >= r.fixed_to]
        return [(0, e) for e in ends]
    starts = [0] + turn
    opts: list[tuple[int, int] | None] = [None]
    for s in starts:
        for e in [n - 1] + turn:
            if e > s:
                opts.append((s, e))
    return opts


class _Model:
    def __init__(self, task: Task, interactions: bool = True, horizon_pad: int = 300,
                 unit_floor: dict[str, int] | None = None):
        self.task = task
        self.net = network(task)
        self.m = cp_model.CpModel()
        self.H = task.end + horizon_pad
        self.interactions = interactions
        self.L = task.rule("stop_loss")
        self.turn = task.rule("turn_min")
        self.cost_terms: list = []
        self.const_cost = 0
        self.unit_terms: dict[str, list] = {u.id: [] for u in task.units}
        self.unit_const: dict[str, int] = {u.id: 0 for u in task.units}
        self.unit_floor = unit_floor or {}
        self.rv: dict[str, dict] = {}
        self._build()

    # ---------------------------------------------------------------- per run
    def _bounds(self, r: Run):
        """(lo, hi_arr, hi_dep): earliest minute at each route point, and the latest arrival and departure there in
        any plan that obeys max_delay (a run that may end before its next stop is bounded only by the horizon)."""
        n = len(r.route)
        lo = [0] * n
        k = r.fixed_to if r.started else 0
        t = max(r.fixed_arr, self.task.start) if r.started else r.dep[0]
        lo[k] = r.fixed_arr if r.started else t
        for i in range(k, n - 1):
            if i > k and r.stops[i]:
                t = max(t, r.dep[i])
            t += r.seg[i]
            if r.arr[i + 1] is not None:
                t = max(t, r.arr[i + 1])
            lo[i + 1] = t
        maxd = self.task.rule("max_delay")
        H = self.H
        hi_arr, hi_dep = [H] * n, [H] * n
        cap, travel, open_end = None, 0, False     # the next stop's latest arrival, minutes to it, could end before it
        for i in range(n - 1, k - 1, -1):
            p = self.net.by_code[r.route[i]]
            if i < n - 1:
                travel += r.seg[i]
                if cap is not None and not open_end:
                    hi_dep[i] = cap - travel
            if r.stops[i] and r.arr[i] is not None and i > 0:
                hi_arr[i] = r.arr[i] + maxd
                cap, travel, open_end = r.arr[i] + maxd, 0, False
            elif i < n - 1:
                hi_arr[i] = hi_dep[i] if not p.turn else H
                if p.turn:
                    open_end = True
        for i in range(n):
            hi_dep[i] = max(hi_dep[i], lo[i])
            hi_arr[i] = max(hi_arr[i], lo[i])
        return lo, hi_arr, hi_dep

    def _run_vars(self, r: Run):
        m, net = self.m, self.net
        n = len(r.route)
        opts = _options(self.task, r)
        o = [m.NewBoolVar(f"o_{r.id}_{a}") for a in range(len(opts))]
        m.AddExactlyOne(o)
        lo, hi_arr, hi_dep = self._bounds(r)
        f0 = r.fixed_to if r.started else 0
        arr, dep = {}, {}
        for i in range(f0, n):
            arr[i] = m.NewIntVar(lo[i], hi_arr[i] + (self.L if not r.stops[i] else 0), f"a_{r.id}_{i}")
            if i < n - 1:
                dep[i] = m.NewIntVar(lo[i], hi_dep[i] + self.L, f"d_{r.id}_{i}")

        def bsum(pred, name):
            terms = [o[a] for a, op in enumerate(opts) if op is not None and pred(op)]
            if not terms:
                return 0
            if len(terms) == 1:
                return terms[0]
            b = m.NewBoolVar(name)
            m.Add(b == sum(terms))
            return b

        v = {i: bsum(lambda op, i=i: op[0] <= i <= op[1], f"v_{r.id}_{i}") for i in range(f0, n)}
        u = {i: bsum(lambda op, i=i: op[0] <= i and op[1] >= i + 1, f"u_{r.id}_{i}") for i in range(f0, n - 1)}
        st = {} if r.started else {i: bsum(lambda op, i=i: op[0] == i, f"s_{r.id}_{i}") for i in range(n)}
        en = {i: bsum(lambda op, i=i: op[1] == i, f"e_{r.id}_{i}") for i in range(f0, n)}
        cancel = o[0] if opts[0] is None else 0
        w = {}
        inner = {}
        for i in range(f0, n):
            code = r.route[i]
            vi, si, ei = v[i], st.get(i, 0), en[i]
            if isinstance(vi, int) and vi == 0:
                inner[i] = 0
                continue
            if r.started and i == f0:
                inner[i] = m.NewBoolVar(f"in_{r.id}_{i}")
                m.Add(inner[i] == 1 - ei)
            else:
                inner[i] = m.NewBoolVar(f"in_{r.id}_{i}")
                m.Add(inner[i] == vi - si - ei)
            if not r.stops[i] and net.by_code[code].kind != "junction" and i < n - 1:
                w[i] = m.NewBoolVar(f"w_{r.id}_{i}")
                m.AddImplication(w[i], inner[i])
        # departures and dwell
        for i in range(f0, n - 1):
            code = r.route[i]
            pt = net.by_code[code]
            if not isinstance(inner[i], int):
                lit = inner[i]
                if r.started and i == f0:
                    if r.stops[i]:
                        m.Add(dep[i] >= max(r.dep[i], self.task.start)).OnlyEnforceIf(lit)
                    m.Add(dep[i] >= self.task.start).OnlyEnforceIf(lit)
                if r.stops[i]:
                    m.Add(dep[i] >= arr[i]).OnlyEnforceIf(lit)
                    m.Add(dep[i] >= r.dep[i]).OnlyEnforceIf(lit)
                elif pt.kind == "junction":
                    m.Add(dep[i] == arr[i]).OnlyEnforceIf(lit)
                else:
                    m.Add(dep[i] == arr[i]).OnlyEnforceIf([lit, w[i].Not()])
                    m.Add(dep[i] >= arr[i]).OnlyEnforceIf(w[i])
            if not r.started and not isinstance(st[i], int):
                m.Add(dep[i] >= r.dep[i]).OnlyEnforceIf(st[i])
        if r.started:
            k = f0
            if k in w:
                m.Add(arr[k] == r.fixed_arr + self.L * w[k])
            else:
                m.Add(arr[k] == r.fixed_arr)
        # running
        slow = [c for c in self.task.closures if c.kind == "slow"]
        for i in range(f0, n - 1):
            if isinstance(u[i], int):
                continue
            acc_terms = []
            if i in w:
                acc_terms.append(w[i])
            if not r.started and not r.stops[i] and not isinstance(st.get(i, 0), int):
                acc_terms.append(st[i])
            brk_terms = []
            if (i + 1) in w:
                brk_terms.append(w[i + 1])
            if not r.stops[i + 1] and not isinstance(en[i + 1], int):
                brk_terms.append(en[i + 1])
            extra = 0
            sec = self.net.section(r.route[i], r.route[i + 1])
            for c in slow:
                lo_i, hi_i = sorted((self.net.index[c.where[0]], self.net.index[c.where[-1]]))
                if not lo_i <= sec < hi_i:
                    continue
                y = m.NewBoolVar(f"slow_{r.id}_{i}_{c.start}")
                b1 = m.NewBoolVar("")
                b2 = m.NewBoolVar("")
                m.Add(dep[i] >= c.start).OnlyEnforceIf(b1)
                m.Add(dep[i] < c.start).OnlyEnforceIf(b1.Not())
                m.Add(dep[i] < c.end).OnlyEnforceIf(b2)
                m.Add(dep[i] >= c.end).OnlyEnforceIf(b2.Not())
                m.AddBoolAnd([b1, b2]).OnlyEnforceIf(y)
                m.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(y.Not())
                extra = extra + c.extra * y
            run_t = m.NewIntVar(0, 2 * self.H, f"rt_{r.id}_{i}")
            m.Add(run_t == dep[i] + r.seg[i] + self.L * sum(acc_terms) + extra)
            if r.arr[i + 1] is not None:          # never ahead of the timetable
                reach = m.NewIntVar(0, 2 * self.H, f"rc_{r.id}_{i}")
                m.AddMaxEquality(reach, [run_t, r.arr[i + 1]])
            else:
                reach = run_t
            m.Add(arr[i + 1] == reach + self.L * sum(brk_terms)).OnlyEnforceIf(u[i])
        # cost: delays at served stops (not where it starts), cancelled stops
        C = self.task.rule("cancel_cost")
        for i in range(f0, n):
            if not r.stops[i] or r.arr[i] is None or i == 0:
                continue
            if r.started and i == f0:
                if r.fixed_arr >= self.task.start:
                    self.const_cost += max(0, r.fixed_arr - r.arr[i])
                    self.unit_const[r.unit] += max(0, r.fixed_arr - r.arr[i])
                continue
            lit = m.NewBoolVar(f"late_{r.id}_{i}")
            m.Add(lit == v[i] - st.get(i, 0)) if not isinstance(v[i], int) else m.Add(lit == 0)
            dl = m.NewIntVar(0, self.H, f"dl_{r.id}_{i}")
            m.Add(dl >= arr[i] - r.arr[i]).OnlyEnforceIf(lit)
            self.cost_terms.append(dl)
            self.unit_terms[r.unit].append(dl)
        for a, op in enumerate(opts):
            if op is None:
                self.cost_terms.append(C * r.n_stops * o[a])
                self.unit_terms[r.unit].append(C * r.n_stops * o[a])
            else:
                s, e = op
                lost = sum(1 for j, x in enumerate(r.stops) if x and not (s <= j <= e))
                if lost:
                    self.cost_terms.append(C * lost * o[a])
                    self.unit_terms[r.unit].append(C * lost * o[a])
        self.rv[r.id] = dict(run=r, opts=opts, o=o, arr=arr, dep=dep, v=v, u=u, st=st, en=en, w=w, inner=inner,
                             lo=lo, hi_arr=hi_arr, hi_dep=hi_dep, cancel=cancel, f0=f0)

    # ---------------------------------------------------------------- train sets
    def _units(self):
        m, net = self.m, self.net
        self.waits = []           # (station, start expr/var, end var, literal)
        for u in self.task.units:
            pos = net.index[u.at] if u.at else None
            rdy = u.ready
            arrived = self.task.start if u.at else None
            for k, rid in enumerate(u.runs):
                R = self.rv[rid]
                r = R["run"]
                npos = m.NewIntVar(0, len(net.points) - 1, f"pos_{rid}")
                nrdy = m.NewIntVar(0, self.H + self.turn, f"rdy_{rid}")
                narr = m.NewIntVar(0, self.H, f"arrd_{rid}")
                c = R["cancel"]
                if not isinstance(c, int):
                    if pos is None:
                        m.Add(c == 0)
                    else:
                        m.Add(npos == pos).OnlyEnforceIf(c)
                        m.Add(nrdy == (rdy if rdy is not None else 0)).OnlyEnforceIf(c)
                        m.Add(narr == (arrived if arrived is not None else 0)).OnlyEnforceIf(c)
                if not r.started:
                    for i, sv in R["st"].items():
                        if isinstance(sv, int):
                            continue
                        want = net.index[r.route[i]]
                        if pos is None or (isinstance(pos, int) and pos != want):
                            m.Add(sv == 0)
                        elif not isinstance(pos, int):
                            m.Add(pos == want).OnlyEnforceIf(sv)
                        if rdy is not None:
                            m.Add(R["dep"][i] >= rdy).OnlyEnforceIf(sv)
                        code = r.route[i]
                        p = net.by_code[code]
                        if p.tracks is not None and p.kind != "junction" and self.interactions:
                            self.waits.append((code, arrived, R["dep"][i], sv))
                for i, ev in R["en"].items():
                    if isinstance(ev, int):
                        continue
                    m.Add(npos == net.index[r.route[i]]).OnlyEnforceIf(ev)
                    m.Add(nrdy == R["arr"][i] + self.turn).OnlyEnforceIf(ev)
                    m.Add(narr == R["arr"][i]).OnlyEnforceIf(ev)
                pos, rdy, arrived = npos, nrdy, narr
            if u.after:
                m.Add(pos == net.index[u.after["station"]])
                cr = m.NewIntVar(0, (self.H + self.turn) * int(u.after["stops"]), f"carry_{u.id}")
                m.Add(cr >= int(u.after["stops"]) * (rdy - u.after["dep"]))
                self.cost_terms.append(cr)
                self.unit_terms[u.id].append(cr)
            # a set that ends the window at a station with few tracks keeps one there
            if self.interactions and pos is not None and not isinstance(pos, int):
                for p in net.points:
                    if p.tracks is not None and p.kind != "junction" and p.turn:
                        b = m.NewBoolVar(f"park_{u.id}_{p.code}")
                        m.Add(pos == net.index[p.code]).OnlyEnforceIf(b)
                        m.Add(pos != net.index[p.code]).OnlyEnforceIf(b.Not())
                        self.waits.append((p.code, arrived, self.H, b))

    # ---------------------------------------------------------------- interactions
    def _sections(self):
        m, net = self.m, self.net
        hf, ho = self.task.rule("h_follow"), self.task.rule("h_opp")
        occ: dict[int, list] = {}
        for rid, R in self.rv.items():
            r = R["run"]
            d = run_dir(net, r)
            for i, ui in R["u"].items():
                if isinstance(ui, int) and ui == 0:
                    continue
                sec = net.section(r.route[i], r.route[i + 1])
                occ.setdefault(sec, []).append((rid, d, R["dep"][i], R["arr"][i + 1], ui,
                                                R["lo"][i], R["hi_dep"][i] + self.L, R["lo"][i + 1],
                                                R["hi_arr"][i + 1] + self.L))
            if r.started and r.fixed_from is not None:
                k = r.fixed_to
                sec = net.section(r.route[k - 1], r.route[k])
                occ.setdefault(sec, []).append((rid, d, r.fixed_from, R["arr"][k], 1, r.fixed_from, r.fixed_from,
                                                r.fixed_arr, r.fixed_arr + self.L))
        self.n_pairs = 0
        order: dict[tuple, object] = {}       # (section, first run id, second run id) -> "first goes first"
        for sec, lst in occ.items():
            single = net.sections[sec].single
            for x in range(len(lst)):
                for y in range(x + 1, len(lst)):
                    a, b = lst[x], lst[y]
                    if a[0] == b[0]:
                        continue
                    same = a[1] == b[1]
                    if not same and not single:
                        continue
                    lits = [l for l in (a[4], b[4]) if not isinstance(l, int)]
                    if same:
                        a_first = b[5] >= a[6] + hf and b[7] >= a[8] + hf
                        b_first = a[5] >= b[6] + hf and a[7] >= b[8] + hf
                    else:
                        a_first = b[5] >= a[8] + ho
                        b_first = a[5] >= b[8] + ho
                    if a_first or b_first:
                        order[(sec, a[0], b[0])] = bool(a_first)
                        continue
                    self.n_pairs += 1
                    z = m.NewBoolVar("")
                    order[(sec, a[0], b[0])] = z
                    if same:
                        m.Add(b[2] >= a[2] + hf).OnlyEnforceIf([z] + lits)
                        m.Add(b[3] >= a[3] + hf).OnlyEnforceIf([z] + lits)
                        m.Add(a[2] >= b[2] + hf).OnlyEnforceIf([z.Not()] + lits)
                        m.Add(a[3] >= b[3] + hf).OnlyEnforceIf([z.Not()] + lits)
                    else:
                        m.Add(b[2] >= a[3] + ho).OnlyEnforceIf([z] + lits)
                        m.Add(a[2] >= b[3] + ho).OnlyEnforceIf([z.Not()] + lits)
        self._tie_orders(order, occ)

    def _tie_orders(self, order: dict, occ: dict):
        """Redundant but exact: where two trains cannot pass each other at the point between two sections (a junction,
        a one-track halt, a station with one track each way), their order is the same in both sections. Tying the
        order variables removes symmetric search the solver would otherwise do."""
        m, net = self.m, self.net
        self.n_ties = 0
        dirs = {e[0]: e[1] for lst in occ.values() for e in lst}

        def lit(sec, a, b):
            if (sec, a, b) in order:
                return order[(sec, a, b)]
            if (sec, b, a) in order:
                v = order[(sec, b, a)]
                return (not v) if isinstance(v, bool) else v.Not()
            return None

        for sec in range(len(net.sections) - 1):
            p = net.points[sec + 1]
            no_pass = p.kind == "junction" or p.directional or p.tracks == 1
            one_track = p.tracks == 1 and net.sections[sec].single and net.sections[sec + 1].single
            if not (no_pass or one_track):
                continue
            for (s, a, b) in list(order):
                if s != sec:
                    continue
                same = dirs.get(a) == dirs.get(b)
                if (same and not no_pass) or (not same and not one_track):
                    continue
                x, y = lit(sec, a, b), lit(sec + 1, a, b)
                if x is None or y is None or (isinstance(x, bool) and isinstance(y, bool)):
                    continue
                if isinstance(x, bool):
                    m.AddBoolOr([y if x else y.Not()])
                elif isinstance(y, bool):
                    m.AddBoolOr([x if y else x.Not()])
                else:
                    m.AddImplication(x, y)
                    m.AddImplication(y, x)
                self.n_ties += 1

    def _closures(self):
        m, net = self.m, self.net
        for c in self.task.closures:
            if c.kind != "closed":
                continue
            lo_s, hi_s = sorted((net.index[c.where[0]], net.index[c.where[-1]]))
            for rid, R in self.rv.items():
                r = R["run"]
                for i, ui in R["u"].items():
                    if isinstance(ui, int) and ui == 0:
                        continue
                    sec = net.section(r.route[i], r.route[i + 1])
                    if not lo_s <= sec < hi_s:
                        continue
                    if R["lo"][i] >= c.end or R["hi_arr"][i + 1] + self.L <= c.start:
                        continue
                    lits = [ui] if not isinstance(ui, int) else []
                    z = m.NewBoolVar("")
                    m.Add(R["arr"][i + 1] <= c.start).OnlyEnforceIf([z] + lits)
                    m.Add(R["dep"][i] >= c.end).OnlyEnforceIf([z.Not()] + lits)

    def _stations(self):
        m, net = self.m, self.net
        by_pt: dict[str, list] = {}
        for rid, R in self.rv.items():
            r = R["run"]
            d = run_dir(net, r)
            for i, lit in R["inner"].items():
                if isinstance(lit, int) and lit == 0:
                    continue
                code = r.route[i]
                p = net.by_code[code]
                if p.tracks is None or p.kind == "junction" or i == len(r.route) - 1:
                    continue
                end = m.NewIntVar(0, self.H + 1, "")
                m.Add(end == R["dep"][i] + 1)
                size = m.NewIntVar(1, self.H + 1, "")
                iv = m.NewOptionalIntervalVar(R["arr"][i], size, end, lit, f"p_{rid}_{i}")
                by_pt.setdefault(code, []).append((iv, d))
        for code, start, depv, lit in self.waits:
            s0 = start if start is not None else 0
            if isinstance(s0, int):
                sv = m.NewConstant(s0)
            else:
                sv = s0
            end = m.NewIntVar(0, self.H + 2, "")
            m.Add(end == depv + 1)
            size = m.NewIntVar(0, self.H + 2, "")
            iv = m.NewOptionalIntervalVar(sv, size, end, lit, f"wait_{code}")
            by_pt.setdefault(code, []).append((iv, 0))
        for code, lst in by_pt.items():
            p = net.by_code[code]
            outs = [c for c in self.task.closures if c.kind == "track" and c.where[0] == code]
            fixed = [m.NewFixedSizeIntervalVar(c.start, c.end - c.start, "") for c in outs]
            if p.directional:
                for d in (1, -1):
                    ivs = [iv for iv, dd in lst if dd in (d, 0)]
                    if len(ivs) + len(fixed) > 1:
                        m.AddNoOverlap(ivs + fixed)
            else:
                ivs = [iv for iv, _ in lst] + fixed
                if len(ivs) > p.tracks:
                    m.AddCumulative(ivs, [1] * len(ivs), p.tracks)

    def _build(self):
        for r in self.task.runs:
            self._run_vars(r)
        self._units()
        self._closures()
        self.n_pairs = 0
        if self.interactions:
            self._sections()
            self._stations()
        self.obj = sum(self.cost_terms) + self.const_cost
        # each train set costs at least what it costs alone with the closures (the floor's part): exact, and it
        # lifts the solver's bound to the floor at once, so only the interactions are left to prove
        for uid, f in self.unit_floor.items():
            if uid in self.unit_terms:
                self.m.Add(sum(self.unit_terms[uid]) + self.unit_const[uid] >= f)
        if self.unit_floor:                    # CP-SAT's bound does not add the parts up by itself
            self.m.Add(self.obj >= sum(self.unit_floor.values()))
        self.m.Minimize(self.obj)

    def unit_costs(self, solver) -> dict[str, int]:
        return {uid: int(sum(solver.Value(t) for t in terms)) + self.unit_const[uid]
                for uid, terms in self.unit_terms.items()}

    # ---------------------------------------------------------------- solution
    def plan(self, solver) -> Plan:
        plan: Plan = {}
        for rid, R in self.rv.items():
            r = R["run"]
            a = next(a for a, ov in enumerate(R["o"]) if solver.Value(ov))
            op = R["opts"][a]
            if op is None:
                plan[rid] = Order(cancel=True)
                continue
            s, e = op
            o = Order(start=r.route[s] if s > 0 else None, end=r.route[e] if e < len(r.route) - 1 else None)
            first = R["f0"] if r.started else s
            for i in range(first, e):
                if self.net.by_code[r.route[i]].kind == "junction":
                    continue
                o.hold[r.route[i]] = solver.Value(R["dep"][i])
            plan[rid] = o
        return _tidy(self.task, plan)

    def hint(self, plan: Plan):
        """Hint the solver with a plan's times (from `check.simulate`)."""
        from .check import simulate
        res, *_ = simulate(self.task, plan)
        for rid, R in self.rv.items():
            x = res.get(rid)
            o = plan.get(rid) or Order()
            r = R["run"]
            if x is None:
                continue
            if x.cancelled:
                key = None
            else:
                key = (0 if r.started else x.first, x.last)
            for a, op in enumerate(R["opts"]):
                self.m.AddHint(R["o"][a], int(op == key))
            if x.cancelled:
                continue
            for i, t in x.arr.items():
                if i in R["arr"]:
                    self.m.AddHint(R["arr"][i], t)
            for i, t in x.dep.items():
                if i in R["dep"]:
                    self.m.AddHint(R["dep"][i], t)
            for i, wv in R["w"].items():
                self.m.AddHint(wv, int(bool(x.stopped.get(i)) and not r.stops[i] and i != x.first and i != x.last))


def _tidy(task: Task, plan: Plan) -> Plan:
    """Drop holds that change nothing (the train would leave then anyway), so reference plans read cleanly."""
    from .check import simulate
    out: Plan = {}
    for rid, o in plan.items():
        if o.cancel:
            out[rid] = o
            continue
        out[rid] = Order(start=o.start, end=o.end, hold=dict(o.hold))
    base, *_ = simulate(task, out)
    for rid, o in list(out.items()):
        if o.cancel:
            continue
        for code in list(o.hold):
            trial = Order(start=o.start, end=o.end, hold={k: v for k, v in o.hold.items() if k != code})
            res, *_ = simulate(task, {**out, rid: trial})
            if res[rid].arr == base[rid].arr and res[rid].dep == base[rid].dep:
                out[rid] = trial
                o = trial
    return {rid: o for rid, o in out.items() if o.cancel or o.start or o.end or o.hold}


def solve(task: Task, time_limit: float = 60.0, workers: int = 4, hint: Plan | None = None,
          interactions: bool = True, log: bool = False, unit_floor: dict[str, int] | None = None) -> Solution:
    t0 = time.time()
    M = _Model(task, interactions=interactions, unit_floor=unit_floor)
    if hint is not None:
        M.hint(hint)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = log
    st = solver.Solve(M.m)
    name = solver.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Solution(None, None, int(solver.BestObjectiveBound()) if st != cp_model.INFEASIBLE else 0, False, name,
                        time.time() - t0)
    plan = M.plan(solver) if interactions else None
    return Solution(plan, int(round(solver.ObjectiveValue())), int(round(solver.BestObjectiveBound())),
                    st == cp_model.OPTIMAL, name, time.time() - t0)


def floor_cost(task: Task, time_limit: float = 30.0) -> int:
    """Lower bound: each train set alone with the closures (no interactions)."""
    return floor_parts(task, time_limit)[0]


def floor_parts(task: Task, time_limit: float = 30.0) -> tuple[int, dict[str, int]]:
    """(floor, each train set's own part). Without interactions the sets are independent, so a proven optimum is
    the sum of each set's optimum alone; the parts are only returned when that is proven."""
    M = _Model(task, interactions=False)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 4
    st = solver.Solve(M.m)
    if st == cp_model.OPTIMAL:
        return int(round(solver.ObjectiveValue())), M.unit_costs(solver)
    return int(solver.BestObjectiveBound()) if st == cp_model.FEASIBLE else 0, {}


def optimal_plan(task: Task, time_limit: float = 60.0, workers: int = 4, hint: Plan | None = None,
                 unit_floor: dict[str, int] | None = None) -> Solution:
    sol = solve(task, time_limit=time_limit, workers=workers, hint=hint, unit_floor=unit_floor)
    if sol.plan is not None:
        res = evaluate(task, sol.plan)
        if not res.feasible or res.cost != sol.cost:
            raise AssertionError(f"{task.task_id}: solver plan re-checks to feasible={res.feasible} cost={res.cost}, "
                                 f"solver said {sol.cost}: {res.violations[:3]}")
    return sol
