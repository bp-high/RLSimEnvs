"""What the agent reads: `rules(task)` is the system prompt, `situation(task)` what get_situation returns.

Both are rendered at runtime from the task, so changing them never needs a pack rebuild.
"""

from __future__ import annotations

from .check import network
from .model import Run, Task, hhmm

PROMPT_VERSION = "rail-v1.0"
KIND_NAME = {"stop": "Stoptrein (stopping)", "fast": "Sneltrein (fast)"}


def rules(task: Task, max_checks: int = 10) -> str:
    R = task.rules
    net = network(task)
    turns = ", ".join(p.code for p in net.points if p.turn)
    return f"""You are the train dispatcher (treindienstleider) for Arriva's line between Leeuwarden and Groningen in the \
Netherlands, on a real day in 2025. Long stretches of the line are single track: trains going opposite ways can only \
pass each other at stations with two tracks or on the double-track parts. Something has gone wrong and you must re-plan \
the trains for the rest of the window. Everything you need is in the situation; it is the whole truth (no surprises \
later).

How trains move (the physics; deterministic):
- Every train set works its runs in the order listed. A run starts where its train set is, no earlier than when the set \
is ready there and never before its scheduled departure.
- Unless your plan says otherwise, a train runs at its minimum running times and leaves as soon as it can, but it never \
leaves a scheduled stop before its scheduled departure and never reaches any point before its timetable minute there \
(the passing minutes marked * included): arrival at the next point = max(departure + running time, timetable minute). \
A late train makes up time this way.
- A train held where it has no scheduled stop (a fast train at a station it normally passes) loses \
{R['stop_loss']} minute braking and {R['stop_loss']} minute accelerating. Trains cannot be held at junctions (J1, J2).
- After a run ends, its train set is ready {R['turn_min']} minutes after arriving, at the point where the run ended.

Rules (a plan that breaks any of them is infeasible):
1. Single track: two trains going opposite ways may never be in a single-track section at the same time: the second \
may enter only when the first has left it (arrived at the far end){' - in that same minute at the earliest' if not R['h_opp'] else f" - {R['h_opp']} minute(s) later at the earliest"}. \
A train is in a section from its departure at one end to its arrival at the other.
2. Headway: trains going the same way stay at least {R['h_follow']} minutes apart at both ends of every section, so a \
train can only overtake another at a station where both fit.
3. Platforms: never more trains standing at (or passing through) a station than it has tracks. A directional station \
(on double track) has one track each way. A train set waiting at a station between runs keeps a track there. \
Terminals (LW, GN) have room for everyone.
4. Closures: no train may be in a closed section while it is closed. A train stuck inside when it closed waits there.
5. Continuity: a run can only start where its train set is. You may short-turn trains only at the turn points \
({turns}). A train already running cannot be cancelled (short-turn it instead). At the end of the window every train \
set must be where its next run (listed) leaves from.
6. No train may reach a stop more than {R['max_delay']} minutes late.

What you decide, per train (run), in the plan:
- "hold": {{"HDG": "07:31"}} keeps it at that point until 07:31 (it leaves then, or later if it is not ready). Use \
holds to make trains wait for each other or for a closure to end.
- "to": "HDG" ends the run early at a turn point (short-turn); its train set is then at HDG, so its next run must \
start there.
- "from": "HDG" starts a run at a turn point instead of its origin (where its set was short-turned).
- "cancel": true does not run it at all; its train set stays where it is (so usually its next run is cancelled too, \
or starts from where the set is).
Trains you leave out of the plan run in full, as early as they can. An empty plan [] changes nothing.

Cost, in stop-minutes (lower is better): every minute a train reaches one of its scheduled stops late (arrival at each \
stop it serves after where it starts), plus {R['cancel_cost']} for every scheduled stop a train does not serve \
(cancelled or cut off by a short-turn: passengers wait for the next train), plus, for each train set, the minutes it is \
late for its next run after the window times that run's number of stops.

Tools:
- get_situation(): the line, the closures, the trains and the train sets.
- check_plan(plan): simulates your plan and returns every train's times, delay, the rule breaks and the cost. You \
have {max_checks} checks. It never grades the plan.
- submit_plan(plan): your final plan. It ends the episode and is graded once against the best possible plan (proved \
by a solver). A plan that breaks a rule scores below 0.2; a plan that breaks none scores 0.2-1.0 by how close its cost \
is to the best.

Plan format, a JSON list with one entry per train you change, for example:
[{{"train": "37413", "to": "HDG"}}, {{"train": "37424", "from": "HDG", "hold": {{"HDG": "07:31"}}}}, \
{{"train": "37326", "hold": {{"FWD": "07:55", "BP": "08:05"}}}}, {{"train": "37419", "cancel": true}}]
Times are "HH:MM" (after midnight: 24:10)."""


def _line_diagram(task: Task) -> list[str]:
    net = network(task)
    parts = []
    for i, p in enumerate(net.points):
        if p.kind == "terminal":
            tag = f"{p.code}[terminal]"
        elif p.kind == "junction":
            tag = f"{p.code}"
        elif p.directional:
            tag = f"{p.code}[1 each way]"
        else:
            tag = f"{p.code}[{p.tracks}{' turn' if p.turn else ''}]"
        if p.turn and p.kind != "terminal" and "turn" not in tag:
            tag = tag[:-1] + " turn]"
        parts.append(tag)
        if i < len(net.sections):
            parts.append(" ---single--- " if net.sections[i].single else " ===double=== ")
    lines = ["West to east (km from Leeuwarden in brackets):"]
    lines.append("  " + ", ".join(f"{p.code} {p.name} ({p.km:g})" for p in net.points))
    lines.append("  " + "".join(parts))
    lines.append("  [n] = tracks at a station; [1 each way] = one track per direction (no overtaking); turn = trains "
                 "can be short-turned there; J1, J2 = junctions where double track becomes single (no stopping).")
    return lines


def _pattern(r: Run) -> tuple:
    return (r.kind, tuple(r.route), tuple(r.stops), tuple(r.seg))


def situation(task: Task) -> str:
    net = network(task)
    out = [f"Groningen - Leeuwarden, {task.day}. It is now {hhmm(task.start)}. You plan every run that leaves "
           f"before {hhmm(task.end)} (and those already running)."]
    out.append("")
    out.append("THE LINE")
    out += _line_diagram(task)
    out.append("")
    out.append("WHAT HAS HAPPENED")
    for n in task.notices:
        out.append(f"- {n}")
    for c in task.closures:
        where = "-".join(c.where) if c.kind != "track" else c.where[0]
        if c.kind == "closed":
            out.append(f"- CLOSED: every section between {c.where[0]} and {c.where[-1]}, {hhmm(c.start)} to "
                       f"{hhmm(c.end)} ({c.cause}).")
        elif c.kind == "slow":
            out.append(f"- SLOW: trains entering a section between {c.where[0]} and {c.where[-1]} from "
                       f"{hhmm(c.start)} to {hhmm(c.end)} take {c.extra} minutes longer ({c.cause}).")
        else:
            out.append(f"- TRACK OUT OF USE: one track at {where} cannot be used {hhmm(c.start)} to {hhmm(c.end)} "
                       f"({c.cause}).")
    if not task.notices and not task.closures:
        out.append("- Nothing beyond the delays below.")
    out.append("")
    out.append("MINIMUM RUNNING TIMES (minutes from each point to the next, for each stopping pattern)")
    patterns: dict[tuple, list[str]] = {}
    for r in task.runs:
        patterns.setdefault(_pattern(r), []).append(r.id)
    for k, (pat, ids) in enumerate(patterns.items()):
        kind, route, stops, seg = pat
        legs = " ".join(f"{route[i]}{'' if stops[i] else '*'} {seg[i]}" for i in range(len(seg))) + f" {route[-1]}"
        out.append(f"P{k + 1} {KIND_NAME[kind]} {route[0]}->{route[-1]}: {legs}")
    out.append("  (* = no scheduled stop there: the train passes; held there it loses "
               f"{task.rule('stop_loss')}+{task.rule('stop_loss')} min)")
    pid = {rid: f"P{k + 1}" for k, ids in enumerate(patterns.values()) for rid in ids}
    out.append("")
    out.append("TRAINS (timetable: arrival/departure at stops, passing minute at points marked *; the pattern gives the "
               "running times)")
    for r in sorted(task.runs, key=lambda r: (r.fixed_arr if r.started else r.dep[0])):
        stops = []
        for i, code in enumerate(r.route):
            if not r.stops[i]:
                stops.append(f"{code}* {hhmm(r.arr[i])}")
                continue
            a, d = r.arr[i], r.dep[i]
            if a is None:
                stops.append(f"{code} {hhmm(d)}")
            elif d is None:
                stops.append(f"{code} {hhmm(a)}")
            elif a == d:
                stops.append(f"{code} {hhmm(a)}")
            else:
                stops.append(f"{code} {hhmm(a)}/{hhmm(d)}")
        head = f"{r.id} {r.kind} {r.origin}->{r.dest} [{pid[r.id]}, set {r.unit}]"
        if r.started:
            k = r.fixed_to
            code = r.route[k]
            sched = r.arr[k] if r.arr[k] is not None else r.dep[k]
            late = r.fixed_arr - sched if sched is not None else 0
            if r.fixed_arr <= task.start and r.stops[k]:
                pos = f"standing at {code} since {hhmm(r.fixed_arr)}"
            elif r.stuck:
                pos = f"STUCK in the closed section before {code}; reaches {code} at {hhmm(r.fixed_arr)} at the earliest"
            else:
                pos = f"running, reaches {code} at {hhmm(r.fixed_arr)}"
            head += f" RUNNING ({pos}; {late:+d} min on its timetable there)"
        out.append(f"{head}: " + ", ".join(stops))
    out.append("")
    out.append("TRAIN SETS (each works its runs in this order)")
    for u in task.units:
        first = task.run(u.runs[0])
        if first.started:
            where = f"running {first.id}"
        else:
            where = f"at {u.at}, ready {hhmm(u.ready)}"
        nxt = (f"; must then be at {u.after['station']} for {u.after['run']} at {hhmm(u.after['dep'])} "
               f"({u.after['stops']} stops)") if u.after else "; no further runs today"
        out.append(f"{u.id}: {where}; runs " + " -> ".join(u.runs) + nxt)
    return "\n".join(out)


def brief(task: Task) -> str:
    return f"{task.task_id}: {len(task.runs)} runs, {len(task.units)} train sets, {len(task.closures)} closures"
