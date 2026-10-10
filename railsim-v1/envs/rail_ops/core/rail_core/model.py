"""Tasks and plans.

A task is a window of one real 2025 day on the Groningen - Leeuwarden line after something went wrong. The agent is
the dispatcher: every train (a *run* from one end of the line to the other) runs by default exactly as scheduled and
as early as the physics allow (never ahead of its timetable). The plan says only what to change:

    hold     keep a train at a station until a minute (to let an opposing train through a single-track section,
             or to wait for a section to reopen)
    to       short-turn: end the run early at a turn point, so the same train can start back from there
    from     start a run late at a turn point (where its train was short-turned)
    cancel   do not run it at all (its train stays where it is)

The physics, given a plan, are deterministic (`check.simulate`): a train leaves each point at the latest of when it
is ready, its scheduled departure (never early at a scheduled stop) and its hold; it runs at its minimum running
time. A train held where it has no scheduled stop loses `stop_loss` minutes braking and `stop_loss` accelerating.
Whether the result is legal (single-track and headway conflicts, platform tracks, closed sections, where each train
ends up) is `check.evaluate`.

Times are whole minutes after 00:00 of the service day (00:30 the next night is 24:30, minute 1470); plans may give
them as integers or "HH:MM".
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

RULES = {
    "h_follow": 3,       # minutes between two trains going the same way, at both ends of a section
    "h_opp": 0,          # minutes after a train leaves a single-track section before an opposing train may enter it
                         # (0: in the same minute, as the timetable's crossings are planned)
    "turn_min": 4,       # minutes for a train to turn back (driver changes ends)
    "stop_loss": 1,      # minutes lost braking, and again accelerating, at a stop that is not in the timetable
    "cancel_cost": 30,   # per scheduled stop a train does not serve (the next train of its kind is 30 minutes later)
    "max_delay": 120,    # no train may reach a stop more than this many minutes late
    "gap_k": 100,        # reward: added to the avoidable cost (about three unserved stops)
}


@dataclass
class Run:
    """One scheduled train from one end of its route to the other."""
    id: str                       # train number
    kind: str                     # stop (Stoptrein: every station) | fast (Sneltrein)
    route: list[str]              # every point from origin to destination in travel order, junctions included
    stops: list[bool]             # scheduled stop at route[i]
    arr: list[int | None]         # scheduled arrival at a stop, the timetable's passing minute elsewhere; None at origin
    dep: list[int | None]         # scheduled departure at a stop, the passing minute elsewhere; None at destination
    seg: list[int]                # minimum running minutes route[i] -> route[i + 1] for this train's stopping pattern
    unit: str                     # the train (rolling-stock unit) that runs it
    fixed_to: int = 0             # in progress: route index of the point it is at or approaching when the task starts
    fixed_arr: int | None = None  # in progress: its arrival there (cannot change)
    fixed_from: int | None = None # in progress and between points: the minute it left route[fixed_to - 1]
    stuck: bool = False           # in progress inside a section that closed: it waits there for the reopening

    @property
    def origin(self) -> str:
        return self.route[0]

    @property
    def dest(self) -> str:
        return self.route[-1]

    @property
    def started(self) -> bool:
        return self.fixed_arr is not None

    @property
    def n_stops(self) -> int:
        return sum(self.stops)

    def idx(self, code: str) -> int:
        return self.route.index(code)


@dataclass
class Unit:
    """A train set and the runs it works through the window, in order."""
    id: str
    runs: list[str]
    at: str | None                # where it stands before its first run (None: its first run is already moving)
    ready: int | None             # when it may leave `at`
    after: dict | None = None     # its first run after the window: {"run", "station", "dep", "stops"}


@dataclass
class Closure:
    kind: str                     # closed (no train may be in the sections) | slow (+extra minutes) | track (one
                                  # platform track out of use at a station)
    where: list[str]              # closed/slow: [a, b], every section between points a and b; track: [station]
    start: int
    end: int                      # exclusive
    extra: int = 0
    cause: str = ""
    real: bool = False            # from the 2025 disruption record (else generated)


@dataclass
class Task:
    task_id: str
    split: str
    line: str
    day: str
    start: int                    # "now": the minute the agent takes over
    end: int                      # runs leaving their origin before this minute are in the task
    network: dict
    runs: list[Run]
    units: list[Unit]
    closures: list[Closure]
    notices: list[str]
    disruptions: list[dict]
    difficulty: str
    reference: dict = field(default_factory=dict)   # optimum, floor, baselines, the real outcome: never shown
    rules: dict = field(default_factory=lambda: dict(RULES))

    def to_dict(self, public: bool = False) -> dict:
        d = asdict(self)
        if public:
            d.pop("reference")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        d = dict(d)
        d["runs"] = [Run(**r) for r in d["runs"]]
        d["units"] = [Unit(**u) for u in d["units"]]
        d["closures"] = [Closure(**c) for c in d["closures"]]
        d.setdefault("reference", {})
        d["rules"] = {**RULES, **(d.get("rules") or {})}
        return cls(**d)

    def run(self, rid: str) -> Run:
        return self.run_map[rid]

    @property
    def run_map(self) -> dict[str, Run]:
        m = self.__dict__.get("_run_map")
        if m is None:
            m = self.__dict__["_run_map"] = {r.id: r for r in self.runs}
        return m

    def rule(self, k: str) -> int:
        return int(self.rules.get(k, RULES[k]))


# ---------------------------------------------------------------- plans

@dataclass
class Order:
    """What the plan changes for one run (absent from the plan: runs in full, as early as possible)."""
    cancel: bool = False
    start: str | None = None      # "from": start at this turn point instead of the origin
    end: str | None = None        # "to": end at this turn point instead of the destination
    hold: dict[str, int] = field(default_factory=dict)   # point -> do not leave before this minute

    def to_json(self, rid: str) -> dict:
        d: dict[str, Any] = {"train": rid}
        if self.cancel:
            d["cancel"] = True
            return d
        if self.start:
            d["from"] = self.start
        if self.end:
            d["to"] = self.end
        if self.hold:
            d["hold"] = {k: hhmm(v) for k, v in self.hold.items()}
        return d


Plan = dict[str, Order]


class PlanError(ValueError):
    """The plan cannot be read at all."""


def hhmm(m: int | None) -> str | None:
    if m is None:
        return None
    return f"{m // 60:02d}:{m % 60:02d}"


_TIME = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*$")


def parse_time(v: Any) -> int:
    if isinstance(v, bool):
        raise ValueError(f"not a time: {v!r}")
    if isinstance(v, int):
        m = v
    elif isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")) or v != int(v):
            raise ValueError(f"not a whole minute: {v!r}")
        m = int(v)
    elif isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"-?\d+", s):
            m = int(s)
        else:
            g = _TIME.match(s)
            if not g:
                raise ValueError(f'not a time: {v!r} (use "HH:MM" or minutes after midnight)')
            h, mi = int(g.group(1)), int(g.group(2))
            if mi > 59:
                raise ValueError(f"not a time: {v!r}")
            m = h * 60 + mi
    else:
        raise ValueError(f"not a time: {v!r}")
    if not 0 <= m <= 48 * 60:
        raise ValueError(f"time out of range: {v!r}")
    return m


def _load(raw: Any) -> Any:
    if isinstance(raw, str):
        s = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
        if fence:
            s = fence.group(1).strip()
        if not s.startswith(("[", "{")):
            i = min([k for k in (s.find("["), s.find("{")) if k >= 0], default=-1)
            if i < 0:
                raise PlanError("the plan is not JSON: give a list like "
                                '[{"train": "37413", "hold": {"HDG": "07:31"}}, {"train": "37324", "cancel": true}]')
            s = s[i:]
        try:
            raw, _ = json.JSONDecoder().raw_decode(s)
        except json.JSONDecodeError as e:
            raise PlanError(f"the plan is not valid JSON: {e}") from None
    if isinstance(raw, dict):
        if "plan" in raw and isinstance(raw["plan"], (list, dict)):
            return _load(raw["plan"])
        # {"37413": {...}, ...}
        raw = [{"train": k, **(v if isinstance(v, dict) else {"cancel": v} if isinstance(v, bool) else {"bad": v})}
               for k, v in raw.items()]
    if not isinstance(raw, list):
        raise PlanError("the plan must be a list of entries, one per train you change")
    return raw


def parse_plan(task: Task, raw: Any) -> tuple[Plan, list[str]]:
    """(plan, entry problems). Raises PlanError only when nothing can be read. An empty list is a valid plan:
    every train runs in full, as early as it can."""
    entries = _load(raw)
    runs = task.run_map
    plan: Plan = {}
    problems: list[str] = []
    for n, e in enumerate(entries):
        if not isinstance(e, dict):
            problems.append(f"entry {n}: not an object")
            continue
        rid = e.get("train", e.get("run", e.get("id")))
        if isinstance(rid, bool) or rid is None:
            problems.append(f"entry {n}: no train number")
            continue
        rid = str(rid).strip()
        if rid not in runs:
            problems.append(f"entry {n}: unknown train {rid!r}")
            continue
        if rid in plan:
            problems.append(f"entry {n}: train {rid} appears twice")
            continue
        r = runs[rid]
        o = Order()
        unknown = set(e) - {"train", "run", "id", "cancel", "from", "start", "to", "end", "hold", "note", "reason"}
        if unknown:
            problems.append(f"train {rid}: unknown field(s) {sorted(unknown)}")
        c = e.get("cancel", False)
        if c not in (True, False, None):
            problems.append(f"train {rid}: cancel must be true or false")
        o.cancel = c is True
        for key, alt, attr in (("from", "start", "start"), ("to", "end", "end")):
            v = e.get(key, e.get(alt))
            if v is None:
                continue
            if not isinstance(v, str) or v.strip().upper() not in r.route:
                problems.append(f"train {rid}: {key} {v!r} is not on its route {'-'.join(r.route)}")
                continue
            setattr(o, attr, v.strip().upper())
        h = e.get("hold") or {}
        if not isinstance(h, dict):
            problems.append(f'train {rid}: hold must be an object like {{"HDG": "07:31"}}')
            h = {}
        for k, v in h.items():
            code = str(k).strip().upper()
            if code not in r.route:
                problems.append(f"train {rid}: hold at {k!r}, which is not on its route")
                continue
            try:
                o.hold[code] = parse_time(v)
            except ValueError as ex:
                problems.append(f"train {rid}: hold at {code}: {ex}")
        plan[rid] = o
    return plan, problems


def plan_to_list(plan: Plan) -> list[dict]:
    return [o.to_json(rid) for rid, o in plan.items()]


def plan_from_list(task: Task, entries: list[dict]) -> Plan:
    plan, problems = parse_plan(task, entries)
    if problems:
        raise PlanError("; ".join(problems))
    return plan
