"""Disruptions and tiers: what can go wrong on the line, from the 2025 record, and how tasks are drawn.

A disruption in the Rijden de Treinen record has a cause, a start, an end and the stations it was linked to. Its
effect in the model depends on the cause (an assumption, stated in SOURCE.md):

    broken down train, collision, emergency call, defective track, person or animal on the track, power failure
        -> the sections between the outermost stations named are closed for the whole disruption
    signal failure, level crossing failure
        -> trains run on sight there: every section in the span takes `SLOW_EXTRA` minutes longer
    points failure
        -> one track out of use at the passing station(s) in the span (no crossing there if it had two)

A disruption linked to every station of the line has no usable location; such causes are replayed on a section
drawn with the task's seed.
"""

from __future__ import annotations

import random

from .model import Closure
from .network import Network

SLOW_EXTRA = 4
CLOSE = ("broken down", "collision", "emergency", "defective railway track", "person", "animal", "power failure",
         "damaged", "fire", "accident", "vandalism", "copper theft")
SLOW = ("signal", "level crossing")
POINTS = ("points",)
LOOPS = ("HDG", "FWD", "BP", "ZH")


def effect(cause: str) -> str:
    c = cause.lower()
    if any(k in c for k in POINTS) and not any(k in c for k in SLOW):
        return "track"
    if any(k in c for k in SLOW):
        return "slow"
    return "closed"


def span(net: Network, codes: list[str]) -> list[str] | None:
    """[west, east] of the stations named, or None when the record names (nearly) the whole line."""
    idx = sorted(net.index[c] for c in codes if c in net.index)
    if len(idx) < 2 or (idx[0] == 0 and idx[-1] == len(net.points) - 1 and len(idx) >= 8):
        return None
    return [net.points[idx[0]].code, net.points[idx[-1]].code]


def closures_for(net: Network, cause: str, codes: list[str], start: int, end: int, rng: random.Random,
                 real: bool = True) -> list[Closure]:
    kind = effect(cause)
    where = span(net, codes)
    if where is None:                      # no location in the record: draw one
        where = _draw_span(net, kind, rng)
    if kind == "track":
        lo, hi = net.index[where[0]], net.index[where[-1]]
        loops = [p.code for p in net.points[lo:hi + 1] if p.code in LOOPS] or [rng.choice(LOOPS)]
        return [Closure("track", [c], start, end, cause=cause, real=real) for c in loops[:1]]
    if kind == "slow":
        return [Closure("slow", where, start, end, extra=SLOW_EXTRA, cause=cause, real=real)]
    return [Closure("closed", where, start, end, cause=cause, real=real)]


def _draw_span(net: Network, kind: str, rng: random.Random) -> list[str]:
    stations = [p.code for p in net.points if p.kind != "junction"]
    if kind == "track":
        c = rng.choice(LOOPS)
        return [c, c]
    i = rng.randrange(len(stations) - 1)
    j = min(len(stations) - 1, i + rng.choice((1, 1, 2)))
    return [stations[i], stations[j]]


# ---------------------------------------------------------------- tiers and candidates

import csv
from datetime import date as _date
from functools import lru_cache

from .data import LINE_DIR, days as _days, disruptions as _real_disruptions
from .model import hhmm

# Eval days lie in these ISO weeks of 2025 (42 days); train days are every other day of January-June.
HELD_OUT = {4, 7, 11, 17, 22, 25}
PEAKS = [(6 * 60 + 30, 9 * 60), (15 * 60 + 30, 18 * 60 + 30)]
LATE_CAUSES = ("a technical inspection", "a late driver", "a door fault being fixed", "a cleaning crew overrun",
               "coupling problems")

TIERS = {
    # window minutes; what is seeded (each disruption's cause and duration drawn from the 2024-2025 record)
    "real":     dict(window=(150, 210)),                       # the record's own disruption, on its own day
    "blockage": dict(window=(150, 210), closed=1, late=(0, 1)),
    "loop":     dict(window=(150, 210), track=1, slow=(0, 1), late=(0, 1)),
    "compound": dict(window=(180, 240), closed=1, track=(0, 1), slow=(0, 1), late=(1, 3)),
}


def split_of(day: str) -> str:
    return "eval" if _date.fromisoformat(day).isocalendar()[1] in HELD_OUT else "train"


def split_days(split: str) -> list[str]:
    return [d for d in _days() if split_of(d) == split]


@lru_cache(maxsize=1)
def profiles() -> list[dict]:
    with open(LINE_DIR / "disruption_profiles.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["duration_minutes"]]
    for r in rows:
        r["effect"] = effect(r["cause_en"])
        r["minutes"] = max(10, min(240, round(float(r["duration_minutes"]))))
    return rows


def real_disruptions(split: str | None = None) -> list[dict]:
    """The record's own disruptions on the line, as (day, start, end, cause, stations)."""
    out = []
    for d in _real_disruptions():
        day = d["start_time"][:10]
        if split and split_of(day) != split:
            continue
        hh, mm = int(d["start_time"][11:13]), int(d["start_time"][14:16])
        start = hh * 60 + mm
        dur = round(float(d["duration_minutes"] or 0))
        if dur < 15 or not 5 * 60 <= start <= 22 * 60:
            continue
        out.append({"day": day, "start": start, "end": start + dur, "cause": d["cause_en"],
                    "stations": [c.strip() for c in d["rdt_station_codes"].split(",")]})
    return out


def draw(rng: random.Random, kind: str) -> dict:
    pool = [p for p in profiles() if p["effect"] == kind] or profiles()
    return rng.choice(pool)


def seeded_closures(net: Network, tier: str, start: int, window: int, rng: random.Random) -> tuple[list, list[str]]:
    """Closures for a seeded tier, starting within the first half hour of the window."""
    r = TIERS[tier]
    out: list[Closure] = []

    def count(key):
        v = r.get(key, 0)
        return rng.randint(*v) if isinstance(v, tuple) else v

    for kind, key in (("closed", "closed"), ("track", "track"), ("slow", "slow")):
        for _ in range(count(key)):
            p = draw(rng, kind)
            t0 = start + rng.choice((0, 0, 5, 10, 20, 30))
            minutes = min(p["minutes"], window - 20) if kind == "closed" else p["minutes"]
            codes = [c.strip() for c in p["rdt_station_codes"].split(",")]
            where = span(net, codes) or _draw_span(net, kind, rng)
            out += closures_for(net, p["cause_en"], where, t0, t0 + minutes, rng, real=False)
    notes = []
    return out, notes


def candidate_id(day: str, start: int, window: int, tier: str, seed: int) -> str:
    return f"gnlw-{day}-{hhmm(start).replace(':', '')}-{window}m-{tier}-{seed}"
