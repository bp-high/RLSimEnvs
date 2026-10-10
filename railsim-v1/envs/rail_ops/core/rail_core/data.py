"""The curated record (data/gnlw): every Arriva train between Leeuwarden and Groningen, January-June 2025, with its
schedule, delays and cancellations; the derived rolling-stock links; the real disruptions; the minimum running times.

`day_runs(day)` turns one service day into model runs: the route through every point (junctions included), the
scheduled stops, the timetable's passing minutes at the points in between (interpolated by distance) and the
minimum running time of every section (the calibrated stop-to-stop minimum, split by distance).
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .model import Run
from .network import DATA, Network, load_network

LINE_DIR = DATA / "gnlw"
KIND = {"Stoptrein": "stop", "Sneltrein": "fast"}


@dataclass
class Stop:
    code: str
    sa: int | None            # scheduled arrival
    sd: int | None            # scheduled departure
    ad: int | None            # arrival delay
    dd: int | None            # departure delay
    arr_cxl: bool
    dep_cxl: bool

    @property
    def aa(self) -> int | None:          # actual arrival
        return None if self.sa is None or self.ad is None else self.sa + self.ad

    @property
    def adep(self) -> int | None:        # actual departure
        return None if self.sd is None or self.dd is None else self.sd + self.dd


@dataclass
class Service:
    train: str
    type: str
    stops: list[Stop]
    full_cxl: bool
    part_cxl: bool

    @property
    def kind(self) -> str:
        return KIND.get(self.type, "stop")


@dataclass
class Day:
    date: str
    services: dict[str, Service]
    links: dict[str, str]                 # arriving train -> the train it forms next
    runs: dict[str, Run] = field(default_factory=dict)
    chains: list[list[str]] = field(default_factory=list)


def _int(x: str) -> int | None:
    return int(x) if x not in ("", None) else None


@lru_cache(maxsize=1)
def _all_services() -> dict[str, dict[str, Service]]:
    by_day: dict[str, dict[str, Service]] = {}
    rows: dict[tuple, list[dict]] = {}
    with gzip.open(LINE_DIR / "services_2025h1.csv.gz", "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["date"], r["svc"]), []).append(r)
    for (day, _), rs in rows.items():
        rs.sort(key=lambda r: int(r["seq"]))
        s = Service(rs[0]["train"], rs[0]["type"],
                    [Stop(r["code"], _int(r["sched_arr"]), _int(r["sched_dep"]), _int(r["arr_delay"]),
                          _int(r["dep_delay"]), r["arr_cxl"] == "1", r["dep_cxl"] == "1") for r in rs],
                    rs[0]["full_cxl"] == "1", rs[0]["part_cxl"] == "1")
        by_day.setdefault(day, {})
        if s.train in by_day[day]:          # a number used twice in a day: keep the first, number the second
            s.train = f"{s.train}b"
        by_day[day][s.train] = s
    return by_day


@lru_cache(maxsize=1)
def _all_links() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with open(LINE_DIR / "links_2025h1.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["date"], {})[r["arriving"]] = r["forms"]
    return out


@lru_cache(maxsize=1)
def runtimes() -> dict:
    return json.loads((LINE_DIR / "runtimes.json").read_text())


def disruptions() -> list[dict]:
    with open(LINE_DIR / "disruptions_2025h1.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def days() -> list[str]:
    return sorted(_all_services())


def span_minutes(net: Network, kind: str, a: str, b: str) -> int:
    """Minimum running minutes between consecutive scheduled stops a and b."""
    rt = runtimes()
    hit = rt.get(kind, {}).get(f"{a}>{b}")
    if hit:
        return int(hit["min"])
    # fallback: the stopping train's legs, less a minute per stop skipped
    route = [c for c in net.route(a, b) if net.by_code[c].kind != "junction"]
    total = 0
    for x, y in zip(route, route[1:]):
        leg = rt["stop"].get(f"{x}>{y}")
        if leg is None:
            raise KeyError(f"no running time for {kind} {a}>{b}")
        total += int(leg["min"])
    return max(1, total - (len(route) - 2))


def clean_stops(net: Network, s: Service) -> list[Stop]:
    """The service's stops in route order with both times where it calls on the way. On disrupted days the record
    holds services changed on the day: an extra stop with only one time, or stops listed out of order."""
    timed = sorted(s.stops, key=lambda st: st.sd if st.sd is not None else (st.sa if st.sa is not None else 10**9))
    first, last = timed[0].code, timed[-1].code
    d = net.direction(first, last)
    seen: dict[str, Stop] = {}
    for st in s.stops:
        seen.setdefault(st.code, st)
    out = sorted(seen.values(), key=lambda st: d * net.index[st.code])
    for i, st in enumerate(out):
        if 0 < i < len(out) - 1:
            if st.sa is None:
                st.sa = st.sd
            if st.sd is None:
                st.sd = st.sa
    if out[0].sd is None:
        out[0].sd = out[0].sa
    if out[-1].sa is None:
        out[-1].sa = out[-1].sd
    times = [t for st in out for t in (st.sa, st.sd) if t is not None]
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f"service {s.train}: times not in route order")
    return out


def make_run(net: Network, s: Service, unit: str = "") -> Run:
    stops_rec = clean_stops(net, s)
    codes = [st.code for st in stops_rec]
    route = net.route(codes[0], codes[-1])
    stop_at = {st.code: st for st in stops_rec}
    n = len(route)
    stops = [c in stop_at for c in route]
    arr: list[int | None] = [None] * n
    dep: list[int | None] = [None] * n
    seg = [0] * (n - 1)
    idx = [i for i in range(n) if stops[i]]
    for i in idx:
        st = stop_at[route[i]]
        arr[i] = st.sa if i > 0 else None
        dep[i] = st.sd if i < n - 1 else None
    for ia, ib in zip(idx, idx[1:]):
        a, b = stop_at[route[ia]], stop_at[route[ib]]
        t_min = span_minutes(net, s.kind, route[ia], route[ib])
        t_sched = (b.sa if b.sa is not None else b.sd) - (a.sd if a.sd is not None else a.sa)
        ka, kb = net.by_code[route[ia]].km, net.by_code[route[ib]].km
        cum = [round(t_min * abs(net.by_code[route[i]].km - ka) / abs(kb - ka)) for i in range(ia, ib + 1)]
        for j in range(1, len(cum)):                # every section takes at least a minute
            cum[j] = max(cum[j], cum[j - 1] + 1)
        for j in range(ia, ib):
            seg[j] = cum[j - ia + 1] - cum[j - ia]
        t0 = a.sd if a.sd is not None else a.sa
        for i in range(ia + 1, ib):
            arr[i] = dep[i] = t0 + round(t_sched * abs(net.by_code[route[i]].km - ka) / abs(kb - ka))
    return Run(id=s.train, kind=s.kind, route=route, stops=stops, arr=arr, dep=dep, seg=seg, unit=unit)


@lru_cache(maxsize=64)
def load_day(date: str) -> Day:
    net = load_network()
    services = dict(_all_services()[date])
    for t in list(services):                  # a handful of services changed on the day are unreadable: drop them
        try:
            clean_stops(net, services[t])
        except ValueError:
            del services[t]
    links = {a: f for a, f in _all_links().get(date, {}).items() if a in services and f in services}
    day = Day(date, services, links)
    formed = set(links.values())
    starts = sorted((t for t in services if t not in formed), key=lambda t: services[t].stops[0].sd or 0)
    for k, t in enumerate(starts):
        chain = [t]
        while chain[-1] in links:
            chain.append(links[chain[-1]])
        day.chains.append(chain)
        for x in chain:
            day.runs[x] = make_run(net, services[x], unit=f"T{k + 1:02d}")
    return day
