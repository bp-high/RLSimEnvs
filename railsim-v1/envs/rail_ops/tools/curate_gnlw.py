"""Curate the Groningen-Leeuwarden line from the Rijden de Treinen open data (CC BY 4.0).

    python3 -I tools/curate_gnlw.py --services 'RAW/services-2025-0[1-6].csv.gz' \
        --disruptions RAW/disruptions-2025.csv --profiles RAW/disruptions-2024.csv RAW/disruptions-2025.csv \
        --out ../../data/gnlw

Writes, for 1 January - 30 June 2025 (one timetable; from July the line runs on to Groningen Europapark and is
partly closed for works):

    services_2025h1.csv.gz   every stop of every Arriva train that runs only between Leeuwarden and Groningen:
                             schedule in minutes after 00:00 of the service date, delays, cancellations, platforms
    disruptions_2025h1.csv   every disruption Rijden de Treinen linked to the line, with cause, start, end, stations
    disruption_profiles.csv  the same for 2024 and 2025 (--profiles): the distribution seeded disruptions are drawn from
    links_2025h1.csv         the rolling-stock circulation (derived): which run each arriving train forms next

Rolling-stock links are not in any public record. They are derived here by first-in-first-out matching at each turn
point on the *schedule*: a departure is formed by the train that arrived there earliest and has stood at least
TURN_MIN minutes. Arriva runs one fleet on both service types, so a stopping train may come back as a fast one.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LINE = ["LW", "LWC", "HDG", "FWD", "DWE", "BP", "GK", "ZH", "GN"]
COLUMNS = ["svc", "date", "type", "company", "train", "full_cxl", "part_cxl", "maxdelay", "stop", "code", "name",
           "arr", "arr_delay", "arr_cxl", "dep", "dep_delay", "dep_cxl", "plat_change", "plan_plat", "act_plat"]
START, END = "2025-01-01", "2025-07-01"
TURN_MIN = 4
AMS = ZoneInfo("Europe/Amsterdam")


def minutes(stamp: str, day: str) -> int | None:
    """RFC 3339 local time -> whole minutes after 00:00 of the service date (25:10 is 1510)."""
    if not stamp:
        return None
    t = datetime.fromisoformat(stamp).astimezone(AMS)
    d0 = datetime.fromisoformat(day).replace(tzinfo=AMS)
    return round((t - d0).total_seconds() / 60)


def read_services(pattern: str) -> list[dict]:
    keep: list[dict] = []
    line = set(LINE)
    for path in sorted(glob.glob(pattern)):
        by_svc: dict[str, list[dict]] = defaultdict(list)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                d = dict(zip(COLUMNS, row))
                if not (START <= d["date"] < END):
                    continue
                if "ipv" in d["type"] or "Bus" in d["type"]:
                    continue
                by_svc[d["svc"]].append(d)
        n0 = len(keep)
        for svc, stops in by_svc.items():
            if len(stops) >= 2 and all(s["code"] in line for s in stops) and stops[0]["company"] == "Arriva":
                keep += stops
        print(f"{path}: {len(keep) - n0} stops", flush=True)
    return keep


def curate(rows: list[dict]) -> list[dict]:
    out = []
    seq: dict[str, int] = defaultdict(int)
    for d in rows:
        i = seq[d["svc"]]
        seq[d["svc"]] += 1
        out.append({
            "date": d["date"], "svc": d["svc"], "train": d["train"], "type": d["type"], "seq": i, "code": d["code"],
            "sched_arr": minutes(d["arr"], d["date"]), "arr_delay": d["arr_delay"] or None,
            "arr_cxl": int(d["arr_cxl"] == "true") if d["arr_cxl"] else None,
            "sched_dep": minutes(d["dep"], d["date"]), "dep_delay": d["dep_delay"] or None,
            "dep_cxl": int(d["dep_cxl"] == "true") if d["dep_cxl"] else None,
            "full_cxl": int(d["full_cxl"] == "true"), "part_cxl": int(d["part_cxl"] == "true"),
            "plan_plat": d["plan_plat"] or None, "act_plat": d["act_plat"] or None,
        })
    return out


def links(stops: list[dict]) -> list[dict]:
    """FIFO matching per (date, station) on the schedule: departures in time order each take the earliest-arrived
    unit that has stood at least TURN_MIN minutes."""
    by_svc: dict[tuple, list[dict]] = defaultdict(list)
    for s in stops:
        by_svc[(s["date"], s["svc"])].append(s)
    arrivals: dict[tuple, list] = defaultdict(list)
    departures: dict[tuple, list] = defaultdict(list)
    for (day, svc), ss in by_svc.items():
        # in time order: services changed on the day sometimes list a stop out of order
        ss.sort(key=lambda s: (s["sched_dep"] if s["sched_dep"] is not None else s["sched_arr"], s["seq"]))
        first, last = ss[0], ss[-1]
        departures[(day, first["code"])].append((first["sched_dep"], first["train"]))
        arrivals[(day, last["code"])].append((last["sched_arr"], last["train"]))
    out = []
    for key in sorted(departures):
        waiting = sorted(arrivals.get(key, []))
        used = [False] * len(waiting)
        for t_dep, train in sorted(departures[key]):
            for i, (t_arr, a_train) in enumerate(waiting):
                if not used[i] and t_arr + TURN_MIN <= t_dep:
                    used[i] = True
                    out.append({"date": key[0], "station": key[1], "arriving": a_train, "forms": train,
                                "arrives": t_arr, "departs": t_dep, "turn": t_dep - t_arr})
                    break
    return out


def write_csv(path: Path, rows: list[dict], gz: bool = False):
    if not rows:
        raise ValueError(f"nothing to write to {path}")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    data = buf.getvalue().encode()
    path.write_bytes(gzip.compress(data, mtime=0) if gz else data)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--services", required=True, help="glob of services-YYYY-MM.csv.gz files")
    ap.add_argument("--disruptions", required=True)
    ap.add_argument("--profiles", nargs="*", help="disruptions-YYYY.csv files for the seeding distribution")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stops = curate(read_services(a.services))
    write_csv(out / "services_2025h1.csv.gz", stops, gz=True)
    write_csv(out / "links_2025h1.csv", links(stops))
    dis = []
    with open(a.disruptions, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if "Groningen - Leeuwarden" in (d["rdt_lines"] or "") and START <= d["start_time"][:10] < END:
                dis.append({k: d[k] for k in ("rdt_id", "cause_nl", "cause_en", "statistical_cause_en", "cause_group",
                                              "start_time", "end_time", "duration_minutes", "rdt_station_codes")})
    write_csv(out / "disruptions_2025h1.csv", dis)
    # every disruption on the line in the profile files (2024 and all of 2025): the distribution seeded tasks draw from
    prof = []
    for path in a.profiles or []:
        with open(path, encoding="utf-8") as f:
            for d in csv.DictReader(f):
                if "Groningen - Leeuwarden" in (d["rdt_lines"] or ""):
                    prof.append({"start_time": d["start_time"], "duration_minutes": d["duration_minutes"],
                                 "cause_en": d["cause_en"], "statistical_cause_en": d["statistical_cause_en"],
                                 "cause_group": d["cause_group"], "rdt_station_codes": d["rdt_station_codes"]})
    if prof:
        write_csv(out / "disruption_profiles.csv", prof)


if __name__ == "__main__":
    main()
