"""Derive the minimum running times from the curated record and write data/gnlw/runtimes.json.

    python3 -I tools/calibrate.py

For every pair of consecutive scheduled stops of a service type (Stoptrein or Sneltrein) the actual running time is
(scheduled arrival + arrival delay) - (scheduled departure + departure delay). The minimum running time is its 5th
percentile over January-June 2025, never more than the scheduled time: trains that are late run at it to make up
time. Spans seen fewer than 30 times are left out; the generator then falls back to the stopping train's times minus
a minute per skipped stop (see rail_core.data.span_minutes).
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data" / "gnlw"
KIND = {"Stoptrein": "stop", "Sneltrein": "fast"}


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def main():
    by_svc: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(DATA / "services_2025h1.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            by_svc[r["svc"]].append(r)
    act: dict[tuple, list[int]] = defaultdict(list)
    sched: dict[tuple, list[int]] = defaultdict(list)
    for svc, ss in by_svc.items():
        ss.sort(key=lambda r: int(r["seq"]))
        kind = KIND.get(ss[0]["type"])
        if kind is None:
            continue
        for a, b in zip(ss, ss[1:]):
            if a["dep_cxl"] == "1" or b["arr_cxl"] == "1" or not a["sched_dep"] or not b["sched_arr"]:
                continue
            if a["dep_delay"] == "" or b["arr_delay"] == "":
                continue
            s = int(b["sched_arr"]) - int(a["sched_dep"])
            x = s + int(b["arr_delay"]) - int(a["dep_delay"])
            key = (kind, a["code"], b["code"])
            sched[key].append(s)
            if x > 0:
                act[key].append(x)
    out: dict[str, dict] = {"stop": {}, "fast": {}}
    for key, xs in sorted(act.items()):
        if len(xs) < 30:
            continue
        kind, a, b = key
        s = max(set(sched[key]), key=sched[key].count)
        out[kind][f"{a}>{b}"] = {"min": min(s, pct(xs, 0.05)), "sched": s, "n": len(xs), "p50": pct(xs, 0.5)}
    body = {"source": "Rijden de Treinen services, 1 January - 30 June 2025 (CC BY 4.0)",
            "rule": "5th percentile of actual stop-to-stop running time, capped at the scheduled time", **out}
    (DATA / "runtimes.json").write_text(json.dumps(body, indent=1) + "\n")
    for k in ("stop", "fast"):
        print(k, {s: v["min"] for s, v in out[k].items()})


if __name__ == "__main__":
    main()
