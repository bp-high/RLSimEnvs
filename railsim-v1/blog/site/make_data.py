"""The article's chart data, from the committed results (results/rollouts/*/index.json, results/train/*/metrics.jsonl)
and, for the banner, one plan's replay from a running RailSimEnv server.

    cd railsim-v1
    envs/rail_ops/.venv/bin/uvicorn rail_openenv.server:app --port 8031 &      # from envs/rail_ops, for the banner
    envs/rail_ops/.venv/bin/python blog/site/make_data.py

Writes content/assets/data/{results.csv, curves.csv, paired.json, banner.json}.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "results" / "rollouts"
OUT = HERE / "content" / "assets" / "data"
P = "2026-11-12-inkling-2-small-"
INK = {  # run -> (label, short label)
    f"{P}base-eval45": ("Inkling 2 Small · 64K · base", "64K base"),
    f"{P}base-max-eval45": ("Inkling 2 Small · 64K · base, max effort", "64K base, max effort"),
    f"{P}rl-a-eval45": ("Inkling 2 Small · 64K · after RL", "64K after RL"),
    f"{P}256k-max-eval45": ("Inkling 2 Small · 262K · base", "262K base"),
    f"{P}256k-max-rl-b-eval45": ("Inkling 2 Small · 262K · after RL", "262K after RL"),
}


def episodes(run: str) -> list[dict]:
    return json.loads((RUNS / run / "index.json").read_text())["episodes"]


def ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    xs = sorted(xs)
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def results() -> None:
    rows = []                                            # Inkling 2 Small and the rule-based baselines only
    for run, (name, _) in INK.items():
        r = [e["reward"] for e in episodes(run)]
        lo, hi = ci(r)
        rows.append({"model": name, "group": "after RL" if "after RL" in name else "Inkling",
                     "reward": round(statistics.mean(r), 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    rows.append({"model": "Greedy dispatcher (rules)", "group": "baseline", "reward": 0.180, "lo": "", "hi": ""})
    rows.append({"model": "Change nothing", "group": "baseline", "reward": 0.105, "lo": "", "hi": ""})
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "group", "reward", "lo", "hi"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["reward"]))


def curves() -> None:
    rows = []
    for run, label in (("rl-a", "64K run (rl-a)"), ("rl-b", "262K run (rl-b)")):
        seen = {}
        for line in (ROOT / "results" / "train" / run / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("phase") == "val":
                seen[r["step"]] = r["reward_mean"]
        rows += [{"series": label, "x": s, "y": round(y, 4)} for s, y in sorted(seen.items())]
    with open(OUT / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["series", "x", "y"])
        w.writeheader()
        w.writerows(rows)


def paired() -> None:
    out = {}
    for key, (a, b) in {"64K": (f"{P}base-eval45", f"{P}rl-a-eval45"),
                        "262K": (f"{P}256k-max-eval45", f"{P}256k-max-rl-b-eval45")}.items():
        before = {e["task_id"]: e for e in episodes(a)}
        after = {e["task_id"]: e for e in episodes(b)}
        out[key] = [{"task": t, "day": before[t]["day"], "tier": before[t]["difficulty"],
                     "before": round(before[t]["reward"], 4), "after": round(after[t]["reward"], 4),
                     "legal_before": before[t]["feasible"], "legal_after": after[t]["feasible"]}
                    for t in sorted(before)]
    (OUT / "paired.json").write_text(json.dumps(out, indent=1))


def banner(server: str = "http://127.0.0.1:8031") -> None:
    task = "gnlw-2025-03-13-0706-180m-real-0"
    plan = f"run:{P}256k-max-rl-b-eval45:tinker:thinkingmachines/Inkling-2-Small:peft:262144#rl-b@max"
    url = f"{server}/api/tasks/{task}/replay?plan={urllib.parse.quote(plan, safe='')}"
    R = json.load(urllib.request.urlopen(url))
    km = {p["code"]: p["km"] for p in R["points"]}
    runs = []
    for r in R["runs"]:
        if r.get("cancelled"):
            continue
        pts = []
        for i in range(r["first"], r["last"] + 1):
            for t in (r["arr"][i], r["dep"][i]):
                if t is not None:
                    pts.append([t, km[r["route"][i]]])
        if len(pts) > 1:
            runs.append({"train": r["train"], "kind": r["kind"], "pts": pts})
    body = {"task": task, "start": R["start"], "end": R["end"], "length": max(km.values()),
            "stations": [{"code": p["code"], "name": p["name"], "km": p["km"]} for p in R["points"]
                         if p["kind"] != "junction"],
            "closures": [{"from": km[c["where"][0]], "to": km[c["where"][-1]], "start": c["start"], "end": c["end"]}
                         for c in R["closures"] if c["kind"] == "closed"],
            "runs": runs}
    (OUT / "banner.json").write_text(json.dumps(body, separators=(",", ":")))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    results()
    curves()
    paired()
    banner()
    for p in sorted(OUT.iterdir()):
        print(p.name, p.stat().st_size)
