"""Build task packs (candidates -> references -> filters), resumable, deterministic for the same data and arguments.

    python -m rail_core.build --split eval --out ../tasks/pool-eval --n 400      # a candidate pool, every tier
    python -m rail_core.build --split train --out ../tasks/rail-v1-train --n 300

Candidates: the record's own disruptions on their own days (tier `real`), then seeded tiers (`ops.TIERS`) on every
day of the split at peak-hour starts, cycling seeds. A candidate is kept when:
    the window can be rebuilt from the record (generate.build_task) and has at least MIN_RUNS runs
    CP-SAT's plan is within 1 % of its proven bound, measured on the avoidable cost (cost - floor + gap_k)
    the naive plan (change nothing) scores at most MAX_NAIVE_REWARD and the greedy dispatchers at most
    MAX_GREEDY_REWARD
Every candidate and why it was dropped goes to build_log.jsonl, with the hardness features of every kept task
(`hardness.features`). Choosing the eval set from a pool is `rail_core.select`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .generate import build_task
from .hardness import features
from .model import Closure, hhmm
from .network import load_network
from .ops import (LATE_CAUSES, PEAKS, TIERS, candidate_id, closures_for, real_disruptions, seeded_closures,
                  split_days)
from .reference import add_reference

MIN_RUNS = 15
MAX_NAIVE_REWARD = 0.6
MAX_GREEDY_REWARD = 0.85
MAX_GAP = 0.01


def _rng(cid: str) -> random.Random:
    return random.Random(int(hashlib.sha256(cid.encode()).hexdigest()[:12], 16))


def make_candidate(day: str, start: int, window: int, tier: str, seed: int, split: str,
                   real: dict | None = None):
    """The task for one candidate (no reference yet), or (None, reason)."""
    net = load_network()
    cid = candidate_id(day, start, window, tier, seed)
    rng = _rng(cid)
    notices: list[str] = []
    why: list[str] = []
    task = None
    if tier == "real":
        # The record's start is when the disruption was announced; dispatchers have often cut trains short a few
        # minutes before. Start the window (and the disruption) earlier until no train was cut before it.
        for early in (0, 5, 10, 15, 20, 30, 45):
            s0 = real["start"] - early
            closures = closures_for(net, real["cause"], real["stations"], s0, real["end"], _rng(cid), real=True)
            why = []
            task = build_task(day, s0, s0 + window, closures, task_id=cid, split=split, difficulty=tier,
                              notices=notices, why=why)
            if task is not None or "cut short" not in (why[0] if why else ""):
                break
        if task is not None and early:
            task.notices.append(f"The record announces this disruption at {hhmm(real['start'])}; trains were already "
                                f"being cut short, so it starts here at {hhmm(task.start)}.")
    else:
        closures, notices = seeded_closures(net, tier, start, window, rng)
        task = build_task(day, start, start + window, closures, task_id=cid, split=split, difficulty=tier,
                          notices=notices, why=why)
    if task is None:
        return None, "rebuild: " + (why[0] if why else "?")
    lo, hi = TIERS[tier].get("late", (0, 0)) if isinstance(TIERS[tier].get("late"), tuple) else (0, 0)
    waiting = [u for u in task.units if u.at and not task.run(u.runs[0]).started]
    for u in rng.sample(waiting, min(len(waiting), rng.randint(lo, hi))):
        extra = rng.choice((10, 15, 20, 25, 30))
        first = task.run(u.runs[0])
        u.ready = max(u.ready or start, first.dep[0]) + extra
        task.notices.append(f"Train set {u.id} at {u.at} is held for {rng.choice(LATE_CAUSES)}: it cannot leave "
                            f"before {hhmm(u.ready)} (its next run {first.id} is due at {hhmm(first.dep[0])}).")
    if len(task.runs) < MIN_RUNS:
        return None, f"too small: {len(task.runs)} runs"
    maxd = task.rule("max_delay")
    for r in task.runs:
        if r.started and r.arr[r.fixed_to] is not None and r.fixed_arr - r.arr[r.fixed_to] > maxd - 30:
            return None, f"{r.id} is already {r.fixed_arr - r.arr[r.fixed_to]} min late"
    return task, ""


def _one(args):
    day, start, window, tier, seed, split, real, limit, workers = args
    t0 = time.time()
    stats = {"day": day, "start": start, "window": window, "tier": tier, "seed": seed}
    try:
        task, reason = make_candidate(day, start, window, tier, seed, split, real)
        if task is None:
            stats["reason"] = reason
            return None, stats
        ref = add_reference(task, time_limit=limit, workers=workers)
        gap = (ref["optimal_cost"] - ref["lower_bound"]) / (ref["optimal_cost"] - ref["floor_cost"] + task.rule("gap_k"))
        stats.update(task_id=task.task_id, runs=len(task.runs), optimal=ref["optimal_cost"], bound=ref["lower_bound"],
                     floor=ref["floor_cost"], proven=ref["proven_optimal"], naive=ref["naive_reward"],
                     greedy=ref["greedy_reward"], greedy_fast=ref["greedy_fast_reward"],
                     real_cost=ref.get("real", {}).get("cost"))
        if gap > MAX_GAP:
            stats["reason"] = f"not solved within 1% ({gap:.3f})"
            return None, stats
        if ref["naive_reward"] > MAX_NAIVE_REWARD:
            stats["reason"] = f"too easy: naive {ref['naive_reward']}"
            return None, stats
        if max(ref["greedy_reward"], ref["greedy_fast_reward"]) > MAX_GREEDY_REWARD:
            stats["reason"] = f"too easy: greedy {max(ref['greedy_reward'], ref['greedy_fast_reward'])}"
            return None, stats
        ref["hardness"] = features(task)
        stats["hardness"] = ref["hardness"]
        stats["reason"] = "kept"
        return task.to_dict(), stats
    except ValueError as e:
        if "no feasible plan" in str(e):   # the record's state at `start` already breaks a rule (e.g. two trains
            stats["reason"] = "infeasible: no legal plan from the rebuilt state"   # stuck head-on)
        else:
            stats["reason"] = f"error {type(e).__name__}: {e}"
        return None, stats
    except Exception as e:      # a generator or solver bug on one candidate is logged, not fatal
        stats["reason"] = f"error {type(e).__name__}: {e}"
        return None, stats
    finally:
        stats["seconds"] = round(time.time() - t0, 1)


def candidates(split: str, max_seed: int, tiers: list[str]) -> list[tuple]:
    out = []
    if "real" in tiers:
        for d in real_disruptions(split):
            for window in (180,):
                out.append((d["day"], d["start"], window, "real", 0, split, d))
    seeded = [t for t in tiers if t != "real"]
    for seed in range(max_seed):
        batch = []
        for day in split_days(split):
            for a, b in PEAKS:
                for start in range(a, b + 1, 30):
                    for tier in seeded:
                        lo, hi = TIERS[tier]["window"]
                        window = lo + 30 * ((seed + start // 30) % ((hi - lo) // 30 + 1))
                        batch.append((day, start, window, tier, seed, split, None))
        key = lambda c: hashlib.sha1(repr(c[:5]).encode()).hexdigest()
        out += sorted(batch, key=key)
    return out


def write_pack(out: Path, tasks: list[dict], log: list[dict], split: str, extra: dict | None = None):
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(t, ensure_ascii=False, separators=(",", ":")) + "\n" for t in tasks).encode()
    name = "tasks.jsonl"
    if len(body) > 8_000_000:
        (out / "tasks.jsonl.gz").write_bytes(gzip.compress(body, mtime=0))
        (out / "tasks.jsonl").unlink(missing_ok=True)
        name = "tasks.jsonl.gz"
    else:
        (out / name).write_bytes(body)
        (out / "tasks.jsonl.gz").unlink(missing_ok=True)
    tiers: dict[str, int] = {}
    for t in tasks:
        tiers[t["difficulty"]] = tiers.get(t["difficulty"], 0) + 1
    manifest = {"name": out.name, "split": split, "tasks": len(tasks), "file": name, "tiers": tiers,
                "sha256": hashlib.sha256(body).hexdigest(), "candidates": len(log),
                "days": sorted({t["day"] for t in tasks}),
                "proven_optimal": sum(bool(t["reference"].get("proven_optimal")) for t in tasks),
                "runs": [min((len(t["runs"]) for t in tasks), default=0), max((len(t["runs"]) for t in tasks), default=0)],
                "source": "Rijden de Treinen train archive and disruptions, 2024-2025 (CC BY 4.0); OpenStreetMap "
                          "(ODbL); see data/gnlw/SOURCE.md", **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "build_log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in log))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["eval", "train"])
    ap.add_argument("--n", type=int, required=True, help="tasks to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", nargs="*", default=list(TIERS))
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2, help="CP-SAT threads per candidate")
    ap.add_argument("--max-seed", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=0, help="stop after trying this many (0: no cap)")
    args = ap.parse_args()
    out = Path(args.out)
    part = out / "partial"
    part.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    kept: list[dict] = []
    done = set()
    if (part / "log.jsonl").is_file():
        for line in (part / "log.jsonl").read_text().splitlines():
            st = json.loads(line)
            log.append(st)
            done.add((st["day"], st["start"], st["window"], st["tier"], st["seed"]))
    if (part / "kept.jsonl").is_file():
        kept = [json.loads(line) for line in (part / "kept.jsonl").read_text().splitlines()]
    queue = [c for c in candidates(args.split, args.max_seed, args.tiers) if c[:5] not in done]
    print(f"resuming: {len(log)} tried, {len(kept)} kept, {len(queue)} queued", flush=True)
    from concurrent.futures import FIRST_COMPLETED, wait

    def more() -> bool:
        return bool(queue) and len(kept) < args.n and (not args.max_candidates or len(log) + len(running)
                                                         < args.max_candidates)

    with ProcessPoolExecutor(args.procs) as ex:
        running = set()
        while True:
            while more() and len(running) < args.procs:      # keep every worker busy
                running.add(ex.submit(_one, queue.pop(0) + (args.time_limit, args.workers)))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                task, stats = f.result()
                log.append(stats)
                with open(part / "log.jsonl", "a") as fh:
                    fh.write(json.dumps(stats) + "\n")
                if task and len(kept) < args.n:
                    kept.append(task)
                    with open(part / "kept.jsonl", "a") as fh:
                        fh.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
            reasons: dict[str, int] = {}
            for s_ in log:
                k = s_["reason"].split(":")[0].split(" (")[0]
                reasons[k] = reasons.get(k, 0) + 1
            print(f"{len(log)} tried, {len(kept)} kept; {reasons}", flush=True)
            if len(kept) >= args.n:
                for f in running:
                    f.cancel()
                break
    kept.sort(key=lambda t: (t["difficulty"], t["day"], t["start"], t["task_id"]))
    print(json.dumps(write_pack(out, kept, log, args.split), indent=2))


if __name__ == "__main__":
    main()
