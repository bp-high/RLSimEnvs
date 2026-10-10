"""A task's reference: the proven optimum, the floor, the lazy policies' costs and rewards, and the real outcome.

Nothing here is shown to the agent.
"""

from __future__ import annotations

from pathlib import Path

from .baselines import POLICIES
from .check import evaluate
from .model import Task, plan_to_list
from .reward import score
from .solve import floor_parts, optimal_plan


def add_reference(task: Task, time_limit: float = 90.0, workers: int = 4, extend: int = 2) -> dict:
    hints = {}
    for name in ("greedy", "greedy_fast"):
        try:
            hints[name] = POLICIES[name](task)
        except Exception as e:        # a baseline that crashes is a bug, but must not stop a build
            hints[name] = None
            task.reference.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {e}")
    best_hint = None
    for name, p in hints.items():
        if p is None:
            continue
        r = evaluate(task, p)
        if r.feasible and (best_hint is None or r.cost < best_hint[1]):
            best_hint = (p, r.cost)
    floor, parts = floor_parts(task)
    sol = optimal_plan(task, time_limit=time_limit, workers=workers, hint=best_hint[0] if best_hint else None,
                       unit_floor=parts)
    if sol.plan is None:
        raise ValueError(f"{task.task_id}: no feasible plan found ({sol.status})")
    floor = min(floor, sol.bound)
    # large windows: warm-start from the best plan and give the proof more time, while the gap is closable
    limit, spent = time_limit, sol.seconds
    for _ in range(extend):
        gap = (sol.cost - sol.bound) / (sol.cost - floor + task.rule("gap_k"))
        if sol.optimal or gap <= 0.01 or gap > 0.3:     # close enough to finish, or hopeless
            break
        limit *= 2
        nxt = optimal_plan(task, time_limit=limit, workers=workers, hint=sol.plan, unit_floor=parts)
        spent += nxt.seconds
        if nxt.plan is not None and (nxt.cost, -nxt.bound) <= (sol.cost, -sol.bound):
            sol = nxt
        elif nxt.plan is not None:
            sol = type(sol)(sol.plan, sol.cost, max(sol.bound, nxt.bound), sol.optimal, sol.status, sol.seconds)
    floor = min(floor, sol.bound)
    ref = task.reference
    ref.update(optimal_cost=sol.cost, optimal_plan=plan_to_list(sol.plan), proven_optimal=sol.optimal,
               lower_bound=sol.bound, floor_cost=floor, solver_seconds=round(spent, 1), solver=sol.status)
    score_baselines(task, hints)
    if "real" in ref:
        rr = ref["real"]
        rew, _ = score(rr["cost"], True, 1.0, sol.cost, floor, gap_k)
        rr["reward_if_legal"] = round(rew, 4)
    return ref


def score_baselines(task: Task, hints: dict | None = None) -> None:
    """The lazy policies' feasibility, cost, reward and rule breaks, against the stored optimum and floor."""
    ref = task.reference
    if hints is None:
        hints = {}
        for name in ("greedy", "greedy_fast"):
            try:
                hints[name] = POLICIES[name](task)
            except Exception as e:
                hints[name] = None
                ref.setdefault("errors", []).append(f"{name}: {type(e).__name__}: {e}")
    gap_k = task.rule("gap_k")
    for name, p in [("naive", POLICIES["naive"](task))] + list(hints.items()):
        if p is None:
            ref[f"{name}_feasible"] = False
            ref[f"{name}_reward"] = 0.0
            continue
        r = evaluate(task, p)
        clean = sum(1 for x in r.runs if not x.problems) / max(1, len(r.runs))
        rew, _ = score(r.cost, r.feasible, clean, ref["optimal_cost"], ref["floor_cost"], gap_k)
        ref[f"{name}_feasible"] = r.feasible
        ref[f"{name}_cost"] = r.cost
        ref[f"{name}_reward"] = round(rew, 4)
        ref[f"{name}_violations"] = len(r.violations)


BASELINE_KEYS = [f"{n}_{k}" for n in ("naive", "greedy", "greedy_fast") for k in ("feasible", "cost", "reward",
                                                                                     "violations")]


def refresh_baselines(pack_dir: Path) -> list[str]:
    """Re-score the lazy policies of every task in a pack (after a baseline fix), without re-solving. Rewrites the
    pack only if something changed; returns the ids of the tasks that changed."""
    import json

    from .build import write_pack
    from .hardness import features
    from .pack import _read

    tasks, manifest = _read(pack_dir)
    raw = [t.to_dict() for t in tasks]
    changed = []
    for t, d in zip(tasks, raw):
        before = {k: t.reference.get(k) for k in BASELINE_KEYS}
        t.reference.pop("errors", None)
        score_baselines(t)
        if {k: t.reference.get(k) for k in BASELINE_KEYS} != before or "errors" in d["reference"]:
            t.reference["hardness"] = features(t)
            d["reference"] = t.reference
            changed.append(t.task_id)
    if changed:
        log = [json.loads(x) for x in (pack_dir / "build_log.jsonl").read_text().splitlines()]
        for st in log:
            for t in tasks:
                if st.get("task_id") == t.task_id and "hardness" in st:
                    st["hardness"] = t.reference["hardness"]
        std = {"name", "split", "tasks", "file", "tiers", "sha256", "candidates", "days", "proven_optimal", "runs",
               "source"}
        write_pack(pack_dir, raw, log, manifest["split"], {k: v for k, v in manifest.items() if k not in std})
    return changed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="re-score the lazy policies stored in task packs")
    ap.add_argument("packs", nargs="+")
    for pack in ap.parse_args().packs:
        print(pack, refresh_baselines(Path(pack)))
