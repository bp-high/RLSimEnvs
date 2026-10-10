"""The terminal reward, from a submitted plan, the task's proven optimum and its floor.

    unreadable, malformed or infeasible   0.2 x clean runs / (runs + entry problems)                 in [0, 0.2)
    feasible                              0.2 + 0.8 x exp(-g / 0.5)                                  in (0.2, 1]
                                          g = (cost - optimum) / (optimum - floor + gap_k)

`floor` is each train set's best cost alone with the closures (`solve.floor_cost`): what the disruption forces no
matter how well the trains are sequenced. The gap is measured against the part a dispatcher controls, so a plan that
wastes as much as the optimum spends above the floor (plus gap_k) scores 0.31. gap_k is 100 stop-minutes (about three
unserved stops), which keeps the gap finite when the optimum sits on the floor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .check import evaluate
from .model import Plan, Task

INFEASIBLE_CAP = 0.2
GAP_TAU = 0.5


@dataclass
class Grade:
    reward: float
    feasible: bool
    clean_fraction: float
    quality: float
    cost: int | None
    delay_cost: int | None
    cancel_cost: int | None
    carry_cost: int | None
    cancelled: int | None
    short_turned: int | None
    optimal_cost: int
    floor_cost: int
    violations: list[dict]
    parse_problems: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def score(cost: int | None, feasible: bool, clean_fraction: float, optimum: int, floor: int,
          gap_k: int) -> tuple[float, float]:
    """(reward, quality)."""
    if not feasible or cost is None:
        return INFEASIBLE_CAP * clean_fraction, 0.0
    g = max(0, cost - optimum) / (max(0, optimum - floor) + gap_k)
    quality = max(math.exp(-g / GAP_TAU), 1e-6)
    return INFEASIBLE_CAP + (1.0 - INFEASIBLE_CAP) * quality, quality


def grade(task: Task, plan: Plan, parse_problems: list[str] | None = None) -> Grade:
    optimum = int(task.reference["optimal_cost"])
    floor = int(task.reference["floor_cost"])
    res = evaluate(task, plan)
    problems = list(parse_problems or [])
    feasible = res.feasible and not problems
    clean = sum(1 for r in res.runs if not r.problems) / max(1, len(res.runs) + len(problems))
    reward, quality = score(res.cost, feasible, clean, optimum, floor, task.rule("gap_k"))
    ok = feasible
    return Grade(round(reward, 6), feasible, round(clean, 4), round(quality, 6),
                 res.cost if ok else None, res.delay_cost if ok else None, res.cancel_cost if ok else None,
                 res.carry_cost if ok else None, res.cancelled if ok else None, res.short_turned if ok else None,
                 optimum, floor, res.violations, problems)
