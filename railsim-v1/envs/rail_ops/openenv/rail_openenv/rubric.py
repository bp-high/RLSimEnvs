"""The reward as OpenEnv rubrics (RFC 004).

`RailPlanRubric` scores the `submit_plan` call and returns 0.0 for every other action, so the grader runs once per
episode. Its two children are the inputs of the score and show up in `named_rubrics()` with their `last_score`:

    runs_clean     fraction of runs with no rule broken (entry problems count as unclean)
    cost_quality   exp(-g / 0.5), g the gap to the proven optimum over the avoidable cost; 0 when infeasible

The combination is `rail_core.reward.score` (infeasible: 0.2 x runs_clean; feasible: 0.2 + 0.8 x cost_quality),
shared with the offline grader so the two can never disagree.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.rubrics import Rubric

from rail_core import Grade, PlanError, Task, grade, parse_plan

SUBMIT_TOOL = "submit_plan"


def submitted_plan(action: Any):
    if isinstance(action, CallToolAction) and action.tool_name == SUBMIT_TOOL:
        return (action.arguments or {}).get("plan")
    return None


class _Graded(Rubric):
    """Grades the submitted plan once per action and shares the result with sibling rubrics."""

    def __init__(self):
        super().__init__()
        self.task: Task | None = None
        self._cache: tuple[int, Grade | None] | None = None

    def graded(self, action: Any) -> Grade | None:
        if self._cache is not None and self._cache[0] == id(action):
            return self._cache[1]
        raw = submitted_plan(action)
        g = None
        if raw is not None and self.task is not None:
            try:
                plan, problems = parse_plan(self.task, raw)
            except PlanError as e:
                plan, problems = {}, [str(e)]
            g = grade(self.task, plan, problems)
        self._cache = (id(action), g)
        return g


class RunsClean(Rubric):
    def __init__(self, source: _Graded):
        super().__init__()
        object.__setattr__(self, "_source", source)  # not a child: avoid a cycle in named_rubrics()

    def forward(self, action, observation) -> float:
        g = self._source.graded(action)
        return 0.0 if g is None else g.clean_fraction


class CostQuality(Rubric):
    def __init__(self, source: _Graded):
        super().__init__()
        object.__setattr__(self, "_source", source)

    def forward(self, action, observation) -> float:
        g = self._source.graded(action)
        return 0.0 if g is None else g.quality


class RailPlanRubric(_Graded):
    """Terminal reward for a dispatch plan: 0.0 until `submit_plan`, then the score in [0, 1]."""

    def __init__(self):
        super().__init__()
        self.runs_clean = RunsClean(self)
        self.cost_quality = CostQuality(self)
        self.last_grade: Grade | None = None

    def set_task(self, task: Task) -> None:
        self.task = task
        self._cache = None
        self.last_grade = None

    def forward(self, action, observation) -> float:
        g = self.graded(action)
        self.runs_clean(action, observation)
        self.cost_quality(action, observation)
        if g is None:
            return 0.0
        self.last_grade = g
        return g.reward

    def reset(self) -> None:
        self._cache = None
        self.last_grade = None
