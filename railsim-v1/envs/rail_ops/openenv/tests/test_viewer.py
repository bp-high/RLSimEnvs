"""The map replay's JavaScript (geometry, motion, track allocation) against real replays, through node --test.

Writes replays for a sample of tasks (the optimum and the lazy baselines) and for every committed model plan, then
runs openenv/tests/viewer/*.test.mjs on them. Skipped when node or the twin data is missing.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

TASKS = Path(__file__).resolve().parents[2] / "tasks"
if not os.environ.get("RAIL_TASKS_DIR"):
    os.environ["RAIL_TASKS_DIR"] = ":".join(str(TASKS / p) for p in ("rail-v1-eval", "rail-v1-train")
                                            if (TASKS / p / "manifest.json").is_file())

from rail_core import load_pack, parse_plan  # noqa: E402
from rail_core.baselines import POLICIES  # noqa: E402
from rail_core.replay import replay  # noqa: E402
from rail_openenv.server import WEB, run_plans  # noqa: E402

VIEWER_TESTS = Path(__file__).resolve().parent / "viewer"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None or not (WEB / "twin" / "twin.json.gz").is_file(), reason="needs node and the twin")
def test_viewer_js(tmp_path):
    pack = load_pack()
    n = 0
    for t in pack.tasks[::12]:
        plan, _ = parse_plan(t, t.reference["optimal_plan"])
        for name, p in [("optimal", plan), ("naive", POLICIES["naive"](t)), ("greedy", POLICIES["greedy"](t))]:
            (tmp_path / f"{n:03d}.json").write_text(json.dumps(replay(t, p, name)))
            n += 1
    for task_id, entries in run_plans().items():
        t = pack.get(task_id)
        for e in entries:
            p, probs = parse_plan(t, e["plan"])
            (tmp_path / f"{n:03d}.json").write_text(json.dumps(replay(t, p, e["label"], probs)))
            n += 1
    res = subprocess.run([NODE, "--test", *sorted(str(f) for f in VIEWER_TESTS.glob("*.test.mjs"))],
                         env={**os.environ, "RAIL_REPLAYS": str(tmp_path)}, capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-2000:]
    assert "# fail 0" in res.stdout and "# skipped 0" in res.stdout, res.stdout[-2000:]
