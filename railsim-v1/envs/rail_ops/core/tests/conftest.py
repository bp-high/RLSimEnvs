import os
from pathlib import Path

import pytest

TASKS = Path(__file__).resolve().parents[2] / "tasks"
if not os.environ.get("RAIL_TASKS_DIR"):
    os.environ["RAIL_TASKS_DIR"] = ":".join(str(TASKS / p) for p in ("rail-v1-eval", "rail-v1-train")
                                            if (TASKS / p / "manifest.json").is_file())


@pytest.fixture(scope="session")
def pack():
    if not os.environ.get("RAIL_TASKS_DIR"):
        pytest.skip("no task pack built")
    from rail_core import load_pack

    return load_pack()


@pytest.fixture(scope="session")
def sample(pack):
    """A spread of tasks: every tier, small and large."""
    by: dict[str, list] = {}
    for t in pack.tasks:
        by.setdefault(t.difficulty, []).append(t)
    out = []
    for tier, ts in sorted(by.items()):
        ts = sorted(ts, key=lambda t: len(t.runs))
        out += [ts[0], ts[len(ts) // 2], ts[-1]]
    return list({t.task_id: t for t in out}.values())
