import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

TASKS = Path(__file__).resolve().parents[2] / "tasks"
if not os.environ.get("RAIL_TASKS_DIR"):
    os.environ["RAIL_TASKS_DIR"] = ":".join(str(TASKS / p) for p in ("rail-v1-eval", "rail-v1-train")
                                            if (TASKS / p / "manifest.json").is_file())

from fastapi.testclient import TestClient  # noqa: E402
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction  # noqa: E402

from rail_core import load_pack  # noqa: E402
from rail_openenv.environment import MAX_CHECKS, MAX_TOOL_CALLS, RailOpsEnvironment  # noqa: E402
from rail_core import parse_plan  # noqa: E402
from rail_core.replay import replay  # noqa: E402
from rail_openenv.server import RUNS_DIR, create_server, model_name, run_plans  # noqa: E402
from rail_openenv.session import LocalSession  # noqa: E402

pytestmark = pytest.mark.skipif(not os.environ.get("RAIL_TASKS_DIR"), reason="no task pack built")
PACK = load_pack() if os.environ.get("RAIL_TASKS_DIR") else None


def _call(env, tool, **args):
    return env.step(CallToolAction(tool_name=tool, arguments=args))


def _text(obs):
    res = obs.result
    if hasattr(res, "content"):
        return res.content[0].text
    if isinstance(res, dict):
        return res["content"][0]["text"]
    return str(res)


def test_tools_and_optimal_submit():
    env = RailOpsEnvironment(PACK)
    task = PACK.tasks[min(3, len(PACK.tasks) - 1)]
    obs = env.reset(task_id=task.task_id)
    assert obs.metadata["task_id"] == task.task_id and "Cost" in obs.metadata["instructions"]
    assert {t.name for t in env.step(ListToolsAction()).tools} == {"get_situation", "check_plan", "submit_plan"}
    assert "Leeuwarden" in _text(_call(env, "get_situation"))
    out = _call(env, "check_plan", plan=[])
    assert out.reward == 0.0 and not out.done
    res = json.loads(_text(out))
    assert res["feasible"] is False and res["violations"] and res["checks_left"] == MAX_CHECKS - 1
    assert "optimal" not in _text(out) and "reference" not in _text(out)
    opt = json.loads(_text(_call(env, "check_plan", plan=task.reference["optimal_plan"])))
    assert opt["feasible"] and opt["cost"] == task.reference["optimal_cost"]
    done = _call(env, "submit_plan", plan=task.reference["optimal_plan"])
    assert done.done and done.reward == 1.0
    assert done.metadata["grade"]["cost"] == task.reference["optimal_cost"]
    assert done.metadata["rubric"]["runs_clean"] == 1.0 and done.metadata["rubric"]["cost_quality"] == 1.0
    after = _call(env, "check_plan", plan=[])
    assert after.error is not None or "over" in _text(after)


def test_infeasible_and_string_plans():
    env = RailOpsEnvironment(PACK)
    task = PACK.tasks[-1]
    env.reset(task_id=task.task_id)
    res = json.loads(_text(_call(env, "check_plan", plan="[]")))
    assert not res["feasible"] and res["violations"]
    done = _call(env, "submit_plan", plan=[])
    assert done.done and 0.0 <= done.reward < 0.2
    assert done.metadata["grade"]["feasible"] is False


def test_check_budget_and_tool_call_limit():
    env = RailOpsEnvironment(PACK)
    task = PACK.tasks[0]
    env.reset(task_id=task.task_id)
    for i in range(MAX_CHECKS + 1):
        res = json.loads(_text(_call(env, "check_plan", plan=[])))
    assert "no checks left" in res["error"]
    obs = None
    for _ in range(MAX_TOOL_CALLS - MAX_CHECKS - 1):
        obs = _call(env, "get_situation")
    assert obs.done and obs.reward == 0.0 and obs.metadata["end_reason"] == "tool_call_limit"


def test_local_session_matches_the_environment():
    task = PACK.tasks[0]
    s = LocalSession(PACK)
    meta = s.reset(task.task_id)
    assert meta["task_id"] == task.task_id and {t.name for t in s.tools} == {"get_situation", "check_plan",
                                                                             "submit_plan"}
    text, done, reward, m = s.call("submit_plan", {"plan": task.reference["optimal_plan"]})
    assert done and reward == 1.0 and json.loads(text)["cost"] == task.reference["optimal_cost"]


def test_http_api_hides_references():
    client = TestClient(create_server(PACK))
    assert client.get("/health").status_code == 200
    tasks = client.get("/api/tasks").json()
    assert len(tasks) == len(PACK.tasks)
    t = client.get(f"/api/tasks/{tasks[0]['task_id']}").json()
    assert "reference" not in t and t["runs"]
    ref = client.get(f"/api/tasks/{tasks[0]['task_id']}/reference").json()
    assert ref["floor_cost"] <= ref["optimal_cost"]
    tl = client.get(f"/api/tasks/{tasks[0]['task_id']}/timeline?plan=optimal").json()
    assert tl["feasible"] and tl["cost"] == ref["optimal_cost"] and tl["paths"]
    split = PACK.splits()[0]
    r = client.post("/rail_ops/num_tasks", json={"split": split})
    assert r.status_code == 200 and r.json()["num_tasks"] == PACK.count(split)
    assert client.get("/viewer/").status_code == 200


def test_replay_api():
    client = TestClient(create_server(PACK))
    tid = PACK.tasks[0].task_id
    plans = client.get(f"/api/tasks/{tid}/plans").json()
    assert [p["id"] for p in plans[:4]] == ["optimal", "greedy", "greedy_fast", "naive"]
    r = client.get(f"/api/tasks/{tid}/replay?plan=optimal").json()
    assert r["feasible"] and r["reward"] == 1.0 and r["cost"] == PACK.tasks[0].reference["optimal_cost"]
    assert sum(c[1] for c in r["costs"]) == r["cost"]
    assert {x["train"] for x in r["runs"]} == {x.id for x in PACK.tasks[0].runs}
    naive = client.get(f"/api/tasks/{tid}/replay?plan=naive").json()
    assert naive["cost"] is None or not naive["feasible"] or naive["reward"] < 1.0
    assert client.get(f"/api/tasks/{tid}/replay?plan=nope").status_code == 400
    assert client.get("/api/episodes/nope/replay").status_code == 404
    page = client.get("/twin/")
    assert page.status_code == 200 and "three" in page.text
    assert client.get("/twin/twin.json.gz").status_code == 200


@pytest.mark.skipif(not RUNS_DIR.is_dir(), reason="no committed rollout runs")
def test_model_plans_replay_to_their_recorded_reward():
    """Every submitted plan in the committed runs replays to the reward the eval recorded for it."""
    plans = run_plans()
    n = 0
    for task_id, entries in plans.items():
        task = PACK.get(task_id)
        for e in entries:
            plan, probs = parse_plan(task, e["plan"])
            assert replay(task, plan, e["label"], probs)["reward"] == pytest.approx(e["reward"], abs=1e-4), e["id"]
            n += 1
    assert n >= 45


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_oracle_over_websocket():
    """A real server and OpenEnv's MCP client: the stored optimum scores 1.0 through the whole stack."""
    import uvicorn

    from oracle_solve import main as oracle  # noqa: F401  (imported below via path)

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_server(PACK), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    try:
        ids = [t.task_id for t in PACK.tasks[:2]]
        argv = ["--url", f"http://127.0.0.1:{port}"] + sum((["--task", i] for i in ids), [])
        assert oracle(argv) == 0
    finally:
        server.should_exit = True
        th.join(timeout=10)


def test_model_names():
    assert model_name("hf:Qwen/Qwen3.8-2.4T-A95B:together@default") == "Qwen3.8-2.4T-A95B (Together)"
    assert model_name("claude-sonnet-5-5@high") == "Claude Sonnet 5.5 (effort high)"
    assert model_name("claude-opus-5-5@default") == "Claude Opus 5.5"
    assert model_name("some-model") == "some-model"


@pytest.mark.skipif(not RUNS_DIR.is_dir(), reason="no committed rollout runs")
def test_runs_api_matches_the_boards():
    """/api/runs (the eval explorer) carries each run's board numbers and every episode, with a plan id the replay
    knows; the pages and the web UI's Explore tab are served."""
    client = TestClient(create_server(PACK))
    runs = client.get("/api/runs").json()
    boards = {p.parent.name: p.read_text() for p in RUNS_DIR.glob("*/board.md")}
    for m in runs["models"]:
        lo, hi = m["ci"]
        assert f"**{m['mean']:.3f}** ({lo:.3f}-{hi:.3f})" in boards[m["run"]]
        assert m["n"] == sum(e["run"] == m["run"] and e["model"] == m["model"] for e in runs["episodes"])
    e = next(e for e in runs["episodes"] if e["plan_id"] and e["task_id"] in {t.task_id for t in PACK.tasks})
    assert e["plan_id"] in {p["id"] for p in client.get(f"/api/tasks/{e['task_id']}/plans").json()}
    summary = client.get("/api/tasks").json()[0]
    assert set(summary["rewards"]) == {"greedy", "greedy_fast", "naive", "real"}
    for page in ("/explorer/", "/viewer/", "/web/"):
        assert client.get(page).status_code == 200


def test_pages_are_revalidated_and_frames_versioned():
    """Browsers must ask for the pages and scripts every time (a deploy shows at once; unchanged files cost a 304), and
    the web UI's frames carry a tag that changes with the pages, so no frame shows a copy cached before a deploy."""
    from rail_openenv.gradio_ui import PAGES, version

    client = TestClient(create_server(PACK))
    for path in ("/explorer/", "/viewer/", "/twin/", "/twin/geo.js"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache", path
        assert client.get(path, headers={"If-None-Match": r.headers["etag"]}).status_code == 304, path
    assert "cache-control" not in client.get("/api/runs").headers
    assert all(f"v={version()}" in src for _, src in PAGES)

