"""The server: OpenEnv's app plus the map replay, a train-graph viewer and their read-only JSON on the same port.

OpenEnv provides `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/mcp`, `/health`, the Task API under `/rail_ops/...`
and its web UI at `/web`, where an "Explore" tab (`gradio_ui.py`) frames the pages below next to OpenEnv's Playground.
We add:

    /twin/                                the map replay: trains moving on the line's OpenStreetMap track, any plan
    /viewer/                              a train graph (time against distance) of any plan, live episodes too
    /explorer/                            the eval explorer: the board and every model's plan on every eval task
    /api/tasks                            task summaries, with the lazy baselines' rewards
    /api/runs                             the committed model runs: each episode's reward and outcome, and a board
    /api/tasks/{id}                       a public task (what the agent sees, structured)
    /api/tasks/{id}/reference             optimum, floor, baselines and the real outcome - for people; no tool shows them
    /api/tasks/{id}/timeline?plan=...     paths for the graph: plan = optimal | naive | greedy | greedy_fast
    /api/episodes                         live episodes on this server (newest first)
    /api/episodes/{episode_id}/timeline   the latest plan an episode checked or submitted, as paths
    /api/tasks/{id}/plans                 the plans the replay can show: optimum, baselines, models' submitted plans
                                          (from results/rollouts/*/index.json, or RAIL_RUNS_DIR), live episodes
    /api/tasks/{id}/replay?plan=...       everything the replay animates (rail_core.replay)
    /api/episodes/{episode_id}/replay     the same for the latest plan of a live episode

    uvicorn rail_openenv.server:app --port 8000

`/` is OpenEnv's: it redirects to the web UI (`/web/`), or to `/explorer/` when ENABLE_WEB_INTERFACE is false.
"""

from __future__ import annotations

import json
import os
import random
import statistics
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from rail_core import TaskPack, evaluate, hhmm, load_pack, parse_plan
from rail_core.baselines import POLICIES
from rail_core.check import network
from rail_core.replay import replay

try:
    from .environment import STORE, RailOpsEnvironment, RailState
    from .gradio_ui import build_ui
except ImportError:
    from environment import STORE, RailOpsEnvironment, RailState
    from gradio_ui import build_ui

ENV_NAME = "rail_ops"
WEB = Path(__file__).resolve().parent / "web"
_UP = Path(__file__).resolve().parents
RUNS_DIR = Path(os.environ.get("RAIL_RUNS_DIR") or (_UP[4] if len(_UP) > 4 else _UP[-1]) / "results" / "rollouts")
BASELINES = {"greedy": "greedy dispatcher", "greedy_fast": "greedy dispatcher (turns early)",
             "naive": "change nothing"}


TIERS = ("real", "blockage", "loop", "compound")
PROVIDERS = {"together": "Together", "baseten": "Baseten", "novita": "Novita", "deepinfra": "DeepInfra",
             "fireworks": "Fireworks"}


def model_name(spec: str) -> str:
    """A person's name for a run's model spec: "hf:Qwen/Qwen3.8-2.4T-A95B:together@default" -> "Qwen3.8-2.4T-A95B
    (Together)", "claude-sonnet-5-5@high" -> "Claude Sonnet 5.5 (effort high)"."""
    model, _, effort = spec.partition("@")
    if model.startswith("hf:"):
        repo, _, provider = model[3:].partition(":")
        name = repo.split("/")[-1]
        return f"{name} ({PROVIDERS.get(provider, provider)})" if provider else name
    if model.startswith("claude-"):
        parts = model.split("-")[1:]
        family = " ".join(p.capitalize() for p in parts if not p.isdigit())
        version = ".".join(p for p in parts if p.isdigit())
        name = f"Claude {family} {version}".strip()
        return f"{name} (effort {effort})" if effort and effort != "default" else name
    return model


def load_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    """The committed rollout runs (results/rollouts/*/index.json): [{run, episodes}]."""
    out = []
    for idx in sorted(runs_dir.glob("*/index.json")) if runs_dir.is_dir() else []:
        try:
            body = json.loads(idx.read_text())
        except (OSError, ValueError):
            continue
        out.append({"run": body["run"], "episodes": body.get("episodes", [])})
    return out


def run_plans(runs_dir: Path = RUNS_DIR) -> dict[str, list[dict]]:
    """Submitted plans in the committed rollout runs, by task: [{id, label, model, run, reward, plan}]."""
    out: dict[str, list[dict]] = {}
    for body in load_runs(runs_dir):
        for e in body["episodes"]:
            if e.get("plan") is None:
                continue
            out.setdefault(e["task_id"], []).append(
                {"id": f"run:{body['run']}:{e['model']}", "label": model_name(e["model"]),
                 "model": e["model"], "run": body["run"], "reward": e.get("reward"), "plan": e["plan"]})
    for entries in out.values():                     # two runs of one model on a task: name the run too
        for e in entries:
            if sum(x["model"] == e["model"] for x in entries) > 1:
                e["label"] = f"{e['label']} ({e['run']})"
    return out


def ci(xs, n=2000, seed=0):
    """95 % bootstrap interval of the mean, the same as eval/report.py's (so the numbers match the README)."""
    xs = sorted(xs)
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def runs_summary(runs: list[dict]) -> dict:
    """What the eval explorer shows: one row per (run, model) with the board's columns, and every episode's outcome."""
    rows, episodes = [], []
    for body in runs:
        by: dict[str, list[dict]] = {}
        for e in body["episodes"]:
            by.setdefault(e["model"], []).append(e)
        for model, es in by.items():
            r = [e["reward"] for e in es]
            lo, hi = ci(r)
            real = [e for e in es if e["difficulty"] == "real" and e.get("real_cost") is not None]
            rows.append({
                "run": body["run"], "model": model, "name": model_name(model), "split": es[0]["split"],
                "n": len(es), "mean": statistics.mean(r), "ci": [lo, hi],
                "tiers": {t: statistics.mean(x["reward"] for x in es if x["difficulty"] == t)
                          for t in TIERS if any(x["difficulty"] == t for x in es)},
                "feasible": sum(bool(e["feasible"]) for e in es),
                "optimal": sum(1 for e in es if e["feasible"] and e["cost"] == e["optimal_cost"]),
                "no_submit": sum(not e["submitted"] for e in es),
                "beat_real": sum(1 for e in real if e["feasible"] and e["cost"] < e["real_cost"]), "real": len(real),
                "median_turns": statistics.median(e["agent_turns"] for e in es)})
            for e in es:
                episodes.append({k: e.get(k) for k in ("model", "task_id", "reward", "feasible", "submitted", "cost",
                                                       "optimal_cost", "end_reason", "agent_turns", "checks",
                                                       "violations")}
                                | {"run": body["run"], "plan_id": f"run:{body['run']}:{model}"
                                   if e.get("plan") is not None else None})
    rows.sort(key=lambda x: (x["split"], -x["mean"]))
    return {"models": rows, "episodes": episodes}


def _summary(t, plans: dict | None = None) -> dict:
    ref = t.reference
    return {"task_id": t.task_id, "split": t.split, "day": t.day, "start": hhmm(t.start), "end": hhmm(t.end),
            "difficulty": t.difficulty, "runs": len(t.runs), "closures": [c.kind for c in t.closures],
            "causes": sorted({c.cause for c in t.closures if c.cause}),
            "model_plans": len((plans or {}).get(t.task_id, [])), "optimal_cost": ref.get("optimal_cost"),
            "rewards": {"greedy": ref.get("greedy_reward"), "greedy_fast": ref.get("greedy_fast_reward"),
                        "naive": ref.get("naive_reward"), "real": (ref.get("real") or {}).get("reward_if_legal")}}


def timeline(task, plan) -> dict:
    """Every served run as a path of (km, minute) points, for a time-distance graph."""
    net = network(task)
    res = evaluate(task, plan)
    paths = []
    for x in res.runs:
        if x.cancelled:
            paths.append({"train": x.train, "set": x.unit, "cancelled": True, "points": []})
            continue
        r = task.run(x.train)
        pts = []
        for i in sorted(set(x.arr) | set(x.dep)):
            km = net.by_code[r.route[i]].km
            if i in x.arr:
                pts.append([km, x.arr[i]])
            if i in x.dep and x.dep[i] != x.arr.get(i):
                pts.append([km, x.dep[i]])
        paths.append({"train": x.train, "set": x.unit, "kind": r.kind, "points": pts, "delay": x.delay,
                      "unserved": x.unserved, "problems": x.problems})
    return {"task_id": task.task_id, "start": task.start, "end": task.end,
            "points": [{"code": p.code, "km": p.km, "kind": p.kind, "tracks": p.tracks} for p in net.points],
            "sections": [{"a": s.a, "b": s.b, "single": s.single} for s in net.sections],
            "closures": [{"kind": c.kind, "where": c.where, "start": c.start, "end": c.end, "cause": c.cause}
                         for c in task.closures],
            "paths": paths, "feasible": res.feasible, "cost": res.cost, "violations": res.violations[:50],
            "optimal_cost": task.reference.get("optimal_cost")}


PAGES = ("/twin", "/explorer", "/viewer")


def mount_api(app, pack: TaskPack) -> None:
    plans_by_task = run_plans()
    runs = runs_summary(load_runs())

    @app.middleware("http")
    async def revalidate_pages(request, call_next):
        """The pages and their scripts change with every deploy. Without a Cache-Control header browsers guess how
        long to reuse a stored copy (often hours), so a new version did not show; `no-cache` makes them ask every time,
        and an unchanged file costs a 304."""
        response = await call_next(request)
        if request.url.path.startswith(PAGES):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.get("/api/tasks", include_in_schema=False)
    def tasks():
        return [_summary(t, plans_by_task) for t in pack.tasks]

    @app.get("/api/runs", include_in_schema=False)
    def model_runs():
        return runs

    def _task(task_id):
        try:
            return pack.get(task_id)
        except KeyError:
            raise HTTPException(404, f"no task {task_id}") from None

    @app.get("/api/tasks/{task_id}", include_in_schema=False)
    def task(task_id: str):
        return pack.public(_task(task_id))

    @app.get("/api/tasks/{task_id}/reference", include_in_schema=False)
    def reference(task_id: str):
        return _task(task_id).reference

    @app.get("/api/tasks/{task_id}/timeline", include_in_schema=False)
    def task_timeline(task_id: str, plan: str = "optimal"):
        t = _task(task_id)
        if plan == "optimal":
            p, _ = parse_plan(t, t.reference["optimal_plan"])
        elif plan in POLICIES:
            p = POLICIES[plan](t)
        else:
            raise HTTPException(400, "plan must be optimal, naive, greedy or greedy_fast")
        return timeline(t, p)

    def _plan(t, plan_id: str):
        """(plan, parse problems, label) for a plan id."""
        if plan_id == "optimal":
            p, probs = parse_plan(t, t.reference["optimal_plan"])
            return p, probs, "CP-SAT optimum"
        if plan_id in POLICIES:
            return POLICIES[plan_id](t), [], BASELINES.get(plan_id, plan_id)
        for e in plans_by_task.get(t.task_id, []):
            if e["id"] == plan_id:
                p, probs = parse_plan(t, e["plan"])
                return p, probs, e["label"]
        raise HTTPException(400, f"unknown plan {plan_id!r}: see /api/tasks/{t.task_id}/plans")

    @lru_cache(maxsize=256)
    def _replay(task_id: str, plan_id: str) -> dict:
        t = _task(task_id)
        p, probs, label = _plan(t, plan_id)
        return replay(t, p, label, probs) | {"plan_id": plan_id}

    @app.get("/api/tasks/{task_id}/plans", include_in_schema=False)
    def task_plans(task_id: str):
        t = _task(task_id)
        ref = t.reference
        out = [{"id": "optimal", "label": "CP-SAT optimum", "reward": 1.0, "cost": ref.get("optimal_cost")}]
        out += [{"id": k, "label": v, "reward": ref.get(f"{k}_reward"),
                 "cost": ref.get(f"{k}_cost") if ref.get(f"{k}_feasible") else None} for k, v in BASELINES.items()]
        out += [{k: e[k] for k in ("id", "label", "reward")} for e in plans_by_task.get(task_id, [])]
        out += [{"id": f"episode:{ep.episode_id}", "label": f"live episode {ep.episode_id}", "reward": None}
                for ep in STORE.recent() if ep.task.task_id == task_id]
        return out

    @app.get("/api/tasks/{task_id}/replay", include_in_schema=False)
    def task_replay(task_id: str, plan: str = "optimal"):
        if plan.startswith("episode:"):
            return episode_replay(plan.split(":", 1)[1])
        return _replay(task_id, plan)

    @app.get("/api/episodes/{episode_id}/replay", include_in_schema=False)
    def episode_replay(episode_id: str):
        ep = STORE.get(episode_id)
        if ep is None:
            raise HTTPException(404, f"no episode {episode_id}")
        last = next((s for s in reversed(ep.steps) if s.get("plan") is not None), None)
        p, probs = parse_plan(ep.task, last["plan"]) if last else ({}, [])
        return replay(ep.task, p, f"live episode {episode_id}", probs) | {"plan_id": f"episode:{episode_id}"}

    @app.get("/api/episodes", include_in_schema=False)
    def episodes():
        return [{k: v for k, v in ep.summary().items() if k != "steps"} | {"steps": len(ep.steps)}
                for ep in STORE.recent()]

    @app.get("/api/episodes/{episode_id}/timeline", include_in_schema=False)
    def episode_timeline(episode_id: str):
        ep = STORE.get(episode_id)
        if ep is None:
            raise HTTPException(404, f"no episode {episode_id}")
        last = next((s for s in reversed(ep.steps) if s.get("plan") is not None), None)
        p, _ = parse_plan(ep.task, last["plan"]) if last else ({}, [])
        return timeline(ep.task, p) | {"episode": {k: v for k, v in ep.summary().items() if k != "steps"}}

    if (WEB / "twin").is_dir():
        app.mount("/twin", StaticFiles(directory=WEB / "twin", html=True), name="twin")
    if (WEB / "explorer").is_dir():
        app.mount("/explorer", StaticFiles(directory=WEB / "explorer", html=True), name="explorer")
    if WEB.is_dir():
        app.mount("/viewer", StaticFiles(directory=WEB, html=True), name="viewer")

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/explorer/" if (WEB / "explorer" / "index.html").is_file() else "/twin/")


def create_server(pack: TaskPack | None = None):
    pack = pack or load_pack()

    def factory():
        return RailOpsEnvironment(pack)

    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    app = create_app(factory, CallToolAction, CallToolObservation, env_name=ENV_NAME,
                     max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "64")),
                     gradio_builder=build_ui, custom_tab_name="Explore", custom_tab_primary=True,
                     show_default_tab=True, title_override="RailSimEnv v1 Environment", state_cls=RailState)
    mount_api(app, pack)

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok", "tasks": {s: pack.count(s) for s in pack.splits()}}

    return app


_app = None


def __getattr__(name: str):
    global _app
    if name == "app":
        if _app is None:
            _app = create_server()
        return _app
    raise AttributeError(name)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    port = port or int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_server(), host=host, port=port, ws_ping_interval=60, ws_ping_timeout=300)


if __name__ == "__main__":
    main()
