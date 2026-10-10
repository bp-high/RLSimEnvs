"""Run models on the rail-dispatch tasks through the env server and save every rollout.

    cd railsim-v1
    envs/rail_ops/.venv/bin/python eval/run_eval.py --run 2026-10-07-sonnet-eval45 \
        --models claude-sonnet-5-5@high --split eval --concurrency 6
    envs/rail_ops/.venv/bin/python eval/run_eval.py --run 2026-10-08-glm-5-3-flash-eval45 \
        --models hf:zai-org/GLM-5.3-Flash:novita --bill-to DeepPPT --split eval

A model is `<model id>[@effort]`. Claude models (effort defaults to high) use the Anthropic API, keyed from
ANTHROPIC_API_KEY or LLM_API_KEY. `hf:<model id>[:<provider>]` models use Hugging Face's OpenAI-compatible router
(`rail_openenv.agent_openai`), keyed from HF_TOKEN or HUGGING_FACE_API_KEY and billed to --bill-to (or HF_BILL_TO);
their effort defaults to "default" (the provider's own reasoning setting; anything else is sent as reasoning_effort).
Starts a server in this process unless --server is given.
Rollouts go to results/rollouts/<run>/<model>/<task>.json (git-ignored) with an index.json and board.md per run;
existing episodes are skipped (resume), --force reruns them.
index.json keeps each episode's submitted plan, so the map replay (/twin/) can show it without the transcripts.
`--reindex` rebuilds index.json and board.md from the transcripts on disk and calls no model.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from rail_core import load_pack  # noqa: E402
from rail_openenv import agent, agent_openai  # noqa: E402
from report import board  # noqa: E402

_LOCK = threading.Lock()


def start_server() -> str:
    import uvicorn

    from rail_openenv.server import create_server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(create_server(), host="127.0.0.1", port=port, log_level="warning",
                            ws_ping_interval=60, ws_ping_timeout=600)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def parse_spec(spec: str) -> tuple[str, str]:
    model, _, effort = spec.partition("@")
    return model, effort or ("default" if agent_openai.is_openai_spec(model) else "high")


def slug(spec: str) -> str:
    return spec.replace("/", "_").replace(":", "_").replace("@", "_")


def submitted_plan(record: dict):
    for step in reversed(record.get("steps") or []):
        if step.get("tool") == "submit_plan":
            return (step.get("arguments") or {}).get("plan")
    return None


def summarize(spec: str, record: dict, task) -> dict:
    g = (record.get("final") or {}).get("grade") or {}
    u = record["usage"]
    ref = task.reference
    return {"model": spec, "task_id": record["task_id"], "split": task.split, "difficulty": task.difficulty,
            "day": task.day, "runs": len(task.runs), "reward": record["reward"] or 0.0,
            "submitted": record["end_reason"] == "submitted", "feasible": bool(g.get("feasible")),
            "cost": g.get("cost"), "optimal_cost": ref["optimal_cost"], "floor_cost": ref["floor_cost"],
            "greedy_cost": ref.get("greedy_cost") if ref.get("greedy_feasible") else None,
            "real_cost": (ref.get("real") or {}).get("cost"), "hardness": ref.get("hardness"),
            "violations": len(g.get("violations") or []), "agent_turns": record["turns"],
            "checks": sum(1 for s in record["steps"] if s["tool"] == "check_plan"), "seconds": record["seconds"],
            "input_tokens": u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"],
            "cache_read_tokens": u["cache_read_input_tokens"], "cache_write_tokens": u["cache_creation_input_tokens"],
            "output_tokens": u["output_tokens"],
            "served_by": record.get("served_by"), "end_reason": record["end_reason"],
            "errors": len(record.get("errors") or []), "plan": submitted_plan(record)}


def write_index(run_dir: Path, rows: dict):
    with _LOCK:
        body = {"run": run_dir.name, "episodes": sorted(rows.values(), key=lambda r: (r["model"], r["task_id"]))}
        tmp = run_dir / "index.json.tmp"
        tmp.write_text(json.dumps(body, indent=1))
        tmp.replace(run_dir / "index.json")


def reindex(run_dir: Path) -> None:
    pack = load_pack()
    rows = {}
    for path in sorted(run_dir.glob("*/*.json")):
        rec = json.loads(path.read_text())
        spec = f"{rec['model']}@{rec['effort']}"
        rows[f"{spec}|{rec['task_id']}"] = summarize(spec, rec, pack.get(rec["task_id"]))
    write_index(run_dir, rows)
    (run_dir / "board.md").write_text(board(list(rows.values())) + "\n")
    print(f"{run_dir.name}: {len(rows)} episodes reindexed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="<model id>[@effort]")
    ap.add_argument("--split", default="eval", choices=["all", "eval", "train"])
    ap.add_argument("--limit", type=int, default=None, help="first N tasks of the split")
    ap.add_argument("--tasks", nargs="*", default=None, help="explicit task ids")
    ap.add_argument("--server", default=None)
    ap.add_argument("--concurrency", type=int, default=6, help="episodes in flight per model")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--wall-clock", type=float, default=3600.0, help="seconds per episode before it is cut off")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bill-to", default=None, help="Hugging Face organisation billed for hf: models (X-HF-Bill-To)")
    ap.add_argument("--progress", action="store_true", help="print a line after every turn (hf: models)")
    ap.add_argument("--reindex", action="store_true", help="rebuild index.json and board.md from the transcripts")
    args = ap.parse_args()
    if args.reindex:
        reindex(ROOT / "results" / "rollouts" / args.run)
        return

    args.models = ["@".join(parse_spec(m)) for m in args.models]   # the same spec as --reindex rebuilds
    pack = load_pack()
    tasks = [t for t in pack.tasks if args.split == "all" or t.split == args.split]
    if args.tasks:
        tasks = [pack.get(t) for t in args.tasks]
    if args.limit:
        tasks = tasks[: args.limit]
    run_dir = ROOT / "results" / "rollouts" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if (run_dir / "index.json").is_file():
        for r in json.loads((run_dir / "index.json").read_text())["episodes"]:
            rows[f"{r['model']}|{r['task_id']}"] = r
    base = args.server or start_server()
    print(f"server {base}; {len(tasks)} tasks x {len(args.models)} models -> {run_dir}", flush=True)

    jobs = []
    for t in tasks:  # task-major, so every model has episodes in flight from the start
        for m in args.models:
            path = run_dir / slug(m) / f"{t.task_id}.json"
            if path.is_file() and not args.force:
                rec = json.loads(path.read_text())
                if not rec.get("errors") and rec.get("end_reason") != "wall_clock":
                    rows[f"{m}|{t.task_id}"] = summarize(m, rec, t)
                    continue
            jobs.append((m, t, path))
    write_index(run_dir, rows)
    clients = {}
    if any(not agent_openai.is_openai_spec(parse_spec(m)[0]) for m, _, _ in jobs):
        clients[agent] = agent.make_client(max_retries=8)
    if any(agent_openai.is_openai_spec(parse_spec(m)[0]) for m, _, _ in jobs):
        clients[agent_openai] = agent_openai.make_openai_client(max_retries=8, bill_to=args.bill_to)

    def work(m, t, path):
        model, effort = parse_spec(m)
        harness = agent_openai if agent_openai.is_openai_spec(model) else agent
        extra = {"progress": lambda line: print(f"  {m} {line}", flush=True)} \
            if args.progress and harness is agent_openai else {}
        try:
            rec = harness.run_episode(base, t.task_id, model=model, effort=effort, max_turns=args.max_turns,
                                      max_tokens=args.max_tokens, wall_clock_s=args.wall_clock,
                                      client=clients[harness], **extra)
        except Exception as e:  # an API or session failure: record it so the resume reruns this episode
            rec = {"task_id": t.task_id, "model": model, "effort": effort, "steps": [], "reward": 0.0,
                   "end_reason": "error", "errors": [f"{type(e).__name__}: {e}"[:2000]], "final": None, "turns": 0,
                   "seconds": 0.0, "served_by": [],
                   "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                             "output_tokens": 0}}
        rec["run"] = args.run
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
        return m, t, rec

    # One pool per model: a slow model's episodes never hold the workers a fast model needs.
    pools = {m: ThreadPoolExecutor(max_workers=args.concurrency) for m in args.models}
    futs = [pools[m].submit(work, m, t, path) for m, t, path in jobs]
    done = 0
    for f in as_completed(futs):
        m, t, rec = f.result()
        rows[f"{m}|{t.task_id}"] = summarize(m, rec, t)
        write_index(run_dir, rows)
        done += 1
        err = f" ERR {rec['errors'][0][:160]}" if rec.get("errors") else ""
        print(f"[{done}/{len(jobs)}] {m} {t.task_id} reward={rec['reward'] or 0:.3f} {rec['end_reason']} "
              f"turns={rec['turns']} {rec['seconds']:.0f}s{err}", flush=True)
    for p in pools.values():
        p.shutdown()
    table = board(list(rows.values()))
    (run_dir / "board.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
