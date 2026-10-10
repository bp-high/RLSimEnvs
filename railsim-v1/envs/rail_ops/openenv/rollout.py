"""Play tasks against a RailSimEnv server and print what happened.

    python rollout.py --policy greedy --tasks 5            # scripted policies, no API key: greedy | greedy_fast | naive | optimal
    python rollout.py --policy claude --tasks 1            # Claude through the MCP tools (ANTHROPIC_API_KEY or LLM_API_KEY)
    python rollout.py --policy claude --model claude-sonnet-5-5 --effort high --base-url http://127.0.0.1:8000

Without --base-url a local server is started on a free port. --out writes every rollout record as JSON lines.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

from rail_core import load_pack, plan_to_list
from rail_core.baselines import POLICIES
from rail_openenv.session import EnvSession


def start_server() -> str:
    import uvicorn

    from rail_openenv.server import create_server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_server(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return f"http://127.0.0.1:{port}"


def scripted(base: str, task, policy: str) -> dict:
    """A scripted episode: read the situation, check the policy's plan once, submit it."""
    plan = task.reference["optimal_plan"] if policy == "optimal" else plan_to_list(POLICIES[policy](task))
    env = EnvSession(base)
    try:
        env.reset(task.task_id)
        env.call("get_situation", {})
        checked, _, _, _ = env.call("check_plan", {"plan": plan})
        _, done, reward, meta = env.call("submit_plan", {"plan": plan})
    finally:
        env.close()
    c = json.loads(checked)
    return {"task_id": task.task_id, "policy": policy, "reward": reward, "end_reason": "submitted" if done else None,
            "final": meta, "checked_feasible": c.get("feasible"), "turns": 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--policy", default="greedy", choices=["claude", "greedy", "greedy_fast", "naive", "optimal"])
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    base = args.base_url or start_server()
    pack = load_pack()
    tasks = [t for t in pack.tasks if t.split == args.split][: args.tasks]
    rewards = []
    for t in tasks:
        if args.policy == "claude":
            from rail_openenv.agent import run_episode

            rec = run_episode(base, t.task_id, model=args.model, effort=args.effort, max_turns=args.max_turns)
            for s in rec["steps"]:
                print(f"  turn {s['turn']} {s['tool']}: {s['result'][:200]}")
        else:
            rec = scripted(base, t, args.policy)
        g = (rec.get("final") or {}).get("grade") or {}
        rewards.append(rec["reward"] or 0.0)
        print(f"{t.task_id} ({t.difficulty}, {len(t.runs)} runs): reward {rec['reward'] or 0:.3f}  "
              f"cost {g.get('cost')} (optimum {t.reference['optimal_cost']}, floor {t.reference['floor_cost']})  "
              f"{rec['end_reason']}")
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
    if rewards:
        print(f"mean reward {sum(rewards) / len(rewards):.3f} over {len(rewards)} tasks")


if __name__ == "__main__":
    main()
