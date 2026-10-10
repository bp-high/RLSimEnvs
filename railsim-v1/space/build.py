"""Assemble the Hugging Face Space for RailSimEnv v1 in dist/space and, with --push, upload it.

    envs/rail_ops/.venv/bin/python space/build.py                            # -> dist/space
    envs/rail_ops/.venv/bin/python space/build.py --push bpHigh/RailSimEnv   # key: HF_TOKEN or HUGGING_FACE_API_KEY

The Space is a Gradio-SDK Space on ZeroGPU hardware (the only Python hosting a free account gets; Docker and cpu-basic
need a paid plan). `app.py` serves the whole OpenEnv server on port 7860 and never asks for a GPU. The bundle holds the
two packages' sources (the agent harnesses left out: the Space runs no model), the eval and train packs and each
committed run's index.json (the plans the explorer and the replay show).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                     # railsim-v1/
ENV = ROOT / "envs" / "rail_ops"
OUT = ROOT / "dist" / "space"
SKIP = {"__pycache__", "agent.py", "agent_openai.py"}
PACKS = {"rail-v1-eval": ("manifest.json", "selection.json"), "rail-v1-train": ("manifest.json",)}
# what never goes to the Hub: the repository's working notes and tooling marks
FORBIDDEN = re.compile(r"claude-sessions|CLAUDE\.md|Claude Code|Co-Authored-By|Claude-Session|claude\.ai/code", re.I)


def _copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if n in SKIP or n.endswith(".pyc")])


def build(out: Path = OUT) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("README.md", "app.py", "requirements.txt"):
        shutil.copy2(ROOT / "space" / name, out / name)
    shutil.copy2(ROOT.parent / "LICENSE", out / "LICENSE")          # the code's MIT licence (the card lists the data's)
    _copytree(ENV / "core" / "rail_core", out / "core" / "rail_core")
    _copytree(ENV / "openenv" / "rail_openenv", out / "openenv" / "rail_openenv")
    spec = (ENV / "openenv" / "openenv.yaml").read_text()
    (out / "openenv.yaml").write_text(spec.replace("port: 8000", "port: 7860"))
    for pack, extra in PACKS.items():
        src, dst = ENV / "tasks" / pack, out / "tasks" / pack
        dst.mkdir(parents=True)
        body = next(n for n in ("tasks.jsonl", "tasks.jsonl.gz") if (src / n).is_file())
        for name in (body, *extra):
            shutil.copy2(src / name, dst / name)
    for idx in sorted((ROOT / "results" / "rollouts").glob("*/index.json")):
        (out / "runs" / idx.parent.name).mkdir(parents=True)
        shutil.copy2(idx, out / "runs" / idx.parent.name / "index.json")
    bad = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.suffix in
           (".py", ".md", ".js", ".html", ".css", ".json", ".yaml", ".txt") and FORBIDDEN.search(p.read_text("utf-8"))]
    if bad:
        raise SystemExit(f"refusing to build: {bad} mention something that stays out of the Space")
    return out


def push(out: Path, repo_id: str, message: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_API_KEY")
    api = HfApi(token=token)
    info = api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(out),
                             commit_message=message,
                             ignore_patterns=["**/__pycache__/**", "*.pyc"],
                             delete_patterns=["core/**", "openenv/**", "tasks/**", "runs/**"])
    print(info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", metavar="REPO_ID", default=None, help="upload to this Space, e.g. bpHigh/RailSimEnv")
    ap.add_argument("-m", "--message", default="Update RailSimEnv v1", help="the Space commit's message")
    args = ap.parse_args()
    out = build()
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{out}: {sum(1 for p in out.rglob('*') if p.is_file())} files, {size / 1e6:.1f} MB")
    if args.push:
        push(out, args.push, args.message)


if __name__ == "__main__":
    main()
