"""Build the Inkling 2 Small article as a static site, review it, and publish it as a Hugging Face Space.

    cd railsim-v1
    PY=envs/rail_ops/.venv/bin/python
    $PY blog/site/build.py --dev                     # edit with live reload on http://127.0.0.1:4321
    $PY blog/site/build.py --serve                   # the exact build that would be pushed, on :4321
    $PY blog/site/build.py --diff                    # what changed since the live Space's build
    $PY blog/site/build.py --draft -m "message"      # push to the private draft Space, to review anywhere
    $PY blog/site/build.py --publish -m "message"    # push to the real Space: only a build already on the draft

Two Spaces, both private until `--public`: the draft (bpHigh/RailSimEnv-Inkling-draft) and the real one
(bpHigh/RailSimEnv-Inkling). Every push writes `build.json` with a fingerprint of everything that goes into the page
(this folder's content, public files, card, build code and the template commit) and the git commit it came from.
`--publish` refuses unless the draft Space carries the same fingerprint, so nothing reaches the real Space without
first being looked at on the draft. `--push REPO` pushes anywhere, without that check.

The page is Hugging Face's research article template (Thibaud Frere, CC BY 4.0), pinned to one commit, with this
folder's `content/` in place of its demo content and `public/` added to its public files. The template is an Astro
site that its own Space builds in Docker; a free account cannot run Docker Spaces, so the site is built here and pushed
as a static Space. Needs node and npm (the template's dependencies are installed once into dist/blog-template/).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # railsim-v1
TEMPLATE = "https://huggingface.co/spaces/tfrere/research-article-template"
TEMPLATE_REV = "5313a101702ce0b762dae664263330f0f05c830e"
CACHE = ROOT / "dist" / "blog-template"
WORK = ROOT / "dist" / "blog-build"
WORK_DEV = ROOT / "dist" / "blog-dev"                    # --dev has its own copy, so builds can run beside it
OUT = ROOT / "dist" / "blog-space"
LIVE = "bpHigh/RailSimEnv-Inkling"
DRAFT = LIVE + "-draft"
PORT = 4321
FORBIDDEN = re.compile(r"claude-sessions|CLAUDE\.md|Claude Code|Co-Authored-By|Claude-Session|claude\.ai/code", re.I)
# What goes into the page, for the fingerprint (make_data.py and the capture tools only feed content/).
INPUTS = ["content", "public", "space-card.md", "build.py", "tools/flatten.cjs"]

# A private Space is shown in a third-party iframe, where a browser that blocks third-party storage makes every
# `localStorage` access throw. The template's theme toggle reads it unguarded, and since its script shares a bundle
# with the table of contents, one throw would leave the page with no contents and a dead toggle.
STORAGE_GUARDS = {
    "src/components/ThemeToggle.astro": [
        ('const saved = localStorage.getItem("theme");',
         'let saved = null;\n    try { saved = localStorage.getItem("theme"); } catch (e) {}'),
        ('localStorage.setItem("theme", next);',
         'try { localStorage.setItem("theme", next); } catch (e) {}'),
    ],
}


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()


def template() -> Path:
    """The template checkout at the pinned commit, with its npm dependencies installed (kept between builds)."""
    if not (CACHE / ".git").is_dir():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "-q", TEMPLATE, str(CACHE)], ROOT)
    run(["git", "-c", "advice.detachedHead=false", "checkout", "-q", TEMPLATE_REV], CACHE)
    run(["git", "lfs", "pull", "--include", "app/public/*"], CACHE)     # icons are LFS files; the Hub refuses pointers
    if not (CACHE / "app" / "node_modules").is_dir():
        run(["npm", "ci", "--no-audit", "--no-fund"], CACHE / "app")
    return CACHE / "app"


def check(folder: Path) -> None:
    bad = [str(p.relative_to(folder)) for p in folder.rglob("*")
           if p.is_file() and p.suffix in (".mdx", ".md", ".html", ".js", ".json", ".bib", ".css")
           and FORBIDDEN.search(p.read_text("utf-8", errors="ignore"))]
    if bad:
        raise SystemExit(f"refusing to build: {bad} mention something that stays out of the Space")


def guard_storage(app: Path) -> None:
    for rel, edits in STORAGE_GUARDS.items():
        f = app / rel
        s = f.read_text("utf-8")
        for old, new in edits:
            if old not in s:
                raise SystemExit(f"{rel}: expected {old!r}; the template changed, update STORAGE_GUARDS")
            s = s.replace(old, new)
        f.write_text(s, "utf-8")


def input_files() -> list[Path]:
    files = []
    for name in INPUTS:
        p = HERE / name
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
    return [f for f in files if "__pycache__" not in f.parts]


def fingerprint() -> str:
    h = hashlib.sha256(TEMPLATE_REV.encode())
    for f in input_files():
        h.update(str(f.relative_to(HERE)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def build_info() -> dict:
    inputs = [str(HERE / n) for n in INPUTS]
    dirty = bool(git("status", "--porcelain", "--", *inputs))
    return {"fingerprint": fingerprint(), "commit": git("rev-parse", "HEAD"), "uncommitted_changes": dirty,
            "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}


def sync_content(work: Path) -> None:
    """Put this folder's content and public files into the template copy."""
    content = work / "src" / "content"
    if content.exists():
        shutil.rmtree(content)
    shutil.copytree(HERE / "content", content)
    data = work / "public" / "data"                      # the template links it to content/assets/data
    if data.is_symlink() or data.is_file():
        data.unlink()
    elif data.is_dir():
        shutil.rmtree(data)
    shutil.copytree(content / "assets" / "data", data)
    shutil.copytree(HERE / "public", work / "public", dirs_exist_ok=True)


def prepare(work: Path = WORK) -> Path:
    check(HERE / "content")
    src = template()
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(src, work, symlinks=True, ignore=shutil.ignore_patterns("node_modules", "dist", ".astro"))
    (work / "node_modules").symlink_to(src / "node_modules")
    guard_storage(work)
    sync_content(work)
    return src


def build() -> Path:
    src = prepare()
    run(["npm", "run", "build"], WORK)
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(WORK / "dist", OUT)
    shutil.copy2(HERE / "space-card.md", OUT / "README.md")
    run(["node", str(HERE / "tools" / "flatten.cjs"), str(OUT), str(src)], ROOT)   # see the note in flatten.cjs
    info = build_info()
    (OUT / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    check(OUT)
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {n} files, {size / 1e6:.1f} MB, fingerprint {info['fingerprint']}")
    return OUT


def dev() -> None:
    """Astro's dev server on the template copy, re-copying this folder's files whenever one changes."""
    prepare(WORK_DEV)
    stamp = lambda: {f: f.stat().st_mtime for f in input_files()}            # noqa: E731
    seen = stamp()
    proc = subprocess.Popen(["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(PORT)], cwd=WORK_DEV)
    print(f"editing: change files in {HERE / 'content'}; the page at http://127.0.0.1:{PORT}/ reloads", flush=True)
    try:
        while proc.poll() is None:
            time.sleep(1)
            now = stamp()
            if now != seen:
                changed = sorted(str(f.relative_to(HERE)) for f in set(now) | set(seen) if now.get(f) != seen.get(f))
                seen = now
                try:
                    check(HERE / "content")
                    sync_content(WORK_DEV)
                    print("synced", ", ".join(changed[:6]), flush=True)
                except SystemExit as e:
                    print(e, flush=True)
    finally:
        proc.terminate()


def deployed(repo_id: str) -> dict | None:
    """The build.json on a Space, or None when it has none (or does not exist)."""
    from huggingface_hub import hf_hub_download
    try:
        return json.loads(Path(hf_hub_download(repo_id, "build.json", repo_type="space")).read_text())
    except Exception:
        return None


def diff(repo_id: str) -> None:
    info, now = deployed(repo_id), fingerprint()
    if not info:
        print(f"{repo_id} has no build.json yet; everything is new")
        return
    print(f"{repo_id}: built {info['built']} from {info['commit'][:10]}"
          f"{' plus uncommitted changes' if info.get('uncommitted_changes') else ''}, fingerprint {info['fingerprint']}")
    if info["fingerprint"] == now:
        print("the current files build exactly what is there: nothing to push")
        return
    paths = [str(HERE / n) for n in INPUTS]
    print(f"now: fingerprint {now}. Changes since that commit (committed and not):\n")
    print(git("diff", "--stat", info["commit"], "--", *paths))
    text = [p for p in paths if p.endswith(("content", "space-card.md"))]
    out = git("diff", "--word-diff=plain", "--word-diff-regex=[^ ]+", info["commit"], "--", *text,
              ":(exclude)*.webp", ":(exclude)*.csv", ":(exclude)*.json")
    print(out or "(no text changes; only code or media)")


def push(out: Path, repo_id: str, message: str, public: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", private=not public, exist_ok=True)
    if public:
        api.update_repo_settings(repo_id, repo_type="space", private=False)
    info = api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="space", commit_message=message,
                             delete_patterns=["*"])
    print(info.commit_url if hasattr(info, "commit_url") else info)
    print(f"https://huggingface.co/spaces/{repo_id} ({'public' if public else 'private unless made public'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dev", action="store_true", help=f"edit with live reload on http://127.0.0.1:{PORT}")
    mode.add_argument("--diff", nargs="?", const=LIVE, metavar="REPO_ID",
                      help=f"show what changed since a Space's build (default {LIVE}; or {DRAFT})")
    mode.add_argument("--draft", action="store_true", help=f"build and push to the draft Space {DRAFT}")
    mode.add_argument("--publish", action="store_true",
                      help=f"build and push to {LIVE}, only if the draft carries the same build")
    mode.add_argument("--push", metavar="REPO_ID", help="build and push to any Space, without the draft check")
    ap.add_argument("-m", "--message", default="Update the article")
    ap.add_argument("--public", action="store_true", help="make the Space public (a new one starts private)")
    ap.add_argument("--serve", action="store_true", help=f"preview the built site on http://127.0.0.1:{PORT}")
    args = ap.parse_args()
    if args.dev:
        return dev()
    if args.diff:
        return diff(args.diff)
    if args.publish:
        draft, now = deployed(DRAFT), fingerprint()
        if not draft or draft["fingerprint"] != now:
            have = draft["fingerprint"] if draft else "nothing"
            raise SystemExit(f"refusing to publish: the draft Space has {have}, these files build {now}. "
                             f"Push them with --draft, review {DRAFT}, then --publish.")
    out = build()
    repo = DRAFT if args.draft else LIVE if args.publish else args.push
    if repo:
        push(out, repo, args.message, args.public)
    if args.serve:
        run(["python3", "-m", "http.server", str(PORT), "--bind", "127.0.0.1"], out)


if __name__ == "__main__":
    main()
