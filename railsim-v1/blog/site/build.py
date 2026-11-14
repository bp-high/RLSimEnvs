"""Build the Inkling 2 Small blog post as a static site and publish it as a Hugging Face Space.

    cd railsim-v1
    envs/rail_ops/.venv/bin/python blog/site/build.py                       # build only: dist/blog-space/
    envs/rail_ops/.venv/bin/python blog/site/build.py --serve               # build, then preview on :4321
    envs/rail_ops/.venv/bin/python blog/site/build.py --push bpHigh/RailSimEnv-Inkling -m "message"
    envs/rail_ops/.venv/bin/python blog/site/build.py --push bpHigh/RailSimEnv-Inkling --public -m "publish"

The page is Hugging Face's research article template (Thibaud Frere, CC BY 4.0), pinned to one commit, with this
folder's `content/` in place of its demo content and `public/` added to its public files. Edit the text in
`content/chapters/*.mdx` (or the title, authors and links in `content/article.mdx`), then build and push again.

The template is an Astro site that its own Space builds in Docker; a free account cannot run Docker Spaces, so the site
is built here and pushed as a static Space. A new Space is created private: `--public` makes it public. Needs node and
npm (the template's dependencies are installed once into dist/blog-template/).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # railsim-v1
TEMPLATE = "https://huggingface.co/spaces/tfrere/research-article-template"
TEMPLATE_REV = "5313a101702ce0b762dae664263330f0f05c830e"
CACHE = ROOT / "dist" / "blog-template"
WORK = ROOT / "dist" / "blog-build"
OUT = ROOT / "dist" / "blog-space"
FORBIDDEN = re.compile(r"claude-sessions|CLAUDE\.md|Claude Code|Co-Authored-By|Claude-Session|claude\.ai/code", re.I)


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


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


# A private Space is shown in a third-party iframe, where a browser that blocks third-party storage makes every
# `localStorage` access throw. The template's theme toggle reads it unguarded, and since its script shares a bundle
# with the table of contents, one throw left the page with no contents and a dead toggle.
STORAGE_GUARDS = {
    "src/components/ThemeToggle.astro": [
        ('const saved = localStorage.getItem("theme");',
         'let saved = null;\n    try { saved = localStorage.getItem("theme"); } catch (e) {}'),
        ('localStorage.setItem("theme", next);',
         'try { localStorage.setItem("theme", next); } catch (e) {}'),
    ],
}


def guard_storage(app: Path) -> None:
    for rel, edits in STORAGE_GUARDS.items():
        f = app / rel
        s = f.read_text("utf-8")
        for old, new in edits:
            if old not in s:
                raise SystemExit(f"{rel}: expected {old!r}; the template changed, update STORAGE_GUARDS")
            s = s.replace(old, new)
        f.write_text(s, "utf-8")


def build() -> Path:
    check(HERE / "content")
    src = template()
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(src, WORK, symlinks=True, ignore=shutil.ignore_patterns("node_modules", "dist", ".astro"))
    (WORK / "node_modules").symlink_to(src / "node_modules")
    guard_storage(WORK)
    content = WORK / "src" / "content"
    shutil.rmtree(content)
    shutil.copytree(HERE / "content", content)
    data = WORK / "public" / "data"                      # the template links it to content/assets/data
    if data.is_symlink() or data.is_file():
        data.unlink()
    elif data.is_dir():
        shutil.rmtree(data)
    shutil.copytree(content / "assets" / "data", data)
    shutil.copytree(HERE / "public", WORK / "public", dirs_exist_ok=True)
    run(["npm", "run", "build"], WORK)
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(WORK / "dist", OUT)
    shutil.copy2(HERE / "space-card.md", OUT / "README.md")
    run(["node", str(HERE / "tools" / "flatten.cjs"), str(OUT), str(src)], ROOT)   # see the note in flatten.cjs
    check(OUT)
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {n} files, {size / 1e6:.1f} MB")
    return OUT


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
    ap.add_argument("--push", metavar="REPO_ID", help="push to this Space, e.g. bpHigh/RailSimEnv-Inkling")
    ap.add_argument("-m", "--message", default="Update the article")
    ap.add_argument("--public", action="store_true", help="make the Space public (a new one starts private)")
    ap.add_argument("--serve", action="store_true", help="preview the built site on http://127.0.0.1:4321")
    args = ap.parse_args()
    out = build()
    if args.push:
        push(out, args.push, args.message, args.public)
    if args.serve:
        run(["python3", "-m", "http.server", "4321", "--bind", "127.0.0.1"], out)


if __name__ == "__main__":
    main()
