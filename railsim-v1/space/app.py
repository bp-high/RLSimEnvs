"""RailSimEnv v1 on a Hugging Face Space: the OpenEnv server (reset, step, state, the MCP tools, the Task API and the
web UI at /web), the eval explorer (/explorer/), the 3D replay (/twin/) and the train graph (/viewer/), on one port.

The Space runs on ZeroGPU hardware but never asks for a GPU: the environment is plain Python and the grader re-simulates
a plan in milliseconds. ZeroGPU wants one decorated function, so `_noop` exists and is never called.

    python app.py            # http://127.0.0.1:7860/web/ (the Space's page), /explorer/, /twin/, /viewer/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "openenv")]
os.environ.setdefault("RAIL_TASKS_DIR", f"{ROOT / 'tasks' / 'rail-v1-eval'}:{ROOT / 'tasks' / 'rail-v1-train'}")
os.environ.setdefault("RAIL_RUNS_DIR", str(ROOT / "runs"))
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
# Spaces turn on Gradio's server-side rendering, whose Node server takes port 7860 before uvicorn can; the web UI
# renders in the browser anyway.
os.environ["GRADIO_SSR_MODE"] = "false"

# The web UI shows a README: this card, without its front matter.
card = (ROOT / "README.md").read_text(encoding="utf-8")
if card.startswith("---"):
    card = card.split("---", 2)[2].lstrip()
readme = Path(os.environ.get("TMPDIR", "/tmp")) / "railsim_readme.md"
readme.write_text(card, encoding="utf-8")
os.environ.setdefault("ENV_README_PATH", str(readme))

try:
    import spaces
except ImportError:  # running elsewhere
    spaces = None

if spaces is not None:
    @spaces.GPU(duration=1)
    def _noop():  # ZeroGPU requires one decorated function; nothing calls it
        pass

import uvicorn  # noqa: E402

from rail_openenv.server import create_server  # noqa: E402

app = create_server()

if __name__ == "__main__":
    if spaces is not None:
        from spaces import zero

        zero.startup()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), ws_ping_interval=60,
                ws_ping_timeout=600)
