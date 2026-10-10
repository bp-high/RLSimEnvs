"""The "Explore" tab of OpenEnv's web UI (/web): the eval explorer, the 3D replay and the train graph, each the
server's own page in a frame, one tab each. OpenEnv's Playground tab (reset, step and the MCP tools) stays next to it.

A frame is only created when its tab is first opened, so the 3D replay does not start WebGL in a hidden tab.
"""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import gradio as gr

WEB = Path(__file__).resolve().parent / "web"


def version() -> str:
    """A tag that changes whenever a page or script does, so a frame never shows a copy cached before a deploy."""
    h = hashlib.sha1()
    for p in sorted(WEB.rglob("*")):
        if p.is_file() and p.suffix in (".html", ".js", ".css"):
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


V = version()
PAGES = [("Eval explorer", f"/explorer/?embed=1&v={V}"), ("3D replay", f"/twin/?v={V}"),
         ("Train graph", f"/viewer/?embed=1&v={V}")]
# OpenEnv's page caps the container width; the pages want the full window.
STYLE = ("<style>.gradio-container .main.fillable{max-width:none!important;padding-left:12px!important;"
         "padding-right:12px!important}.gradio-container .html-container:has(iframe){padding:0!important}</style>")


def frame(src: str, title: str) -> str:
    return (f'{STYLE}<iframe src="{html.escape(src)}" title="RailSimEnv {html.escape(title)}" allow="fullscreen" '
            'style="width:100%;height:calc(100vh - 190px);min-height:720px;border:0;display:block"></iframe>')


def build_ui(*_args, **_kwargs) -> gr.Blocks:
    """OpenEnv's `gradio_builder` hook; its arguments (web manager, action fields, metadata, ...) are not needed."""
    with gr.Blocks(title="RailSimEnv v1") as ui:
        with gr.Tabs():
            for i, (name, src) in enumerate(PAGES):
                with gr.Tab(name) as tab:
                    out = gr.HTML(frame(src, name) if i == 0 else "")
                    loaded = gr.State(i == 0)

                    def show(done: bool, src: str = src, name: str = name):
                        return (gr.update(), True) if done else (frame(src, name), True)

                    tab.select(show, inputs=loaded, outputs=[out, loaded])
    return ui
