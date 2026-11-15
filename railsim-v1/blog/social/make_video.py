"""A 30-second video of the Inkling 2 Small experiments, for social posts, cut from the article's media.

    cd railsim-v1
    envs/rail_ops/.venv/bin/python blog/social/make_video.py        # writes blog/social/inkling-railsim-30s.mp4

1920x1080, 30 fps, H.264, no audio. Every number on screen is from the article (blog/site/content) and the committed
eval runs. The 3D clips carry their own map and data credits; the cards that show the map carry them too.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "site" / "public" / "media"
OUT = HERE / "inkling-railsim-30s.mp4"
W, H, FPS = 1920, 1080, 30
BG = (14, 17, 22)
INK = (240, 242, 245)
MUTED = (160, 168, 180)
ORANGE = (232, 129, 74)
BLUE = (77, 184, 230)
GREY = (120, 128, 140)
CREDIT = "Map © OpenStreetMap contributors (ODbL) · data Rijden de Treinen (CC BY 4.0) · RailSimEnv"
FONTS = "/usr/share/fonts/opentype/inter/"


def font(weight: str, size: int):
    try:
        return ImageFont.truetype(f"{FONTS}Inter-{weight}.otf", size)
    except OSError:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def header(d: ImageDraw.ImageDraw, kicker: str, title: str, sub: str = "", y: int = 56) -> None:
    d.text((80, y), kicker, font=font("Bold", 28), fill=ORANGE)
    d.text((80, y + 44), title, font=font("Bold", 60), fill=INK)
    if sub:
        d.text((80, y + 124), sub, font=font("Regular", 32), fill=MUTED)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


ENC = ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS)]


def fades(dur: float) -> str:
    return f"fade=t=in:st=0:d=0.35,fade=t=out:st={dur - 0.35:.2f}:d=0.35"


def clip(tmp: Path, name: str, bg: Image.Image, src: str, start: float, length: float, speed: float,
         box: tuple[int, int, int], dim: float = 0.0) -> Path:
    """A segment: a still background with a video clip scaled to box = (x, y, width) on top."""
    bgp = tmp / f"{name}-bg.png"
    bg.save(bgp)
    x, y, w = box
    dur = length / speed
    v = f"[1:v]trim=start={start}:duration={length},setpts=(PTS-STARTPTS)/{speed},fps={FPS},scale={w}:-2"
    if dim:
        v += f",eq=brightness={-dim}"
    out = tmp / f"{name}.mp4"
    ffmpeg("-loop", "1", "-t", f"{dur:.3f}", "-i", str(bgp), "-i", str(MEDIA / src),
           "-filter_complex", f"{v}[v];[0:v][v]overlay={x}:{y}:shortest=1,{fades(dur)}[o]",
           "-map", "[o]", "-t", f"{dur:.3f}", *ENC, str(out))
    return out


def still(tmp: Path, name: str, img: Image.Image, dur: float) -> Path:
    p = tmp / f"{name}.png"
    img.save(p)
    out = tmp / f"{name}.mp4"
    ffmpeg("-loop", "1", "-t", f"{dur:.3f}", "-i", str(p), "-vf", fades(dur), *ENC, str(out))
    return out


def poster_bg(src: str, darken: float = 0.72) -> Image.Image:
    im = Image.open(MEDIA / src).convert("RGB")
    w, h = im.size                                          # crop off the clip's label bar and credit line
    im = im.crop((int(w * 0.1), int(h * 0.12), int(w * 0.9), int(h * 0.88))).resize((W, H))
    im = im.filter(ImageFilter.GaussianBlur(3))
    return Image.blend(im, Image.new("RGB", (W, H), BG), darken)


def title_card() -> Image.Image:
    im = poster_bg("hero.webp")
    d = ImageDraw.Draw(im)
    d.text((120, 300), "RAILSIMENV  ×  TINKER", font=font("Bold", 34), fill=ORANGE)
    d.multiline_text((120, 360), "Post-training Inkling 2 Small\nto run a railway", font=font("Black", 104), fill=INK,
                     spacing=10)
    d.text((120, 640), "Reinforcement learning on real disruptions of a Dutch single-track line",
           font=font("Regular", 40), fill=MUTED)
    d.text((120, H - 70), CREDIT, font=font("Regular", 22), fill=MUTED)
    return im


def chart_frames(tmp: Path, dur: float) -> Path:
    rows = [("64K context", 0.119, BLUE), ("64K context + RL", 0.186, ORANGE),
            ("262K context", 0.259, BLUE), ("262K context + RL", 0.316, ORANGE)]
    greedy = 0.180
    x0, x1, top, rh = 560, 1760, 330, 150
    sx = lambda v: x0 + (x1 - x0) * v / 0.4                          # noqa: E731
    n = int(dur * FPS)
    folder = tmp / "chart"
    folder.mkdir()
    for f in range(n):
        t = f / FPS
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        header(d, "RESULTS  ·  45 HELD-OUT TASKS", "More room to think did more than training",
               "Mean reward; 1.0 = the solver's proven optimum on every task")
        for v in (0.0, 0.1, 0.2, 0.3, 0.4):
            d.line([(sx(v), top - 20), (sx(v), top + rh * 4 - 30)], fill=(38, 43, 52), width=2)
            d.text((sx(v), top + rh * 4 - 14), f"{v:.1f}", font=font("Regular", 24), fill=GREY, anchor="mt")
        gx = sx(greedy)
        for yy in range(top - 20, top + rh * 4 - 30, 18):
            d.line([(gx, yy), (gx, yy + 9)], fill=GREY, width=3)
        d.text((gx + 10, top - 46), f"rule-based dispatcher {greedy:.3f}", font=font("Medium", 24), fill=GREY)
        for i, (label, val, col) in enumerate(rows):
            p = min(1.0, max(0.0, (t - 0.4 - i * 0.7) / 0.6))
            p = 1 - (1 - p) ** 3
            y = top + i * rh
            d.text((x0 - 30, y + 38), label, font=font("SemiBold", 36), fill=INK if p > 0 else GREY, anchor="rm")
            if p > 0:
                d.rounded_rectangle([x0, y + 10, sx(val * p), y + 66], radius=8, fill=col)
                d.text((sx(val * p) + 16, y + 38), f"{val * p:.3f}", font=font("Bold", 38), fill=INK, anchor="lm")
        if t > 3.6:
            a = min(1.0, (t - 3.6) / 0.5)
            c = tuple(int(BG[k] + (INK[k] - BG[k]) * a) for k in range(3))
            d.text((80, H - 110), "2.7× the starting score; the longer context alone took it from 0.119 to 0.259.",
                   font=font("Regular", 32), fill=c)
        im.save(folder / f"{f:04d}.png")
    out = tmp / "chart.mp4"
    ffmpeg("-framerate", str(FPS), "-i", str(folder / "%04d.png"), "-vf", fades(dur), *ENC, str(out))
    return out


def caveat_bg() -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((80, 56), "HONEST NUMBERS", font=font("Bold", 28), fill=ORANGE)
    d.text((80, 100), "+0.056 per task after RL: promising, not settled", font=font("Bold", 60), fill=INK)
    d.text((80, 180), "Better on 22 tasks, worse on 13, the same on 10; 95% interval −0.008 to +0.126.",
           font=font("Regular", 32), fill=MUTED)
    d.text((80, 226), "And on a short closure (31 May) it now waits it out instead of turning trains back:",
           font=font("Regular", 32), fill=MUTED)
    return im


def end_card() -> Image.Image:
    im = poster_bg("litter-duel.webp", 0.8)
    d = ImageDraw.Draw(im)
    d.text((120, 260), "TRY IT", font=font("Bold", 34), fill=ORANGE)
    d.text((120, 312), "RailSimEnv: dispatch real Dutch trains", font=font("Black", 80), fill=INK)
    d.text((120, 440), "An OpenEnv environment rebuilt from the public record of every train on the line,",
           font=font("Regular", 38), fill=MUTED)
    d.text((120, 492), "every plan graded against a proven optimum.", font=font("Regular", 38), fill=MUTED)
    d.text((120, 600), "huggingface.co/spaces/bpHigh/RailSimEnv", font=font("SemiBold", 44), fill=INK)
    d.text((120, 664), "Write-up: huggingface.co/spaces/bpHigh/RailSimEnv-Inkling", font=font("SemiBold", 44),
           fill=INK)
    d.text((120, 760), "Inkling 2 Small (Thinking Machines) fine-tuned with LoRA through the Tinker API",
           font=font("Regular", 32), fill=MUTED)
    d.text((120, H - 70), CREDIT, font=font("Regular", 22), fill=MUTED)
    return im


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        parts = [still(tmp, "title", title_card(), 3.0)]

        bg = Image.new("RGB", (W, H), BG)
        header(ImageDraw.Draw(bg), "13 MARCH 2025  ·  THE JOB", "A train breaks down on single track",
               "Leeuwarden – Hurdegaryp closed for an hour: re-plan 26 trains, graded against a proven optimum")
        parts.append(clip(tmp, "hero", bg, "hero.mp4", 0.0, 6.3, 1.4, (315, 310, 1290)))

        parts.append(chart_frames(tmp, 6.0))

        bg = Image.new("RGB", (W, H), BG)
        header(ImageDraw.Draw(bg), "SAME TASK  ·  BEFORE vs AFTER RL", "It learned to turn trains back",
               "Before 0.131: trains queued, head-on conflicts later.  After 0.938: 8 trains turned back at Hurdegaryp")
        parts.append(clip(tmp, "duel", bg, "duel.mp4", 0.0, 5.0, 1.0, (60, 330, 1800)))

        bg = Image.new("RGB", (W, H), BG)
        header(ImageDraw.Draw(bg), "16 FEBRUARY 2025  ·  LITTER ON THE TRACK", "Half the line closed in the evening rush",
               "Before 0.200: 5 trains cancelled.  After 1.000: 1 cancelled, 6 turned back, exactly the proven optimum")
        parts.append(clip(tmp, "litter", bg, "litter-duel.mp4", 1.5, 8.0, 1.6, (60, 330, 1800)))

        parts.append(clip(tmp, "caveat", caveat_bg(), "may-duel.mp4", 4.0, 4.0, 1.0, (210, 320, 1500)))

        parts.append(still(tmp, "end", end_card(), 3.5))

        lst = tmp / "parts.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts))
        ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(OUT))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(OUT)], capture_output=True, text=True).stdout)
    print(f"{OUT}: {dur:.1f} s, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
