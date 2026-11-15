"""Join the 3D replay frames (from tools/shot3d.js) into the article's videos, with a caption bar burned in.

    python blog/site/tools/encode.py <shots dir> <name> [--out blog/site/public/media] [--fps 12]

One plan: the frames as they are, a label (plan, reward) at the top left and the clock at the top right. Two plans
(`base` and `rl`): side by side, each labelled. Every video carries the map and data credits. Writes <name>.mp4
(H.264) and <name>.webp (a poster: the middle frame).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
FONT_R = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
CREDIT = "Map © OpenStreetMap contributors (ODbL) · data Rijden de Treinen (CC BY 4.0) · RailSimEnv"


def font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def hhmm(t: float) -> str:
    return f"{int(t // 60) % 24:02d}:{int(t % 60):02d}"


def pill(draw, xy, text, f, fill=(15, 18, 24, 205), ink=(255, 255, 255, 255), pad=(12, 7)):
    x, y = xy
    w = draw.textlength(text, font=f)
    h = f.size
    draw.rounded_rectangle([x, y, x + w + 2 * pad[0], y + h + 2 * pad[1] + 2], radius=8, fill=fill)
    draw.text((x + pad[0], y + pad[1] - 1), text, font=f, fill=ink)
    return w + 2 * pad[0]


def label(plan: dict) -> tuple[str, str]:
    name = plan["label"].split(" · ")[0].replace(" (effort max)", "")
    tag = "after RL" if ", RL" in name else "no training"
    return f"Inkling 2 Small 256K · {tag}", f"reward {plan['reward']:.2f}" + ("" if plan["feasible"] else " · breaks rules")


def frame(img: Image.Image, plan: dict, t: float, scale: float, clock: bool = True) -> Image.Image:
    img = img.convert("RGBA")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    big, small = font(FONT, round(18 * scale)), font(FONT_R, round(14 * scale))
    title, sub = label(plan)
    good = plan["feasible"] and plan["reward"] > 0.5
    w = pill(d, (round(14 * scale), round(14 * scale)), title, big)
    pill(d, (round(14 * scale) + w + 8, round(14 * scale)), sub, big,
         fill=(32, 120, 74, 220) if good else (176, 48, 48, 225))
    if clock:
        ct = hhmm(t)
        cw = d.textlength(ct, font=big)
        pill(d, (img.width - cw - round(38 * scale), round(14 * scale)), ct, big)
    cw = d.textlength(CREDIT, font=small)
    pill(d, (img.width - cw - round(30 * scale), img.height - small.size - round(28 * scale)), CREDIT, small,
         fill=(15, 18, 24, 150), ink=(235, 235, 235, 255), pad=(10, 5))
    return Image.alpha_composite(img, over).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shots", type=Path)
    ap.add_argument("name")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "public" / "media")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    meta = json.loads((args.shots / "meta.json").read_text())
    keys = list(meta["plans"])
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate(meta["frames"]):
            imgs = [Image.open(args.shots / f"{k}-{i:04d}.png") for k in keys]
            scale = imgs[0].width / 1280 if len(keys) == 1 else imgs[0].width / 900
            parts = [frame(im, meta["plans"][k], t, scale, clock=(len(keys) == 1 or j == len(keys) - 1))
                     for j, (k, im) in enumerate(zip(keys, imgs))]
            if len(parts) == 1:
                out = parts[0]
            else:
                gap = 6
                out = Image.new("RGB", (sum(p.width for p in parts) + gap * (len(parts) - 1), parts[0].height), (12, 14, 18))
                x = 0
                for p in parts:
                    out.paste(p, (x, 0))
                    x += p.width + gap
            if out.width % 2 or out.height % 2:
                out = out.crop((0, 0, out.width - out.width % 2, out.height - out.height % 2))
            out.save(Path(tmp) / f"f{i:04d}.png")
            if i == len(meta["frames"]) // 2:
                out.save(args.out / f"{args.name}.webp", "WEBP", quality=80, method=6)
        mp4 = args.out / f"{args.name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(Path(tmp) / "f%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart",
                        str(mp4)], check=True)
    print(mp4, f"{mp4.stat().st_size / 1e6:.1f} MB", len(meta["frames"]), "frames")


if __name__ == "__main__":
    main()
