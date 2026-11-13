"""Regenerate docs/assets/dashboard.gif from the current local site.

1. start a local static server for ``docs/``;
2. run ``scripts/capture_dashboard_gif.js`` with Playwright to capture the
   five README views (folder, state map, Risk Lab top, tail-by-grade and
   Evidence);
3. assemble the frames with Pillow into an optimised looping GIF.

Usage:
    npm ci
    npx playwright install chromium
    python scripts/make_dashboard_gif.py
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageStat

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "artifacts" / "gif-frames"
TARGET = DOCS / "assets" / "dashboard.gif"
DURATIONS = [1800, 1800, 2200, 2800, 2200]
MAX_BYTES = 2048 * 1024
SOFT_BYTES = 1536 * 1024


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _save_gif(frames: list[Image.Image], colors: int) -> None:
    palette_frames = [
        image.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)
        for image in frames
    ]
    palette_frames[0].save(
        TARGET,
        save_all=True,
        append_images=palette_frames[1:],
        duration=DURATIONS,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> int:
    if shutil.which("node") is None:
        print("node is required (Playwright is installed with npm ci)", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()

    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", str(DOCS)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(1.0)
        env = os.environ.copy()
        env["DASHBOARD_URL"] = f"http://127.0.0.1:{port}/"
        env["DASHBOARD_FRAMES"] = str(OUT)
        subprocess.run(["node", str(ROOT / "scripts" / "capture_dashboard_gif.js")], cwd=ROOT, env=env, check=True)
    finally:
        server.terminate()
        server.wait(timeout=10)

    paths = sorted(OUT.glob("[0-9][0-9]-*.png"))
    if len(paths) != 5:
        raise SystemExit(f"expected 5 GIF frames, found {len(paths)}: {[p.name for p in paths]}")

    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = image.resize((960, 600), Image.Resampling.LANCZOS)
        frames.append(image)

    chart_check = Image.open(OUT / "checks" / "04-chart-only.png").convert("RGB")
    if max(ImageStat.Stat(chart_check).stddev) < 2:
        raise SystemExit("frame 04 chart is blank; refusing to accept a blank/heatmap fallback")

    _save_gif(frames, 160)
    colors = 160
    if TARGET.stat().st_size > SOFT_BYTES:
        _save_gif(frames, 128)
        colors = 128
    size = TARGET.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(f"dashboard.gif is {size / 1024:.0f} KB, above the 2048 KB limit")
    print(f"wrote {TARGET} ({size / 1024:.0f} KB, {len(frames)} frames, {colors} colors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
