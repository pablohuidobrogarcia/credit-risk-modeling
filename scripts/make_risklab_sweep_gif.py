"""Regenerate the optional README Risk Lab rho-sweep GIF.

The output lives at the repository root under ``assets/readme/``, so it is
not part of the deployed ``docs/`` tree.

Usage:
    npm ci
    npx playwright install chromium
    python scripts/make_risklab_sweep_gif.py
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "artifacts" / "risklab-sweep-frames"
TARGET = ROOT / "assets" / "readme" / "risklab-sweep.gif"
DURATIONS = [350] * 12
MAX_BYTES = 3 * 1024 * 1024


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
        env["RISKLAB_URL"] = f"http://127.0.0.1:{port}/"
        env["RISKLAB_FRAMES"] = str(OUT)
        subprocess.run(["node", str(ROOT / "scripts" / "capture_risklab_sweep.js")], cwd=ROOT, env=env, check=True)
    finally:
        server.terminate()
        server.wait(timeout=10)

    paths = sorted(OUT.glob("*.png"))
    if len(paths) != 12:
        raise SystemExit(f"expected 12 sweep frames, found {len(paths)}")

    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        width = 800
        height = round(image.height * width / image.width)
        frames.append(image.resize((width, height), Image.Resampling.LANCZOS))

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    palette_frames = [
        image.convert("P", palette=Image.Palette.ADAPTIVE, colors=160)
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
    size = TARGET.stat().st_size
    if size > MAX_BYTES:
        raise SystemExit(f"risklab-sweep.gif is {size / 1024:.0f} KB, above the 3 MB limit")
    print(f"wrote {TARGET} ({size / 1024:.0f} KB, {len(frames)} frames, {sum(DURATIONS)} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
