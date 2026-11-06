"""Regenerate docs/assets/dashboard.gif from the current local site.

The repository had no GIF generator, so this is the documented approach:
1. start a local static server for ``docs/``;
2. run ``scripts/capture_dashboard_gif.js`` with Playwright to capture the
   Total folder sheet, the State map and the Risk Lab top;
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

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "artifacts" / "gif-frames"
TARGET = DOCS / "assets" / "dashboard.gif"


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
        env["DASHBOARD_URL"] = f"http://127.0.0.1:{port}/"
        env["DASHBOARD_FRAMES"] = str(OUT)
        subprocess.run(["node", str(ROOT / "scripts" / "capture_dashboard_gif.js")], cwd=ROOT, env=env, check=True)
    finally:
        server.terminate()
        server.wait(timeout=10)

    frames = []
    for path in sorted(OUT.glob("*.png")):
        image = Image.open(path).convert("RGB")
        image = image.resize((960, 600), Image.Resampling.LANCZOS)
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    if not frames:
        raise SystemExit("no GIF frames were captured")
    frames[0].save(
        TARGET,
        save_all=True,
        append_images=frames[1:],
        duration=[1800, 1800, 2200],
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {TARGET} ({TARGET.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
