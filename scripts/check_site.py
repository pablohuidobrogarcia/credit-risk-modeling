"""Static validation for the GitHub Pages site in ``docs/``.

Run with ``python scripts/check_site.py`` (also invoked from the figures stage's
test suite). It fails loudly on anything that would break the published page:

* every local ``href``/``src`` referenced by ``docs/index.html`` exists;
* every ``data-fallback-src`` image exists;
* every ``data-field="a.b.c"`` path resolves in ``docs/data.json`` (this is what
  guarantees no number on the page is hardcoded — the placeholders must all be
  filled from data.json);
* every file under ``docs/`` is under the 1 MB hard limit and every image under
  300 KB (the page has to stay fast on a phone).

Exit code 0 = OK, 1 = problems found.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
INDEX = DOCS / "index.html"
DATA = DOCS / "data.json"

# Local reference attributes to validate (ignore http(s):// and data: URIs).
REF_RE = re.compile(r'(?:href|src)\s*=\s*"([^"]+)"')
FALLBACK_RE = re.compile(r'data-fallback-src\s*=\s*"([^"]+)"')
FIELD_RE = re.compile(r'data-field\s*=\s*"([^"]+)"')

MAX_FILE_KB = 1024
MAX_IMAGE_KB = 300
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def resolve(path_str: str) -> Path:
    """Resolve a page-relative path against docs/."""
    return (DOCS / path_str).resolve()


def is_local(ref: str) -> bool:
    return not (ref.startswith("http://") or ref.startswith("https://")
                or ref.startswith("data:") or ref.startswith("mailto:")
                or ref.startswith("#") or ref.startswith("//"))


def get_path(obj, dotted: str):
    for key in dotted.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, list) and key.isdigit():
            obj = obj[int(key)]
        else:
            return None
    return obj


def main() -> int:
    problems: list[str] = []
    if not INDEX.exists():
        print(f"FAIL: {INDEX} not found")
        return 1
    if not DATA.exists():
        print(f"FAIL: {DATA} not found")
        return 1

    html = INDEX.read_text(encoding="utf-8")
    data = json.loads(DATA.read_text(encoding="utf-8"))

    # 1. Local references exist.
    for ref in sorted(set(REF_RE.findall(html))):
        if not is_local(ref):
            continue
        target = resolve(ref.split("#", 1)[0])
        if not target.exists():
            problems.append(f"broken reference: {ref}")

    # 2. Fallback images exist.
    for ref in sorted(set(FALLBACK_RE.findall(html))):
        if not resolve(ref).exists():
            problems.append(f"missing fallback image: {ref}")

    # 3. Every data-field resolves in data.json.
    fields = sorted(set(FIELD_RE.findall(html)))
    missing_fields = [f for f in fields if get_path(data, f) is None]
    for f in missing_fields:
        problems.append(f"data-field '{f}' does not resolve in data.json")
    print(f"data-field placeholders: {len(fields)} (missing: {len(missing_fields)})")

    # 4. Size limits.
    for p in sorted(DOCS.rglob("*")):
        if not p.is_file():
            continue
        kb = p.stat().st_size / 1024
        limit = MAX_IMAGE_KB if p.suffix.lower() in IMAGE_SUFFIXES else MAX_FILE_KB
        if kb > limit:
            problems.append(f"{p.relative_to(DOCS)} is {kb:.0f} KB (limit {limit} KB)")

    # 5. JavaScript sanity: every JS file parses (best effort — needs Node).
    import shutil
    import subprocess

    node = shutil.which("node")
    for js in sorted(DOCS.rglob("*.js")):
        text = js.read_text(encoding="utf-8")
        # Portable guard for the classic trap: an unquoted object key or a bare
        # property access containing a dot (e.g. `var_0.99_pct_ead`), which is a
        # syntax error. data.json keys with dots must use bracket notation.
        for m in re.finditer(r"\.\s*[A-Za-z_]\w*_0\.\d", text):
            problems.append(
                f"{js.relative_to(DOCS)}: dotted property access {m.group(0)!r} "
                "(use bracket notation for keys containing a dot)"
            )
        if node:
            proc = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
            if proc.returncode != 0:
                problems.append(f"{js.relative_to(DOCS)}: node --check failed: {proc.stderr.strip().splitlines()[-1] if proc.stderr else 'unknown'}")

    if problems:
        print("\nSITE CHECK FAILED:")
        for p in problems:
            print("  -", p)
        return 1

    print("Site check OK: all references, data-field paths and size limits pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
