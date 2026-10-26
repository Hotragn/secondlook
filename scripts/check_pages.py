"""Check that docs/ works from a subpath, the way GitHub Pages serves it.

Copies docs/ into a temporary folder as secondlook/, serves the parent folder on a free local
port and drives the real UI in headless Chromium: load /secondlook/, pick a menu and a person,
open a quote. Fails on a missing .nojekyll, an absolute local URL ("/..." in src, href,
url() or fetch) in the site's HTML, CSS or JS, any HTTP error, any console error, any
request that leaves the local server, or a self-hosted font that did not load.

Run: uv run python -I scripts/check_pages.py   (or: make pages-check)
"""

import functools
import http.server
import re
import shutil
import socketserver
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SUBPATH = "secondlook"
ABSOLUTE = re.compile(r"""(?:src|href)\s*=\s*["']/(?!/)|url\(\s*["']?/(?!/)|fetch\(\s*["'`]/(?!/)""")
RELATIVE = re.compile(r"""(?:src|href)\s*=\s*["'](?![a-z]+:|#|/)([^"']+)["']""")
FONTS = {"Newsreader", "IBM Plex Sans", "IBM Plex Mono"}


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main() -> None:
    problems = []
    if not (DOCS / ".nojekyll").is_file():
        problems.append("docs/.nojekyll is missing")
    for p in sorted(DOCS.rglob("*")):
        if p.suffix in (".html", ".css", ".js") and ABSOLUTE.search(p.read_text()):
            problems.append(f"absolute local URL in {p.relative_to(ROOT)}")
        if p.suffix == ".html":
            # Every relative link must exist, including ones a headless browser never fetches (the favicon).
            for ref in RELATIVE.findall(p.read_text()):
                if not (p.parent / ref.split("?")[0]).exists():
                    problems.append(f"{p.relative_to(ROOT)} links to missing {ref}")

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(DOCS, Path(tmp) / SUBPATH)
        httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=tmp))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        origin = f"http://127.0.0.1:{httpd.server_address[1]}"
        responses = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1180, "height": 860})
            page.on("console", lambda m: problems.append(f"console error: {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
            page.on("request", lambda r: None if r.url.startswith((origin + "/", "data:"))
                    else problems.append(f"request outside the local server: {r.url}"))
            page.on("response", lambda r: responses.append((r.status, r.url)))
            page.goto(f"{origin}/{SUBPATH}/")
            page.wait_for_selector("#menus .row-choice")
            page.click("label.row-choice:has(input[value='wa-pebb'])")
            page.click("#tab-p")
            page.locator("#panel-p label.row-choice").first.click()
            page.wait_for_selector("#result-card .verdict")
            plan = page.locator("#result-card .plan").first
            plan.locator("details.facts summary").click()
            plan.locator("button.quote-num").first.click()
            page.wait_for_selector("dialog#quote[open]")
            loaded = set(page.evaluate("""async () => { await document.fonts.ready;
                return [...document.fonts].filter(f => f.status === 'loaded').map(f => f.family.replace(/["']/g, '')); }"""))
            browser.close()
        httpd.shutdown()

    problems += [f"HTTP {s}: {u}" for s, u in responses if s >= 400]
    problems += [f"font not loaded: {f}" for f in sorted(FONTS - loaded)]
    paths = sorted({u.removeprefix(origin) for _, u in responses})
    print(f"served docs/ at {origin}/{SUBPATH}/: {len(responses)} responses, all from the local server" if not problems
          else f"served docs/ at {origin}/{SUBPATH}/")
    for p in paths:
        print("  " + p)
    if problems:
        raise SystemExit("pages check failed:\n  " + "\n  ".join(problems))
    print(f"ok: no HTTP errors, no console errors, no outside requests; fonts loaded: {', '.join(sorted(loaded & FONTS))}")


if __name__ == "__main__":
    main()
