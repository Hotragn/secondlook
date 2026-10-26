"""Screenshots and GIFs of the static site, taken with headless Chromium (Playwright).

Serves docs/ on a free local port, drives the real UI, and writes PNGs, two GIFs and
docs/media/captions.md into docs/media/. Console errors are printed and fail the run.

Run: uv run python -I scripts/capture.py   (or: make capture)
"""

import functools
import http.server
import io
import re
import socketserver
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MEDIA = DOCS / "media"
MENU = "wa-pebb"
PEOPLE = [("2810297103", "healthy"), ("2810385102", "prescriptions"), ("2810035101", "high-spender")]
CAPTIONS = []


def serve() -> tuple[socketserver.TCPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(DOCS)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def shot(page, name: str, caption: str, full: bool = False, el=None) -> None:
    path = MEDIA / f"{name}.png"
    (el or page).screenshot(path=str(path), **({"full_page": True} if full and el is None else {}))
    CAPTIONS.append((path.name, caption))


def frame(page, frames: list, el=None) -> None:
    frames.append(Image.open(io.BytesIO((el or page).screenshot())).convert("RGB"))


def gif(frames: list, name: str, caption: str, width: int = 720) -> None:
    sized = [f.resize((width, round(f.height * width / f.width))) for f in frames]
    h = max(f.height for f in sized)
    canvas = [Image.new("RGB", (width, h), (245, 240, 230)) for _ in sized]
    for c, f in zip(canvas, sized):
        c.paste(f, (0, 0))
    canvas[0].save(MEDIA / f"{name}.gif", save_all=True, append_images=canvas[1:], duration=[1100] * (len(canvas) - 1) + [2200],
                   loop=0, optimize=True)
    CAPTIONS.append((f"{name}.gif", caption))


def pick_menu(page) -> None:
    page.click(f"label.row-choice:has(input[value='{MENU}'])")


def pick_person(page, pid: str) -> None:
    page.click("#tab-p")
    page.click(f"label.row-choice:has(input[value='{pid}'])")
    page.wait_for_selector("#result-card .verdict")
    page.wait_for_timeout(250)


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    httpd, port = serve()
    url = f"http://127.0.0.1:{port}/"
    errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1180, "height": 860}, device_scale_factor=2, color_scheme="light")
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_selector("#menus .row-choice")
        shot(page, "01-home", "Home: pick your employer's 2027 plans, then describe yourself.")
        pick_menu(page)
        shot(page, "02-menu", "Washington PEBB's nine 2027 plans, with what you pay each month.", el=page.locator("section.step").first)

        # GIF 1: pick a person, get the result.
        frames = []
        page.click("#tab-p")
        page.locator("#panel-p").scroll_into_view_if_needed()
        frame(page, frames)
        page.click(f"label.row-choice:has(input[value='{PEOPLE[0][0]}'])")
        page.wait_for_selector("#result-card .verdict")
        page.wait_for_timeout(700)
        frame(page, frames)
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(300)
        frame(page, frames)
        gif(frames, "gif-1-person-to-result", "Pick a real person and Second Look says whether the cheapest plan is safe for them.")

        for pid, slug in PEOPLE:
            pick_person(page, pid)
            label = page.locator(f"label.row-choice:has(input[value='{pid}']) .t").inner_text()
            verdict = page.locator("#result-card .verdict").inner_text()
            shot(page, f"03-result-{slug}", f"{label}: \"{verdict}\"", el=page.locator("#result"))

        # GIF 2: tap a number, see its line in the plan document.
        pick_person(page, PEOPLE[0][0])
        frames = []
        first = page.locator("#result-card .plan").first
        first.scroll_into_view_if_needed()
        frame(page, frames)
        first.locator("details.facts summary").click()
        page.wait_for_timeout(250)
        frame(page, frames)
        first.locator("button.quote-num").nth(1).click()
        page.wait_for_selector("dialog#quote[open]")
        page.wait_for_timeout(300)
        frame(page, frames)
        shot(page, "04-quote-highlight", "Every plan number opens the line it came from, highlighted, with the page number.",
             el=page.locator("dialog#quote"))
        gif(frames, "gif-2-number-to-quote", "Tap a plan number to see the source line in the plan PDF.")
        page.keyboard.press("Escape")

        page.click("#sure summary")
        page.wait_for_timeout(250)
        shot(page, "05-how-sure", "How sure is this? Real numbers from the evaluations, and what the estimates leave out.",
             el=page.locator("#sure"))

        # The five-question path.
        page.click("#tab-q")
        for q, i in (("age", 1), ("health", 0), ("conditions", 0), ("prescriptions", 1), ("urgent", 0)):
            page.click(f"label.opt:has(input[name='{q}'][value='{i}'])")
        page.click("#go")
        page.wait_for_selector("#result-card .verdict")
        page.wait_for_timeout(250)
        shot(page, "06-five-answers", "Five answers: 35 to 49, very good health, no conditions, a few prescriptions.",
             el=page.locator("#result"))

        # Phone width and dark mode.
        phone = browser.new_page(viewport={"width": 360, "height": 780}, device_scale_factor=2, color_scheme="light")
        phone.on("pageerror", lambda e: errors.append(str(e)))
        phone.goto(url)
        phone.wait_for_selector("#menus .row-choice")
        pick_menu(phone)
        pick_person(phone, PEOPLE[1][0])
        phone.locator("#result").scroll_into_view_if_needed()
        shot(phone, "07-mobile", "At 360 px: the same result card on a phone.")
        dark = browser.new_page(viewport={"width": 1180, "height": 860}, device_scale_factor=2, color_scheme="dark")
        dark.goto(url)
        dark.wait_for_selector("#menus .row-choice")
        pick_menu(dark)
        pick_person(dark, PEOPLE[2][0])
        shot(dark, "08-dark", "Dark mode follows the system setting.", el=dark.locator("#result"))
        browser.close()
    httpd.shutdown()

    # Keep caption lines for media this script does not make (the launch video) while the file is there.
    path = MEDIA / "captions.md"
    made = {n for n, _ in CAPTIONS}
    kept = [line for line in (path.read_text().splitlines() if path.exists() else [])
            if (m := re.match(r"- `([^`]+)`", line)) and m.group(1) not in made and (MEDIA / m.group(1)).exists()]
    path.write_text("# Second Look screenshots and GIFs\n\nScreenshots and GIFs are made by `make capture` (scripts/capture.py); "
                    "other entries are added by hand.\n\n" + "\n".join([f"- `{n}`: {c}" for n, c in CAPTIONS] + kept) + "\n")
    for n, c in CAPTIONS:
        print(f"{n}: {c}")
    if errors:
        raise SystemExit("console errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
