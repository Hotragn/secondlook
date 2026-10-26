"""Post media in docs/media, made only from the committed results.

- chart-1-summary.png: "same as last year" against the lowest premium, pooled over the three
  2027 menus in eval/results/step4_decision.json. Drawn as HTML in the brand's fonts and
  colours and captured 1600 px wide with headless Chromium. The rows carry the labels;
  colour is emphasis only (ledger green for the lowest premium, a neutral gray for same as
  last year). The pair passes the palette validator's CVD, normal-vision and contrast
  checks; the gray fails its chroma floor by design.
- chart-2-plan-choice.png, chart-3-learning-curve.png, chart-4-plan-reader.png: the
  evaluation charts in eval/results, resized to 1600 px wide.
Also writes docs/media/charts.md with a caption and source for each image.

Run: uv run python -I scripts/media.py   (or: make media)
"""

import base64
import json
import math
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MEDIA = DOCS / "media"
RESULTS = ROOT / "eval" / "results"
NAME = "Lekka"
WIDTH = 1600
FONTS = [("Newsreader", 600, "Newsreader-600"), ("IBM Plex Sans", 400, "IBMPlexSans-400"),
         ("IBM Plex Sans", 500, "IBMPlexSans-500"), ("IBM Plex Mono", 400, "IBMPlexMono-400"),
         ("IBM Plex Mono", 500, "IBMPlexMono-500")]
ROWS = [("same_as_last_year", "Same as last year", "The plan that would have been cheapest for their 2023 care", "context"),
        ("lowest_premium", "Lowest premium", "The plan with the lowest employee premium", "accent")]
CHARTS = [("chart-2-plan-choice", "step4_decision"), ("chart-3-learning-curve", "step2b_learning_curve"),
          ("chart-4-plan-reader", "step3_reader")]


def pct(x):
    return f"{x:.0%}"


def pct1(x):
    return f"{x:.1%}"


def usd(x):
    return f"${x:,.0f}"


def summary_html(r: dict) -> str:
    pooled = r["pooled"]
    faces = "\n".join(
        f"@font-face {{ font-family: '{family}'; font-weight: {weight}; src: url(data:font/woff2;base64,"
        f"{base64.b64encode((DOCS / 'fonts' / f'{stem}.woff2').read_bytes()).decode()}) format('woff2'); }}"
        for family, weight, stem in FONTS)
    mark = (DOCS / "assets" / "mark.svg").read_text()
    top = 50 * math.ceil(max(pooled[key]["mean_regret"]["ci95"][1] for key, *_ in ROWS) / 50)

    def cell(m: dict, scale: float, fmt) -> str:
        v, (lo, hi) = m["value"], m["ci95"]
        return (f'<div class="cell"><div class="num">{fmt(v)}</div>'
                f'<div class="track"><i style="width:{v / scale:.2%}"></i>'
                f'<s style="left:{lo / scale:.2%};width:{(hi - lo) / scale:.2%}"></s></div>'
                f'<div class="ci">95% interval {fmt(lo)} to {fmt(hi)}</div></div>')

    rows = "".join(f'<div class="row {role}"><div class="label"><b>{title}</b><span>{note}</span></div>'
                   f'{cell(pooled[key]["hit_rate"], 1.0, pct)}{cell(pooled[key]["mean_regret"], top, usd)}</div>'
                   for key, title, note, role in ROWS)
    tab = pooled["tabpfn_expected"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{faces}
:root {{ --paper: #F5F0E6; --ink: #1E2521; --ledger: #2E6B57; --mute: #6F7671; --line: #E3DCCD; --context: #8A877E; }}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: var(--paper); color: var(--ink); font: 400 14px/1.45 'IBM Plex Sans', sans-serif; width: 800px; }}
.frame {{ padding: 30px 36px 26px; background: var(--paper); }}
.brand {{ display: flex; align-items: center; gap: 8px; font: 600 19px 'Newsreader', serif; letter-spacing: -0.02em; }}
.brand svg {{ width: 26px; height: 26px; }}
h1 {{ font: 600 31px/1.15 'Newsreader', serif; letter-spacing: -0.015em; margin-top: 18px; }}
.sub {{ color: var(--mute); margin-top: 6px; }}
.grid {{ margin-top: 18px; }}
.head, .row {{ display: grid; grid-template-columns: 200px 1fr 1fr; column-gap: 32px; }}
.head {{ font-size: 12.5px; color: var(--mute); padding-bottom: 8px; }}
.row {{ border-top: 1px solid var(--line); padding: 14px 0; }}
.label b {{ display: block; font-weight: 500; font-size: 16px; }}
.label span {{ display: block; color: var(--mute); font-size: 12.5px; line-height: 1.35; margin-top: 3px; }}
.num {{ font: 500 26px/1 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}
.track {{ position: relative; height: 10px; margin-top: 10px; background: color-mix(in srgb, var(--line) 70%, transparent); border-radius: 0 4px 4px 0; }}
.track i {{ position: absolute; left: 0; top: 0; height: 10px; border-radius: 0 4px 4px 0; }}
.context .track i {{ background: var(--context); }}
.accent .track i {{ background: var(--ledger); }}
.track s {{ position: absolute; top: 4px; height: 2px; background: var(--ink); }}
.track s::before, .track s::after {{ content: ""; position: absolute; top: -3px; width: 2px; height: 8px; background: var(--ink); }}
.track s::before {{ left: 0; }}
.track s::after {{ right: 0; }}
.ci {{ font: 400 11.5px 'IBM Plex Mono', monospace; color: var(--mute); margin-top: 7px; }}
.foot {{ margin-top: 4px; padding-top: 12px; border-top: 1px solid var(--line); font-size: 11.5px; color: var(--mute); }}
.foot code {{ font-family: 'IBM Plex Mono', monospace; }}
</style></head><body><div class="frame">
<div class="brand">{mark}<span>{NAME}</span></div>
<h1>&ldquo;Same as last year&rdquo;, or the lowest premium?</h1>
<p class="sub">{r['people']:,} held-out adults from the US Medical Expenditure Panel Survey, priced on three real 2027 employer menus.</p>
<div class="grid"><div class="head"><span></span><span>Got the truly cheapest plan</span><span>Average overpayment</span></div>{rows}</div>
<p class="foot">Overpayment is what a plan cost the person (employee premium, plus their real 2024 care under its rules, less any employer HSA deposit) above the cheapest plan on the same menu. Black lines are 95% intervals. TabPFN's expected-cost pick: {pct(tab['hit_rate']['value'])} and {usd(tab['mean_regret']['value'])}. Source: <code>eval/results/step4_decision.json</code></p>
</div></body></html>"""


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    r4 = json.loads((RESULTS / "step4_decision.json").read_text())
    r2b = json.loads((RESULTS / "step2b_learning_curve.json").read_text())
    r3 = json.loads((RESULTS / "step3_reader.json").read_text())

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 800, "height": 450}, device_scale_factor=WIDTH / 800)
        page.set_content(summary_html(r4))
        page.evaluate("document.fonts.ready")
        page.locator(".frame").screenshot(path=str(MEDIA / "chart-1-summary.png"))
        browser.close()
    for name, source in CHARTS:
        im = Image.open(RESULTS / f"{source}.png").convert("RGB")
        im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS).save(MEDIA / f"{name}.png", optimize=True)

    p = r4["pooled"]
    saly, low = p["same_as_last_year"], p["lowest_premium"]
    size = r2b["reaches_full_data_accuracy"]["TabPFN v2"]["pinball_q50"]["smallest_size_within_5pct"]
    e4b = r3["results"]["gemma4:e4b-it-qat"]["gold_fields"]
    captions = [
        ("chart-1-summary.png", f"\"Same as last year\" gave {r4['people']:,} held-out adults the truly cheapest 2027 plan "
         f"{pct(saly['hit_rate']['value'])} of the time, against {pct(low['hit_rate']['value'])} for the lowest premium. "
         f"Average overpayment: {usd(saly['mean_regret']['value'])} against {usd(low['mean_regret']['value'])}.",
         "eval/results/step4_decision.json"),
        ("chart-2-plan-choice.png", "Every plan-choosing strategy on the three real 2027 menus: share given the truly "
         "cheapest plan, mean and 90th-percentile overpayment, with 95% intervals.", "eval/results/step4_decision.json"),
        ("chart-3-learning-curve.png", "Pinball loss and 10-90% band coverage against training-set size for three spend "
         f"models. TabPFN v2 is within 5% of its full-data median loss at {size} people.",
         "eval/results/step2b_learning_curve.json"),
        ("chart-4-plan-reader.png", f"Three Gemma 4 builds on {e4b['n']} key values from {len(r3['documents'])} held-out "
         f"plan documents. gemma4:e4b-it-qat read {pct1(e4b['accuracy'])} right, {pct1(e4b['wrong_rate'])} wrong, and "
         f"flagged {pct1(e4b['needs_check_rate'])} for checking.", "eval/results/step3_reader.json"),
    ]
    (MEDIA / "charts.md").write_text(
        f"# {NAME} charts\n\nMade by `make media` (scripts/media.py) from eval/results. Each image is {WIDTH} px wide.\n\n"
        + "\n".join(f"- `{n}`: {c} Source: `{s}`." for n, c, s in captions) + "\n")
    for n, *_ in captions:
        w, h = Image.open(MEDIA / n).size
        print(f"{n}: {w} x {h}")


if __name__ == "__main__":
    main()
