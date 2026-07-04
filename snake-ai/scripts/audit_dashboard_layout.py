import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


VIEWPORTS = [
    ("desktop", 1440, 900),
    ("tablet", 920, 900),
    ("mobile", 390, 844),
    ("narrow", 360, 740),
]


def audit_page(page, url, name, width, height, screenshot_dir):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(url, wait_until="networkidle")
    page.screenshot(path=str(screenshot_dir / f"snake-dashboard-{name}.png"), full_page=True)
    return page.evaluate(
        """
        ({ name, width, height }) => {
          const selectors = [
            "body",
            ".app-shell",
            ".stage",
            ".controls",
            ".board-wrap",
            "#board",
            ".hud",
            ".chart-section",
            ".actions",
            ".grid-two",
            "button",
            "input",
            "select",
            "label"
          ];
          const overflow = [];
          const clipped = [];
          for (const selector of selectors) {
            for (const el of document.querySelectorAll(selector)) {
              const rect = el.getBoundingClientRect();
              if (rect.width > width + 0.5 || rect.right > width + 0.5 || rect.left < -0.5) {
                overflow.push({
                  selector,
                  text: (el.innerText || el.value || el.id || "").slice(0, 48),
                  left: Math.round(rect.left),
                  right: Math.round(rect.right),
                  width: Math.round(rect.width)
                });
              }
              if (el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1) {
                clipped.push({
                  selector,
                  text: (el.innerText || el.value || el.id || "").slice(0, 48),
                  scrollWidth: el.scrollWidth,
                  clientWidth: el.clientWidth,
                  scrollHeight: el.scrollHeight,
                  clientHeight: el.clientHeight
                });
              }
            }
          }
          const controls = document.querySelector(".controls").getBoundingClientRect();
          const board = document.querySelector("#board").getBoundingClientRect();
          const chart = document.querySelector(".chart-section").getBoundingClientRect();
          return {
            name,
            viewport: { width, height },
            documentWidth: document.documentElement.scrollWidth,
            bodyWidth: document.body.scrollWidth,
            hasHorizontalOverflow: document.documentElement.scrollWidth > width + 1,
            controls: {
              top: Math.round(controls.top),
              left: Math.round(controls.left),
              width: Math.round(controls.width),
              height: Math.round(controls.height)
            },
            board: {
              top: Math.round(board.top),
              left: Math.round(board.left),
              width: Math.round(board.width),
              height: Math.round(board.height)
            },
            chart: {
              top: Math.round(chart.top),
              left: Math.round(chart.left),
              width: Math.round(chart.width),
              height: Math.round(chart.height)
            },
            overflow,
            clipped
          };
        }
        """,
        {"name": name, "width": width, "height": height},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7861/")
    parser.add_argument("--screenshot-dir", default="/tmp/snake-dashboard-audit")
    args = parser.parse_args()

    screenshot_dir = Path(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        results = [
            audit_page(page, args.url, name, width, height, screenshot_dir)
            for name, width, height in VIEWPORTS
        ]
        browser.close()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
