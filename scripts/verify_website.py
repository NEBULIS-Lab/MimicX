#!/usr/bin/env python3
"""Optional CPU-only Playwright audit of local assets, layout and video controls."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chromium", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/website-qa")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "docs")))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    reports = []
    try:
        with sync_playwright() as driver:
            browser = driver.chromium.launch(executable_path=args.chromium, headless=True,
                                             args=["--disable-gpu", "--disable-dev-shm-usage"])
            try:
                for name, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
                    page = browser.new_page(viewport={"width": width, "height": height})
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.on("response", lambda response: errors.append(f"HTTP {response.status} {response.url}") if response.status >= 400 else None)
                    page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")
                    page.wait_for_function("document.querySelector('#hero-video').readyState >= 2")
                    assert page.locator("#core-results tr").count() == 8
                    assert page.locator("#result-downloads a").count() == 9
                    for selector in ("#method", "#policies", "#results", "#resources"):
                        page.locator(selector).scroll_into_view_if_needed()
                    page.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)")
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal page overflow"
                    for task in ("tennis", "football", "dance", "kungfu"):
                        page.select_option("#task-select", task)
                        page.wait_for_function("['fixed-video','ours-video'].every(id => document.getElementById(id).readyState >= 2)")
                        page.click("#pair-toggle")
                        page.wait_for_function("document.getElementById('fixed-video').currentTime > 0.1 && !document.getElementById('ours-video').paused")
                        page.click("#pair-toggle")
                        page.locator("#pair-seek").evaluate("input => {input.value='500';input.dispatchEvent(new Event('input'));}")
                        page.wait_for_function("Math.abs(document.getElementById('fixed-video').currentTime - document.getElementById('ours-video').currentTime) < 0.2")
                        page.click("#pair-reset")
                    page.select_option("#task-select", "tennis")
                    page.wait_for_function("document.getElementById('fixed-video').readyState >= 2")
                    page.locator("#top").scroll_into_view_if_needed()
                    page.evaluate("document.getElementById('hero-video').play()")
                    page.wait_for_function("document.getElementById('hero-video').readyState >= 2 && !document.getElementById('hero-video').paused")
                    page.wait_for_timeout(500)
                    page.evaluate("document.getElementById('hero-video').pause()")
                    page.screenshot(path=str(args.output / f"{name}.png"), full_page=True)
                    for section in ("top", "method", "policies", "results"):
                        page.locator(f"#{section}").screenshot(path=str(args.output / f"{name}-{section}.png"))
                    assert not errors, errors
                    reports.append({"viewport": name, "width": width, "height": height,
                                    "table_rows": 8, "download_csvs": 9, "playback_pairs": 4,
                                    "errors": errors, "horizontal_overflow": False})
                    page.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
