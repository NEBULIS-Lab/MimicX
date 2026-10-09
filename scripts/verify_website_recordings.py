"""CPU browser checks for the all-visible recordings release."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright
from verify_website import QuietHandler


def verify(site, output, chromium):
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reports = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium, args=["--disable-gpu", "--disable-dev-shm-usage"])
            for width, height in ((1440, 960), (390, 844), (320, 740)):
                page = browser.new_page(viewport=dict(width=width, height=height), is_mobile=width < 700, has_touch=width < 700)
                errors, missing = [], []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("response", lambda r: missing.append(r.url) if r.status >= 400 else None)
                page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
                assert page.locator("html").get_attribute("data-theme") == "dark"
                assert page.locator("[data-policy-task]").count() == 4
                assert page.locator("[data-comparison-task]").count() == 0
                assert page.locator("#baseline-select, [role=tabpanel]").count() == 0
                for task in ("tennis", "football", "dance", "kungfu"):
                    selector = f'[data-policy-task="{task}"]'
                    row = page.locator(selector)
                    row.scroll_into_view_if_needed()
                    page.wait_for_function("s => [...document.querySelectorAll(s+' video')].every(v => !v.paused && v.currentTime > .05 && v.videoWidth > 0)", arg=selector, timeout=30000)
                    frames = row.locator('.recording-frame').evaluate_all("ns=>ns.map(n=>{const b=n.getBoundingClientRect(); return {width:b.width,height:b.height}})")
                    assert max(f['width'] for f in frames) - min(f['width'] for f in frames) < 1
                    assert len(frames) == 5
                    assert all(abs(f['width']/f['height']-1)<.02 for f in frames)
                    row.locator('.recording-toggle').click()
                    page.wait_for_function("s => [...document.querySelectorAll(s+' video')].every(v => v.paused)", arg=selector)
                    row.screenshot(path=str(output / f"{width}-dark-{task}.png"))
                    if width < 900:
                        row.locator('.recording-grid').evaluate('n => n.scrollLeft = n.scrollWidth')
                        page.wait_for_timeout(250)
                        row.screenshot(path=str(output / f"{width}-dark-{task}-right.png"))
                        row.locator('.recording-grid').evaluate('n => n.scrollLeft = 0')
                    if task == 'kungfu':
                        assert row.locator('[data-crop="top-only"] video').evaluate('v=>getComputedStyle(v).objectPosition') == '50% 100%'
                    row.locator('[data-method="ours"] a[data-viewer="video"]').click()
                    page.wait_for_function("document.querySelector('#media-viewer').open && document.querySelector('#viewer-content video').videoWidth > 0")
                    assert page.locator('.recording-row video').evaluate_all("vs=>vs.every(v=>v.paused)")
                    page.locator('#media-viewer').evaluate("async n => { await Promise.all(n.getAnimations().map(a => a.finished.catch(() => {}))); }")
                    box = page.locator('#media-viewer').bounding_box()
                    assert box['x'] >= -1 and box['x'] + box['width'] <= width + 1, box
                    if task == "tennis":
                        page.locator('#media-viewer').screenshot(path=str(output / f"{width}-reference-overlay.png"))
                    page.click('#viewer-close')
                    page.wait_for_function("!document.querySelector('#media-viewer').open")
                    row.locator('.recording-toggle').click()
                    page.wait_for_function("s => [...document.querySelectorAll(s+' video')].some(v=>!v.paused)", arg=selector)
                scenes = page.locator('.rx-scene-recordings')
                scenes.scroll_into_view_if_needed()
                page.wait_for_function('[...document.querySelectorAll(".rx-scene-grid img")].every(n=>n.complete && n.naturalWidth>0)')
                for link in scenes.locator('[data-viewer="video"]').all():
                    link.click()
                    page.wait_for_function('document.querySelector("#viewer-content video")?.videoWidth > 0')
                    assert page.locator('#viewer-content video').evaluate('v=>v.videoWidth === v.videoHeight')
                    page.click('#viewer-close')
                    page.wait_for_function("!document.querySelector('#media-viewer').open")
                scenes.screenshot(path=str(output / f'{width}-collision-scenes.png'))
                page.click('#theme-toggle')
                page.locator('[data-policy-task="tennis"]').scroll_into_view_if_needed()
                page.locator('[data-policy-task="tennis"]').screenshot(path=str(output / f"{width}-light-tennis.png"))
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
                page.emulate_media(reduced_motion='reduce')
                page.wait_for_function("[...document.querySelectorAll('.recording-row video')].every(v=>v.paused)")
                assert not errors, errors
                assert not missing, missing
                reports.append(dict(width=width, errors=errors, missing=missing, four_tasks_visible=True, autoplay_and_modal_passed=True))
                page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    (output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromium", required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
