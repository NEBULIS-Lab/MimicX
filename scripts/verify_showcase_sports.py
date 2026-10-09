"""CPU-only browser checks for mixed image/video showcase slides."""
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
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    reports = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium,
                args=['--disable-gpu', '--disable-dev-shm-usage'])
            for width, height in ((1440, 960), (390, 844)):
                context = browser.new_context(viewport=dict(width=width, height=height))
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                assert page.locator('.showcase-track figure').count() == 7
                assert page.locator('.showcase-track video').count() == 2
                assert page.locator('.showcase-pagination button').all_text_contents() == [
                    'Football Juggling', 'Parkour', 'Track Run', 'Stair Ascent',
                    'Platform Jump', 'Forest Traversal', 'Tennis Swing']
                page.locator('.showcase-viewport').scroll_into_view_if_needed()
                for label, index in (('Tennis Swing', 6), ('Football Juggling', 0)):
                    page.locator('.showcase-pagination button').filter(has_text=label).click()
                    page.locator('.showcase-viewport').evaluate("e=>e.scrollIntoView({block:'center',behavior:'instant'})")
                    page.wait_for_function(f'document.querySelector(".showcase-track").dataset.active==="{index}"')
                    page.wait_for_function('''() => {
                        const v = document.querySelector('.showcase-track .is-active video');
                        return v && !v.paused && v.readyState >= 2 && v.currentTime > .05;
                    }''')
                    page.wait_for_timeout(400)
                    video = page.locator('.showcase-track .is-active video')
                    page.screenshot(path=str(output / f'{width}-{index}-showcase.png'))
                    box = video.bounding_box()
                    assert abs(box['width'] / box['height'] - 16 / 9) < .05, (width, index, box,
                        video.evaluate('v=>({width:getComputedStyle(v).width,height:getComputedStyle(v).height})'))
                    assert page.locator('.showcase-track video').evaluate_all(
                        'vs=>vs.filter(v=>!v.paused).length') == 1
                    page.screenshot(path=str(output / f'{width}-{index}-showcase.png'))
                    page.locator('.showcase-track .is-active a').click()
                    page.wait_for_function('document.querySelector("#media-viewer").open')
                    page.wait_for_function('document.querySelector("#viewer-content video").readyState>=2')
                    assert page.locator('.showcase-track video').evaluate_all('vs=>vs.every(v=>v.paused)')
                    page.screenshot(path=str(output / f'{width}-{index}-viewer.png'))
                    page.click('#viewer-close')
                    page.wait_for_function('!document.querySelector("#media-viewer").open')
                    page.wait_for_function('!document.querySelector(".showcase-track .is-active video").paused')
                page.locator('.showcase-track .is-active video').hover()
                page.wait_for_function('document.querySelector(".showcase-track").dataset.active!=="0"', timeout=7000)
                page.locator('#overview').scroll_into_view_if_needed()
                page.wait_for_function('Array.from(document.querySelectorAll(".showcase-track video")).every(v=>v.paused)')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                assert not errors, errors
                reports.append(dict(width=width, slides=7, videos=2, modal=True,
                                    hover_autoplay=True, inactive_videos_paused=True, errors=errors))
                context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    (output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
