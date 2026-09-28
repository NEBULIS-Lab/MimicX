"""CPU browser QA for the editorial website; accepts either release directory."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright
from verify_website import QuietHandler


def verify(site, output, chromium, quick=False):
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reports = []
    widths = [390, 1440] if quick else [320, 390, 768, 1024, 1440, 1920]
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium, args=['--disable-gpu', '--disable-dev-shm-usage', '--disable-accelerated-video-decode', '--disable-accelerated-video-encode'])
            for width in widths:
                page = browser.new_page(viewport={'width': width, 'height': 960}, reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('response', lambda r: errors.append(f'{r.status}: {r.url}') if r.status >= 400 else None)
                page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='networkidle')
                page.evaluate('document.querySelectorAll("img").forEach(i => i.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)')
                assert page.locator('#showcase-toggle').get_attribute('aria-pressed') == 'false'
                assert page.locator('[data-clone]').count() == 8
                assert page.locator('[data-clone] a[tabindex="-1"]').count() == 8
                page.locator('#overview [data-viewer]').first.click()
                assert page.locator('#media-viewer').evaluate('n=>n.open')
                page.keyboard.press('Escape')
                assert not page.locator('#media-viewer').evaluate('n=>n.open')
                for theme in ('dark', 'light'):
                    if theme == 'light': page.locator('#theme-toggle').click()
                    assert page.locator('html').get_attribute('data-theme') == theme
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, theme, 'overflow')
                    if width >= 1024:
                        fraction = page.locator('#method-overview').bounding_box()['width'] / page.locator('#method .rx-container').evaluate('n=>n.clientWidth-56')
                        assert .64 < fraction < .66, fraction
                    for section in ('showcase', 'overview', 'method', 'results', 'hloop'):
                        page.locator('#'+section).scroll_into_view_if_needed()
                        page.wait_for_timeout(120)
                        page.locator('#'+section).screenshot(path=str(output / f'{width}-{theme}-{section}.png'))
                    page.locator('#top').scroll_into_view_if_needed()
                    page.screenshot(path=str(output / f'{width}-{theme}-full.png'), full_page=True)
                page.reload(wait_until='networkidle')
                assert page.locator('html').get_attribute('data-theme') == 'light'
                for task in ('tennis', 'football', 'dance', 'kungfu'):
                    page.click(f'[data-task="{task}"]')
                    page.click('#pair-toggle')
                    page.wait_for_function('document.getElementById("ours-video").currentTime > .1')
                    page.click('#pair-toggle')
                    page.click('#pair-reset')
                assert not errors, errors
                reports.append(dict(width=width, themes=['dark', 'light'], playback_pairs=4, overflow=False, errors=errors))
                page.close()
            # Motion is enabled only on desktop without the accessibility preference.
            page = browser.new_page(viewport={'width': 1440, 'height': 960})
            page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='networkidle')
            page.locator('#showcase').scroll_into_view_if_needed()
            page.mouse.move(0, 0)
            left = page.locator('.showcase-viewport').evaluate('n=>n.scrollLeft')
            page.wait_for_timeout(1500)
            assert page.locator('.showcase-viewport').evaluate('n=>n.scrollLeft') > left
            assert page.locator('.showcase-track video').evaluate_all('vs=>vs.filter(v=>!v.paused).length') <= 1
            page.click('#showcase-toggle')
            left = page.locator('.showcase-viewport').evaluate('n=>n.scrollLeft')
            page.mouse.move(0, 0)
            page.wait_for_timeout(600)
            assert page.locator('.showcase-viewport').evaluate('n=>n.scrollLeft') == left
            page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    (output / 'report.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1]/'docs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium, args.quick)
