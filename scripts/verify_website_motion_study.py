"""Browser QA for additive motion evidence and its existing media interactions."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def verify(site, output, chromium):
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    reports = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium,
                                        args=['--disable-gpu', '--disable-dev-shm-usage'])
            for width in (320, 390, 1440):
                page = browser.new_page(viewport=dict(width=width, height=1000))
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='domcontentloaded')
                assert page.locator('html').get_attribute('data-theme') == 'dark'
                page.locator('#motion-study').scroll_into_view_if_needed()
                page.wait_for_function('Array.from(document.querySelectorAll("#motion-study video")).some(v=>v.currentTime>.1)')
                for theme in ('dark', 'light'):
                    if theme == 'light':
                        page.locator('#theme-toggle').click()
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    if theme == 'dark':
                        color = page.locator('.motion-table tbody th').first.evaluate('n=>getComputedStyle(n).color')
                        assert color == 'rgb(241, 240, 238)', color
                    for selector, name in (('.motion-comparisons', 'comparisons'),
                                           ('#method', 'method'), ('#motion-study > div > .motion-table-frame', 'summary')):
                        page.locator('.research-nav').evaluate('n=>n.style.visibility="hidden"')
                        page.locator(selector).screenshot(path=str(output / f'{width}-{theme}-{name}.png'))
                        page.locator('.research-nav').evaluate('n=>n.style.visibility=""')
                    page.locator('.motion-detail').evaluate('n=>n.open=true')
                    assert page.locator('#motion-completion-rows tr').count() == 12
                    page.locator('.motion-detail').screenshot(path=str(output / f'{width}-{theme}-details.png'))
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                page.locator('.motion-recording [data-viewer]').first.click()
                page.wait_for_function('document.querySelector("#media-viewer video")?.videoWidth > 0')
                assert page.locator('#media-viewer').evaluate('n=>n.open')
                page.keyboard.press('Escape')
                page.wait_for_function('!document.querySelector("#media-viewer").open')
                assert not errors, errors
                reports.append(dict(width=width, themes=['dark', 'light'], overflow=False,
                                    autoplay=True, viewer=True, javascript_errors=errors))
                page.close()
            browser.close()
        (output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    finally:
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'docs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
