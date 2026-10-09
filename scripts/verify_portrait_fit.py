"""Check that portrait media and tables fit without horizontal scrolling (CPU-only)."""
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
    reports, failures = [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium,
                args=['--disable-gpu', '--disable-dev-shm-usage'])
            for width, height in ((320, 568), (360, 800), (390, 844), (430, 932), (768, 1024), (1440, 960)):
                page = browser.new_page(viewport=dict(width=width, height=height),
                    is_mobile=width < 768, has_touch=True, reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                page.evaluate('document.querySelectorAll("img").forEach(n=>n.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(n=>n.complete&&n.naturalWidth>0)')
                for theme in ('dark', 'light'):
                    if theme == 'light':
                        page.click('#theme-toggle')
                    measurements = page.evaluate('''() => {
                        const selectors = ['.recording-grid', '.recording-grid figure', '.rx-paper-stages',
                            '.rx-plot-grid', '.result-plot', '.rx-table-scroll', '#results .rx-table',
                            '#results .rx-table th', '#results .rx-table td', '.rx-workflow-tabs',
                            '.research-nav nav', '.showcase-pagination'];
                        const overflow = selectors.flatMap(s => Array.from(document.querySelectorAll(s))
                            .filter(n => n.scrollWidth > n.clientWidth + 2)
                            .map(n => ({selector:s, visible:n.clientWidth, content:n.scrollWidth})));
                        const clipped = [...document.querySelectorAll('.recording-frame, .rx-paper-stages img, .result-plot img, #results .rx-table')]
                            .filter(n => { const r = n.getBoundingClientRect(); return r.left < -1 || r.right > innerWidth + 1; })
                            .map(n => n.className || n.tagName);
                        return {overflow, clipped, pageOverflow:document.documentElement.scrollWidth > innerWidth,
                            videoColumns:getComputedStyle(document.querySelector('.recording-grid')).gridTemplateColumns,
                            plotColumns:getComputedStyle(document.querySelector('.rx-plot-grid')).gridTemplateColumns};
                    }''')
                    if measurements['overflow'] or measurements['clipped'] or measurements['pageOverflow']:
                        failures.append(dict(width=width, theme=theme, **measurements))
                    if width in (320, 390, 1440):
                        for selector, name in (('#top', 'hero'), ('.research-nav', 'navigation'),
                            ('.recording-row', 'recordings'), ('#overview', 'workflow'),
                            ('.rx-plot-grid', 'plots'), ('.rx-table-scroll', 'table'), ('#hloop', 'hloop')):
                            page.locator(selector).first.screenshot(path=str(output / f'{width}-{theme}-{name}.png'),
                                style='' if name in ('hero', 'navigation') else '.research-nav { visibility: hidden; }')
                    reports.append(dict(width=width, theme=theme, **measurements))
                # Fitting inline content must not remove the full-size inspection tools.
                for selector, kind in (('.result-plot a', 'image'), ('#table-expand', 'table'),
                    ('.recording-grid [data-viewer="video"]', 'video')):
                    page.locator(selector).first.click()
                    page.wait_for_function('document.querySelector("#media-viewer").open')
                    if kind == 'video':
                        page.wait_for_function('document.querySelector("#viewer-content video").readyState >= 2')
                    else:
                        page.wait_for_function('document.querySelector("#viewer-content").dataset.ready === "true"')
                        initial = float(page.locator('#viewer-content').get_attribute('data-scale'))
                        page.click('#viewer-plus')
                        page.wait_for_function('n => Number(document.querySelector("#viewer-content").dataset.scale) > n', arg=initial)
                        page.click('#viewer-reset')
                        page.wait_for_function('n => Math.abs(Number(document.querySelector("#viewer-content").dataset.scale) - n) < .01', arg=initial)
                    box = page.locator('#media-viewer').bounding_box()
                    assert box['x'] >= 0 and box['x'] + box['width'] <= width, (width, kind, box)
                    page.click('#viewer-close')
                    page.wait_for_function('!document.querySelector("#media-viewer").open')
                assert not errors, errors
                page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    (output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    assert not failures, json.dumps(failures, indent=2)
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
