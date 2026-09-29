"""CPU-only touch, album and mobile-layout checks for the project page."""
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
            browser = p.chromium.launch(executable_path=chromium, args=['--disable-gpu', '--disable-dev-shm-usage'])
            for width, height in [(320, 568), (390, 844), (430, 932), (768, 1024), (1440, 960)]:
                page = browser.new_page(viewport={'width':width, 'height':height},
                                        is_mobile=width < 768, has_touch=True, reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.on('response', lambda r: errors.append(f'{r.status} {r.url}') if r.status >= 400 else None)
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                page.evaluate('document.querySelectorAll("img").forEach(n=>n.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(n=>n.complete&&n.naturalWidth>0)')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'), width
                copy = page.locator('.hero-copy').bounding_box()
                bar = page.locator('.hero-bottom').bounding_box()
                nav = page.locator('.top-nav').bounding_box()
                assert copy['y'] >= nav['y'] + nav['height'] - 2, (width, 'hero/header overlap')
                if page.locator('#authors').count():
                    assert copy['y'] + copy['height'] <= bar['y'] + 2, (width, 'hero/author overlap')
                if page.locator('#authors').count():
                    assert page.locator('.hero-bottom #authors').count() == 1
                for theme in ('dark', 'light'):
                    if theme == 'light': page.click('#theme-toggle')
                    page.locator('#top').screenshot(path=str(output/f'{width}-{theme}-hero.png'))
                    page.locator('.showcase-viewport').scroll_into_view_if_needed()
                    visible = page.locator('.showcase-track figure').evaluate_all('''ns=>ns.filter(n=>{
                        const r=n.querySelector('img').getBoundingClientRect();
                        return Math.min(innerWidth,r.right)-Math.max(0,r.left)>12;
                    }).length''')
                    assert visible >= 3, (width, 'neighbor pages missing', visible)
                    page.locator('.research-nav').evaluate('n=>n.style.visibility="hidden"')
                    for section in ('showcase', 'method', 'results'):
                        page.locator('#'+section).screenshot(path=str(output/f'{width}-{theme}-{section}.png'))
                    page.locator('.research-nav').evaluate('n=>n.style.visibility=""')
                page.locator('.showcase-viewport').scroll_into_view_if_needed()
                page.click('#showcase-next')
                page.wait_for_function('document.querySelector(".showcase-track").dataset.active==="1"')
                page.click('#showcase-prev')
                page.wait_for_function('document.querySelector(".showcase-track").dataset.active==="0"')
                for selector, name in [('.showcase-track .is-active a','panorama'), ('.result-plot a','plot'), ('#table-expand','table')]:
                    page.locator(selector).first.click()
                    page.wait_for_function('document.querySelector("#viewer-content").dataset.ready === "true"')
                    initial = float(page.locator('#viewer-content').get_attribute('data-scale'))
                    if name == 'table':
                        stage = page.locator('.viewer-stage').bounding_box()
                        table = page.locator('.viewer-pan-surface table').bounding_box()
                        assert table['x'] >= stage['x'] - 2, (width, 'table clipped left', table)
                        assert table['x'] + table['width'] <= stage['x'] + stage['width'] + 2, (width, 'table clipped right', table)
                        assert abs(table['x'] + table['width']/2 - stage['x'] - stage['width']/2) < 2, (width, 'table not centered')
                        assert page.locator('.viewer-pan-surface th').first.evaluate('n=>getComputedStyle(n).position') == 'static'
                        page.locator('#media-viewer').screenshot(path=str(output/f'{width}-fit-table.png'))
                    page.click('#viewer-plus')
                    page.wait_for_function('(n)=>Number(document.querySelector("#viewer-content").dataset.scale)>n', arg=initial)
                    page.locator('#media-viewer').screenshot(path=str(output/f'{width}-zoom-{name}.png'))
                    page.click('#viewer-reset')
                    page.wait_for_function('(n)=>Math.abs(Number(document.querySelector("#viewer-content").dataset.scale)-n)<.01', arg=initial)
                    # Exercise genuine two-touch input, not only the zoom buttons.
                    if width == 390 and name == 'plot':
                        client = page.context.new_cdp_session(page)
                        box = page.locator('.viewer-stage').bounding_box()
                        x, y = box['x']+box['width']/2, box['y']+box['height']/2
                        def fingers(delta):
                            return [dict(x=x-delta, y=y, id=1), dict(x=x+delta, y=y, id=2)]
                        client.send('Input.dispatchTouchEvent', dict(type='touchStart',touchPoints=fingers(30)))
                        for delta in (40, 50, 65, 85):
                            client.send('Input.dispatchTouchEvent', dict(type='touchMove',touchPoints=fingers(delta)))
                        client.send('Input.dispatchTouchEvent', dict(type='touchEnd',touchPoints=[]))
                        assert float(page.locator('#viewer-content').get_attribute('data-scale')) > initial*1.2
                        client.detach()
                    page.click('#viewer-close')
                    assert not page.locator('#media-viewer').evaluate('n=>n.open')
                assert not errors, errors
                reports.append(dict(width=width,height=height,neighbors=visible,zoom=['panorama','plot','table'],errors=errors))
                page.close()
            browser.close()
    finally:
        server.shutdown(); server.server_close(); worker.join()
    (output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site,args.output,args.chromium)
