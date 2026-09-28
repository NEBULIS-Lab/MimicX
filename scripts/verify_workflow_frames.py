"""Check native image framing and aligned stage panels in both website themes."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright
from verify_website import QuietHandler


def verify(site, output, chromium):
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium, args=['--disable-gpu', '--disable-dev-shm-usage'])
            for width in (320, 390, 1440):
                page = browser.new_page(viewport={'width': width, 'height': 960}, reduced_motion='reduce')
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                page.locator('#overview').scroll_into_view_if_needed()
                for theme in ('dark', 'light'):
                    if theme == 'light':
                        page.click('#theme-toggle')
                    for case in ('tennis', 'forest'):
                        page.click(f'[data-workflow="{case}"]')
                        page.evaluate('document.querySelectorAll("#workflow-stages img").forEach(n=>n.loading="eager")')
                        page.wait_for_function('Array.from(document.querySelectorAll("#workflow-stages img")).every(n=>n.complete && n.naturalWidth>0)')
                        dimensions = page.locator('#workflow-stages img').evaluate_all('''ns=>ns.map(n=>{
                            const im=n.getBoundingClientRect(), a=n.parentElement.getBoundingClientRect();
                            return {height:im.height, distortion:Math.abs(im.width/im.height-n.naturalWidth/n.naturalHeight),
                                    frame:Math.abs(a.height-im.height)+Math.abs(a.width-im.width)};
                        })''')
                        assert all(d['distortion'] < .01 and d['frame'] < 1 for d in dimensions), dimensions
                        assert max(d['height'] for d in dimensions)-min(d['height'] for d in dimensions) < 1, dimensions
                        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                        page.locator('.research-nav').evaluate('n=>n.style.visibility="hidden"')
                        page.locator('#overview').screenshot(path=str(output/f'{width}-{theme}-{case}-stages.png'))
                        if case == 'tennis':
                            page.locator('#overview').screenshot(path=str(output/f'{width}-{theme}-overview.png'))
                        page.locator('.research-nav').evaluate('n=>n.style.visibility=""')
                page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    print('12 workflow layouts: native ratio, tight frames, aligned heights, no overflow')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
