"""CPU browser regression checks for the clean hero, album motion and theme."""
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
            for width, height in [(320, 700), (390, 844), (1440, 960)]:
                context = browser.new_context(viewport=dict(width=width,height=height),
                    color_scheme='light', reduced_motion='no-preference', is_mobile=width<700, has_touch=True)
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                assert page.locator('html').get_attribute('data-theme') == 'dark'
                assert page.locator('.hero-context, .hero-expand').count() == 0
                assert 'View policy rollouts' not in page.locator('#top').inner_text()
                assert page.locator('.showcase-viewport').get_attribute('data-interval') == '4000'
                page.evaluate('document.querySelectorAll("img").forEach(n=>n.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(n=>n.complete&&n.naturalWidth>0)')
                page.screenshot(path=str(output/f'{width}-dark-first-visit.png'))
                page.locator('.showcase-viewport').scroll_into_view_if_needed()
                page.mouse.move(0,0)
                page.wait_for_function('document.querySelector(".showcase-track").dataset.active!=="0"', timeout=6500)
                page.click('#showcase-toggle')
                page.wait_for_timeout(1000)
                page.screenshot(path=str(output/f'{width}-album-before.png'))
                page.click('#showcase-next')
                page.wait_for_timeout(140)
                page.screenshot(path=str(output/f'{width}-album-transition.png'))
                visual = page.locator('.showcase-track figure').evaluate_all('''ns=>{
                    const rail=document.querySelector('.showcase-viewport').getBoundingClientRect();
                    const c=rail.left+rail.width/2, base=innerWidth<=700?.94:.88;
                    return ns.map(n=>{const r=n.getBoundingClientRect(), f=Math.max(0,1-Math.abs((r.left+r.width/2-c)/r.width));
                        return {expected:base+(1-base)*f*f*(3-2*f), actual:Number(n.style.getPropertyValue('--album-scale'))};});
                }''')
                assert all(abs(r['expected']-r['actual'])<.025 for r in visual), visual
                page.wait_for_timeout(1100)
                page.screenshot(path=str(output/f'{width}-album-settled.png'))
                assert float(page.locator('.showcase-track .is-active').evaluate('n=>n.style.getPropertyValue("--album-scale")'))>.995
                page.locator('.showcase-pagination button').last.click()
                page.wait_for_timeout(1300)
                page.click('#showcase-next')
                page.wait_for_function('document.querySelector(".showcase-track").dataset.active==="0"')
                page.wait_for_timeout(1100)
                visible = page.locator('.showcase-track img').evaluate_all('''ns=>ns.filter(n=>{
                    const r=n.getBoundingClientRect();return Math.min(innerWidth,r.right)-Math.max(0,r.left)>12;
                }).length''')
                assert visible>=3, (width, visible)
                page.screenshot(path=str(output/f'{width}-album-loop.png'))
                page.click('#theme-toggle')
                page.reload(wait_until='networkidle')
                assert page.locator('html').get_attribute('data-theme') == 'light'
                page.evaluate('localStorage.removeItem("mimicx-theme")')
                page.reload(wait_until='networkidle')
                assert page.locator('html').get_attribute('data-theme') == 'dark'
                page.emulate_media(reduced_motion='reduce')
                page.wait_for_function('document.querySelector("#showcase-toggle").getAttribute("aria-pressed")==="false"')
                assert page.locator('#showcase-toggle').get_attribute('aria-pressed') == 'false'
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                assert not errors, errors
                reports.append(dict(width=width,first_visit='dark',saved_light=True,autoplay_ms=4000,
                                    scroll_linked=True,loop_neighbors=visible,errors=errors))
                context.close()
            browser.close()
    finally:
        server.shutdown(); server.server_close(); worker.join()
    (output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site,args.output,args.chromium)
