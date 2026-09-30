"""Check album dwell and origin-linked viewing in a CPU browser."""
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
            for width, height in [(390, 844), (1440, 960)]:
                context = browser.new_context(viewport=dict(width=width, height=height),
                    is_mobile=width < 700, has_touch=width < 700,
                    record_video_dir=str(output), record_video_size=dict(width=width, height=height))
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                page.evaluate('document.querySelectorAll("img").forEach(n=>n.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(n=>n.complete&&n.naturalWidth>0)')
                page.locator('.showcase-viewport').scroll_into_view_if_needed()
                page.mouse.move(0, 0)
                assert page.locator('.showcase-viewport').get_attribute('data-moving') == 'false'
                for theme in ['dark', 'light']:
                    if page.locator('html').get_attribute('data-theme') != theme:
                        page.click('#theme-toggle')
                    page.locator('.showcase-viewport').scroll_into_view_if_needed()
                    page.click('#showcase-next')
                    assert page.locator('.showcase-viewport').get_attribute('data-moving') == 'true'
                    assert page.locator('.showcase-viewport').get_attribute('data-playing') == 'false'
                    page.screenshot(path=str(output/f'{width}-{theme}-album-turn.png'))
                    page.wait_for_function('document.querySelector(".showcase-viewport").dataset.moving==="false"')
                    page.wait_for_timeout(160)
                    progress = page.locator('.showcase-pagination [aria-pressed="true"]').evaluate(
                        'n=>Number(n.style.getPropertyValue("--progress"))')
                    assert progress < .15, progress
                    page.screenshot(path=str(output/f'{width}-{theme}-album-rest.png'))
                    if page.locator('#showcase-toggle').get_attribute('aria-pressed') == 'true':
                        page.click('#showcase-toggle')
                    link = page.locator('.showcase-track .is-active a')
                    link.click()
                    page.wait_for_function('document.querySelector("#viewer-content").dataset.ready==="true"')
                    keyframes = page.locator('#media-viewer').evaluate('''n=>{
                        const a=n.getAnimations().find(a=>a.id==='media-carry');
                        return a ? a.effect.getKeyframes() : [];
                    }''')
                    assert len(keyframes) == 2, keyframes
                    assert keyframes[0]['transform'] != keyframes[1]['transform']
                    page.wait_for_timeout(350)
                    page.screenshot(path=str(output/f'{width}-{theme}-viewer.png'))
                    bounds = page.locator('#media-viewer').bounding_box()
                    assert bounds['x'] >= 0 and bounds['x'] + bounds['width'] <= width
                    page.click('#viewer-close')
                    page.screenshot(path=str(output/f'{width}-{theme}-return.png'))
                    page.wait_for_function('!document.querySelector("#media-viewer").open')
                    assert link.evaluate('n=>n===document.activeElement')
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')

                # Rapid navigation, wraparound and repeated closing must settle cleanly.
                page.evaluate('''()=>{for(let i=0;i<11;i++) document.querySelector('#showcase-next').click();}''')
                page.wait_for_function('document.querySelector(".showcase-viewport").dataset.moving==="false"')
                page.locator('.showcase-track .is-active a').click()
                page.keyboard.press('Escape')
                page.keyboard.press('Escape')
                page.wait_for_function('''!document.querySelector('#media-viewer').open &&
                    !document.documentElement.classList.contains('media-open')''')
                assert not page.locator('html').evaluate('n=>n.classList.contains("media-open")')

                page.emulate_media(reduced_motion='reduce')
                page.click('#showcase-next')
                assert page.locator('.showcase-viewport').get_attribute('data-moving') == 'false'
                page.locator('.showcase-track .is-active a').click()
                page.wait_for_function('document.querySelector("#viewer-content").dataset.ready==="true"')
                assert page.locator('#media-viewer').evaluate('n=>n.getAnimations().length') == 0
                page.keyboard.press('Escape')
                page.wait_for_function('!document.querySelector("#media-viewer").open')
                assert not errors, errors
                reports.append(dict(width=width,settled_dwell=True,origin_carry=True,
                    rapid_navigation=True,reduced_motion=True,errors=errors))
                video = page.video
                context.close()
                Path(video.path()).rename(output/f'{width}-continuity-demo.webm')
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    (output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chromium',required=True)
    args = parser.parse_args()
    verify(args.site,args.output,args.chromium)
