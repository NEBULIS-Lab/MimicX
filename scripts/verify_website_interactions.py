"""CPU-only browser checks for motion, navigation and media transitions."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
from io import BytesIO

from PIL import Image

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
            for width, height in [(390, 844), (1440, 960)]:
                context = browser.new_context(viewport=dict(width=width, height=height),
                    is_mobile=width < 700, has_touch=width < 700, color_scheme='light',
                    record_video_dir=str(output), record_video_size=dict(width=width, height=height))
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.add_init_script('''{
                    const request = window.requestAnimationFrame;
                    window.albumTicks = 0;
                    window.requestAnimationFrame = callback => request.call(window, time => {
                        if (callback.name === 'frame') window.albumTicks++;
                        callback(time);
                    });
                }''')
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                page.evaluate('document.querySelectorAll("img").forEach(n=>n.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(n=>n.complete&&n.naturalWidth>0)')
                assert page.locator('html').get_attribute('data-theme') == 'dark'

                def scroll_to(selector):
                    page.locator(selector).evaluate('''n=>window.scrollTo({
                        top:scrollY+n.getBoundingClientRect().top-document.querySelector('.research-nav').offsetHeight-12,
                        behavior:'instant'})''')
                    page.wait_for_timeout(250)

                scroll_to('.showcase-viewport')
                page.mouse.move(0, 0)
                page.wait_for_function('document.querySelector(".showcase-viewport").dataset.playing==="true"')
                page.click('#showcase-toggle')
                page.wait_for_function('document.querySelector(".showcase-viewport").dataset.playing==="false"')
                ticks = page.evaluate('albumTicks')
                page.wait_for_timeout(250)
                assert page.evaluate('albumTicks') == ticks
                page.locator('.showcase-pagination button').last.click()
                page.wait_for_timeout(1100)
                assert page.locator('.showcase-pagination').evaluate('''n=>{
                    const a=n.querySelector('[aria-pressed="true"]').getBoundingClientRect(),b=n.getBoundingClientRect();
                    return a.left>=b.left-1 && a.right<=b.right+1;
                }''')
                page.screenshot(path=str(output/f'{width}-album-follow.png'))
                page.click('#showcase-next')
                page.wait_for_timeout(1100)
                page.locator('.showcase-track .is-active a').click()
                page.wait_for_function('document.querySelector("#viewer-content").dataset.ready==="true"')
                page.wait_for_timeout(300)
                page.screenshot(path=str(output/f'{width}-viewer-open.png'))
                for icon in page.locator('.viewer-toolbar .rx-icon').all():
                    pixels = Image.open(BytesIO(icon.screenshot())).convert('RGB')
                    assert sum(min(pixels.getpixel((x, y))) > 145
                               for x in range(pixels.width) for y in range(pixels.height)) > 8
                bounds = page.locator('#media-viewer').bounding_box()
                assert bounds['x'] >= 0 and bounds['x'] + bounds['width'] <= width
                page.click('#viewer-close')
                page.wait_for_function('!document.querySelector("#media-viewer").open')
                assert not page.locator('html').evaluate('n=>n.classList.contains("media-open")')
                page.locator('.showcase-track .is-active a').click()
                page.wait_for_function('document.querySelector("#viewer-content").dataset.ready==="true"')
                page.keyboard.press('Escape')
                page.wait_for_function('!document.querySelector("#media-viewer").open')

                scroll_to('#policies')
                assert page.locator('.research-nav a[aria-current]').get_attribute('href') == '#policies'
                page.click('#task-kungfu')
                page.wait_for_timeout(350)
                assert 'Kung Fu' in page.locator('#ours-video').get_attribute('aria-label')
                page.screenshot(path=str(output/f'{width}-policy-switch.png'))
                scroll_to('#overview')
                assert page.locator('.research-nav a[aria-current]').get_attribute('href') == '#overview'
                page.click('[data-workflow="forest"]')
                page.wait_for_function('document.querySelector("#workflow-stages").getAttribute("aria-busy")==="false"')
                page.wait_for_timeout(400)
                assert page.locator('#workflow-stages img').evaluate_all('ns=>ns.every(n=>n.src.includes("forest-stage"))')
                page.screenshot(path=str(output/f'{width}-workflow-forest.png'))
                page.evaluate('''()=>{
                    document.querySelector('[data-workflow="tennis"]').click();
                    document.querySelector('[data-workflow="forest"]').click();
                    document.querySelector('[data-workflow="tennis"]').click();
                }''')
                page.wait_for_function('document.querySelector("#workflow-stages").getAttribute("aria-busy")==="false"')
                assert page.locator('#workflow-stages img').evaluate_all('ns=>ns.every(n=>n.src.includes("tennis-stage"))')
                ticks = page.evaluate('albumTicks')
                page.wait_for_timeout(250)
                assert page.evaluate('albumTicks') == ticks

                for theme in ['light', 'dark']:
                    if page.locator('html').get_attribute('data-theme') != theme:
                        page.click('#theme-toggle')
                    for section in ['method', 'results', 'hloop']:
                        scroll_to('#' + section)
                        page.screenshot(path=str(output/f'{width}-{theme}-{section}.png'))
                        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                page.emulate_media(reduced_motion='reduce')
                scroll_to('#overview')
                page.click('[data-workflow="forest"]')
                page.wait_for_function('document.querySelector("#workflow-stages").getAttribute("aria-busy")==="false"')
                assert page.locator('#workflow-stages').evaluate('n=>n.getAnimations({subtree:true}).length') == 0
                page.locator('#workflow-stages a').first.click()
                page.wait_for_function('document.querySelector("#viewer-content").dataset.ready==="true"')
                assert page.locator('#media-viewer').evaluate('n=>getComputedStyle(n).animationName') == 'none'
                page.keyboard.press('Escape')
                page.wait_for_function('!document.querySelector("#media-viewer").open')
                assert not errors, errors
                reports.append(dict(width=width, default_dark=True, idle_ticks=0,
                    navigation=True, atomic_frames=True, reduced_motion=True, errors=errors))
                video = page.video
                context.close()
                original = Path(video.path())
                original.rename(output/f'{width}-interaction-demo.webm')
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    (output/'report.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
