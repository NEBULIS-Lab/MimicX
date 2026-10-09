"""CPU-only author-baseline, inline readout and native-ratio dialog checks."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading

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
            for width, height in [(320, 568), (390, 844), (768, 1024), (1024, 768), (1440, 960)]:
                page = browser.new_page(viewport=dict(width=width, height=height),
                                        is_mobile=width < 768, has_touch=True, reduced_motion='reduce')
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='networkidle')
                page.evaluate('document.querySelectorAll("img").forEach(n=>n.loading="eager")')
                page.wait_for_function('Array.from(document.images).every(n=>n.complete&&n.naturalWidth>0)')
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                if page.locator('#authors').count():
                    baselines = page.locator('#authors > p:first-child > span').evaluate_all('''ns=>ns.map(n=>{
                        const r=document.createRange();r.selectNodeContents(n.firstChild);
                        return r.getBoundingClientRect().bottom;
                    })''')
                    assert len(baselines) == 10
                    for a, b in zip(baselines, baselines[1:]):
                        assert abs(a-b) < .25 or abs(a-b) > 10, (width, 'misaligned author text', baselines)
                    if width >= 1440:
                        assert max(baselines)-min(baselines) < .25
                assert page.locator('[data-policy-task]').count() == 4
                assert page.locator('[data-comparison-task]').count() == 2
                assert page.locator('.rx-policy-summary p, .rx-policy-summary div').count() == 0
                assert 'additional policy continuation' in page.locator('.rx-policy-summary').inner_text()
                for theme in ('dark', 'light'):
                    if theme == 'light': page.click('#theme-toggle')
                    page.locator('.research-nav').evaluate('n=>n.style.visibility="hidden"')
                    page.evaluate('window.scrollTo({top:0,behavior:"instant"})')
                    page.wait_for_function('Array.from(document.images).every(n=>n.complete&&n.naturalWidth>0)')
                    # Crop one complete page capture; mobile element screenshots can
                    # clip below the viewport when scroll margins reserve header space.
                    full = Image.open(BytesIO(page.screenshot(full_page=True)))
                    for section in ('top', 'showcase', 'policies', 'overview', 'method', 'results', 'hloop'):
                        box = page.locator('#'+section).bounding_box()
                        x, y = round(box['x']), round(box['y'])
                        full.crop((x,y,x+round(box['width']),y+round(box['height']))).save(output/f'{width}-{theme}-{section}.png')
                    page.locator('.research-nav').evaluate('n=>n.style.visibility=""')
                dialogs = []
                for selector, name in [('.showcase-track .is-active a', 'panorama'), ('#overview li:first-child a', 'source'),
                                       ('#method-overview a', 'method'), ('.result-plot a', 'plot'),
                                       ('.result-plot-wide a', 'trace'), ('#hloop figure a', 'hloop'),
                                       ('#table-expand', 'table')]:
                    page.locator(selector).first.click()
                    page.wait_for_function('document.querySelector("#viewer-content").dataset.ready==="true"')
                    dialog = page.locator('#media-viewer').bounding_box()
                    stage = page.locator('.viewer-stage').bounding_box()
                    assert dialog['x'] >= -1 and dialog['y'] >= -1, (width, name, dialog)
                    assert dialog['x']+dialog['width'] <= width+1, (width, name, dialog)
                    assert dialog['y']+dialog['height'] <= height+1, (width, name, dialog)
                    if name != 'table':
                        ratio = page.locator('#viewer-content img').evaluate('n=>n.naturalWidth/n.naturalHeight')
                        assert abs(stage['width']/stage['height']/ratio-1)<.005, (width, name, 'wrong aspect ratio', stage, ratio)
                    if width in (390, 1440):
                        page.locator('#media-viewer').screenshot(path=str(output/f'{width}-dialog-{name}.png'))
                    dialogs.append(dict(name=name, width=dialog['width'], height=dialog['height']))
                    page.click('#viewer-close')
                assert max(d['height'] for d in dialogs)-min(d['height'] for d in dialogs)>50
                assert not errors, errors
                reports.append(dict(width=width, height=height, dialogs=dialogs, errors=errors))
                page.close()
            browser.close()
    finally:
        server.shutdown(); server.server_close(); worker.join()
    (output/'report.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
