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
                assert page.locator('.showcase-track figure').count() == 8
                assert page.locator('.showcase-track figure[aria-hidden="false"]').count() == 1
                assert page.locator('.showcase-track video').count() == 0
                assert page.locator('#showcase #policies').count() == 1
                assert page.locator('.rx-scene-grid [data-viewer="video"]').count() == 4
                if width >= 1440 and page.locator('#authors').count():
                    assert page.locator('#authors > p:first-child span').evaluate_all('ns=>new Set(ns.map(n=>Math.round(n.getBoundingClientRect().top))).size') <= 2
                page.locator('#overview [data-viewer]').first.click()
                assert page.locator('#media-viewer').evaluate('n=>n.open')
                page.keyboard.press('Escape')
                assert not page.locator('#media-viewer').evaluate('n=>n.open')
                page.click('[data-workflow="forest"]')
                page.wait_for_function('Array.from(document.querySelectorAll("#workflow-stages img")).every(i=>i.complete && i.naturalWidth>0)')
                page.locator('#overview').screenshot(path=str(output/f'{width}-dark-forest-stages.png'))
                page.click('[data-workflow="tennis"]')
                for theme in ('dark', 'light'):
                    if theme == 'light': page.locator('#theme-toggle').click()
                    assert page.locator('html').get_attribute('data-theme') == theme
                    page.wait_for_function('(theme)=>Array.from(document.querySelectorAll("[data-plot]")).every(n=>n.complete && n.naturalWidth>0 && n.currentSrc.endsWith(`-${theme}.svg`))', arg=theme)
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, theme, 'overflow')
                    plots = page.locator('[data-plot]').evaluate_all('(ns)=>ns.map(n=>({src:n.currentSrc, width:n.getBoundingClientRect().width}))')
                    assert len(plots)==5
                    for plot in plots:
                        assert f'-{theme}.svg' in plot['src']
                        if width <= 700:
                            assert '-mobile-' in plot['src']
                    if width >= 1100:
                        fraction = page.locator('#method-overview').bounding_box()['width'] / page.locator('#method .rx-container').evaluate('n=>n.clientWidth-56')
                        assert .60 < fraction < .65, fraction
                        rects = page.locator('.result-plot:not(.result-plot-wide)').evaluate_all('ns=>ns.map(n=>Math.round(n.getBoundingClientRect().top))')
                        assert len(set(rects)) == 1, rects
                        art = page.locator('#method-overview').bounding_box()
                        notes = page.locator('.method-ledger').bounding_box()
                        assert abs(art['y']-notes['y']) < 3
                    for section in ('showcase', 'overview', 'method', 'results', 'hloop'):
                        page.locator('#'+section).scroll_into_view_if_needed()
                        page.wait_for_timeout(120)
                        page.locator('.research-nav').evaluate('n=>n.style.visibility="hidden"')
                        page.locator('#'+section).screenshot(path=str(output / f'{width}-{theme}-{section}.png'))
                        page.locator('.research-nav').evaluate('n=>n.style.visibility=""')
                    page.locator('#top').scroll_into_view_if_needed()
                    page.screenshot(path=str(output / f'{width}-{theme}-full.png'), full_page=True)
                page.reload(wait_until='networkidle')
                assert page.locator('html').get_attribute('data-theme') == 'light'
                for task in ('tennis', 'football', 'dance', 'kungfu'):
                    row = page.locator(f'[data-policy-task="{task}"]')
                    row.scroll_into_view_if_needed()
                    row.locator('.recording-toggle').click()
                    page.wait_for_function('s => [...document.querySelectorAll(s + " video")].some(v => v.currentTime > .1)', arg=f'[data-policy-task="{task}"]')
                    row.locator('.recording-toggle').click()
                for link in page.locator('.rx-scene-grid [data-viewer]').all():
                    link.click()
                    page.wait_for_function('document.querySelector("#viewer-content video")?.currentTime > .1')
                    page.click('#viewer-close')
                for selector in ('#method-overview a', '#hloop figure a', '.result-plot a'):
                    page.locator(selector).first.click()
                    assert page.locator('#media-viewer').evaluate('n=>n.open')
                    page.wait_for_function('document.querySelector("#viewer-content img")?.naturalWidth > 0')
                    page.keyboard.press('Escape')
                assert not errors, errors
                reports.append(dict(width=width, themes=['dark', 'light'], playback_pairs=7, scene_videos=4, overflow=False, errors=errors))
                page.close()
            # Autoplay advances one image at a time and respects a manual pause.
            page = browser.new_page(viewport={'width': 1440, 'height': 960})
            page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='networkidle')
            page.locator('.showcase-viewport').scroll_into_view_if_needed()
            page.mouse.move(0, 0)
            active = page.locator('.showcase-track').get_attribute('data-active')
            page.wait_for_timeout(5200)
            assert page.locator('.showcase-track').get_attribute('data-active') != active
            page.click('#showcase-toggle')
            active = page.locator('.showcase-track').get_attribute('data-active')
            page.mouse.move(0, 0)
            page.wait_for_timeout(600)
            assert page.locator('.showcase-track').get_attribute('data-active') == active
            page.locator('.showcase-pagination button').nth(3).click()
            assert page.locator('.showcase-track').get_attribute('data-active') == '3'
            page.close()
            page = browser.new_page(viewport={'width': 844, 'height': 390}, reduced_motion='reduce')
            page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='networkidle')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.locator('#showcase').scroll_into_view_if_needed()
            page.screenshot(path=str(output/'landscape-showcase.png'))
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
