"""CPU-only browser regression for lazy previews, row playback and HD viewing."""
import argparse
from functools import partial
import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer

from playwright.sync_api import sync_playwright
from verify_website_recordings import RecordingHandler


def verify(site, output, chromium):
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(RecordingHandler, directory=str(site)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    reports = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chromium,
                args=['--disable-gpu', '--disable-dev-shm-usage'])
            for width, height, low in ((1440, 960, False), (390, 844, True)):
                page = browser.new_page(viewport=dict(width=width, height=height))
                errors, requests = [], []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.on('request', lambda r: requests.append(r.url) if '.mp4' in r.url else None)
                page.add_init_script('''Object.defineProperty(navigator, 'connection', {value: {
                    saveData: LOW, effectiveType: LOW ? '3g' : '4g', downlink: LOW ? 1 : 10,
                    addEventListener() {} }});'''.replace('LOW', str(low).lower()))
                if low:
                    cdp = page.context.new_cdp_session(page)
                    cdp.send('Network.enable')
                    cdp.send('Network.emulateNetworkConditions', dict(offline=False, latency=120,
                        downloadThroughput=125000, uploadThroughput=62500, connectionType='cellular3g'))
                page.goto(f'http://127.0.0.1:{server.server_port}', wait_until='domcontentloaded')
                page.wait_for_timeout(1000)
                assert all('/previews/showcase--football-contact-slow4x.mp4' in url for url in requests), requests
                initial_requests = len(requests)
                row = page.locator('[data-policy-task="tennis"]')
                started = time.monotonic()
                row.scroll_into_view_if_needed()
                expected = 1 if low else 5
                page.wait_for_function('n => [...document.querySelectorAll("[data-policy-task=tennis] video")].filter(v=>!v.paused && v.currentTime > .1).length === n', arg=expected, timeout=60000)
                startup_seconds = time.monotonic() - started
                videos = row.locator('video')
                if low:
                    assert videos.evaluate_all('vs=>vs.filter(v=>v.hasAttribute("src")).length') == 1
                    assert row.locator('[data-method=ours] video').evaluate('v=>!v.paused')
                    playing = row.locator('[data-method=ours] video')
                    initial_time = playing.evaluate('v=>v.currentTime')
                    page.wait_for_timeout(3500)
                    before = videos.evaluate_all('vs=>vs.map(v=>v.currentTime)')
                    playback_advance = before[-1] - initial_time
                    row.locator('[data-method=sonic] .recording-load').click()
                    page.wait_for_function('!document.querySelector("[data-policy-task=tennis] [data-method=sonic] video").paused', timeout=60000)
                    assert row.locator('[data-method=ours] video').evaluate('v=>v.paused && !v.hasAttribute("src")')
                else:
                    playback_advance = None
                    page.wait_for_timeout(1800)
                    before = videos.evaluate_all('vs=>vs.map(v=>v.currentTime)')
                    assert max(before) - min(before) < .2, before
                    row.locator('.recording-toggle').click()
                    page.wait_for_function('[...document.querySelectorAll("[data-policy-task=tennis] video")].every(v=>v.paused)')
                    row.locator('.recording-toggle').click()
                    page.wait_for_function('[...document.querySelectorAll("[data-policy-task=tennis] video")].every(v=>!v.paused)')
                assert all('/previews/' in url for url in requests), requests
                row.screenshot(path=str(output / f'{width}-recordings.png'))
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                # Explicit viewer action alone requests the untouched HD recording.
                link = row.locator('[data-method=ours] a[data-viewer=video]')
                original = link.get_attribute('href')
                link.click()
                page.wait_for_function('document.querySelector("#viewer-content video")?.videoWidth > 0', timeout=60000)
                assert page.locator('#viewer-content video').get_attribute('src').endswith(original)
                assert page.locator('.recording-row video').evaluate_all('vs=>vs.every(v=>v.paused)')
                page.click('#viewer-close')
                page.wait_for_function('!document.querySelector("#media-viewer").open')
                page.locator('#low-data-mode').set_checked(not low)
                row.scroll_into_view_if_needed()
                page.wait_for_function('n => [...document.querySelectorAll("[data-policy-task=tennis] video")].filter(v=>!v.paused).length === n', arg=5 if low else 1, timeout=60000)
                if not low:
                    page.locator('#low-data-mode').uncheck()
                    for task in ('football', 'dance', 'kungfu', 'track', 'stairs', 'forest', 'bench', 'parkour', 'descent'):
                        other = page.locator(f'[data-policy-task="{task}"]')
                        other.scroll_into_view_if_needed()
                        page.wait_for_function('task => [...document.querySelectorAll(`[data-policy-task="${task}"] video`)].every(v=>!v.paused && v.videoWidth > 0 && v.currentTime > .03)', arg=task, timeout=30000)
                        other.screenshot(path=str(output / f'{width}-{task}.png'))
                    assert row.locator('video').evaluate_all('vs=>vs.every(v=>!v.hasAttribute("src"))')
                    row.scroll_into_view_if_needed()
                    page.wait_for_function('[...document.querySelectorAll("[data-policy-task=tennis] video")].every(v=>!v.paused)', timeout=30000)
                    row.locator('video').first.evaluate('v=>v.pause()')
                    page.wait_for_timeout(750)
                    assert row.locator('video').evaluate_all('vs=>vs.every(v=>v.paused)'), 'Native pause must survive a prior unload'
                    page.locator('[data-policy-task=descent]').scroll_into_view_if_needed()
                    page.wait_for_function('[...document.querySelectorAll("[data-policy-task=tennis] video")].every(v=>!v.hasAttribute("src"))')
                    row.scroll_into_view_if_needed()
                    row.locator('video').first.focus()
                    page.keyboard.press('Space')
                    page.wait_for_function('[...document.querySelectorAll("[data-policy-task=tennis] video")].every(v=>!v.paused)', timeout=30000)
                    page.locator('.showcase-viewport').scroll_into_view_if_needed()
                    page.locator('.showcase-pagination button').first.click()
                    page.wait_for_function('document.querySelector(".showcase-track video").currentTime > 1', timeout=15000)
                    page.locator('.showcase-viewport').screenshot(path=str(output / f'{width}-showcase.png'))
                    page.locator('.showcase-pagination button').nth(1).click()
                    page.locator('.showcase-pagination button').first.click()
                    page.wait_for_function('document.querySelector(".showcase-track video").currentTime > .1', timeout=15000)
                    assert page.locator('.showcase-track video').first.evaluate('v=>v.currentTime < 1'), 'Reselecting a clip must restart it'
                page.emulate_media(reduced_motion='reduce')
                page.wait_for_function('[...document.querySelectorAll(".recording-row video")].every(v=>v.paused)')
                if not low:
                    row.scroll_into_view_if_needed()
                    row.locator('.recording-load').first.click()
                    page.wait_for_function('[...document.querySelectorAll("[data-policy-task=tennis] video")].every(v=>!v.paused)', timeout=30000)
                assert not errors, errors
                reports.append(dict(width=width, low_data=low, throttled_mbps=1 if low else None,
                                    initial_video_requests=initial_requests, inline_previews_only=True,
                                    hd_only_on_click=True, startup_seconds=startup_seconds,
                                    playback_advance_over_3_5s=playback_advance, playback_times=before, errors=errors))
                page.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    (output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chromium', required=True)
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium)
