"""Website-only regression checks; optional CPU browser QA writes outside the site."""

import functools
import hashlib
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import unittest


SITE = Path(__file__).resolve().parents[1]
TASKS = ('track', 'stairs', 'forest', 'bench', 'parkour')
MEDIA = 'assets/media/terrain-release/'


class Recordings(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.rows = {}
        self.current = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article' and attrs.get('data-policy-task') in TASKS:
            self.current = {'attrs': attrs, 'videos': [], 'figures': [], 'links': [], 'text': []}
            self.rows[attrs['data-policy-task']] = self.current
        if self.current is not None:
            if tag in ('video', 'figure', 'a'):
                self.current[{'video': 'videos', 'figure': 'figures', 'a': 'links'}[tag]].append(attrs)

    def handle_endtag(self, tag):
        if tag == 'article':
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current['text'].append(data)


class CollisionScenes(unittest.TestCase):
    def test_two_recordings_per_task_and_no_visible_fixed_reference(self):
        rows = Recordings((SITE / 'index.html').read_text()).rows
        self.assertEqual(tuple(rows), TASKS)
        for task, row in rows.items():
            with self.subTest(task=task):
                self.assertEqual([f['data-method'] for f in row['figures']], ['input', 'ours'])
                self.assertEqual(len(row['videos']), 2)
                self.assertEqual(row['attrs']['data-sync'], 'independent')
                self.assertNotIn('Fixed Reference', json.dumps(row))
                expected = [MEDIA + task + '-input.mp4', MEDIA + (task + '-ours.mp4' if task != 'parkour' else 'parkour-mpc.mp4')]
                self.assertEqual([v['src'] for v in row['videos']], expected)
                for video in row['videos']:
                    self.assertTrue((SITE / video['poster']).is_file())
                    self.assertIn(video['src'], [link['href'] for link in row['links']])

    def test_parkour_control_label_and_provenance(self):
        row = Recordings((SITE / 'index.html').read_text()).rows['parkour']
        text = ' '.join(row['text'])
        for phrase in ('MimicX + MPC', 'locally retimed 5.38 s', 'robot joint control', 'diagnostic control demonstration'):
            self.assertIn(phrase, text)
        self.assertNotIn('policy completes', text)
        self.assertEqual(row['videos'][1]['aria-label'], 'MimicX + MPC')
        self.assertFalse(any('parkour-robot.mp4' in link['href'] for link in row['links']))
        path = SITE / MEDIA / 'parkour-mpc.json'
        self.assertTrue(path.is_file(), 'Parkour hybrid-control provenance is missing')
        manifest = json.loads(path.read_text())
        self.assertFalse(manifest['updates_paper_metrics'])
        self.assertFalse(manifest['updates_published_weights'])
        self.assertEqual(manifest['purpose'], 'Diagnostic control demonstration')
        self.assertIn('model predictive control', manifest['control'])
        for media in manifest['media']:
            content = (SITE / MEDIA / media['file']).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), media['sha256'])
        self.assertNotIn('/data/', path.read_text())
        self.assertNotIn('/home/', path.read_text())

    def test_existing_terrain_media_are_unchanged(self):
        manifest = json.loads((SITE / MEDIA / 'manifest.json').read_text())
        for task in manifest['tasks']:
            for media in task['media']:
                with self.subTest(file=media['file']):
                    content = (SITE / MEDIA / media['file']).read_bytes()
                    self.assertEqual(hashlib.sha256(content).hexdigest(), media['sha256'])

    @unittest.skipUnless(os.environ.get('MIMICX_BROWSER_QA_DIR'), 'Set MIMICX_BROWSER_QA_DIR for CPU browser QA')
    def test_responsive_browser_and_playback(self):
        from playwright.sync_api import sync_playwright
        from RangeHTTPServer import RangeRequestHandler

        output = Path(os.environ['MIMICX_BROWSER_QA_DIR']).resolve()
        self.assertFalse(output.is_relative_to(SITE), 'QA artifacts must stay outside the site')
        output.mkdir(parents=True, exist_ok=True)

        class QuietHandler(RangeRequestHandler):
            def log_message(self, *args):
                pass

            def copyfile(self, source, outputfile):
                try:
                    super().copyfile(source, outputfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Browser seeking cancels obsolete range requests.

        server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(SITE)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        report = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    executable_path=os.environ.get('MIMICX_CHROMIUM'),
                    args=['--disable-gpu', '--disable-gpu-compositing', '--disable-webgl',
                          '--disable-accelerated-video-decode', '--disable-accelerated-video-encode'],
                )
                try:
                    page = browser.new_page(reduced_motion='reduce')
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{server.server_port}/', wait_until='networkidle')
                    scene = page.locator('.rx-scene-recordings')
                    for width in (320, 390, 768, 1024, 1440, 1920):
                        with self.subTest(width=width):
                            page.set_viewport_size({'width': width, 'height': 1000})
                            scene.scroll_into_view_if_needed()
                            page.evaluate('document.fonts.ready')
                            boxes = page.locator('.terrain-row .recording-frame').evaluate_all('''elements => elements.map(el => {
                                const r = el.getBoundingClientRect();
                                const v = el.querySelector('video');
                                return {x:r.x, y:r.y, width:r.width, height:r.height, fit:getComputedStyle(v).objectFit};
                            })''')
                            self.assertEqual(len(boxes), 10)
                            self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), width)
                            self.assertNotIn('Fixed Reference', scene.inner_text())
                            for box in boxes:
                                self.assertAlmostEqual(box['width'] / box['height'], 16 / 9, delta=.01)
                                self.assertEqual(box['fit'], 'contain')
                                self.assertGreaterEqual(box['x'], 0)
                                self.assertLessEqual(box['x'] + box['width'], width)
                                self.assertAlmostEqual(box['width'], boxes[0]['width'], delta=1)
                            self.assertAlmostEqual(boxes[0]['y'], boxes[1]['y'], delta=1)
                            if width > 900:
                                self.assertAlmostEqual(boxes[0]['y'], boxes[2]['y'], delta=1)
                                self.assertAlmostEqual(boxes[0]['y'], boxes[3]['y'], delta=1)
                            else:
                                self.assertGreater(boxes[2]['y'], boxes[0]['y'] + boxes[0]['height'])
                            scene.screenshot(path=str(output / f'collision-{width}.png'),
                                             style='.research-nav { visibility: hidden; }')
                            report.append({'width': width, 'frames': boxes})

                    for width in (390, 1440):
                        page.set_viewport_size({'width': width, 'height': 1000})
                        for target, name in ((scene, 'collision'), (page.locator('[data-policy-task="parkour"]'), 'parkour')):
                            target.evaluate('''el => {
                                const nav = document.querySelector('.research-nav').getBoundingClientRect();
                                window.scrollTo({top: window.scrollY + el.getBoundingClientRect().top - nav.height - 18, behavior:'instant'});
                            }''')
                            nav = page.locator('.research-nav').bounding_box()
                            self.assertAlmostEqual(nav['y'], 0, delta=1)
                            self.assertGreaterEqual(target.bounding_box()['y'], nav['height'])
                            self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), width)
                            captions = target.locator('.recording-note, figcaption > span').evaluate_all('''elements => elements.map(el => {
                                const r = document.createRange(); r.selectNodeContents(el);
                                const box = el.getBoundingClientRect();
                                return {right:box.right, lines:Array.from(r.getClientRects(), line => line.right),
                                        client:el.clientWidth, scroll:el.scrollWidth};
                            })''')
                            for caption in captions:
                                self.assertLessEqual(caption['scroll'], caption['client'] + 1)
                                self.assertLessEqual(caption['right'], width)
                                self.assertTrue(all(right <= caption['right'] + 1 for right in caption['lines']))
                            page.screenshot(path=str(output / f'viewport-{name}-{width}.png'))
                            report.append({'viewport': width, 'target': name, 'navigation': nav, 'captions': captions})

                    page.set_viewport_size({'width': 1440, 'height': 1000})
                    parkour = page.locator('[data-policy-task="parkour"]')
                    parkour.scroll_into_view_if_needed()
                    video = parkour.locator('[data-method="ours"] video')
                    video.evaluate('v => {v.load();}')
                    page.wait_for_function('document.querySelector("[data-policy-task=parkour] [data-method=ours] video").readyState >= 2')
                    self.assertAlmostEqual(video.evaluate('v => v.duration'), 5.38, delta=.001)
                    self.assertEqual(video.evaluate('v => [v.videoWidth, v.videoHeight]'), [1920, 1080])
                    samples = []
                    for time in (.6, 4.5):
                        video.evaluate('''(v, time) => new Promise(resolve => {
                            v.requestVideoFrameCallback(resolve); v.currentTime = time;
                        })''', time)
                        self.assertAlmostEqual(video.evaluate('v => v.currentTime'), time, delta=.03)
                        samples.append(video.evaluate('''v => {
                            const c = document.createElement('canvas'); c.width=160; c.height=90;
                            const ctx = c.getContext('2d'); ctx.drawImage(v, 0, 0, 160, 90);
                            return Array.from(ctx.getImageData(0, 0, 160, 90).data);
                        }'''))
                    self.assertGreater(len(set(samples[0])), 50, 'Video must not be blank')
                    self.assertNotEqual(hashlib.sha256(bytes(samples[0])).hexdigest(),
                                        hashlib.sha256(bytes(samples[1])).hexdigest(), 'Video frames must change')
                    parkour.screenshot(path=str(output / 'parkour-decoded.png'))
                    button = parkour.locator('.recording-toggle')
                    button.click()
                    page.wait_for_function('Array.from(document.querySelectorAll("[data-policy-task=parkour] video")).every(v => !v.paused)')
                    button.click()
                    self.assertTrue(parkour.locator('video').evaluate_all('videos => videos.every(v => v.paused)'))
                    parkour.locator('[data-method="ours"] [data-viewer="video"]').click()
                    dialog = page.locator('#media-viewer')
                    self.assertTrue(dialog.evaluate('el => el.open'))
                    self.assertIn('parkour-mpc.mp4', dialog.locator('video').get_attribute('src'))
                    self.assertIn('MimicX + MPC', dialog.inner_text())
                    page.keyboard.press('Escape')
                    page.wait_for_function('!document.getElementById("media-viewer").open')
                    page.locator('#theme-toggle').click()
                    scene.screenshot(path=str(output / 'collision-light-1440.png'),
                                     style='.research-nav { visibility: hidden; }')
                    self.assertEqual(errors, [])
                    (output / 'responsive.json').write_text(json.dumps(report, indent=2) + '\n')
                finally:
                    browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
