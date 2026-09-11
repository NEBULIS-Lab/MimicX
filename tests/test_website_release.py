import hashlib
from html.parser import HTMLParser
import json
import struct
from pathlib import Path
from urllib.parse import unquote, urlsplit

from scripts.build_website_results import build
from scripts.package_website_media import MEDIA, PROMPT_MEDIA

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "docs"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.ids.add(attrs.get("id", ""))
        self.targets.extend(attrs[key] for key in ("href", "src", "poster") if key in attrs)


def test_website_local_links_and_fragments_exist():
    parser = Links()
    parser.feed((WEBSITE / "index.html").read_text())
    for target in parser.targets:
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        if url.path:
            assert (WEBSITE / unquote(url.path)).is_file(), target
        elif url.fragment:
            assert url.fragment in parser.ids, target


def test_media_match_source_manifest():
    directory = WEBSITE / "assets/media"
    records = json.loads((directory / "manifest.json").read_text())
    assert {row['file'] for row in records} == set(MEDIA) | set(PROMPT_MEDIA)
    for row in records:
        content = (directory / row["file"]).read_bytes()
        assert len(content) == row["bytes"]
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
    for task in ("tennis", "football", "dance", "kungfu"):
        for method in ("fixed", "ours"):
            assert (directory / f"{task}-{method}.mp4").is_file()


def test_website_tables_are_generated_from_all_core_tasks(tmp_path):
    source = (WEBSITE / "index.html").read_text()
    page = tmp_path / "index.html"
    page.write_text(source)
    build(page, WEBSITE / "assets/results")
    assert page.read_text() == source, "Rebuild the website after changing numerical CSVs"
    table = source.split('<tbody id="core-results">')[1].split('</tbody>')[0]
    assert table.count('<tr') == 8
    for task in ("Tennis Swing", "Football Juggling", "Dance Sequence", "Kung Fu Sequence"):
        assert table.count(task) == 2


def test_results_are_website_only_and_template_has_no_trackers():
    assert not (ROOT / "benchmarks").exists()
    assert not (ROOT / "results").exists()
    assert "Result Snapshot" not in (ROOT / "README.md").read_text()
    source = (WEBSITE / "index.html").read_text()
    assert "RoboSplat" in source
    assert "googletagmanager" not in source
    assert "google-analytics" not in source


def test_core_asset_manifest_is_path_free():
    payload = json.loads((ROOT / "configs/paper/core_inputs.json").read_text())
    assert len(payload["tasks"]) == 4
    assert len(payload["assets"]) == 22
    assert payload["seeds"] == [101, 202, 303]
    paths = {row["path"] for row in payload["assets"]}
    for row in payload["assets"]:
        assert not Path(row["path"]).is_absolute()
        assert ".." not in Path(row["path"]).parts
        assert len(row["sha256"]) == 64
    for task in payload["tasks"]:
        for key in ("motion_file", "base_checkpoint", "hloop_checkpoint"):
            assert task[key] in paths


def test_huggingface_links_are_in_header_and_resources():
    source = (WEBSITE / "index.html").read_text()
    header = source.split('<nav class="top-nav"')[1].split('</nav>')[0]
    resources = source.split('id="resources"')[1].split('</section>')[0]
    for url in ("https://huggingface.co/Shuaijun/MimicX-Policies",
                "https://huggingface.co/datasets/Shuaijun/MimicX-Assets"):
        assert f'href="{url}"' in header
        assert f'href="{url}"' in resources
    assert "recommended/README.md" in resources


def test_static_hero_clean_workflow_and_four_result_plots():
    source = (WEBSITE / "index.html").read_text()
    hero = source.split('<header')[1].split('</header>')[0]
    assert '<video' not in hero
    assert 'id="hero-image"' in hero
    assert 'id="method-overview"' in source
    assert source.count('class="result-plot"') == 4
    assert 'Inspect the method. Run the loop.' not in source
    assert 'hero-video' not in (WEBSITE / 'assets/js/mimicx.js').read_text()
    for name in ('stage-human.png', 'stage-world.png', 'stage-reference.png'):
        assert MEDIA[name].endswith('__ALPHA.png')
    assert all('PIPELINE__V9__' not in value for value in MEDIA.values())


def test_transparent_workflow_derivatives_have_recorded_sources():
    directory = WEBSITE / 'assets/media'
    records = {r['file']: r for r in json.loads((directory / 'manifest.json').read_text())}
    for name in ('stage-human.png', 'stage-world.png', 'stage-reference.png'):
        content = (directory / name).read_bytes()
        assert content[:8] == b'\x89PNG\r\n\x1a\n'
        assert struct.unpack('>II', content[16:24]) == (1200, 900)
        assert content[25] == 6, 'Workflow PNG must retain RGBA transparency'
        assert len(records[name]['source_sha256']) == 64
        assert len(records[name]['alpha_bounds']) == 4
