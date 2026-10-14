import importlib.util
from pathlib import Path


def audit_module():
    spec = importlib.util.spec_from_file_location('audit', Path(__file__).parents[1] / 'scripts/check_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_code_scope_skips_website_not_guides(tmp_path, monkeypatch):
    module = audit_module()
    (tmp_path / 'docs/assets').mkdir(parents=True)
    (tmp_path / 'docs/assets/font.woff2').write_bytes(b'\x80')
    (tmp_path / 'docs/guide.md').write_text('portable documentation')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: b'docs/assets/font.woff2\0docs/guide.md\0')
    assert module.main(['--scope', 'code']) == 0
    (tmp_path / 'docs/guide.md').write_text('/' + 'home' + '/test/private.py')
    assert module.main(['--scope', 'code']) == 1


def test_unexpected_binary_is_reported(tmp_path, monkeypatch, capsys):
    module = audit_module()
    (tmp_path / 'payload.bin').write_bytes(b'\x80')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: b'payload.bin\0')
    assert module.main(['--scope', 'code']) == 1
    assert 'Unexpected binary' in capsys.readouterr().out


def test_nested_guides_are_scanned(tmp_path, monkeypatch):
    module = audit_module()
    path = tmp_path / 'docs/reproduction/ENV.md'
    path.parent.mkdir(parents=True)
    path.write_text('/' + 'home' + '/test/private.py')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: b'docs/reproduction/ENV.md\0')
    assert module.main(['--scope', 'code']) == 1
