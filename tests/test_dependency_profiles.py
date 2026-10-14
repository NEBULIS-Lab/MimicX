import importlib.util
from pathlib import Path
import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/setup_dependencies.py'
    spec = importlib.util.spec_from_file_location('setup_profiles', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_profiles_are_explicit_and_disjoint(tmp_path, monkeypatch):
    mod = module()
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **kw: None)
    mod.setup('unitree_rl_mjlab', tmp_path / 'paper')
    mod.setup('unitree_rl_mjlab', tmp_path / 'demos', profile='demonstrations')
    assert (tmp_path / 'paper/src/tasks/tracking/mdp/commands.py').is_file()
    assert (tmp_path / 'demos/src/tasks/tracking/rl/autorefine_lora.py').is_file()
    assert not (tmp_path / 'paper/src/tasks/tracking/rl/autorefine_lora.py').exists()


def test_missing_profile_rejected_before_clone(tmp_path, monkeypatch):
    mod = module()
    def unexpected(*a, **kw):
        pytest.fail('clone was started for an unavailable profile')
    monkeypatch.setattr(mod.subprocess, 'run', unexpected)
    with pytest.raises(ValueError, match='profile'):
        mod.setup('GMR', tmp_path / 'clone', profile='demonstrations')
