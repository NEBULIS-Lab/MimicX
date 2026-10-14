"""Optional CPU physics tests; no viewer or renderer is created."""
import importlib

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

ROBOT = '''<mujoco><worldbody><body name="platform"><freejoint name="root"/>
<geom name="surface" type="box" size="3 3 .02" pos="0 0 -.02" mass="1000"/>
</body></worldbody></mujoco>'''


def engine(speed=0., floor=False, duration=.6):
    api = importlib.import_module("mimicx.visualization.contact_replay")
    root = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, speed * duration, 1, 0, 0, 0.]])
    if not floor:
        root[:, 0] = 100
    return api.BallReplay(ROBOT, [0, duration], root, np.empty((2, 0)), [],
                          config=api.BallConfig(dt=.001, mass=.1, radius=.05, floor_z=None))


def test_freefall_has_no_contacts_and_never_changes_recorded_robot():
    replay = engine(duration=.2)
    before = replay.robot_qpos.copy()
    out = replay.simulate([0, 0, 2, 0, 0, 0])
    np.testing.assert_allclose(out["velocity"][-1], [0, 0, -1.962], atol=1e-8)
    assert out["contact_count"].sum() == 0
    np.testing.assert_array_equal(replay.robot_qpos, before)
    np.testing.assert_array_equal(replay.data.qpos[:-7], before[-1])


def test_rebound_has_normal_force_impulse_and_consistent_time():
    replay = engine(floor=True)
    out = replay.simulate([0, 0, .5, 0, 0, -1])
    change = np.diff(out["velocity"][:, 2])
    assert np.any(change > .05)
    assert np.all(out["contact_count"][change > .05] > 0)
    assert out["position"][:, 2].min() > .045
    api = importlib.import_module("mimicx.visualization.contact_replay")
    audit = api.audit_replay(out, mass=.1)
    assert audit["force_balance_residual_max_m_s"] < 1e-6
    for event in out["contacts"]:
        np.testing.assert_allclose(event["normal_world"], [0, 0, 1], atol=1e-8)
        assert event["time"] == pytest.approx(out["time"][event["step"]])
        assert event["force_world_n"][2] > 0


def test_audit_rejects_contact_duration_that_changes_reported_impulse():
    api = importlib.import_module("mimicx.visualization.contact_replay")
    out = engine(floor=True).simulate([0, 0, .5, 0, 0, -1])
    out["contacts"][0]["dt"] *= 10
    with pytest.raises(ValueError):
        api.audit_replay(out, mass=.1)


def test_moving_recorded_surface_transfers_velocity():
    stationary = engine(floor=True, duration=.1).simulate([0, 0, .16, 0, 0, -1])
    moving = engine(speed=1, floor=True, duration=.1).simulate([0, 0, .16, 0, 0, -1])
    assert moving["velocity"][-1, 2] > stationary["velocity"][-1, 2] + .5


def test_non_grid_aligned_replay_keeps_exact_last_timestamp():
    replay = engine(duration=.1035)
    out = replay.simulate([0, 0, 2, 0, 0, 0], start=.0005)
    assert out["time"][0] == pytest.approx(.0005)
    assert out["time"][-1] == pytest.approx(.1035)
    np.testing.assert_allclose(out["velocity"][-1, 2], -9.81 * .103, atol=1e-8)


def test_named_joint_mapping_reorders_qpos_and_rejects_missing():
    api = importlib.import_module("mimicx.visualization.baseline_recording")
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
    <geom size=".1"/><body><joint name="hip"/><geom size=".1"/>
    <body><joint name="knee"/><geom size=".1"/></body></body></body></worldbody></mujoco>''')
    pos, vel = api.named_joint_addresses(model, ["knee", "hip"])
    np.testing.assert_array_equal(pos, [8, 7]); np.testing.assert_array_equal(vel, [7, 6])
    with pytest.raises(ValueError):
        api.named_joint_addresses(model, ["missing", "hip"])


def test_fit_adjusts_initial_state_only_with_bounded_reproducible_search():
    api = importlib.import_module("mimicx.visualization.contact_replay")
    replay = engine(duration=.2)
    before = replay.robot_qpos.copy()
    observed = replay.simulate([.05, 0, 2, .1, 0, 0])
    kwargs = dict(lower=[-.1, 0, 2, -.2, 0, 0], upper=[.1, 0, 2, .2, 0, 0],
                  rounds=2, population=8, seed=5)
    fitted = api.fit_initial_conditions(replay, observed["time"], observed["position"], **kwargs)
    again = api.fit_initial_conditions(replay, observed["time"], observed["position"], **kwargs)
    assert fitted["position_rmse_m"] < .05
    assert fitted == again
    assert fitted["trials"] == 16
    np.testing.assert_array_equal(replay.robot_qpos, before)


def test_contact_cli_exports_and_independently_audits(tmp_path):
    import json
    import subprocess
    import sys
    (tmp_path / "robot.xml").write_text(ROBOT)
    root = np.tile([100., 0, 0, 1, 0, 0, 0], (2, 1))
    np.savez(tmp_path / "robot.npz", timestamps=[0., .1], robot_root_pose=root,
             robot_joint_positions=np.empty((2, 0)), joint_names=np.array([], dtype=str))
    config = dict(model="robot.xml", robot="robot.npz", initial=[0, 0, 2, 0, 0, 0],
                  physics=dict(dt=.001, mass=.1, radius=.05, floor_z=None))
    (tmp_path / "config.json").write_text(json.dumps(config))
    script = "scripts/visualization/contact_replay.py"
    result = subprocess.run([sys.executable, script, "simulate", "--config", str(tmp_path / "config.json"),
                             "--output", str(tmp_path / "out")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    audit = subprocess.run([sys.executable, script, "audit", "--input", str(tmp_path / "out")],
                           capture_output=True, text=True)
    assert audit.returncode == 0, audit.stderr
    assert json.loads(audit.stdout)["unexplained_velocity_change_substeps"] == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["robot_input_sha256"]
    assert manifest["policy_evaluation"] is False
    # Independently corrupted artifact must be rejected, even if still valid JSON.
    (tmp_path / "out" / "contacts.json").write_text("[{}]")
    bad = subprocess.run([sys.executable, script, "audit", "--input", str(tmp_path / "out")],
                         capture_output=True, text=True)
    assert bad.returncode != 0


def test_recorder_body_velocity_is_about_each_body_origin():
    from mimicx.visualization.baseline_recording import StateRecorder
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="base"><freejoint/><geom size=".1" mass="1"/>
    <body name="tip" pos="2 0 0"><geom size=".1" mass="1"/></body>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[5] = 1
    mujoco.mj_forward(model, data)
    recorder = StateRecorder(model, [], body_names=['base', 'tip'])
    recorder.append(data, monotonic_time=10., elastic_band_enabled=False)
    np.testing.assert_allclose(recorder.samples[0]['body_lin_vel'], [[0, 0, 0], [0, 2, 0]], atol=1e-12)
    np.testing.assert_allclose(recorder.samples[0]['body_ang_vel'], [[0, 0, 1], [0, 0, 1]], atol=1e-12)
