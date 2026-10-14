"""CPU-only MuJoCo integration, skipped in the lightweight package environment."""
import importlib
import json
import os

import numpy as np
import pytest


@pytest.fixture
def fixture():
    mj = pytest.importorskip("mujoco")
    names = ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint"]
    limbs = "".join(f'<body pos="{i * .1} 0 0"><joint name="robot/{name}"/>'
                    '<geom size=".05" mass="1"/></body>' for i, name in enumerate(names))
    actuators = "".join(f'<position joint="robot/{name}" kp="10" kv="1"/>' for name in reversed(names))
    model = mj.MjModel.from_xml_string(
        '<mujoco><option timestep=".005" gravity="0 0 0" integrator="implicitfast"/>'
        '<worldbody><geom type="plane" size="0 0 .1"/>'
        '<body name="robot/pelvis" pos="0 0 1"><freejoint/>'
        '<geom size=".1" mass="2"/>' + limbs +
        '<body name="robot/torso_link" pos="0 0 .2"><geom size=".1" mass="1"/></body>'
        '<body name="robot/left_ankle_roll_link" pos="0 .1 -.2"><geom size=".05" mass="1"/></body>'
        '<body name="robot/right_ankle_roll_link" pos="0 -.1 -.2"><geom size=".05" mass="1"/></body>'
        '</body></worldbody><actuator>' + actuators + '</actuator></mujoco>')
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    selected = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
    ids = [model.body("robot/" + n).id for n in selected]
    motion = dict(joint_pos=np.zeros((270, 4)), joint_vel=np.zeros((270, 4)),
                  body_pos_w=np.repeat(data.xpos[ids][None], 270, axis=0),
                  body_quat_w=np.repeat(data.xquat[ids][None], 270, axis=0),
                  body_ang_vel_w=np.zeros((270, 4, 3)), body_lin_vel_w=np.zeros((270, 4, 3)))
    return mj, model, data, dict(joint_names=names, body_names=selected,
                                 reference_body_names=selected), motion


def test_real_planner_rollout_replans_and_preserves_observed_state(fixture):
    _, model, data, names, motion = fixture
    p = importlib.import_module("mimicx.parkour.predictive")
    affine = importlib.import_module("mimicx.parkour.servo").ActionAffine(names["joint_names"], 1., 0.)
    cfg = p.PlannerConfig(start=0, horizon=4, population=5, generations=0, threads=1)
    before = data.qpos.copy()
    with p.Planner(model, motion, names, affine, np.zeros((269, 4)), cfg) as planner:
        a = planner.step(0, data.qpos, data.qvel)
        assert a.shape == (4,) and np.isfinite(a).all()
        assert planner.history[0]["rollouts"] == 5
        planner.step(1, data.qpos, data.qvel)
        assert len(planner.history) == 1
        planner.step(3, data.qpos, data.qvel)
        assert len(planner.history) == 2
        assert len(planner.prediction_errors) == 1  # Only consecutive predictions are comparable.
    np.testing.assert_array_equal(data.qpos, before)


def test_state_replay_is_exact_but_never_classified_as_physical_execution(fixture):
    _, model, data, names, motion = fixture
    h = importlib.import_module("mimicx.parkour.hybrid")
    root = np.tile(data.qpos[:7], (270, 1, 1))
    root[:, 0, 0] = np.linspace(0, .1, 270)
    traces = dict(root_pose=root, joint_pos=np.zeros((270, 1, 4)), active=np.ones((270, 1), bool))
    result = h.replay_states(model, traces, names, motion, index=0)
    np.testing.assert_array_equal(result["qpos"][:, :7], root[:, 0])
    assert result["report"]["execution"] == "kinematic_state_replay"
    assert result["report"]["physical_execution_verified"] is False


def test_physical_servo_execution_uses_only_reset_pose(fixture):
    _, model, data, names, motion = fixture
    h = importlib.import_module("mimicx.parkour.hybrid")
    affine = importlib.import_module("mimicx.parkour.servo").ActionAffine(names["joint_names"], 1., 0.)
    targets = np.tile([.2, -.1, .1, -.2], (269, 1))
    result = h.run_cpu(model, motion, names, affine, targets, data.qpos, data.qvel, steps=3)
    assert result["qpos"].shape == (4, model.nq)
    assert np.max(np.abs(result["qpos"][-1, 7:])) > 0
    assert not result["report"]["completion"]["horizon_complete"]
    assert result["report"]["pose_writes_after_reset"] == 0


def test_world_penetration_and_xml_model_loading(fixture, tmp_path):
    mj, model, data, _, _ = fixture
    servo = importlib.import_module("mimicx.parkour.servo")
    p = importlib.import_module("mimicx.parkour.predictive")
    path = tmp_path / "model.mjb"
    mj.mj_saveModel(model, str(path))
    assert servo.load_model(model_mjb=path).nu == 4
    xml = tmp_path / "model.xml"
    mj.mj_saveLastXML(str(xml), model)
    assert servo.load_model(model_xml=xml).nu == 4
    with pytest.raises(ValueError, match="exactly one"):
        servo.load_model()
    data.qpos[2] = .02
    mj.mj_forward(model, data)
    assert p.world_penetration(model, data) > .01


def test_initial_state_rotates_world_angular_velocity_to_local(fixture):
    _, model, _, names, motion = fixture
    h = importlib.import_module("mimicx.parkour.hybrid")
    motion["body_quat_w"][0, 0] = [2**-.5, 0, 0, 2**-.5]
    motion["body_ang_vel_w"][0, 0] = [1, 0, 0]
    motion["body_lin_vel_w"][0, 0] = [2, 3, 4]
    _, velocity = h.initial_state(model, motion, names)
    np.testing.assert_allclose(velocity[:6], [2, 3, 4, 0, -1, 0], atol=1e-12)


def test_cpu_callback_changes_only_selected_active_action(fixture):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    _, model, data, names, motion = fixture
    p = importlib.import_module("mimicx.parkour.predictive")
    affine = importlib.import_module("mimicx.parkour.servo").ActionAffine(names["joint_names"], 1., 0.)
    cfg = p.PlannerConfig(start=1, horizon=3, population=5, generations=0, threads=1)
    base = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(
        qpos=torch.tensor(np.tile(data.qpos, (2, 1)), device="cpu"),
        qvel=torch.tensor(np.tile(data.qvel, (2, 1)), device="cpu"))))
    action = SimpleNamespace(target_names=names["joint_names"], scale=1., offset=0.)
    actions = torch.ones((2, 4), device="cpu")
    with p.Planner(model, motion, names, affine, np.zeros((269, 4)), cfg) as planner:
        assert planner.callback(0, actions, base, None, action, [True, True]) is actions
        assert planner.callback(1, actions, base, None, action, [True, False]) is actions
        result = planner.callback(1, actions, base, None, action, [True, True])
        np.testing.assert_array_equal(result[0], [1, 1, 1, 1])
        assert not torch.equal(result[1], actions[1])
        np.testing.assert_array_equal(actions, np.ones((2, 4)))


def test_cpu_evaluation_rejects_different_planner_clock(fixture):
    _, model, data, names, motion = fixture
    p = importlib.import_module("mimicx.parkour.predictive")
    h = importlib.import_module("mimicx.parkour.hybrid")
    affine = importlib.import_module("mimicx.parkour.servo").ActionAffine(names["joint_names"], 1., 0.)
    cfg = p.PlannerConfig(start=0, horizon=3, control_dt=.01, substeps=2, population=5, generations=0, threads=1)
    with p.Planner(model, motion, names, affine, np.zeros((269, 4)), cfg) as planner:
        with pytest.raises(ValueError, match="clock"):
            h.run_cpu(model, motion, names, affine, np.zeros((269, 4)), data.qpos, data.qvel, planner=planner, steps=1)


def test_native_cpu_model_build_when_assets_are_provided(tmp_path):
    pytest.importorskip("mujoco")
    scene = os.environ.get("MIMICX_PARKOUR_SCENE")
    motion = os.environ.get("MIMICX_PARKOUR_MOTION")
    if not scene or not motion:
        pytest.skip("Set MIMICX_PARKOUR_SCENE/MOTION and runtime PYTHONPATH for the native build check")
    servo = importlib.import_module("mimicx.parkour.servo")
    model = servo.prepare_model(scene, motion, tmp_path / "build")
    assert (model.nq, model.nv, model.nu, model.na) == (36, 35, 29, 0)
    assert model.opt.timestep == .005
    affine = servo.ActionAffine(**json.loads((tmp_path / "build/affine.json").read_text()))
    assert affine.to_controls(np.zeros(29), servo.actuator_names(model)).shape == (29,)
    np.testing.assert_allclose(model.geom("parkour_ground").friction, [.8, .005, .0001])
    recording = os.environ.get("MIMICX_PARKOUR_RECORDING")
    if recording:
        from pathlib import Path
        h = importlib.import_module("mimicx.parkour.hybrid")
        with np.load(Path(recording) / "traces.npz", allow_pickle=False) as archive:
            traces = dict(archive)
        with np.load(motion, allow_pickle=False) as archive:
            reference = dict(archive)
        names = json.loads((Path(recording) / "names.json").read_text())
        result = h.replay_states(model, traces, names, reference)
        np.testing.assert_array_equal(result["qpos"][:, :7], traces["root_pose"][:, 2])
        np.testing.assert_allclose(result["completion_metrics"], traces["completion_metrics"][:, 2], atol=2e-5)
        assert result["report"]["completion"]["motion_complete"]
        assert result["report"]["contacts"]["max_sampled_penetration_m"] == pytest.approx(.05344, abs=1e-6)


def test_cpu_termination_uses_pre_advance_reference_but_scores_next_frame(fixture):
    _, model, data, names, motion = fixture
    from mimicx.parkour import hybrid, servo
    motion['body_pos_w'][1, :, 2] += .5
    affine = servo.ActionAffine(names['joint_names'], 1., 0.)
    result = hybrid.run_cpu(model, motion, names, affine, np.zeros((269, 4)),
                            data.qpos, data.qvel, steps=1)
    assert result['report']['termination_channels'] == []
    assert result['completion_metrics'][-1, 5] > .4
