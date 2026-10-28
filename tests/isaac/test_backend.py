"""CPU-only contracts; these tests do not certify physical Isaac execution."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from mimicx.adapters.isaac_motion import convert_motion, validate_motion, names
from mimicx.backends.isaaclab.configuration import apply_patch, policy_observations, reapply_learning_rate
from mimicx.backends.isaaclab.compat import (
    prepare_upstream_import, torch_view, TorchArticulationView,
    canonical_to_native_quaternion, native_to_canonical_quaternion,
    contact_paths_by_env,
)
from mimicx.backends.isaaclab.runner import runner_config, main as isaac_main
from mimicx.backends.isaaclab.telemetry import RolloutTelemetry
from mimicx.refinement.closed_loop import ClosedLoopCoordinator
from mimicx.refinement.gate import load_rollout_metrics
from mimicx.refinement.manifest import load_loop_manifest
from mimicx.refinement.manifest import hash_file
from mimicx.refinement.executor import CommandRecord, run_command
from mimicx.runtime.backends import isaac_rollout_outputs
from mimicx.refinement.proposals import candidate_patch
from tests.autorefine.test_closed_loop import write_loop_manifest


def test_noise_rename_bridge_is_idempotent_and_preserves_legacy():
    replacement = object()
    current = NS(UniformNoiseCfg=replacement)
    assert prepare_upstream_import(noise_module=current)
    assert current.AdditiveUniformNoiseCfg is replacement
    assert prepare_upstream_import(noise_module=current) == ()
    legacy = NS(AdditiveUniformNoiseCfg=object(), UniformNoiseCfg=replacement)
    before = legacy.AdditiveUniformNoiseCfg
    assert prepare_upstream_import(noise_module=legacy) == ()
    assert legacy.AdditiveUniformNoiseCfg is before
    with pytest.raises(ImportError, match="neither"):
        prepare_upstream_import(noise_module=NS())


def test_native_array_and_quaternion_boundary():
    quaternion = np.array([[1., 0., 0., 0.], [.70710678, 0., 0., .70710678]])
    native = canonical_to_native_quaternion(quaternion, xyzw=True)
    assert native[0].tolist() == [0, 0, 0, 1]
    np.testing.assert_array_equal(native_to_canonical_quaternion(native, xyzw=True), quaternion)
    assert canonical_to_native_quaternion(quaternion, xyzw=False) is quaternion
    link_velocity, com_velocity = object(), object()
    calls = []
    asset = NS(data=NS(joint_pos=NS(torch=quaternion),
                       body_lin_vel_w=NS(torch=com_velocity),
                       body_link_lin_vel_w=NS(torch=link_velocity)),
               write_root_link_state_to_sim=lambda state, **kw: calls.append((state, kw)))
    proxy = TorchArticulationView(asset)
    assert proxy.data.joint_pos is quaternion
    assert proxy.data.body_lin_vel_w is link_velocity
    assert torch_view(quaternion) is quaternion
    asset.data.joint_pos = quaternion  # legacy tensor path does not need conversion
    proxy.write_root_state_to_sim("native-state", env_ids=[0])
    assert calls == [("native-state", {"env_ids": [0]})]


def test_lab3_reset_translates_link_velocity_and_uses_keyword_only_api(monkeypatch):
    class Array(np.ndarray):
        def clone(self): return self.copy()
    # Exercise the backend boundary without importing Torch, Warp or Kit.
    monkeypatch.setitem(sys.modules, "torch", NS(linalg=NS(cross=lambda a, b, dim: np.cross(a, b, axis=dim))))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", NS(quat_apply=lambda q, v: np.array([[0., 1., 0.]])))
    calls = {}
    def write_pose(*, root_pose, env_ids): calls["pose"] = root_pose.copy()
    def write_velocity(*, root_velocity, env_ids): calls["velocity"] = root_velocity.copy()
    asset = NS(data=NS(joint_pos=NS(torch=object()), body_com_pos_b=NS(torch=np.array([[[1., 0., 0.]]]))),
               write_root_link_pose_to_sim_index=write_pose,
               write_root_com_velocity_to_sim_index=write_velocity)
    state = np.array([[1, 2, 3, 0, 0, .70710678, .70710678, 2, 3, 4, 0, 0, 2]], dtype=float).view(Array)
    original = state.copy()
    TorchArticulationView(asset).write_root_state_to_sim(state, env_ids=[0])
    np.testing.assert_array_equal(calls["pose"], state[:, :7])
    np.testing.assert_allclose(calls["velocity"], [[0, 3, 4, 0, 0, 2]])
    np.testing.assert_array_equal(state, original)


def test_nested_contact_paths_preserve_hierarchy_and_env_major_order():
    paths = ["/World/envs/env_.*/Robot/Geometry/pelvis",
             "/World/envs/env_.*/Robot/Geometry/pelvis/hip/knee/ankle"]
    result = contact_paths_by_env(paths, 2)
    assert result == [path.replace("env_.*", f"env_{i}") for i in range(2) for path in paths]
    with pytest.raises(ValueError, match="namespace"):
        contact_paths_by_env(["/Other/Robot/pelvis"], 2)
    with pytest.raises(ValueError, match="unique"):
        contact_paths_by_env(paths + paths, 2)


@pytest.mark.parametrize("backend", ["mjlab", "isaaclab", "typo"])
def test_manifest_backend_selection(tmp_path, backend):
    path = write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=.2)
    payload = yaml.safe_load(path.read_text())
    assert load_loop_manifest(path).runtime.backend == "mjlab"
    payload["runtime"]["backend"] = backend
    path.write_text(yaml.safe_dump(payload))
    if backend == "typo":
        with pytest.raises(ValueError, match="Unsupported runtime backend"):
            load_loop_manifest(path)
    else:
        assert load_loop_manifest(path).runtime.backend == backend


def motion():
    return dict(fps=np.array(50.0), joint_names=np.array(["a", "b"]),
                body_names=np.array(["root", "tip"]),
                joint_pos=np.tile([1., 2.], (5, 1)), joint_vel=np.zeros((5, 2)),
                body_pos_w=np.zeros((5, 2, 3)), body_quat_w=np.tile([0., 0., 0., 1.], (5, 2, 1)),
                body_lin_vel_w=np.zeros((5, 2, 3)), body_ang_vel_w=np.zeros((5, 2, 3)))


def test_named_reorder_quaternion_and_no_overwrite(tmp_path):
    source, output = tmp_path / "source.npz", tmp_path / "output.npz"
    np.savez(source, **motion())
    before = source.read_bytes()
    kwargs = dict(target_joints=["b", "a"], target_bodies=["tip", "root"], quaternion_order="xyzw")
    convert_motion(source, output, **kwargs)
    assert source.read_bytes() == before
    with np.load(output) as data:
        validate_motion(data, joint_names=["b", "a"], body_names=["tip", "root"], fps=50)
        assert data["joint_pos"][0].tolist() == [2, 1]
        assert data["body_quat_w"][0, 0].tolist() == [1, 0, 0, 0]
        assert data["velocity_reference"].item() == "link_origin"
    with pytest.raises(FileExistsError):
        convert_motion(source, output, **kwargs)
    with pytest.raises(FileExistsError):
        convert_motion(source, source, **kwargs)


@pytest.mark.parametrize("defect", ["unnamed", "rotation", "timing", "shape", "nan", "duplicate", "missing_body"])
def test_motion_rejects_invalid_inputs(tmp_path, defect):
    data = motion()
    kwargs = dict(target_joints=["a", "b"], target_bodies=["root", "tip"])
    if defect == "unnamed":
        del data["joint_names"]
    elif defect == "rotation":
        data["body_quat_w"] *= 2
    elif defect == "timing":
        data["timestamps"] = np.array([0, .02, .04, .05, .08])
    elif defect == "shape":
        data["body_pos_w"] = np.zeros((5, 1, 3))
    elif defect == "nan":
        data["joint_vel"][0, 0] = np.nan
    elif defect == "duplicate":
        data["joint_names"] = np.array(["a", "a"])
    else:
        kwargs["target_bodies"] = ["root"]
    source = tmp_path / "source.npz"
    np.savez(source, **data)
    with pytest.raises(ValueError):
        convert_motion(source, tmp_path / "out.npz", **kwargs)


@pytest.mark.parametrize("field,value", [("quaternion_order", "xyzw"), ("units", "cm"),
                                         ("angular_velocity_frame", "body"), ("velocity_reference", "com")])
def test_conversion_cannot_relabel_conflicting_conventions(tmp_path, field, value):
    source = tmp_path / "source.npz"
    np.savez(source, **motion(), **{field: np.asarray(value)})
    with pytest.raises(ValueError, match="contradicts input metadata"):
        convert_motion(source, tmp_path / "out.npz", target_joints=["a", "b"], target_bodies=["root", "tip"])
    assert not (tmp_path / "out.npz").exists()


def test_runtime_order_and_frequency_rejected():
    assert names(np.array([b"left", b"right"]), "joints") == ["left", "right"]
    with pytest.raises(ValueError, match="articulation"):
        validate_motion(motion(), joint_names=["b", "a"])
    with pytest.raises(ValueError, match="control frequency"):
        validate_motion(motion(), fps=60)


@pytest.mark.parametrize("collision", ["metrics", "csv", "states", "marker", "input", "hardlink"])
def test_rollout_paths_reject_collisions_and_input_overwrite(tmp_path, collision):
    metrics, steps, states, marker = isaac_rollout_outputs(tmp_path / "metrics.json", tmp_path / "steps.csv")
    source = tmp_path / "motion.npz"
    source.write_bytes(b"input")
    if collision in {"metrics", "csv"}:
        steps = metrics
    elif collision == "states":
        steps = states
    elif collision == "marker":
        steps = marker
    elif collision == "input":
        metrics = source
    else:
        metrics.hardlink_to(source)
    with pytest.raises(ValueError, match="cannot overwrite"):
        isaac_rollout_outputs(metrics, steps, protected_paths=[source])
    assert source.read_bytes() == b"input"


def test_invalid_task_and_unsafe_rollout_fail_before_optional_import(tmp_path):
    with pytest.raises(ValueError, match="Use a MimicX task"):
        isaac_main("train", ["--task", "not-a-task", "--motion-file", "missing", "--output-root", str(tmp_path), "--run-name", "invalid"])
    source = tmp_path / "motion.npz"
    np.savez(source, **motion(), units="SI", quaternion_order="wxyz", angular_velocity_frame="world")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="cannot overwrite"):
        isaac_main("rollout", ["--motion-file", str(source), "--checkpoint-file", str(checkpoint),
                               "--steps", "4", "--metrics-output", str(checkpoint),
                               "--step-metrics-output", str(tmp_path / "steps.csv")])
    assert checkpoint.read_bytes() == b"checkpoint"


def test_reward_translation_and_resume_lr():
    patch = candidate_patch("cpu_patch", {"mode": "window", "start": 10, "end": 20, "ratio": .5}, [], [])
    rewards = NS(**{key: NS(weight=1, params={"std": 1}) for key in
                   ("motion_global_anchor_pos", "motion_global_anchor_ori", "motion_body_pos", "motion_body_ori", "action_rate_l2")})
    cfg = NS(rewards=rewards, commands=NS(motion=NS(body_names=["pelvis", "torso_link"])))
    functions = {key: object() for key in ("body_pos_rewards", "body_z_rewards", "body_lin_vel_rewards", "body_ang_vel_rewards")}
    apply_patch(cfg, patch, frames=101, reward_factory=lambda **kw: NS(**kw), functions=functions)
    assert rewards.motion_global_anchor_pos.weight == 5.2
    assert rewards.mimicx_body_pos_rewards_0.params["body_names"] == ["pelvis", "torso_link"]
    assert cfg.commands.motion.mimicx_sampling["ratio"] == .5
    runner = NS(alg=NS(learning_rate=.1, optimizer=NS(param_groups=[{"lr": .1}, {"lr": .2}])))
    reapply_learning_rate(runner, 2e-6)
    assert runner.alg.learning_rate == 2e-6
    assert all(group["lr"] == 2e-6 for group in runner.alg.optimizer.param_groups)


def test_versioned_runner_configuration():
    kwargs = dict(seed=2, device="cpu", learning_rate=1e-4)
    new = runner_config(rsl_version="5.0.1", **kwargs)
    assert new["actor"]["distribution_cfg"]["class_name"] == "GaussianDistribution"
    assert new["obs_groups"] == {"actor": ["policy"], "critic": ["critic"]}
    assert new["algorithm"]["rnd_cfg"] is None
    assert "policy" not in new and "wandb" not in json.dumps(new)
    assert "policy" in runner_config(rsl_version="2.3.0", **kwargs)
    with pytest.raises(ValueError, match="Unaudited"):
        runner_config(rsl_version="3.0.0", **kwargs)
    observation = {"policy": [1]}
    assert policy_observations((observation, {})) is observation
    assert policy_observations(observation) is observation


@pytest.mark.parametrize("script", ["train", "rollout", "convert_motion"])
def test_cli_help_without_simulator(script):
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, str(root / "scripts/isaac" / f"{script}.py"), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout


@pytest.mark.parametrize("accepted", [True, False])
def test_cpu_isaac_routed_complete_loop(tmp_path, accepted):
    path = write_loop_manifest(tmp_path, candidate_frontier=90, candidate_body=.2, executable_runtime=True)
    manifest = load_loop_manifest(path)
    script = manifest.runtime.train_script
    script.write_text(script.read_text().replace('parser.add_argument("task")', 'parser.add_argument("--task")')
                      .replace('"--agent.run-name"', '"--run-name"').replace('"agent.run_name"', '"run_name"')
                      + '\n(root / f"{run_name}.completed.json").write_text("{}")\n')
    if not accepted:
        rollout = manifest.runtime.rollout_script
        rollout.write_text(rollout.read_text().replace('0.19, 1.4', '0.90, 1.4'))
    rollout = manifest.runtime.rollout_script
    rollout.write_text(rollout.read_text() + '\nargs.metrics_output.with_suffix(".states.npz").write_bytes(b"fake-state")\n'
                       + 'args.metrics_output.with_suffix(".completed.json").write_text("{}")\n')
    manifest = replace(manifest, runtime=replace(manifest.runtime, backend="isaaclab"),
                       search=replace(manifest.search, max_iterations=1))
    coordinator = ClosedLoopCoordinator(manifest, tmp_path / "loop")
    record = coordinator._train_record({"id": "a", "run_name": "candidate", "patch_file": "patch.json"})
    assert "--checkpoint-file" in record.argv and "--agent.resume" not in record.argv
    assert "--headless" in coordinator._rollout_command(1, manifest.task.base_checkpoint)["argv"]
    assert len(coordinator._rollout_command(1, manifest.task.base_checkpoint)["expected_outputs"]) == 4
    result = coordinator.execute()
    assert result.accepted is accepted
    state = json.loads((tmp_path / "loop/state.json").read_text())
    if not accepted:
        assert state["incumbent"]["id"] == "incumbent"
    else:
        assert state["incumbent"]["id"].startswith("auto_01")


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def item(self): return self.value.item()
    def __getitem__(self, index): return Tensor(self.value[index])
    def __sub__(self, other): return Tensor(self.value - other.value)


def test_telemetry_preserves_terminal_state_for_gate_and_replay(tmp_path):
    command = NS(body_pos_w=Tensor([[[0, 0, 0], [0, 0, 1]]]),
                 robot_body_pos_w=Tensor([[[1, 0, 0], [1, 0, 1]]]),
                 body_pos_relative_w=Tensor([[[0, 0, 0], [0, 0, 1]]]),
                 anchor_pos_w=Tensor([[0, 0, 0]]), robot_anchor_pos_w=Tensor([[1, 0, 0]]),
                 joint_pos=Tensor([[0, 0]]), robot_joint_pos=Tensor([[1, 1]]),
                 time_steps=Tensor([0]), cfg=NS(body_names=["pelvis", "torso_link"]))
    robot = NS(data=NS(joint_pos=Tensor([[1, 1]]), joint_vel=Tensor([[0, 0]]),
                       body_lin_vel_w=Tensor([[[0, 0, 0]] * 2]), body_ang_vel_w=Tensor([[[0, 0, 0]] * 2]),
                       body_pos_w=command.robot_body_pos_w, body_quat_w=Tensor([[[1, 0, 0, 0]] * 2])))
    class Scene(dict):
        env_origins = Tensor([[0, 0, 0]])
    env = NS(command_manager=NS(get_term=lambda _: command), scene=Scene(robot=robot),
             step_dt=.02, termination_manager=NS(dones=Tensor([True]), terminated=Tensor([True]),
                                                 time_outs=Tensor([False]), active_terms=["anchor_pos"], get_term=lambda _: Tensor([True])))
    telemetry = RolloutTelemetry()
    telemetry.capture(env)
    env.termination_manager.dones = Tensor([False])
    env.termination_manager.terminated = Tensor([False])
    command.time_steps = Tensor([1])
    telemetry.capture(env)
    command.robot_body_pos_w.value[:] = 999  # simulated reset must not alter captured data
    for row in telemetry.rows:
        row["rew_mean"] = 1.0
    output = tmp_path / "metrics.json"
    telemetry.write(output, tmp_path / "metrics_steps.csv", metadata={
        "joint_names": ["a", "b"], "body_names": ["pelvis", "torso_link"],
        "contact_body_names": [], "reference_body_names": ["pelvis", "torso_link"], "fps": 50.0})
    record = load_rollout_metrics(output, horizon=2, metric_keys=["error_body_pos"], seed=1)
    assert record.done_count == 1 and record.first_failure_step == 1
    assert record.metrics["error_body_pos"] == 1
    assert [(r["episode_id"], r["episode_step"], r["motion_frame"]) for r in telemetry.rows] == [(0, 1, 0), (1, 1, 1)]
    assert telemetry.rows[0]["sample_time_s"] == .02 and telemetry.rows[0]["command_time_s"] == 0
    assert json.loads(output.read_text())["terminated_count"] == 1
    with np.load(output.with_suffix(".states.npz")) as data:
        assert data["body_pos_w"][0, 0, 0] == 1
        assert data["joint_names"].tolist() == ["a", "b"]
        assert data["reference_body_names"].tolist() == ["pelvis", "torso_link"]
        assert data["fps"].item() == 50
    completion = json.loads(output.with_suffix(".completed.json").read_text())
    assert completion["artifact_sha256"][output.name] == hash_file(output)


def telemetry_sample():
    telemetry = RolloutTelemetry()
    telemetry.rows = [{"step": 1, "motion_frame": 0, "episode_id": 0, "episode_step": 1,
                       "sample_time_s": .02, "episode_time_s": .02, "command_time_s": 0,
                       "done_any": True, "terminated_any": False, "truncated_any": True,
                       "error_body_pos": .1, "rew_mean": 1.0}]
    telemetry.states = [{"joint_pos": np.zeros(2), "actions": np.zeros(2)}]
    return telemetry


def test_timeout_recording_is_distinct_but_gate_remains_conservative(tmp_path):
    output = tmp_path / "metrics.json"
    result = telemetry_sample().write(output, tmp_path / "metrics_steps.csv", metadata={})
    assert result["done_any_count"] == 1
    assert result["terminated_count"] == 0 and result["truncated_count"] == 1
    assert load_rollout_metrics(output, horizon=1, metric_keys=[], seed=1).done_count == 1


@pytest.mark.parametrize("failure", ["states", "metrics", "nonfinite"])
def test_incomplete_telemetry_cannot_commit_or_look_successful(tmp_path, monkeypatch, failure):
    output, steps, states, completed = isaac_rollout_outputs(tmp_path / "metrics.json", tmp_path / "metrics_steps.csv")
    for path in (output, steps, states, completed):
        path.write_text("stale")
    telemetry = telemetry_sample()
    if failure == "nonfinite":
        telemetry.states[0]["actions"][0] = np.nan
    elif failure == "states":
        def fail(*args, **kwargs): raise OSError("state export failed")
        monkeypatch.setattr(np, "savez_compressed", fail)
    else:
        def fail(*args, **kwargs): raise OSError("JSON commit failed")
        monkeypatch.setattr("mimicx.backends.isaaclab.telemetry.atomic_write_json", fail)
    with pytest.raises((ValueError, OSError)):
        telemetry.write(output, steps, metadata={})
    assert not completed.exists()
    record = CommandRecord("masked-kit-exit", (sys.executable, "-c", "pass"), tmp_path,
                           tmp_path / "log.txt", expected_outputs=(output, steps, states, completed))
    result = run_command(record)
    assert result.process_exit_code == 0 and result.exit_code != 0
