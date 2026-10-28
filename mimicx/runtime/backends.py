"""Simulator selection without importing optional simulator dependencies."""

from pathlib import Path


def validate_backend(name: str) -> str:
    if name not in {"mjlab", "isaaclab"}:
        raise ValueError(f"Unsupported runtime backend: {name}")
    return name


def isaac_train_argv(*, python, script, task, motion_file, checkpoint, output_root,
                     run_name, iterations, num_envs, learning_rate, seed):
    return [str(python), str(script), "--task", str(task), "--motion-file", str(motion_file),
            "--checkpoint-file", str(checkpoint), "--output-root", str(output_root),
            "--run-name", str(run_name), "--iterations", str(iterations),
            "--num-envs", str(num_envs), "--learning-rate", str(learning_rate),
            "--seed", str(seed), "--headless"]


def isaac_script(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "isaac" / f"{name}.py"


def isaac_rollout_outputs(metrics, steps, *, protected_paths=()):
    metrics, steps = Path(metrics).expanduser().resolve(), Path(steps).expanduser().resolve()
    outputs = (metrics, steps, metrics.with_suffix(".states.npz"), metrics.with_suffix(".completed.json"))
    protected = [Path(p).expanduser().resolve() for p in protected_paths]
    for index, path in enumerate(outputs):
        if path.is_dir():
            raise ValueError(f"Rollout output is a directory: {path}")
        for other in (*outputs[:index], *protected):
            if path == other or (path.exists() and other.exists() and path.samefile(other)):
                raise ValueError(f"Rollout outputs must be distinct and cannot overwrite inputs: {path}")
    return outputs
