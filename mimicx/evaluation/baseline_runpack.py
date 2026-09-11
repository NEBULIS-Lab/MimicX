"""Prepare provenance-checked external baseline runpacks without launching them."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BaselineRunpackReport:
    baselines: int
    ready: int
    staged: int
    output_dir: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _probe_imports(python: Path, modules: list[str]) -> dict[str, Any]:
    if not modules:
        return {
            "python": str(python),
            "required_imports": [],
            "missing_imports": [],
            "imports_ready": python.is_file(),
        }
    if not python.is_file():
        return {
            "python": str(python),
            "required_imports": modules,
            "missing_imports": modules,
            "imports_ready": False,
        }
    script = (
        "import importlib.util,json,sys; "
        "print(json.dumps([name for name in sys.argv[1:] "
        "if importlib.util.find_spec(name) is None]))"
    )
    completed = subprocess.run(
        [str(python), "-c", script, *modules],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    missing = json.loads(completed.stdout)
    return {
        "python": str(python),
        "required_imports": modules,
        "missing_imports": missing,
        "imports_ready": not missing,
    }


def _guarded_script(repo: Path, command: list[str]) -> str:
    rendered = " ".join(shlex.quote(str(item)) for item in command)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'if [[ "${MIMICX_ALLOW_BASELINE_RUN:-0}" != "1" ]]; then',
            '  echo "Baseline launch is guarded. Set MIMICX_ALLOW_BASELINE_RUN=1 explicitly." >&2',
            "  exit 2",
            "fi",
            ': "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES explicitly}"',
            f"cd {shlex.quote(str(repo))}",
            f"exec env CUDA_VISIBLE_DEVICES=\"$CUDA_VISIBLE_DEVICES\" {rendered}",
            "",
        ]
    )


def prepare_baseline_runpack(
    manifest_path: Path,
    output_dir: Path,
) -> BaselineRunpackReport:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mimicx.external-baselines.v1":
        raise ValueError("Unsupported external baseline manifest schema")
    forbidden = tuple(str(item) for item in payload.get("forbidden_path_tokens", []))
    resolved: list[dict[str, Any]] = []
    scripts: list[tuple[str, str]] = []

    for baseline in payload.get("baselines", []):
        repo = Path(baseline["repo_path"]).expanduser().resolve()
        repo_text = str(repo)
        for token in forbidden:
            if token and token in repo_text:
                raise ValueError(
                    f"forbidden provenance token {token!r} in baseline path {repo}"
                )
        if not repo.is_dir():
            raise FileNotFoundError(f"Baseline repository is missing: {repo}")
        observed_commit = _git_commit(repo)
        expected_commit = str(baseline["expected_commit"])
        if observed_commit != expected_commit:
            raise ValueError(
                f"Commit mismatch for {baseline['id']}: "
                f"expected={expected_commit}, observed={observed_commit}"
            )

        files = []
        for relative in baseline.get("required_files", []):
            path = (repo / relative).resolve()
            if repo not in path.parents:
                raise ValueError(f"Required file escapes baseline repository: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"Required baseline file is missing: {path}")
            files.append(
                {
                    "relative_path": str(relative),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

        blockers = [str(item) for item in baseline.get("blockers", [])]
        python = Path(baseline.get("python", "/usr/bin/python3")).expanduser().resolve()
        runtime_probe = _probe_imports(
            python, [str(item) for item in baseline.get("required_imports", [])]
        )
        matched_ready = bool(baseline.get("matched_protocol_ready", False))
        if not runtime_probe["imports_ready"]:
            blockers.append(
                "missing runtime imports: "
                + ", ".join(runtime_probe["missing_imports"])
            )
            matched_ready = False
        if not matched_ready and not blockers:
            raise ValueError(f"Staged baseline {baseline['id']} must record blockers")
        command = [str(item) for item in baseline.get("command", [])]
        if not command:
            raise ValueError(f"Baseline {baseline['id']} has no command template")
        scripts.append((f"{baseline['id']}.sh", _guarded_script(repo, command)))
        resolved.append(
            {
                "id": str(baseline["id"]),
                "display_name": str(baseline["display_name"]),
                "repo_path": repo_text,
                "expected_commit": expected_commit,
                "observed_commit": observed_commit,
                "protocol_role": str(baseline.get("protocol_role", "external")),
                "matched_protocol_ready": matched_ready,
                "status": "ready" if matched_ready else "staged",
                "blockers": blockers,
                "runtime_probe": runtime_probe,
                "required_files": files,
                "command": command,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in scripts:
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
    (output_dir / "resolved_baselines.json").write_text(
        json.dumps(
            {
                "schema": payload["schema"],
                "source_manifest": str(manifest_path.resolve()),
                "source_manifest_sha256": _sha256(manifest_path),
                "baselines": resolved,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    ready = sum(item["status"] == "ready" for item in resolved)
    staged = len(resolved) - ready
    report = BaselineRunpackReport(
        baselines=len(resolved),
        ready=ready,
        staged=staged,
        output_dir=str(output_dir.resolve()),
    )
    lines = [
        "# External Baseline Preflight",
        "",
        f"- Baselines: **{report.baselines}**",
        f"- Matched-protocol ready: **{report.ready}**",
        f"- Staged with explicit blockers: **{report.staged}**",
        "- GPU jobs launched by this preparation step: **0**",
        "",
        "| Baseline | Commit | Status | Blockers |",
        "|---|---|---|---|",
    ]
    for item in resolved:
        blockers = "; ".join(item["blockers"]) or "--"
        lines.append(
            f"| {item['display_name']} | `{item['observed_commit'][:12]}` | "
            f"{item['status']} | {blockers} |"
        )
    lines.extend(
        [
            "",
            "Generated shell files are launch-guarded and require both "
            "`MIMICX_ALLOW_BASELINE_RUN=1` and explicit `CUDA_VISIBLE_DEVICES`.",
        ]
    )
    (output_dir / "BASELINE_PREFLIGHT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report
