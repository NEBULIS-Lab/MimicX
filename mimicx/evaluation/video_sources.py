"""Immutable source-video admission for native MimicX campaigns."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from mimicx.refinement.state import atomic_write_json


SCHEMA = "mimicx.video-sources.v1"


@dataclass(frozen=True)
class AdmittedSource:
    id: str
    source_path: str
    source_sha256: str
    materialized_path: str
    role: str
    replaces: str | None = None


@dataclass(frozen=True)
class AdmissionReport:
    manifest_path: str
    manifest_sha256: str
    records: tuple[AdmittedSource, ...]

    @property
    def admitted_ids(self) -> tuple[str, ...]:
        return tuple(record.id for record in self.records)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported source manifest schema in {path}")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source manifest must contain a non-empty sources list")
    return payload


def _resolve(path_value: object, root: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _verify_record(record: Mapping[str, Any], root: Path) -> tuple[str, Path, str]:
    source_id = str(record["id"])
    source = _resolve(record["source"], root)
    if not source.is_file():
        raise FileNotFoundError(f"source video does not exist: {source}")
    expected = str(record["sha256"]).lower()
    observed = sha256(source)
    if expected != observed:
        raise ValueError(
            f"SHA-256 mismatch for {source_id}: expected {expected}, observed {observed}"
        )
    return source_id, source, observed


def _materialize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise ValueError(f"existing admitted source has different payload: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def admit_sources(
    manifest: Path,
    output_dir: Path,
    *,
    reference_rejected_ids: set[str] | None = None,
) -> AdmissionReport:
    source_manifest = manifest.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    payload = _load(source_manifest)
    rejected = set(reference_rejected_ids or ())
    seen: set[str] = set()
    admitted: list[AdmittedSource] = []
    for raw in payload["sources"]:
        if not isinstance(raw, Mapping):
            raise ValueError("each source record must be a mapping")
        primary_id = str(raw["id"])
        if primary_id in seen:
            raise ValueError(f"duplicate source id: {primary_id}")
        seen.add(primary_id)
        selected = raw
        replaces: str | None = None
        if primary_id in rejected:
            fallback = raw.get("fallback")
            if not isinstance(fallback, Mapping):
                raise ValueError(f"reference-rejected source has no fallback: {primary_id}")
            selected = fallback
            replaces = primary_id
        source_id, source, observed = _verify_record(selected, source_manifest.parent)
        if source_id in {record.id for record in admitted}:
            raise ValueError(f"duplicate admitted source id: {source_id}")
        extension = source.suffix.lower() or ".mp4"
        destination = output / f"{source_id}{extension}"
        _materialize(source, destination)
        admitted.append(
            AdmittedSource(
                id=source_id,
                source_path=str(source),
                source_sha256=observed,
                materialized_path=str(destination),
                role=str(raw.get("role", "confirmation")),
                replaces=replaces,
            )
        )
    report = AdmissionReport(
        manifest_path=str(source_manifest),
        manifest_sha256=sha256(source_manifest),
        records=tuple(admitted),
    )
    atomic_write_json(
        output / "resolved_sources.json",
        {
            "schema": "mimicx.resolved-video-sources.v1",
            "manifest_path": report.manifest_path,
            "manifest_sha256": report.manifest_sha256,
            "records": [asdict(record) for record in report.records],
        },
    )
    return report
