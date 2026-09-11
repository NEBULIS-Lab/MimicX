from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class MotionEntry:
    id: str
    category: str
    source: Path | None
    output: Path | None

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_dataset_manifest(path: Path) -> tuple[dict, list[MotionEntry]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mimicx.dataset-screen.v1":
        raise ValueError("unsupported dataset manifest schema")
    entries = []
    for raw in payload.get("motions", []):
        source = Path(raw["source"]).expanduser().resolve() if raw.get("source") else None
        output = Path(raw["output"]).expanduser().resolve() if raw.get("output") else None
        entries.append(MotionEntry(str(raw["id"]), str(raw["category"]), source, output))
    if len({entry.id for entry in entries}) != len(entries):
        raise ValueError("duplicate motion id")
    return payload, entries

def audit_dataset_manifest(path: Path) -> dict:
    payload, entries = load_dataset_manifest(path)
    records, missing = [], []
    for entry in entries:
        if entry.source is None or not entry.source.is_file():
            missing.append(entry.id)
            records.append({"id": entry.id, "category": entry.category, "source": str(entry.source) if entry.source else None, "status": "missing"})
        else:
            records.append({"id": entry.id, "category": entry.category, "source": str(entry.source), "bytes": entry.source.stat().st_size, "sha256": sha256(entry.source), "status": "ready"})
    categories = sorted({entry.category for entry in entries})
    return {"schema": "mimicx.dataset-audit.v1", "dataset": payload["dataset"], "desired_count": int(payload["desired_count"]), "declared_count": len(entries), "ready_count": len(entries) - len(missing), "missing": missing, "categories": categories, "records": records}
