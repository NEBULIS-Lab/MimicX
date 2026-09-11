from pathlib import Path
import yaml
from mimicx.datasets.manifest import audit_dataset_manifest, load_dataset_manifest

def test_manifest_audit_hashes_ready_and_marks_missing(tmp_path: Path) -> None:
    ready = tmp_path / "ready.csv"; ready.write_text("1,2\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"schema": "mimicx.dataset-screen.v1", "dataset": "demo", "desired_count": 2, "motions": [{"id": "ready", "category": "walk", "source": str(ready)}, {"id": "missing", "category": "jump", "source": str(tmp_path / "missing.csv")}]}))
    report = audit_dataset_manifest(manifest)
    assert report["ready_count"] == 1
    assert report["missing"] == ["missing"]
    assert len(report["records"][0]["sha256"]) == 64

def test_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"schema": "mimicx.dataset-screen.v1", "dataset": "demo", "desired_count": 2, "motions": [{"id": "same", "category": "walk"}, {"id": "same", "category": "run"}]}))
    try: load_dataset_manifest(manifest)
    except ValueError as error: assert "duplicate" in str(error)
    else: raise AssertionError("duplicate ids accepted")
