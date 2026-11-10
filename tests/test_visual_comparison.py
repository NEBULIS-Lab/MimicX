"""Synthetic CPU media only: no experiment recordings or graphics contexts."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.fixture
def request_data(tmp_path, request):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("PIL")
    fps, rate = getattr(request, "param", (25, "25"))
    frames = 121 if hasattr(request, "param") else 4
    clips = []
    for index, color in enumerate(((200, 20, 20), (20, 200, 20), (20, 20, 200), (200, 200, 20))):
        path = tmp_path / f"input-{index}.mp4"
        writer = ffmpeg.write_frames(str(path), (320, 180), fps=fps, macro_block_size=1,
                                    input_params=["-r", rate])
        writer.send(None)
        try:
            for frame in range(frames):
                offset = (frame % 4) * 10
                writer.send(np.full((180, 320, 3), np.asarray(color) + offset, np.uint8))
        finally:
            writer.close()
        clips.append(path)
    camera = dict(camera_origin_world_m=np.zeros((frames, 3)).tolist(),
                  world_from_camera_linear=np.tile(np.eye(3), (frames, 1, 1)).tolist(),
                  intrinsics=np.tile(np.eye(3), (frames, 1, 1)).tolist())
    return dict(clips=clips, labels=["A", "B", "C", "D"], physical_complete=[True, False, True, False],
                calibrations=[copy.deepcopy(camera) for _ in range(4)], frames=frames, fps=fps,
                source_size=(320, 180), tile_size=(320, 180), output=tmp_path / "comparison.mp4")


def test_comparison_preserves_all_clips_clocks_outcomes_and_tile_order(request_data):
    module = importlib.import_module("mimicx.visualization.comparison")
    before = [path.read_bytes() for path in request_data["clips"]]
    report = module.export_comparison(**request_data)
    assert report["full_clock_passed"]
    assert report["output_size"] == [640, 440]
    assert report["decoded_frames"] == 4
    assert report["visual_review"] == "pending"
    assert report["tiles"][1]["header"] == "B | physical failure"
    assert report["tiles"][2]["header"] == "C | motion complete"
    assert report["tiles"][2]["image_box"] == [0, 260, 320, 440]
    assert [path.read_bytes() for path in request_data["clips"]] == before
    ffmpeg = importlib.import_module("imageio_ffmpeg")
    reader = ffmpeg.read_frames(str(request_data["output"]), pix_fmt="rgb24")
    try:
        next(reader)
        images = [np.frombuffer(frame, np.uint8).reshape(440, 640, 3) for frame in reader]
    finally:
        reader.close()
    # Interior color probes verify all four sources, not only a nonblank canvas.
    for x, y, color in ((100, 100, [200, 20, 20]), (420, 100, [20, 200, 20]),
                        (100, 320, [20, 20, 200]), (420, 320, [200, 200, 20])):
        np.testing.assert_allclose(images[0][y, x], color, atol=6)
        assert np.mean(images[-1][y, x].astype(float) - images[0][y, x]) > 20
    assert not list(request_data["output"].parent.glob(".comparison-*"))


@pytest.mark.parametrize("bad", ["camera", "clock", "outcome", "aspect", "header", "count"])
def test_comparison_rejects_incomparable_inputs_without_publishing(request_data, bad):
    module = importlib.import_module("mimicx.visualization.comparison")
    if bad == "camera":
        request_data["calibrations"][1]["camera_origin_world_m"][2][0] = .1
    elif bad == "clock":
        request_data["frames"] = 5
        for camera in request_data["calibrations"]:
            for sequence in camera.values():
                sequence.append(copy.deepcopy(sequence[-1]))
    elif bad == "outcome":
        request_data["physical_complete"][0] = "true"
    elif bad == "aspect":
        request_data["tile_size"] = (320, 160)
    elif bad == "header":
        request_data["labels"][0] = "long label " * 100
    else:
        request_data["clips"].pop()
    with pytest.raises(ValueError):
        module.export_comparison(**request_data)
    assert not request_data["output"].exists()


def test_comparison_refuses_existing_output_before_reading_sources(tmp_path):
    module = importlib.import_module("mimicx.visualization.comparison")
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        module.export_comparison([], labels=[], physical_complete=[], calibrations=[], frames=4, fps=25,
                                 source_size=(320, 180), output=output)
    assert output.read_bytes() == b"preserve"


def test_comparison_cli_uses_config_relative_inputs_and_writes_receipt(request_data, tmp_path):
    config = {key: request_data[key] for key in ("frames", "fps", "source_size", "tile_size")}
    config["clips"] = []
    for i, path in enumerate(request_data["clips"]):
        camera = tmp_path / f"camera-{i}.json"
        camera.write_text(json.dumps(request_data["calibrations"][i]))
        config["clips"].append(dict(video=path.name, camera=camera.name, label=request_data["labels"][i],
                                    physical_complete=request_data["physical_complete"][i]))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    script = Path(__file__).resolve().parents[1] / "scripts/visualization/comparison.py"
    result = subprocess.run([sys.executable, str(script), "--config", str(config_path),
                             "--output", str(request_data["output"])], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(request_data["output"].with_suffix(".json").read_text())
    assert receipt["decoded_frames"] == 4 and receipt["full_clock_passed"]
    assert str(tmp_path) not in json.dumps(receipt)


def test_comparison_preserves_output_published_during_encoding(request_data, monkeypatch):
    module = importlib.import_module("mimicx.visualization.comparison")
    audit = module.audit_video
    output = request_data["output"]

    def publish_competing_output(path, **kwargs):
        report = audit(path, **kwargs)
        if Path(path).name == "video.mp4":
            output.write_bytes(b"other export")
        return report

    monkeypatch.setattr(module, "audit_video", publish_competing_output)
    with pytest.raises(FileExistsError):
        module.export_comparison(**request_data)
    assert output.read_bytes() == b"other export"


def test_comparison_cli_preserves_existing_receipt_without_reading_config(tmp_path):
    receipt = tmp_path / "comparison.json"
    receipt.write_text('{"existing": true}')
    script = Path(__file__).resolve().parents[1] / "scripts/visualization/comparison.py"
    result = subprocess.run([sys.executable, str(script), "--config", str(tmp_path / "absent.json"),
                             "--output", str(tmp_path / "comparison.mp4")], capture_output=True, text=True)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert json.loads(receipt.read_text()) == {"existing": True}
    assert not (tmp_path / "comparison.mp4").exists()


@pytest.mark.parametrize("request_data", [(50., "50"), (29.97, "2997/100"), (59.94, "2997/50"),
                                          (30000 / 1001, "30000/1001"),
                                          (60000 / 1001, "60000/1001")], indirect=True)
def test_montage_preserves_50hz_decimal_and_ntsc_clocks(request_data):
    module = importlib.import_module("mimicx.visualization.comparison")
    report = module.export_comparison(**request_data)
    assert report["full_clock_passed"] and report["decoded_frames"] == 121
    assert report["last_pts_seconds"] == pytest.approx(120 / request_data["fps"], abs=1e-9)
    assert all(audit["full_clock_passed"] for audit in report["input_audits"])
