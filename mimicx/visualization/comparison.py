"""Bounded, synchronous 2x2 review montage of four already-rendered recordings."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .source_camera import validate_camera_sequence
from .video_audit import audit_video


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_comparison(clips, *, labels, physical_complete, calibrations, frames, fps,
                      source_size, output, tile_size=(640, 360), font_path=None, ffmpeg=None):
    """Preserve complete zero-based clocks and add external method/outcome headers.

    Inputs must have the same fixed camera and dimensions. Each source and the
    encoded output are fully decoded on CPU. No input is overwritten, shortened,
    looped or retimed. Camera/physical-outcome provenance remains the caller's
    responsibility: supplied calibration cannot prove which camera made a video.
    """
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if any(len(values) != 4 for values in (clips, labels, physical_complete, calibrations)):
        raise ValueError("Comparison requires exactly four clips, labels, outcomes and cameras")
    if any(type(value) is not bool for value in physical_complete):
        raise ValueError("Physical completion must be an explicit boolean, not strict qualification")
    if any(not isinstance(label, str) or not label.strip() or any(ord(c) < 32 for c in label) for label in labels):
        raise ValueError("Method labels must be nonempty single-line text")
    for size in (source_size, tile_size):
        if (len(size) != 2 or not np.isfinite(size).all()
                or any(v < 2 or int(v) != v or v % 2 for v in size)):
            raise ValueError("Comparison sizes must be positive even integer dimensions")
    width, height = map(int, tile_size)
    if width * source_size[1] != height * source_size[0]:
        raise ValueError("Tile aspect ratio must match source; cropping and stretching are not supported")
    camera_reports = [validate_camera_sequence(camera, frames=frames, require_constant=True,
                                               shared_with=calibrations[0]) for camera in calibrations]
    clips = [Path(path) for path in clips]
    hashes = [_sha256(path) for path in clips]
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont

    ffmpeg = ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()
    font = ImageFont.truetype(str(font_path) if font_path else "DejaVuSans.ttf", 24)
    header_height = 40
    headers = []
    header_images = []
    for label, complete in zip(labels, physical_complete):
        header = label + (" | motion complete" if complete else " | physical failure")
        image = Image.new("RGB", (width, header_height), "#f5f5f5")
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((12, 3), header, font=font)
        if box[0] < 0 or box[1] < 0 or box[2] > width - 12 or box[3] > header_height - 4:
            raise ValueError("24px outcome header does not fit; shorten label or increase tile size")
        draw.text((12, 3), header, font=font, fill="#161616")
        headers.append(header)
        header_images.append(image)
    input_audits = [audit_video(path, frames=frames, fps=fps, size=source_size, ffmpeg=ffmpeg) for path in clips]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as temporary:
        directory = Path(temporary)
        header_paths = [directory / f"header-{i}.png" for i in range(4)]
        for image, path in zip(header_images, header_paths):
            image.save(path)
        partial = directory / "video.mp4"
        command = [str(ffmpeg), "-nostdin", "-hide_banner", "-v", "error", "-xerror", "-n",
                   "-copyts", "-filter_complex_threads", "1"]
        for path in [*clips, *header_paths]:
            command.extend(["-hwaccel", "none", "-threads", "1", "-i", str(path)])
        filters = []
        for i in range(4):
            filters.append(f"[{i}:v]scale={width}:{height}:flags=lanczos,"
                           f"pad={width}:{height + header_height}:0:{header_height}:color=white[t{i}]")
            filters.append(f"[t{i}][{i+4}:v]overlay=0:0:eof_action=repeat:shortest=0[v{i}]")
        filters.extend(["[v0][v1]hstack=inputs=2:shortest=1[top]",
                        "[v2][v3]hstack=inputs=2:shortest=1[bottom]",
                        "[top][bottom]vstack=inputs=2:shortest=1[grid]"])
        command.extend(["-filter_complex", ";".join(filters), "-map", "[grid]", "-an", "-sn", "-dn",
                        "-vsync", "0", "-c:v", "libx264", "-threads", "2", "-preset", "medium", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)])
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=3600,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES=""))
        if result.returncode or result.stderr.strip():
            raise ValueError(f"CPU comparison encoding failed: {result.stderr[:4096]}")
        audit = audit_video(partial, frames=frames, fps=fps, size=(width * 2, (height + header_height) * 2),
                            ffmpeg=ffmpeg)
        if [_sha256(path) for path in clips] != hashes:
            raise ValueError("Comparison source changed during export")
        output_hash = _sha256(partial)
        os.link(partial, output)
    tiles = [dict(label=labels[i], physical_complete=physical_complete[i], header=headers[i],
                  source_sha256=hashes[i], camera_sha256=camera_reports[i]["camera_sha256"],
                  image_box=[(i % 2) * width, (i // 2) * (height + header_height) + header_height,
                             (i % 2 + 1) * width, (i // 2 + 1) * (height + header_height)]) for i in range(4)]
    return dict(audit, tiles=tiles, input_audits=input_audits, sha256=output_hash,
                retime_factor=1., uniform_scale=width / source_size[0], header_height=header_height,
                visual_review="pending", kind="review_montage_not_new_evaluation")
