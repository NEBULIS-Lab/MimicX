"""CPU full-decode checks that retain every presentation timestamp."""
from __future__ import annotations

import csv
from fractions import Fraction
import math
import os
import re
import subprocess
import tempfile


def validate_framehash(lines, *, frames, fps, size):
    """Validate FFmpeg RGB24 SHA256 framehash output on a zero-based CFR clock.

    No timestamps are shifted, rounded to frame indices, or inferred from an
    average FPS. Duration includes the final frame's display interval.
    """
    if (isinstance(frames, bool) or int(frames) != frames or frames < 1
            or not math.isfinite(fps) or fps <= 0 or len(size) != 2
            or any(not math.isfinite(v) or int(v) != v or v < 1 for v in size)):
        raise ValueError("Expected positive frames, FPS and integer dimensions")
    timebase, dimensions, count = None, None, 0
    interval = 1 / Fraction(str(fps))
    tolerance = min(Fraction(1, 1000000), interval / 1000)
    for line in lines:
        if line.startswith("#tb 0:"):
            timebase = Fraction(line.split(":", 1)[1].strip())
        elif line.startswith("#dimensions 0:"):
            dimensions = tuple(map(int, line.split(":", 1)[1].strip().split("x")))
        elif line.strip() and not line.startswith("#"):
            row = next(csv.reader([line], skipinitialspace=True))
            if (timebase is None or timebase <= 0 or dimensions != tuple(size)
                    or len(row) != 6 or int(row[0]) != 0 or int(row[4]) != size[0] * size[1] * 3
                    or re.fullmatch(r"[0-9a-fA-F]{64}", row[5].strip()) is None):
                raise ValueError("Invalid decoded RGB frame, dimensions or timebase")
            pts, duration = int(row[2]) * timebase, int(row[3]) * timebase
            if abs(pts - count * interval) > tolerance or abs(duration - interval) > tolerance:
                raise ValueError(f"Decoded frame clock differs at frame {count}")
            count += 1
    if count != frames:
        raise ValueError(f"Decoded frame count differs: {count} != {frames}")
    return dict(decoded_frames=count, fps=float(fps), output_size=list(size),
                first_pts_seconds=0., last_pts_seconds=float(pts), duration_seconds=float(pts + duration),
                full_decode_passed=True, full_clock_passed=True)


def audit_video(video, *, frames, fps, size, ffmpeg=None, timeout=3600):
    """Decode with software FFmpeg; reject errors and clocks before publication."""
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [str(ffmpeg), "-nostdin", "-hide_banner", "-v", "error", "-xerror", "-copyts",
               "-hwaccel", "none", "-threads", "2", "-i", str(video), "-map", "0:v:0",
               "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-enc_time_base", "demux",
               "-c:v", "rawvideo", "-threads", "1", "-pix_fmt", "rgb24",
               "-f", "framemd5", "-hash", "sha256", "-"]
    with tempfile.TemporaryFile(mode="w+t") as output, tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES=""),
                                 timeout=timeout)
        errors.seek(0)
        error = errors.read(4096)
        if process.returncode or error.strip():
            raise ValueError(f"Full video decode failed: {error.strip()}")
        output.seek(0)
        return validate_framehash(output, frames=frames, fps=fps, size=size)
