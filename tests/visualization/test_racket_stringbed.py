from __future__ import annotations

import numpy as np

from mimicx.visualization.racket_stringbed import elliptical_stringbed_segments


def test_stringbed_has_requested_mains_and_crosses_clipped_to_ellipse() -> None:
    segments = elliptical_stringbed_segments(
        center_z=0.495,
        half_width=0.122,
        half_height=0.158,
        main_count=16,
        cross_count=19,
    )

    mains = [segment for segment in segments if segment.family == "main"]
    crosses = [segment for segment in segments if segment.family == "cross"]

    assert len(mains) == 16
    assert len(crosses) == 19
    assert all(np.linalg.norm(np.subtract(segment.end, segment.start)) > 0.02 for segment in segments)

    for segment in segments:
        for point in (segment.start, segment.end):
            _, y, z = point
            normalized_radius = (y / 0.122) ** 2 + ((z - 0.495) / 0.158) ** 2
            assert normalized_radius <= 1.0 + 1e-9


def test_stringbed_lies_in_canonical_racket_yz_plane() -> None:
    segments = elliptical_stringbed_segments()

    assert segments
    assert all(np.isclose(segment.start[0], 0.0) for segment in segments)
    assert all(np.isclose(segment.end[0], 0.0) for segment in segments)
