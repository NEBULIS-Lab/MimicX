from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class StringSegment:
    family: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]


def _interior_positions(radius: float, count: int, coverage: float) -> list[float]:
    if radius <= 0.0:
        raise ValueError("ellipse radii must be positive")
    if count < 1:
        raise ValueError("string counts must be positive")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    if count == 1:
        return [0.0]
    extent = radius * coverage
    return [-extent + index * 2.0 * extent / (count - 1) for index in range(count)]


def elliptical_stringbed_segments(
    *,
    center_z: float = 0.495,
    half_width: float = 0.122,
    half_height: float = 0.158,
    main_count: int = 16,
    cross_count: int = 19,
    coverage: float = 0.9,
) -> tuple[StringSegment, ...]:
    """Generate a string pattern in the racket's canonical Y-Z plane."""
    segments: list[StringSegment] = []
    for y in _interior_positions(half_width, main_count, coverage):
        z_extent = half_height * sqrt(max(0.0, 1.0 - (y / half_width) ** 2))
        segments.append(
            StringSegment("main", (0.0, y, center_z - z_extent), (0.0, y, center_z + z_extent))
        )
    for z_offset in _interior_positions(half_height, cross_count, coverage):
        y_extent = half_width * sqrt(max(0.0, 1.0 - (z_offset / half_height) ** 2))
        segments.append(
            StringSegment(
                "cross",
                (0.0, -y_extent, center_z + z_offset),
                (0.0, y_extent, center_z + z_offset),
            )
        )
    return tuple(segments)
