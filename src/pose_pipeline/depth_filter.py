"""Shared, deterministic uint16 depth filtering for every RGB-D consumer.

The production default is deliberately ``off``.  Filtering never changes the
image shape, dtype, or invalid-pixel mask and never fills depth holes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time
from typing import Iterable

import numpy as np


DEPTH_FILTER_PROFILES = (
    "off",
    "range_v1",
    "bilateral_light_v1",
    "bilateral_medium_v1",
)


@dataclass(frozen=True)
class DepthFilterConfig:
    profile: str = "off"
    minimum_depth_m: float | None = None
    maximum_depth_m: float | None = None
    bilateral_diameter_px: int | None = None
    bilateral_sigma_color_m: float | None = None
    bilateral_sigma_space_px: float | None = None

    def __post_init__(self) -> None:
        if self.profile not in DEPTH_FILTER_PROFILES:
            raise ValueError(
                f"unknown depth filter profile {self.profile!r}; "
                f"choose from {DEPTH_FILTER_PROFILES}"
            )
        expected = self.from_profile(self.profile, _validate=False)
        if self != expected:
            raise ValueError(
                f"profile {self.profile!r} has non-canonical parameters"
            )

    @classmethod
    def from_profile(
        cls, profile: str, *, _validate: bool = True,
    ) -> "DepthFilterConfig":
        parameters = {
            "off": {},
            "range_v1": {
                "minimum_depth_m": 0.30,
                "maximum_depth_m": 4.50,
            },
            "bilateral_light_v1": {
                "minimum_depth_m": 0.30,
                "maximum_depth_m": 4.50,
                "bilateral_diameter_px": 5,
                "bilateral_sigma_color_m": 0.015,
                "bilateral_sigma_space_px": 2.0,
            },
            "bilateral_medium_v1": {
                "minimum_depth_m": 0.30,
                "maximum_depth_m": 4.50,
                "bilateral_diameter_px": 5,
                "bilateral_sigma_color_m": 0.030,
                "bilateral_sigma_space_px": 2.0,
            },
        }
        if profile not in parameters:
            raise ValueError(
                f"unknown depth filter profile {profile!r}; "
                f"choose from {DEPTH_FILTER_PROFILES}"
            )
        if _validate:
            return cls(profile=profile, **parameters[profile])
        value = object.__new__(cls)
        for field, field_value in {
            "profile": profile,
            "minimum_depth_m": None,
            "maximum_depth_m": None,
            "bilateral_diameter_px": None,
            "bilateral_sigma_color_m": None,
            "bilateral_sigma_space_px": None,
            **parameters[profile],
        }.items():
            object.__setattr__(value, field, field_value)
        return value

    def as_dict(self) -> dict:
        return {
            "schema": "depth_filter_config.v1",
            **asdict(self),
            "fills_holes": False,
            "temporal_filtering": False,
        }

    @property
    def parameters_sha256(self) -> str:
        raw = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class DepthFilterStats:
    profile: str
    elapsed_ms: float
    pixel_count: int
    input_valid_pixels: int
    output_valid_pixels: int
    clipped_below_pixels: int
    clipped_above_pixels: int
    changed_pixels: int
    strong_edge_pairs: int
    retained_strong_edge_pairs: int
    input_sha256: str
    filtered_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _strong_edge_counts(
    before_m: np.ndarray, after_m: np.ndarray, comparable: np.ndarray,
) -> tuple[int, int]:
    strong = 0
    retained = 0
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = slice(None, -1)
        second[axis] = slice(1, None)
        first = tuple(first)
        second = tuple(second)
        eligible = comparable[first] & comparable[second]
        raw_gradient = np.abs(before_m[first] - before_m[second])
        strong_mask = eligible & (raw_gradient >= 0.050)
        strong += int(np.count_nonzero(strong_mask))
        filtered_gradient = np.abs(after_m[first] - after_m[second])
        retained += int(np.count_nonzero(
            strong_mask & (filtered_gradient >= 0.5 * raw_gradient)
        ))
    return strong, retained


def apply_depth_filter(
    depth: np.ndarray,
    depth_scale: float,
    config: DepthFilterConfig = DepthFilterConfig(),
) -> tuple[np.ndarray, DepthFilterStats]:
    """Apply one canonical profile and return uint16 depth plus audit stats."""
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("depth filter requires a uint16 HxW image")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    source = np.ascontiguousarray(depth)
    input_valid = source != 0
    input_sha256 = _array_sha256(source)
    started = time.perf_counter()

    if config.profile == "off":
        filtered = source.copy()
        below = above = 0
        comparable = input_valid
    else:
        minimum_raw = int(np.ceil(
            float(config.minimum_depth_m) * depth_scale,
        ))
        maximum_raw = int(np.floor(
            float(config.maximum_depth_m) * depth_scale,
        ))
        below_mask = input_valid & (source < minimum_raw)
        above_mask = input_valid & (source > maximum_raw)
        comparable = input_valid & ~below_mask & ~above_mask
        if config.bilateral_diameter_px is None:
            filtered = source.copy()
            filtered[~comparable] = 0
        else:
            import cv2

            # Bilateral weights are invariant when depth and sigmaColor are
            # scaled together. Staying in raw units avoids a full-frame
            # divide/multiply round trip while preserving metric parameters.
            working_raw = source.astype(np.float32)
            working_raw[~comparable] = 0.0
            working_raw = cv2.bilateralFilter(
                working_raw,
                d=int(config.bilateral_diameter_px),
                sigmaColor=(
                    float(config.bilateral_sigma_color_m) * depth_scale
                ),
                sigmaSpace=float(config.bilateral_sigma_space_px),
                borderType=cv2.BORDER_REPLICATE,
            )
            # Restoring the exact mask is the fail-closed no-hole-fill contract.
            working_raw[~comparable] = 0.0
            np.rint(working_raw, out=working_raw)
            np.clip(
                working_raw, 0, np.iinfo(np.uint16).max,
                out=working_raw,
            )
            filtered = working_raw.astype(np.uint16)
            filtered[~comparable] = 0
        below = int(np.count_nonzero(below_mask))
        above = int(np.count_nonzero(above_mask))

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    before_m = source.astype(np.float32) / np.float32(depth_scale)
    after_m = filtered.astype(np.float32) / np.float32(depth_scale)
    strong, retained = _strong_edge_counts(
        before_m, after_m, comparable,
    )
    stats = DepthFilterStats(
        profile=config.profile,
        elapsed_ms=elapsed_ms,
        pixel_count=int(source.size),
        input_valid_pixels=int(np.count_nonzero(input_valid)),
        output_valid_pixels=int(np.count_nonzero(filtered)),
        clipped_below_pixels=below,
        clipped_above_pixels=above,
        changed_pixels=int(np.count_nonzero(filtered != source)),
        strong_edge_pairs=strong,
        retained_strong_edge_pairs=retained,
        input_sha256=input_sha256,
        filtered_sha256=_array_sha256(filtered),
    )
    return np.ascontiguousarray(filtered), stats


class DepthFilterAccumulator:
    """Streaming audit that avoids retaining full depth frames in memory."""

    def __init__(self, config: DepthFilterConfig):
        self.config = config
        self._stats: list[DepthFilterStats] = []
        self._rolling = hashlib.sha256()

    def update(self, frame_id: int, stats: DepthFilterStats) -> None:
        if stats.profile != self.config.profile:
            raise ValueError("depth filter stats/profile mismatch")
        self._stats.append(stats)
        self._rolling.update(int(frame_id).to_bytes(8, "little", signed=True))
        self._rolling.update(bytes.fromhex(stats.filtered_sha256))

    def extend(self, rows: Iterable[tuple[int, DepthFilterStats]]) -> None:
        for frame_id, stats in rows:
            self.update(frame_id, stats)

    def summary(self) -> dict:
        try:
            import cv2
            opencv_version = cv2.__version__
        except ImportError:
            opencv_version = None

        elapsed = np.asarray(
            [row.elapsed_ms for row in self._stats], dtype=np.float64,
        )
        strong = sum(row.strong_edge_pairs for row in self._stats)
        retained = sum(
            row.retained_strong_edge_pairs for row in self._stats
        )
        input_valid = sum(row.input_valid_pixels for row in self._stats)
        output_valid = sum(row.output_valid_pixels for row in self._stats)
        pixels = sum(row.pixel_count for row in self._stats)
        return {
            "schema": "depth_filter_audit.v1",
            "config": self.config.as_dict(),
            "parameters_sha256": self.config.parameters_sha256,
            "opencv_version": opencv_version,
            "processed_frame_reads": len(self._stats),
            "filter_elapsed_ms_total": float(elapsed.sum()) if len(elapsed) else 0.0,
            "filter_elapsed_ms_median": float(np.median(elapsed)) if len(elapsed) else None,
            "filter_elapsed_ms_p95": float(np.percentile(elapsed, 95)) if len(elapsed) else None,
            "input_valid_pixels": input_valid,
            "output_valid_pixels": output_valid,
            "valid_pixel_retention": (
                output_valid / input_valid if input_valid else None
            ),
            "changed_pixel_fraction": (
                sum(row.changed_pixels for row in self._stats) / pixels
                if pixels else None
            ),
            "clipped_below_pixels": sum(
                row.clipped_below_pixels for row in self._stats
            ),
            "clipped_above_pixels": sum(
                row.clipped_above_pixels for row in self._stats
            ),
            "strong_edge_pairs": strong,
            "strong_edge_half_gradient_retention": (
                retained / strong if strong else None
            ),
            "filtered_depth_rolling_sha256": self._rolling.hexdigest(),
            "fills_holes": False,
            "gt_consumed": False,
        }
