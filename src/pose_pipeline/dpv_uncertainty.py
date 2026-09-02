"""GT-free dynamic uncertainty sidecars for the DPV RGB-D frontend.

The module keeps uncertainty outside the RGB image: no black mask or synthetic
edge is introduced.  Static confidence can be sampled at DPVO patch locations,
used to reject unreliable metric-depth samples, and multiplied into BA weights
by the isolated DPV worker integration.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import load_manifest, sha256_file, stable_json_sha256


@dataclass(frozen=True)
class DynamicUncertaintyConfig:
    minimum_static_confidence: float = 0.20
    depth_prior_sigma_m: float = 0.10
    minimum_weighted_samples: int = 120
    sampling_stride: int = 8

    def validate(self) -> "DynamicUncertaintyConfig":
        if not 0.0 <= self.minimum_static_confidence < 1.0:
            raise ValueError("minimum static confidence must be in [0,1)")
        if self.depth_prior_sigma_m <= 0 or not np.isfinite(self.depth_prior_sigma_m):
            raise ValueError("depth prior sigma must be positive")
        if self.minimum_weighted_samples < 1 or self.sampling_stride < 1:
            raise ValueError("sample counts and stride must be positive")
        return self


class DynamicUncertaintyStore:
    """Load a versioned model artifact and expose aligned static confidence."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        artifact_path: Path,
        config: DynamicUncertaintyConfig = DynamicUncertaintyConfig(),
    ) -> None:
        self.manifest = load_manifest(manifest_path)
        self.artifact_path = Path(artifact_path)
        self.config = config.validate()
        with np.load(self.artifact_path) as payload:
            required = {"frame_ids", "dynamic_uncertainty"}
            if not required <= set(payload.files):
                raise ValueError("uncertainty artifact needs frame_ids and dynamic_uncertainty")
            self.frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
            self.uncertainty = np.asarray(payload["dynamic_uncertainty"], dtype=np.float32)
            self.depth_prior_m = (
                np.asarray(payload["depth_prior_m"], dtype=np.float32)
                if "depth_prior_m" in payload else None
            )
            self.provider = str(payload["provider"].item()) if "provider" in payload else "unknown"
            self.model_commit = str(payload["model_commit"].item()) if "model_commit" in payload else "unknown"
            self.checkpoint_sha256 = (
                str(payload["checkpoint_sha256"].item())
                if "checkpoint_sha256" in payload else "unknown"
            )
            gt_consumed = bool(payload["gt_consumed"].item()) if "gt_consumed" in payload else True
        if gt_consumed:
            raise ValueError("uncertainty artifact must declare gt_consumed=false")
        if self.uncertainty.ndim != 3 or self.uncertainty.shape[0] != len(self.frame_ids):
            raise ValueError("dynamic_uncertainty must have shape [N,H,W]")
        if not np.isfinite(self.uncertainty).all() or np.any((self.uncertainty < 0) | (self.uncertainty > 1)):
            raise ValueError("dynamic uncertainty must be finite in [0,1]")
        if self.depth_prior_m is not None:
            if self.depth_prior_m.shape != self.uncertainty.shape:
                raise ValueError("depth prior must align with uncertainty")
            if np.any(np.isfinite(self.depth_prior_m) & (self.depth_prior_m <= 0)):
                raise ValueError("finite depth priors must be positive")
        expected = np.asarray([frame.frame_id for frame in self.manifest.frames], dtype=np.int64)
        if not np.array_equal(self.frame_ids, expected):
            raise ValueError("uncertainty artifact must cover the complete ordered manifest")
        self._ordinal = {int(frame_id): index for index, frame_id in enumerate(self.frame_ids)}

    def _resize(self, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if value.shape == shape:
            return value.astype(np.float32, copy=True)
        source = np.asarray(value, dtype=np.float32)
        target_h, target_w = shape
        source_h, source_w = source.shape
        y = np.linspace(0.0, source_h - 1.0, target_h)
        x = np.linspace(0.0, source_w - 1.0, target_w)
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        y1 = np.minimum(y0 + 1, source_h - 1)
        x1 = np.minimum(x0 + 1, source_w - 1)
        wy = (y - y0).astype(np.float32)[:, None]
        wx = (x - x0).astype(np.float32)[None, :]
        top = source[y0[:, None], x0[None, :]] * (1.0 - wx)
        top += source[y0[:, None], x1[None, :]] * wx
        bottom = source[y1[:, None], x0[None, :]] * (1.0 - wx)
        bottom += source[y1[:, None], x1[None, :]] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32)

    def static_confidence(
        self,
        frame_id: int,
        shape: tuple[int, int],
        *,
        sensor_depth_m: np.ndarray | None = None,
    ) -> np.ndarray:
        ordinal = self._ordinal[int(frame_id)]
        confidence = 1.0 - self._resize(self.uncertainty[ordinal], shape)
        if sensor_depth_m is not None and self.depth_prior_m is not None:
            sensor = np.asarray(sensor_depth_m, dtype=np.float32)
            if sensor.shape != shape:
                raise ValueError("sensor depth shape differs from requested confidence shape")
            prior = self._resize(self.depth_prior_m[ordinal], shape)
            valid = np.isfinite(sensor) & (sensor > 0) & np.isfinite(prior) & (prior > 0)
            agreement = np.zeros(shape, dtype=np.float32)
            agreement[valid] = np.exp(
                -np.abs(sensor[valid] - prior[valid]) / self.config.depth_prior_sigma_m
            )
            confidence *= agreement
        confidence[confidence < self.config.minimum_static_confidence] = 0.0
        return np.clip(confidence, 0.0, 1.0)

    def patch_weights(
        self,
        frame_id: int,
        patch_uv: np.ndarray,
        image_shape: tuple[int, int],
        *,
        sensor_depth_m: np.ndarray | None = None,
    ) -> np.ndarray:
        uv = np.asarray(patch_uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
            raise ValueError("patch_uv must have finite shape [K,2]")
        confidence = self.static_confidence(
            frame_id, image_shape, sensor_depth_m=sensor_depth_m,
        )
        u = np.rint(uv[:, 0]).astype(np.int64)
        v = np.rint(uv[:, 1]).astype(np.int64)
        inside = (u >= 0) & (u < image_shape[1]) & (v >= 0) & (v < image_shape[0])
        weights = np.zeros(len(uv), dtype=np.float32)
        weights[inside] = confidence[v[inside], u[inside]]
        return weights

    def write_audit(self, path: Path) -> dict:
        rows = []
        for ordinal, frame in enumerate(self.manifest.frames):
            confidence = 1.0 - self.uncertainty[ordinal]
            rows.append({
                "frame_id": frame.frame_id,
                "mean_static_confidence": float(np.mean(confidence)),
                "static_fraction_at_threshold": float(np.mean(
                    confidence >= self.config.minimum_static_confidence
                )),
            })
        unsigned = {
            "schema": "dpv_dynamic_uncertainty_audit.v1",
            "sequence_id": self.manifest.sequence_id,
            "provider": self.provider,
            "model_commit": self.model_commit,
            "checkpoint_sha256": self.checkpoint_sha256,
            "artifact_sha256": sha256_file(self.artifact_path),
            "manifest_payload_sha256": self.manifest.as_dict()["payload_sha256"],
            "config": asdict(self.config),
            "frame_count": len(rows),
            "rows": rows,
            "integration_points": [
                "metric_scale_patch_samples",
                "rgbd_depth_consistency",
                "dpvo_bundle_adjustment_patch_weights",
            ],
            "gt_consumed": False,
            "identity_fallback_used": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return unsigned


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    if not len(values):
        raise ValueError("weighted median has no supported values")
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * weights.sum(), side="left")])


def uncertainty_weighted_metric_scale(
    disparity: Sequence[float],
    sensor_depth_m: Sequence[float],
    static_weights: Sequence[float],
    *,
    minimum_samples: int = 120,
) -> tuple[float, dict]:
    disparity = np.asarray(disparity, dtype=np.float64).reshape(-1)
    depth = np.asarray(sensor_depth_m, dtype=np.float64).reshape(-1)
    weights = np.asarray(static_weights, dtype=np.float64).reshape(-1)
    if not (disparity.shape == depth.shape == weights.shape):
        raise ValueError("scale arrays must share a shape")
    valid = np.isfinite(disparity) & np.isfinite(depth) & np.isfinite(weights)
    valid &= (disparity > 1e-8) & (depth > 0) & (weights > 0)
    if int(valid.sum()) < minimum_samples:
        raise ValueError("uncertainty-weighted metric scale is under-supported")
    scale_samples = disparity[valid] * depth[valid]
    scale = weighted_median(scale_samples, weights[valid])
    return scale, {
        "method": "dynamic_uncertainty_weighted_median",
        "supported_samples": int(valid.sum()),
        "effective_weight": float(weights[valid].sum()),
        "scale": scale,
        "gt_consumed": False,
    }
