from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pose_pipeline.contracts import FrameRecord, SequenceManifest, write_manifest
from pose_pipeline.dpv_uncertainty import (
    DynamicUncertaintyStore, uncertainty_weighted_metric_scale,
)
from scripts.export_droid_w_uncertainty import (
    droid_uncertainty_to_dynamic, interpolate_keyframes,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    frames = []
    for index in range(2):
        color = tmp_path / "color" / f"{index}.jpg"
        depth = tmp_path / "depth" / f"{index}.png"
        color.write_bytes(b"rgb")
        depth.write_bytes(b"depth")
        frames.append(FrameRecord(index, index, color, depth, (500.0, 500.0, 2.0, 2.0)))
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, SequenceManifest("orbbec", "dynamic", tmp_path, 1000.0, tuple(frames), "test"))
    uncertainty = np.zeros((2, 4, 4), dtype=np.float32)
    uncertainty[:, :, 2:] = 0.95
    prior = np.full((2, 4, 4), 2.0, dtype=np.float32)
    artifact = tmp_path / "uncertainty.npz"
    np.savez_compressed(
        artifact,
        frame_ids=np.asarray([0, 1]),
        dynamic_uncertainty=uncertainty,
        depth_prior_m=prior,
        provider=np.asarray("DROID-W"),
        model_commit=np.asarray("a" * 40),
        checkpoint_sha256=np.asarray("b" * 64),
        gt_consumed=np.asarray(False),
    )
    return manifest, artifact


def test_dynamic_region_receives_zero_patch_weight(tmp_path: Path):
    manifest, artifact = _fixture(tmp_path)
    store = DynamicUncertaintyStore(manifest_path=manifest, artifact_path=artifact)
    weights = store.patch_weights(
        0, np.asarray([[0, 0], [3, 0], [1, 3], [3, 3]]), (4, 4),
        sensor_depth_m=np.full((4, 4), 2.0, dtype=np.float32),
    )
    np.testing.assert_allclose(weights, [1.0, 0.0, 1.0, 0.0], atol=1e-6)
    audit = store.write_audit(tmp_path / "audit.json")
    assert audit["gt_consumed"] is False


def test_depth_prior_disagreement_reduces_static_weight(tmp_path: Path):
    manifest, artifact = _fixture(tmp_path)
    store = DynamicUncertaintyStore(manifest_path=manifest, artifact_path=artifact)
    sensor = np.full((4, 4), 2.0, dtype=np.float32)
    sensor[0, 0] = 3.0
    confidence = store.static_confidence(0, (4, 4), sensor_depth_m=sensor)
    assert confidence[0, 0] == 0.0
    assert confidence[1, 1] == pytest.approx(1.0)


def test_uncertainty_weighted_scale_ignores_dynamic_outliers():
    disparity = np.full(200, 0.5)
    depth = np.full(200, 2.0)
    depth[150:] = 200.0
    weights = np.ones(200)
    weights[150:] = 0.0
    scale, report = uncertainty_weighted_metric_scale(
        disparity, depth, weights, minimum_samples=120,
    )
    assert scale == pytest.approx(1.0)
    assert report["supported_samples"] == 150


def test_droid_weight_conversion_matches_udba_kernel():
    raw = np.asarray([0.70, 0.80, 1.00], dtype=np.float32)
    dynamic = droid_uncertainty_to_dynamic(raw)
    np.testing.assert_allclose(dynamic, [0.0, 0.0, 0.9], atol=1e-6)


def test_uncertainty_interpolation_covers_every_frame():
    values = np.asarray([np.zeros((2, 2)), np.ones((2, 2))], dtype=np.float32)
    full = interpolate_keyframes(np.asarray([1, 3]), values, 5)
    assert full.shape == (5, 2, 2)
    np.testing.assert_allclose(full[0], 0.0)
    np.testing.assert_allclose(full[2], 0.5)
    np.testing.assert_allclose(full[4], 1.0)
