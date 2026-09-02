from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pose_pipeline.contracts import FrameRecord, SequenceManifest, load_trajectory, write_manifest
from pose_pipeline.sgad_shadow import audit_sgad_source, import_sgad_shadow


def test_official_style_source_is_rejected(tmp_path: Path):
    (tmp_path / "src/entities").mkdir(parents=True)
    (tmp_path / "mp_Mapper.py").write_text(
        "self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])\n"
        "estimated_c2w = self.dataset[frame_id][-1]\n"
    )
    (tmp_path / "src/entities/datasets.py").write_text(
        'self.load_poses(self.dataset_path / "gt_pose.txt")\n'
    )
    report = audit_sgad_source(tmp_path)
    assert report["passes_no_gt_input_gate"] is False
    assert len(report["blockers"]) == 3


def _inputs(tmp_path: Path, passed: bool = True):
    (tmp_path / "color").mkdir()
    (tmp_path / "depth").mkdir()
    frames = []
    matrices = []
    for index in range(3):
        color = tmp_path / "color" / f"{index}.jpg"
        depth = tmp_path / "depth" / f"{index}.png"
        color.write_bytes(b"rgb")
        depth.write_bytes(b"depth")
        frames.append(FrameRecord(index, index, color, depth, (500., 500., 2., 2.)))
        matrix = np.eye(4)
        matrix[0, 3] = index * .1
        matrices.append(matrix)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, SequenceManifest("orbbec", "sgad", tmp_path, 1000., tuple(frames), "test"))
    matrix_path = tmp_path / "estimated.npy"
    np.save(matrix_path, np.asarray(matrices))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "provider_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "source_audit_sha256": "c" * 64,
        "gt_consumed": False,
        "source_audit_passed": passed,
        "matrix_convention": "T_world_camera_m",
    }))
    return manifest, matrix_path, provenance


def test_imports_only_audited_full_trajectory(tmp_path: Path):
    manifest, matrices, provenance = _inputs(tmp_path)
    report = import_sgad_shadow(
        manifest_path=manifest, matrices_path=matrices,
        provenance_path=provenance, output_path=tmp_path / "out.json",
        audit_path=tmp_path / "audit.json",
    )
    rows, _ = load_trajectory(tmp_path / "out.json")
    assert report["frame_count"] == len(rows) == 3


def test_rejects_failed_source_audit(tmp_path: Path):
    manifest, matrices, provenance = _inputs(tmp_path, passed=False)
    with pytest.raises(ValueError, match="no-GT input gate"):
        import_sgad_shadow(
            manifest_path=manifest, matrices_path=matrices,
            provenance_path=provenance, output_path=tmp_path / "out.json",
            audit_path=tmp_path / "audit.json",
        )
