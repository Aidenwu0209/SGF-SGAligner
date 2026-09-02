"""No-GT source audit and trajectory importer for an SGAD-SLAM shadow arm."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import (
    PoseRecord, load_manifest, sha256_file, stable_json_sha256, validate_se3,
    write_trajectory,
)


FORBIDDEN_SOURCE_SNIPPETS = {
    "first_pose_from_dataset": "self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])",
    "first_two_poses_from_dataset": "estimated_c2w = self.dataset[frame_id][-1]",
    "scannet_pose_loader": "self.load_poses(self.dataset_path / \"gt_pose.txt\")",
}


def audit_sgad_source(source_root: Path) -> dict:
    root = Path(source_root).resolve()
    targets = [root / "mp_Mapper.py", root / "src/entities/datasets.py"]
    rows, blockers = [], []
    for path in targets:
        if not path.is_file():
            blockers.append({"rule": "required_source_missing", "path": str(path)})
            continue
        text = path.read_text(encoding="utf-8")
        rows.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
        for rule, snippet in FORBIDDEN_SOURCE_SNIPPETS.items():
            if snippet in text:
                blockers.append({"rule": rule, "path": str(path.relative_to(root))})
    return {
        "schema": "sgad_no_gt_source_audit.v1",
        "source_root": str(root),
        "files": rows,
        "blockers": blockers,
        "passes_no_gt_input_gate": not blockers,
    }


def import_sgad_shadow(
    *, manifest_path: Path, matrices_path: Path, provenance_path: Path,
    output_path: Path, audit_path: Path,
    maximum_step_translation_m: float = 1.5,
) -> dict:
    manifest = load_manifest(manifest_path)
    provenance = json.loads(Path(provenance_path).read_text())
    required = {
        "provider_commit", "config_sha256", "source_audit_sha256",
        "gt_consumed", "source_audit_passed", "matrix_convention",
    }
    missing = required - set(provenance)
    if missing:
        raise ValueError(f"SGAD provenance missing: {sorted(missing)}")
    if provenance["gt_consumed"] is not False or provenance["source_audit_passed"] is not True:
        raise ValueError("SGAD provider did not pass the no-GT input gate")
    if provenance["matrix_convention"] != "T_world_camera_m":
        raise ValueError("SGAD matrix convention mismatch")
    if len(str(provenance["provider_commit"])) != 40:
        raise ValueError("bad SGAD provider commit")
    for key in ("config_sha256", "source_audit_sha256"):
        if len(str(provenance[key])) != 64:
            raise ValueError(f"bad provenance digest: {key}")

    matrices = np.load(matrices_path, allow_pickle=False)
    if matrices.shape != (len(manifest.frames), 4, 4):
        raise ValueError("SGAD output must contain one matrix per manifest frame")
    matrices = [validate_se3(matrix, "SGAD pose") for matrix in matrices]
    origin_inverse = np.linalg.inv(matrices[0])
    matrices = [validate_se3(origin_inverse @ matrix) for matrix in matrices]
    steps = [float(np.linalg.norm((np.linalg.inv(a) @ b)[:3, 3]))
             for a, b in zip(matrices, matrices[1:])]
    if steps and max(steps) > maximum_step_translation_m:
        raise ValueError("SGAD trajectory exceeds the metric motion gate")

    records = [PoseRecord(
        frame_id=frame.frame_id, timestamp_us=frame.timestamp_us,
        t_world_camera=matrix, source="SGAD-SLAM-shadow-no-gt",
    ) for frame, matrix in zip(manifest.frames, matrices)]
    write_trajectory(
        output_path, records, sequence_id=manifest.sequence_id,
        arm="sgad_slam_shadow", metadata={
            **provenance,
            "matrices_sha256": sha256_file(matrices_path),
            "provenance_sha256": sha256_file(provenance_path),
            "manifest_payload_sha256": manifest.as_dict()["payload_sha256"],
        },
    )
    unsigned = {
        "schema": "sgad_shadow_import_audit.v1",
        "sequence_id": manifest.sequence_id,
        "frame_count": len(records),
        "complete_manifest_coverage": True,
        "maximum_step_translation_m": max(steps, default=0.0),
        "output_trajectory_sha256": sha256_file(output_path),
        "gt_consumed": False,
        "identity_fallback_used": False,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump({**unsigned, "payload_sha256": stable_json_sha256(unsigned)}, stream,
                  indent=2, sort_keys=True)
        stream.write("\n")
    return unsigned
