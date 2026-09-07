"""Development integration of hybrid proposal, visual verification and Guard.

Inference never imports the GT evaluator. A successful invocation always
delivers a complete committed trajectory and its full-frame refusion cloud;
``accepted`` separately states whether the candidate improved the geometry.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .appearance import AppearanceConfig
from .bounded_backend import BoundedBackendConfig
from .contracts import load_manifest, load_trajectory, sha256_file
from .geometry_backend import GeometryBootstrapConfig
from .pose_graph import CorrectionAuditConfig, PoseGraphOptimizationConfig
from .runner import PrecommitGeometryConfig, _write_json, run_sequence
from .submaps import LoopProposalConfig, SubmapConfig
from .visual_verification import VisualVerificationConfig


def load_unified_config(path: Path, *, clip_download_root: Path) -> dict:
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema", "development_only", "promotion_eligible", "submap",
        "proposal", "appearance", "registration", "optimization",
        "visual", "bounded", "correction", "geometry",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("unified config has missing or unknown sections")
    if value["schema"] != "unified_pose_backend.v1":
        raise ValueError("unsupported unified config schema")
    if value["development_only"] is not True or value["promotion_eligible"] is not False:
        raise ValueError("this profile is a development experiment")
    bounded = dict(value["bounded"])
    if "correction_backtracking_scales" in bounded:
        bounded["correction_backtracking_scales"] = tuple(bounded["correction_backtracking_scales"])
    config = {
        "submap_config": SubmapConfig(**value["submap"]),
        "proposal_config": LoopProposalConfig(**value["proposal"]),
        "appearance_config": AppearanceConfig(
            **value["appearance"], download_root=str(clip_download_root.resolve()),
        ),
        "geometry_config": GeometryBootstrapConfig(**value["registration"]),
        "pose_graph_config": PoseGraphOptimizationConfig(**value["optimization"]),
        "visual_config": VisualVerificationConfig(**value["visual"]),
        "bounded_config": BoundedBackendConfig(**bounded),
        "correction_config": CorrectionAuditConfig(**value["correction"]),
        "precommit_geometry_config": PrecommitGeometryConfig(**value["geometry"]),
    }
    if (
        config["proposal_config"].policy != "hybrid36"
        or config["appearance_config"].provider != "openai_clip"
        or config["geometry_config"].decision_version != 2
        or config["geometry_config"].correspondence_policy != "baseline_mutual_fpfh"
    ):
        raise ValueError("unified recipe requires Hybrid36 and baseline geometric registration")
    geometry = config["precommit_geometry_config"]
    correction = config["correction_config"]
    if not (
        config["visual_config"].enabled
        and config["bounded_config"].enabled
        and geometry.enabled and geometry.require_scene_improvement
        and geometry.frame_stride == 1
        and config["pose_graph_config"].robustifier == "huber"
        and correction.propagation == "legacy_slerp_linear"
        and correction.maximum_absolute_correction_translation_m is not None
        and correction.maximum_absolute_correction_rotation_deg is not None
    ):
        raise ValueError("unified profile cannot disable PnP, bounded Huber or full-frame Guard")
    return config


def run_unified_sequence(
    *, manifest_path: Path, trajectory_path: Path, output_dir: Path,
    config_path: Path, clip_download_root: Path,
) -> dict:
    from reconstruction.rgbd_refusion import FullRefusionRequest, run_full_rgbd_refusion

    config = load_unified_config(config_path, clip_download_root=clip_download_root)
    result = run_sequence(
        arm="candidate", manifest_path=manifest_path,
        trajectory_path=trajectory_path, output_dir=output_dir, **config,
    )
    output_dir = Path(output_dir).resolve()
    committed = output_dir / "trajectory.json"
    poses, _ = load_trajectory(committed)
    manifest = load_manifest(manifest_path)
    frame_ids = tuple(frame.frame_id for frame in manifest.frames)
    geometry_config = config["precommit_geometry_config"]
    refusion_arm = "candidate" if result["accepted"] else "baseline"
    receipt_path = output_dir / "precommit_refusion" / refusion_arm / "refusion_result.json"
    refusion = None
    if receipt_path.is_file():
        candidate_receipt = json.loads(receipt_path.read_text())
        if (
            candidate_receipt.get("status") == "completed"
            and candidate_receipt.get("integrated_frame_count") == len(poses) == len(frame_ids)
            and candidate_receipt.get("requested_frame_count") == len(frame_ids)
            and candidate_receipt.get("trajectory_pose_count") == len(poses)
            and candidate_receipt.get("trajectory_sha256") == sha256_file(committed)
            and candidate_receipt.get("manifest_sha256") == sha256_file(manifest_path)
            and candidate_receipt.get("identity_fallback_used") is False
            and candidate_receipt.get("gt_consumed") is False
            and all(candidate_receipt.get(name) == getattr(geometry_config, name)
                    for name in ("voxel_length_m", "sdf_trunc_m", "depth_trunc_m"))
            and Path(candidate_receipt["cloud"]).is_file()
            and sha256_file(Path(candidate_receipt["cloud"])) == candidate_receipt.get("cloud_sha256")
        ):
            refusion = candidate_receipt
    if refusion is None:
        refusion = run_full_rgbd_refusion(FullRefusionRequest(
            manifest=manifest_path, trajectory=committed,
            output_dir=output_dir / "committed_refusion",
            fused_frame_ids=frame_ids,
            voxel_length_m=geometry_config.voxel_length_m,
            sdf_trunc_m=geometry_config.sdf_trunc_m,
            depth_trunc_m=geometry_config.depth_trunc_m,
        ))
    if refusion["integrated_frame_count"] != len(poses) or len(poses) != len(frame_ids):
        raise RuntimeError("unified refusion did not preserve all admitted frames")
    final = {
        **result,
        "schema": "unified_pose_result.v1",
        "development_only": True,
        "promotion_eligible": False,
        "config": {name: asdict(row) for name, row in config.items()},
        "config_sha256": sha256_file(config_path),
        "committed_trajectory": str(committed),
        "committed_trajectory_sha256": sha256_file(committed),
        "source_trajectory_sha256": sha256_file(trajectory_path),
        "rollback_byte_identical": (
            not result["accepted"] and sha256_file(committed) == sha256_file(trajectory_path)
        ),
        "final_cloud": refusion["cloud"],
        "final_cloud_sha256": refusion["cloud_sha256"],
        "integrated_frame_count": refusion["integrated_frame_count"],
        "gt_consumed": False,
    }
    _write_json(output_dir / "unified_result.json", final)
    return final
