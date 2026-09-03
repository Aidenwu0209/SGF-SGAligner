#!/usr/bin/env python3
"""GT-isolated 3RScan reference/rescan registration matrix.

``run`` consumes only transform-free split selection and reconstructed clouds.
``evaluate`` is a separate process that opens 3RScan transforms and frame poses.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from pose_pipeline.contracts import (
    load_trajectory, sha256_file, stable_json_sha256, validate_se3,
)
from pose_pipeline.evaluation import paired_bootstrap_improvement
from pose_pipeline.geometry_backend import (
    GeometryBootstrapConfig, dense_verification, fpfh_correspondences,
    register_submaps_bidirectional,
)
from pose_pipeline.robust_backend import (
    RobustPoseConfig, pygcransac_hypothesis, spatial_support,
    transform_distance,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    points = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 500:
        raise ValueError(f"cloud has fewer than 500 finite points: {path}")
    return np.ascontiguousarray(points)


def icp(source: np.ndarray, reference: np.ndarray, initial: np.ndarray) -> np.ndarray:
    import open3d as o3d

    source_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    reference_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference))
    source_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.30, max_nn=40))
    reference_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.30, max_nn=40))
    result = o3d.pipelines.registration.registration_icp(
        source_cloud, reference_cloud, 0.15, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    return validate_se3(np.asarray(result.transformation))


def baseline_registration(
    source: np.ndarray, reference: np.ndarray,
    geometry: GeometryBootstrapConfig,
) -> dict:
    source_xyz, reference_xyz, source_corr, reference_corr = fpfh_correspondences(
        source, reference, geometry,
    )
    hypothesis, reason = pygcransac_hypothesis(
        source_corr, reference_corr, threshold_m=0.05,
    )
    base = {
        "schema": "scan3r_pair_baseline.v1",
        "solver": "single_pygcransac_plus_icp",
        "accepted": False,
        "gt_consumed": False,
    }
    if hypothesis is None:
        return {**base, "reason": reason or "no_hypothesis"}
    initial = np.asarray(hypothesis["transform"], dtype=np.float64)
    refined = icp(source_xyz, reference_xyz, initial)
    update_rotation, update_translation = transform_distance(initial, refined)
    verification = dense_verification(
        source_xyz, reference_xyz, refined, geometry.verification_distance_m,
    )
    extent, second = spatial_support(source_corr)
    gates = {
        "minimum_support": int(hypothesis["support_count"]) >= 6,
        "spatial_extent": extent >= 2.0,
        "spatial_second_axis": second >= 0.10,
        "icp_rotation_update": update_rotation <= 10.0,
        "icp_translation_update": update_translation <= 0.20,
        "minimum_overlap": verification["minimum_overlap"] >= 0.10,
    }
    accepted = all(gates.values())
    return {
        **base,
        "accepted": accepted,
        "reason": "dense_gates_pass" if accepted else "dense_gates_reject",
        "transform": refined.tolist() if accepted else None,
        "hypothesis": hypothesis,
        "verification": verification,
        "icp_update_rotation_deg": update_rotation,
        "icp_update_translation_m": update_translation,
        "spatial_extent_m": extent,
        "spatial_second_axis_m": second,
        "gates": gates,
    }


def candidate_registration(
    source: np.ndarray, reference: np.ndarray,
    robust: RobustPoseConfig, geometry: GeometryBootstrapConfig,
) -> dict:
    return register_submaps_bidirectional(source, reference, robust, geometry)


def _safe_selection(path: Path) -> dict:
    selection = json.loads(path.read_text())
    if selection.get("schema") != "scan3r_pose_selection.v1":
        raise ValueError("selection schema mismatch")
    unsigned = dict(selection)
    expected = unsigned.pop("payload_sha256", None)
    if expected != stable_json_sha256(unsigned):
        raise ValueError("selection payload SHA mismatch")
    if selection.get("contains_transforms") is not False:
        raise ValueError("inference selection contains transforms")
    def forbidden_key(value: object) -> bool:
        if isinstance(value, dict):
            return any(
                "transform" in str(key).lower() or forbidden_key(child)
                for key, child in value.items()
                if key != "contains_transforms"
            )
        if isinstance(value, list):
            return any(forbidden_key(child) for child in value)
        return False
    if forbidden_key(selection):
        raise ValueError("inference selection has transform-like content")
    return selection


def run_inference(args: argparse.Namespace) -> None:
    selection = _safe_selection(args.selection)
    args.output.mkdir(parents=True, exist_ok=False)
    robust, geometry = RobustPoseConfig(), GeometryBootstrapConfig()
    rows = []
    for group in selection.get("groups", []):
        ids = [group["reference"], *group.get("scans", [])]
        present = {
            row["sequence_id"] for row in selection["sequences"]
            if row.get("present") is True
        }
        ids = [sequence_id for sequence_id in ids if sequence_id in present]
        for target_index, source_index in combinations(range(len(ids)), 2):
            target, source = ids[target_index], ids[source_index]
            pair_id = f"{source}_to_{target}"
            pair_dir = args.output / "pairs" / pair_id
            pair_dir.mkdir(parents=True, exist_ok=False)
            row = {
                "pair_id": pair_id,
                "group_reference": group["reference"],
                "split": group["split"],
                "source": source,
                "reference": target,
                "status": "failed",
            }
            try:
                arms = {}
                for arm in ("baseline", "candidate"):
                    source_path = args.sequence_results / source / f"{arm}_refusion" / "refused.ply"
                    target_path = args.sequence_results / target / f"{arm}_refusion" / "refused.ply"
                    source_points, target_points = read_points(source_path), read_points(target_path)
                    result = (
                        baseline_registration(source_points, target_points, geometry)
                        if arm == "baseline"
                        else candidate_registration(source_points, target_points, robust, geometry)
                    )
                    arms[arm] = {
                        "source_cloud_sha256": sha256_file(source_path),
                        "reference_cloud_sha256": sha256_file(target_path),
                        "registration": result,
                    }
                payload = {
                    "schema": "scan3r_pair_inference.v1",
                    "pair_id": pair_id,
                    "group_reference": group["reference"],
                    "split": group["split"],
                    "source": source,
                    "reference": target,
                    "arms": arms,
                    "gt_consumed": False,
                }
                write_json(pair_dir / "inference.json", payload)
                row.update({
                    "status": "completed",
                    "baseline_accepted": arms["baseline"]["registration"]["accepted"],
                    "candidate_accepted": arms["candidate"]["registration"]["accepted"],
                })
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            write_json(pair_dir / "row.json", row)
            print(json.dumps(row, sort_keys=True), flush=True)
    summary = {
        "schema": "scan3r_pair_inference_matrix.v1",
        "pair_count": len(rows),
        "completed_count": sum(row["status"] == "completed" for row in rows),
        "rows": rows,
        "selection_sha256": sha256_file(args.selection),
        "gt_consumed": False,
    }
    write_json(args.output / "summary.json", summary)


def metadata_groups(path: Path) -> dict[str, dict]:
    metadata = json.loads(path.read_text())
    return {group["reference"]: group for group in metadata}


def scan_to_group_transforms(group: dict) -> dict[str, np.ndarray]:
    transforms = {group["reference"]: np.eye(4)}
    for scan in group.get("scans", []):
        transforms[scan["reference"]] = validate_se3(
            np.asarray(scan["transform"], dtype=np.float64).reshape(4, 4).T,
            f"3RScan group transform {scan['reference']}",
        )
    return transforms


def world_to_scan_alignment(
    data_root: Path, sequence_results: Path, sequence_id: str, arm: str,
) -> np.ndarray:
    trajectory, _ = load_trajectory(
        sequence_results / sequence_id / arm / "trajectory.json",
    )
    sequence = data_root / sequence_id / "sequence"
    for pose in trajectory:
        path = sequence / f"frame-{pose.frame_id:06d}.pose.txt"
        try:
            truth = validate_se3(np.loadtxt(path), f"3RScan pose {sequence_id}/{pose.frame_id}")
        except ValueError:
            continue
        return validate_se3(truth @ np.linalg.inv(pose.t_world_camera))
    raise ValueError(f"no finite pose alignment for {sequence_id}/{arm}")


def evaluate(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    groups = metadata_groups(args.metadata)
    inference_summary = json.loads((args.inference / "summary.json").read_text())
    rows, estimates = [], {"baseline": {}, "candidate": {}}
    alignment_cache = {}
    for item in inference_summary["rows"]:
        if item.get("status") != "completed":
            row = {
                "pair_id": item["pair_id"],
                "group_reference": item["group_reference"],
                "split": item["split"],
                "source": item["source"],
                "reference": item["reference"],
                "official_reference_pair": (
                    item["reference"] == item["group_reference"]
                ),
                "inference_status": "failed",
                "inference_error": item.get("error"),
            }
            for arm in ("baseline", "candidate"):
                row.update({
                    f"{arm}_accepted": False,
                    f"{arm}_rre_deg": None,
                    f"{arm}_rte_m": None,
                    f"{arm}_recall_5deg_02m": False,
                    f"{arm}_catastrophic_accept": False,
                })
            rows.append(row)
            continue
        pair = json.loads(
            (args.inference / "pairs" / item["pair_id"] / "inference.json").read_text()
        )
        group = groups[pair["group_reference"]]
        group_transforms = scan_to_group_transforms(group)
        source, target = pair["source"], pair["reference"]
        truth_scan = validate_se3(
            np.linalg.inv(group_transforms[target]) @ group_transforms[source],
        )
        row = {
            "pair_id": pair["pair_id"], "group_reference": pair["group_reference"],
            "split": pair["split"], "source": source, "reference": target,
            "official_reference_pair": target == pair["group_reference"],
            "inference_status": "completed",
        }
        for arm in ("baseline", "candidate"):
            for sequence_id in (source, target):
                key = (sequence_id, arm)
                if key not in alignment_cache:
                    alignment_cache[key] = world_to_scan_alignment(
                        args.data_root, args.sequence_results, sequence_id, arm,
                    )
            truth = validate_se3(
                np.linalg.inv(alignment_cache[(target, arm)])
                @ truth_scan @ alignment_cache[(source, arm)],
            )
            registration = pair["arms"][arm]["registration"]
            accepted = registration.get("accepted") is True
            rotation = translation = None
            if accepted:
                estimate = validate_se3(registration["transform"])
                rotation, translation = transform_distance(estimate, truth)
                estimates[arm][(target, source)] = estimate
                estimates[arm][(source, target)] = np.linalg.inv(estimate)
            row.update({
                f"{arm}_accepted": accepted,
                f"{arm}_rre_deg": rotation,
                f"{arm}_rte_m": translation,
                f"{arm}_recall_5deg_02m": bool(
                    accepted and rotation <= 5.0 and translation <= 0.20
                ),
                f"{arm}_catastrophic_accept": bool(
                    accepted and (rotation > 20.0 or translation > 0.50)
                ),
            })
        rows.append(row)
    validation_rows = [row for row in rows if row["split"] == "validation"]
    official = [
        row for row in validation_rows if row["official_reference_pair"]
    ]
    cycle_rows = []
    for group_reference, group in groups.items():
        ids = [group_reference, *[scan["reference"] for scan in group.get("scans", [])]]
        for a, b, c in combinations(ids, 3):
            result = {"group_reference": group_reference, "nodes": [a, b, c]}
            for arm in ("baseline", "candidate"):
                available = all(key in estimates[arm] for key in ((a, b), (b, c), (a, c)))
                rotation = translation = None
                if available:
                    rotation, translation = transform_distance(
                        estimates[arm][(a, c)],
                        estimates[arm][(a, b)] @ estimates[arm][(b, c)],
                    )
                result.update({
                    f"{arm}_available": available,
                    f"{arm}_cycle_rotation_deg": rotation,
                    f"{arm}_cycle_translation_m": translation,
                })
            if result["baseline_available"] or result["candidate_available"]:
                cycle_rows.append(result)
    aggregate = {
        "schema": "scan3r_pair_evaluation.v1",
        "official_pairs": len(official),
        "official_pairs_with_completed_inference": sum(
            row["inference_status"] == "completed" for row in official
        ),
        "validation_pair_count": len(validation_rows),
    }
    for arm in ("baseline", "candidate"):
        aggregate[arm] = {
            "accepted": sum(row[f"{arm}_accepted"] for row in official),
            "recall_5deg_02m": sum(row[f"{arm}_recall_5deg_02m"] for row in official) / max(len(official), 1),
            "catastrophic_accepts": sum(row[f"{arm}_catastrophic_accept"] for row in official),
            "all_validation_accepted": sum(
                row[f"{arm}_accepted"] for row in validation_rows
            ),
            "all_validation_catastrophic_accepts": sum(
                row[f"{arm}_catastrophic_accept"] for row in validation_rows
            ),
            "cycle_available": sum(row[f"{arm}_available"] for row in cycle_rows),
        }
    baseline_primary = [
        (row["baseline_rre_deg"] / 5.0 + row["baseline_rte_m"] / 0.20)
        if row["baseline_accepted"] else 100.0 for row in official
    ]
    candidate_primary = [
        (row["candidate_rre_deg"] / 5.0 + row["candidate_rte_m"] / 0.20)
        if row["candidate_accepted"] else 100.0 for row in official
    ]
    aggregate["paired_primary_bootstrap"] = paired_bootstrap_improvement(
        baseline_primary, candidate_primary,
    ) if official else None
    aggregate["passes_gate"] = bool(
        official
        and aggregate["candidate"]["all_validation_catastrophic_accepts"] == 0
        and aggregate["candidate"]["recall_5deg_02m"]
        >= aggregate["baseline"]["recall_5deg_02m"] - 0.01
        and aggregate["paired_primary_bootstrap"]["passes_10pct_and_positive_ci"]
    )
    aggregate["rows"] = rows
    aggregate["cycles"] = cycle_rows
    aggregate["gt_role"] = "evaluation_only"
    aggregate["metadata_sha256"] = sha256_file(args.metadata)
    write_json(args.output / "summary.json", aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--sequence-results", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(handler=run_inference)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--inference", type=Path, required=True)
    evaluation.add_argument("--metadata", type=Path, required=True)
    evaluation.add_argument("--data-root", type=Path, required=True)
    evaluation.add_argument("--sequence-results", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.set_defaults(handler=evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
