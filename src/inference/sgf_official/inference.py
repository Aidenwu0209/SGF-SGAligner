"""Unified three-mode inference CLI (project adaptation code).

    python -m inference_sgf \
        --mode official_oracle|official_sgf_predicted|legacy_geometry_baseline \
        --pair-id <src>_to_<ref> --output <dir>

official_oracle:        3DSSG GT graph -> official pct/gat/rel/attr ->
                        official checkpoint -> official matching ->
                        official GeoTransformer + pygcransac
official_sgf_predicted: SGF predicted graph -> official pct/gat/rel ->
                        official checkpoint (no attribute module) ->
                        official matching -> GeoTransformer
legacy_geometry_baseline: legacy GeometrySGAligner through its own
                        frozen environment (subprocess boundary).

All modes share pair order, frames, seed=42 and metre conventions.
GT enters ONLY the evaluation stage (metrics/strict), never inputs.
Failures return structured status; only accepted decisions write
transform.txt.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters.sgf.data_sources import (
    DATA_ROOT, OracleGraphSource, PredictedGraphSource,
    attribute_vocab_164, load_anchor_ids, load_gt_transform,
    load_oracle_anchor_ids, oracle_gt_transform, _source_inseg_cloud,
)
from adapters.sgf.graph_adapter import adapt_graph, merge_pair_contracts
from adapters.sgf.object_adapter import adapt_objects
from adapters.sgf.tensor_contract import (
    validate_contract, validate_pair_dict,
)
from safety.registration_decision import (
    FROZEN_RULE, evaluate_registration_decision, rotation_angle_deg,
    spatial_support, write_decision_files,
)
from safety import decision_features as dfx

OFFICIAL_SNAPSHOT = (
    "/home/aidenwu/Documents/sgaligner-sgf-official/checkpoints/release/"
    "sgaligner_pct_gat_rel_attr.pth.tar"
)
GEOT_SNAPSHOT = (
    "/home/aidenwu/Documents/sgaligner-sgf-official/checkpoints/"
    "geotransformer/geotransformer-3dmatch.pth.tar"
)
STRICT = (5.0, 0.20)
RELAXED = (10.0, 0.30)
REG_K = 3          # official reg_k (top-k node correspondences)
NUM_P2P = 1000     # official num_p2p_corrs


INFERENCE_FUNCTIONS = (
    "build_pair_inputs", "official_forward", "official_matching",
    "official_registration", "geotransformer_forward",
)


class GTAccessAudit(ast.NodeVisitor):
    """AST guard over the INFERENCE-INPUT functions only.

    The evaluation section (post-registration metrics against GT) is
    exempt by design; GT must never enter feature construction,
    matching, GeoTransformer or RANSAC inputs.
    """

    FORBIDDEN = ("load_gt_transform", "gt_transform", "anchor_pairs",
                 "load_anchor_ids", "load_oracle_anchor_ids",
                 "oracle_gt_transform")

    def __init__(self, path: str):
        self.violations = []
        tree = ast.parse(Path(path).read_text())
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in INFERENCE_FUNCTIONS
            ):
                self.visit(node)

    def visit_Attribute(self, node):
        if node.attr in ("gt_transform", "anchor_pairs"):
            self.violations.append(f"line {node.lineno}: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.FORBIDDEN:
            self.violations.append(f"line {node.lineno}: {node.id}")
        self.generic_visit(node)


def build_pair_inputs(
    pair_id: str, mode: str,
    sampling_mode: str = "deterministic_pcg64",
):
    src_scan, ref_scan = pair_id.split("_to_")
    vocab = attribute_vocab_164() if mode == "official_oracle" else None
    source = (
        OracleGraphSource() if mode == "official_oracle"
        else PredictedGraphSource()
    )
    contracts = []
    iteration_orders = []
    for scan_id in (src_scan, ref_scan):
        result = source.load(scan_id)
        iteration_order = None
        if sampling_mode == "official_mt19937":
            # oracle scans reproduce the official objects.json draw
            # order; predicted graphs have no official reference, so
            # the adapter-defined canonical (sorted-id) order is used
            if hasattr(source, "official_object_order"):
                iteration_order = source.official_object_order(scan_id)
        objects = adapt_objects(
            result.segments, seed=42,
            sampling_mode=sampling_mode,
            scan_seed=0,
            iteration_order=iteration_order,
        )
        iteration_orders.append(objects.sampling_iteration_order)
        kwargs = dict(
            directed_pairs=result.directed_pairs,
            relation_triples=result.relation_triples,
        )
        if mode == "official_oracle":
            kwargs.update(
                attributes_per_object=result.attributes_per_object,
                attribute_vocab=vocab,
            )
        contract = adapt_graph(objects, mode=mode, **kwargs)
        contract.scene_ids = [scan_id]
        errors = validate_contract(contract)
        if errors:
            raise ValueError(f"contract invalid for {scan_id}: {errors}")
        contracts.append(contract)

    # pcl_center per official semantics (Scan3R getitem, test split):
    # the mean of the SOURCE scan's full scene points.
    if mode == "official_oracle":
        from plyfile import PlyData

        ply = PlyData.read(DATA_ROOT / src_scan / "labels.instances.annotated.v2.ply")
        vertex = ply["vertex"]
        pcl_center = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=1
        ).astype(np.float64).mean(axis=0)
        center_definition = "oracle: source full-scene PLY point mean"
    else:
        with np.load(_source_inseg_cloud(src_scan)) as data:
            pcl_center = np.asarray(
                data["xyz"], dtype=np.float64
            ).mean(axis=0)
        center_definition = (
            "predicted: source InSeg full stable surfel cloud mean"
        )
    data_dict = merge_pair_contracts(contracts[0], contracts[1], pcl_center)
    data_dict["pcl_center_definition"] = center_definition
    data_dict["sampling_mode"] = sampling_mode
    data_dict["scan_seed"] = 0 if sampling_mode == "official_mt19937" else None
    data_dict["sampling_iteration_order"] = {
        "src": iteration_orders[0], "ref": iteration_orders[1],
    }
    errors = validate_pair_dict(data_dict)
    if errors:
        raise ValueError(f"pair dict invalid: {errors}")
    return data_dict, contracts


def official_forward(data_dict: dict, mode: str, device: str = "cuda"):
    from aligner.sg_aligner import MultiModalEncoder

    modules = (
        ["pct", "gat", "rel", "attr"] if mode == "official_oracle"
        else ["pct", "gat", "rel"]
    )
    model = MultiModalEncoder(
        modules=modules, rel_dim=41,
        attr_dim=164,
    ).to(device)
    state = torch.load(OFFICIAL_SNAPSHOT, map_location=device,
                       weights_only=False)
    if mode == "official_oracle":
        model.load_state_dict(state["model"], strict=True)
    else:
        # predicted mode ADAPTATION PATCH (recorded in
        # MIGRATION_MAP.md): the official forward cannot run a 4-row
        # fusion with 3 modalities (hard assert in MultiModalFusion),
        # so pct/gat/rel run with a 3-row fusion initialised from the
        # official checkpoint's first three rows (checkpoint module
        # order: pct, gat, rel, attr). The attr head is NOT
        # instantiated and no 164-D input is ever read or fabricated;
        # attribute_available=false is recorded in every output.
        official_state = dict(state["model"])
        fusion_rows = official_state.pop("fusion.weight")[:3].clone()
        model.load_state_dict(official_state, strict=False)
        with torch.no_grad():
            model.fusion.weight.copy_(fusion_rows)
    model.eval()

    # official collate keeps [N,512,3] flat tensors with batch_size=1
    batch = {
        "tot_obj_pts": torch.from_numpy(
            data_dict["tot_obj_pts"]
        ).to(device),
        "tot_bow_vec_object_edge_feats": torch.from_numpy(
            data_dict["tot_bow_vec_object_edge_feats"]
        ).to(device),
        "tot_rel_pose": torch.from_numpy(
            data_dict["tot_rel_pose"]
        ).to(device),
        "edges": torch.from_numpy(
            data_dict["edges"].astype(np.int64)
        ).to(device),
        "graph_per_obj_count": [np.asarray(
            data_dict["graph_per_obj_count"], dtype=np.int64
        )],
        "graph_per_edge_count": [np.asarray(
            data_dict["graph_per_edge_count"], dtype=np.int64
        )],
        "batch_size": 1,
    }
    if data_dict["tot_bow_vec_object_attr_feats"] is not None:
        batch["tot_bow_vec_object_attr_feats"] = torch.from_numpy(
            data_dict["tot_bow_vec_object_attr_feats"]
        ).to(device)
    elif mode == "official_sgf_predicted":
        # SGF has no 164-D attribute source. The official forward reads
        # this key unconditionally, but with 'attr' absent from modules
        # the value is NEVER consumed: meta_embedding_attr never runs,
        # so outputs are bit-identical to a hypothetical read-free
        # official forward. This is an unused read-buffer ONLY —
        # attribute_available=false is recorded in every output's
        # provenance and no oracle equivalence is claimed.
        batch["tot_bow_vec_object_attr_feats"] = torch.zeros(
            (data_dict["tot_obj_pts"].shape[0], 164)
        ).to(device)
    with torch.no_grad():
        output = model(batch)
    embedding = (
        output["joint"] if len(output) > 1 else output[modules[0]]
    )
    return embedding.cpu().numpy(), state.get("epoch")


def OracleGraphSource_anchor_segments(scan_id):
    source = OracleGraphSource()
    return source.load(scan_id).segments


def official_matching(embedding: np.ndarray, src_count: int):
    """Official ranking + top-k cross-graph node correspondences."""
    normed = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    sim = 1.0 - normed @ normed.T
    rank_list = np.argsort(sim, axis=1)
    node_corrs = []
    for idx in range(src_count):
        ranks = [r for r in rank_list[idx] if r != idx][:REG_K]
        for target in ranks:
            if target < src_count:
                continue  # within-graph match
            node_corrs.append((idx, int(target)))
    return node_corrs, rank_list, sim


def geotransformer_forward(
    src_points: np.ndarray, ref_points: np.ndarray, device: str = "cuda"
):
    """Official GeoTransformer point registration (one point pair)."""
    from GeoTransformer.config import make_cfg
    from GeoTransformer.geotransformer.utils.data import (
        registration_collate_fn_stack_mode,
    )
    from GeoTransformer.model import create_model
    from engine.registration_evaluator import RegistrationEvaluator

    if not hasattr(geotransformer_forward, "_model"):
        cfg = make_cfg()
        model = create_model(cfg).to(device)
        state = torch.load(GEOT_SNAPSHOT, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        geotransformer_forward._model = model
        geotransformer_forward._cfg = cfg
        geotransformer_forward._neighbor_limits = (
            RegistrationEvaluator.__init__.__defaults__ or None
        )
        from GeoTransformer.config import make_cfg as _mc
        geotransformer_forward._neighbor_limits = [18, 16, 14, 12]

    model = geotransformer_forward._model
    cfg = geotransformer_forward._cfg
    npoint = 10000
    if len(src_points) > npoint:
        src_points = src_points[
            np.random.default_rng(42).choice(
                len(src_points), npoint, replace=False
            )
        ]
    if len(ref_points) > npoint:
        ref_points = ref_points[
            np.random.default_rng(42).choice(
                len(ref_points), npoint, replace=False
            )
        ]
    # Exact sparsity check AFTER collation, BEFORE the model: read the
    # collated fine/coarse lengths and compare with the official
    # num_points_in_patch — no proxy heuristics, no point duplication.

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": np.ones_like(ref_points[:, :1], dtype=np.float32),
        "src_feats": np.ones_like(src_points[:, :1], dtype=np.float32),
        "transform": np.eye(4, dtype=np.float32),
    }
    with torch.no_grad():
        staged = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size, cfg.backbone.init_radius,
            geotransformer_forward._neighbor_limits,
        )
        from utils import torch_util

        data_dict_stage = staged  # lengths readable pre-cuda
        data_dict = torch_util.to_cuda(staged)
        lengths = data_dict_stage["lengths"]
        min_patch = int(cfg.model.num_points_in_patch)
        angle_k = int(cfg.geotransformer.angle_k)
        # topk(k+1) inside get_embedding_indices runs on EVERY stage's
        # point set; each stage must have >= angle_k+1 points, and the
        # fine stage additionally >= num_points_in_patch
        stage_counts = {
            f"stage{stage}_src": int(lengths[stage][0])
            for stage in range(len(lengths))
        }
        stage_counts.update({
            f"stage{stage}_ref": int(lengths[stage][1])
            for stage in range(len(lengths))
        })
        insufficient = (
            int(lengths[1][0]) < min_patch
            or int(lengths[1][1]) < min_patch
            or any(
                int(lengths[stage][side]) < angle_k + 1
                for stage in range(1, len(lengths))
                for side in (0, 1)
            )
        )
        if insufficient:
            return "insufficient_post_voxel_points", {
                **stage_counts,
                "num_points_in_patch": min_patch,
                "angle_k": angle_k,
            }
        try:
            output = model(data_dict)
        except (RuntimeError, IndexError, ValueError) as exc:
            import traceback

            return (
                "geotransformer_runtime_error",
                {
                    "error": repr(exc),
                    "traceback": traceback.format_exc()[-1500:],
                },
            )
    output = {
        k: (v.cpu().numpy() if torch.is_tensor(v) else v)
        for k, v in output.items()
    }
    return "ok", output


class NodePairFailure(Exception):
    """Typed per-node-pair registration failure."""

    def __init__(self, stage, detail):
        self.stage = stage
        self.detail = detail
        super().__init__(f"{stage}: {detail}")


def official_registration(
    data_dict, node_corrs, mode, device="cuda", pair_id=""
):
    """Aligner-style registration on FULL world-frame object points.

    GeoTransformer reads registration_pts (complete, deduplicated,
    metre, world coordinates), deterministically subsampled to 10k.
    Correspondence points stay in world coordinates end-to-end; the
    pygcransac result therefore needs no centring composition.
    """
    import pygcransac
    from scipy.spatial import cKDTree

    objects = data_dict["registration_pts"]
    id2oid = data_dict["registration_id2oid"]
    point_corrs = {"src": [], "ref": [], "scores": []}
    per_pair_used = []
    node_pair_failures = []
    for src_idx, ref_idx in node_corrs:
        # node_corrs reference GLOBAL merged indices already
        src_oid = id2oid.get(int(src_idx))
        ref_oid = id2oid.get(int(ref_idx))
        detail_head = {
            "pair_id": pair_id,
            "src_index": int(src_idx), "ref_index": int(ref_idx),
            "src_object_id": src_oid, "ref_object_id": ref_oid,
        }
        src_pts = objects.get(int(src_idx))
        ref_pts = objects.get(int(ref_idx))
        if src_pts is None or ref_pts is None:
            node_pair_failures.append({
                **detail_head, "stage": "insufficient_raw_points",
                "reason": "object below min_stable_surfels at build",
            })
            continue
        detail_head.update({
            "src_input_points": int(len(src_pts)),
            "ref_input_points": int(len(ref_pts)),
        })
        if len(src_pts) < 50 or len(ref_pts) < 50:
            node_pair_failures.append({
                **detail_head, "stage": "insufficient_raw_points",
                "reason": "fewer than 50 registration points",
            })
            continue
        status, output = geotransformer_forward(
            src_pts, ref_pts, device=device
        )
        if status == "insufficient_post_voxel_points":
            node_pair_failures.append({
                **detail_head,
                "stage": "insufficient_post_voxel_points",
                "reason": "fewer than 32 occupied 2.5cm voxels",
                "post_voxel_src": None, "post_voxel_ref": None,
            })
            continue
        if status == "geotransformer_runtime_error":
            node_pair_failures.append({
                **detail_head,
                "stage": "geotransformer_runtime_error",
                **(output or {}),
            })
            continue
        src_corr = output["src_corr_points"]
        ref_corr = output["ref_corr_points"]
        scores = output["corr_scores"]
        if len(src_corr) == 0:
            node_pair_failures.append({
                **detail_head, "stage": "empty_point_correspondence",
                "reason": "geotransformer returned zero correspondences",
            })
            continue
        if len(scores) > max(NUM_P2P // max(len(node_corrs), 1), 1):
            keep = np.argsort(-scores)[
                : NUM_P2P // max(len(node_corrs), 1)
            ]
            src_corr, ref_corr, scores = (
                src_corr[keep], ref_corr[keep], scores[keep]
            )
        point_corrs["src"].append(src_corr)
        point_corrs["ref"].append(ref_corr)
        point_corrs["scores"].append(scores)
        per_pair_used.append((int(src_idx), int(ref_idx)))
    if not point_corrs["src"]:
        return None, per_pair_used, node_pair_failures
    src_all = np.concatenate(point_corrs["src"])
    ref_all = np.concatenate(point_corrs["ref"])
    corrs = np.concatenate([src_all, ref_all], axis=1).astype(np.float64)
    shifted = corrs - corrs.min(axis=0)
    try:
            est_transform, _inliers = pygcransac.findRigidTransform(
            np.ascontiguousarray(shifted),
            probabilities=[],
            threshold=0.05, neighborhood_size=4.0, sampler=1,
            min_iters=1000, max_iters=10000,
            spatial_coherence_weight=0.0, use_space_partitioning=True,
            neighborhood=0, conf=0.999, use_sprt=False,
        )
    except (RuntimeError, ValueError) as exc:
        import traceback

        return None, per_pair_used, [{
            "pair_id": pair_id, "stage": "ransac_failure",
            "error": repr(exc),
            "traceback": traceback.format_exc()[-1000:],
            "src_corr_points": int(len(src_all)),
            "ref_corr_points": int(len(ref_all)),
        }]
    if not isinstance(est_transform, np.ndarray) or est_transform.shape != (4, 4):
        return None, per_pair_used, [{
            "pair_id": pair_id, "stage": "ransac_failure",
            "error": f"pygcransac returned {type(est_transform).__name__}",
            "src_corr_points": int(len(src_all)),
            "ref_corr_points": int(len(ref_all)),
        }]
    # exact official composition (registration_evaluator.py:186-190):
    # row-major pygcransac convention + per-side min-shift undo
    min_coordinates = np.min(corrs, axis=0)
    T1 = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [-min_coordinates[0], -min_coordinates[1],
         -min_coordinates[2], 1],
    ])
    T2inv = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [min_coordinates[3], min_coordinates[4],
         min_coordinates[5], 1],
    ])
    est_transform = (T1 @ est_transform @ T2inv).T
    # registration points were WORLD-frame end-to-end; est_transform is
    # already the source->reference world transform (the min-shift was
    # undone by the official T1/T2inv algebra above).
    residual = np.linalg.norm(
        src_all @ est_transform[:3, :3].T + est_transform[:3, 3] - ref_all,
        axis=1,
    )
    inliers = int((residual <= 0.10).sum())
    return {
        "transform": est_transform,
        "inliers": inliers,
        "corrs": int(len(src_all)),
        "inlier_ratio": inliers / max(len(src_all), 1),
        "src_corr_points": src_all,
        "ref_corr_points": ref_all,
        "node_pairs_used": per_pair_used,
        "node_pair_failures": node_pair_failures,
    }, per_pair_used, node_pair_failures


def decision_features_full(
    data_dict, registration, node_corrs, pair_id, device="cuda",
    rule="C", bidirectional=True,
):
    """Real, corr-independent decision features + chosen rule verdict.

    Surfaces are the FULL world-frame registration points of the
    MATCHED objects (never GeoT corr points).  ICP runs on that union;
    bidirectional runs an independent reference->source estimation.
    Every feature carries source/units provenance.
    """
    from scipy.spatial import cKDTree

    objects = data_dict["registration_pts"]
    used = registration["node_pairs_used"]
    successful_pairs = len(used)
    failed_pairs = len(registration.get("node_pair_failures", []))
    total_pairs = successful_pairs + failed_pairs
    success_ratio = (
        successful_pairs / total_pairs if total_pairs else 0.0
    )

    src_surface = np.concatenate(
        [objects[int(a)] for a, _b in used]
    )
    ref_surface = np.concatenate(
        [objects[int(b)] for _a, b in used]
    )

    # matched spatial support (barycentres of used src objects)
    src_bary = np.asarray(
        [objects[int(a)].mean(axis=0) for a, _b in used]
    )
    extent, second = spatial_support(src_bary)

    ransac_transform = registration["transform"]

    evidence = dfx.surface_evidence(
        src_surface, ref_surface, ransac_transform, seed=42
    )
    icp = dfx.segment_icp(
        src_surface, ref_surface, ransac_transform, seed=42
    )

    bidir_rotation = None
    bidir_translation = None
    bidirectional_available = False
    if bidirectional:
        try:
            # true reverse: independent ref->src ICP seeded from the
            # inverse — never the inverse passed off as inference
            t_rs = dfx.segment_icp(
                ref_surface, src_surface, np.linalg.inv(ransac_transform),
                seed=43,
            ).transform
            bidir_rotation, bidir_translation = dfx.transform_discrepancy(
                ransac_transform, t_rs
            )
            bidirectional_available = True
        except Exception:
            bidirectional_available = False

    features = {
        "ransac_inliers": registration["inliers"],
        "ransac_inlier_ratio": registration["inlier_ratio"],
        "spatial_extent_m": float(extent),
        "spatial_second_axis_m": float(second),
        "icp_update_translation_m": icp.update_translation_m,
        "icp_update_rotation_deg": icp.update_rotation_deg,
        "bidirectional_rotation_deg": bidir_rotation,
        "bidirectional_translation_m": bidir_translation,
        "overlap_ratio": evidence.overlap_10cm,
        "icp_converged": icp.converged,
        "overlap_10cm": evidence.overlap_10cm,
        "overlap_5cm": evidence.overlap_5cm,
        "symmetric_trimmed_chamfer_m": evidence.symmetric_trimmed_chamfer_m,
        "median_residual_m": evidence.median_residual_m,
        "p90_residual_m": evidence.p90_residual_m,
        "icp_fitness": icp.fitness,
        "icp_rmse_m": icp.rmse_m,
        "node_pair_success_ratio": success_ratio,
        "successful_node_pairs": successful_pairs,
        "failed_node_pairs": failed_pairs,
        "bidirectional_available": bidirectional_available,
        "_provenance": {
            "surfaces": "matched-object FULL registration points (world, metres)",
            "surface_points": {
                "src": evidence.n_src_points, "ref": evidence.n_ref_points,
            },
            "surface_seed": evidence.seed,
            "icp": "deterministic NN-ICP on surface union",
            "bidirectional": (
                "independent ref->src ICP seeded from inverse"
                if bidirectional_available else "unavailable -> rule B/C reject"
            ),
            "units": "metres / degrees / ratios",
        },
    }
    # normalize None bidirectional values for the rule evaluators
    rule_features = dict(features)
    if not bidirectional_available:
        rule_features["bidirectional_rotation_deg"] = 1e9
        rule_features["bidirectional_translation_m"] = 1e9
    violations = dfx.RULE_EVALUATORS[rule](rule_features)
    decision = {
        "status": "accepted" if not violations else "rejected",
        "usable_for_reconstruction": not violations,
        "rejection_reasons": violations,
        "rule": f"fix2-{rule}",
        "rule_thresholds": dfx.RULE_THRESHOLDS,
        "features": {
            k: v for k, v in features.items() if k != "_provenance"
        },
        "feature_provenance": features["_provenance"],
    }
    return features, decision, icp


def run_pair(pair_id: str, mode: str, output_dir: Path,
             device: str = "cuda", decision_rule: str = "C") -> dict:
    import logging

    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"pair_id": pair_id, "mode": mode}
    try:
        if mode == "legacy_geometry_baseline":
            return run_legacy_baseline(pair_id, output_dir)
        data_dict, contracts = build_pair_inputs(pair_id, mode)
        oracle_segments = None
        (output_dir / "graph_input.json").write_text(json.dumps({
            "src_objects": int(data_dict["graph_per_obj_count"][0]),
            "ref_objects": int(data_dict["graph_per_obj_count"][1]),
            "src_edges": int(data_dict["graph_per_edge_count"][0]),
            "ref_edges": int(data_dict["graph_per_edge_count"][1]),
            "provenance": data_dict["provenance"],
        }, indent=2, default=str) + "\n")
        embedding, ckpt_epoch = official_forward(data_dict, mode, device)
        np.savez_compressed(
            output_dir / "official_embeddings.npz", embedding=embedding
        )
        src_count = data_dict["src_count"]
        node_corrs, rank_list, sim = official_matching(
            embedding, src_count
        )
        (output_dir / "node_matches.json").write_text(json.dumps({
            "node_corrs": [
                [int(a), int(b)] for a, b in node_corrs
            ],
        }, indent=2) + "\n")

        registration, used_pairs, node_failures = official_registration(
            data_dict, node_corrs, mode, device=device, pair_id=pair_id
        )
        if mode == "official_oracle":
            src_scan, ref_scan = pair_id.split("_to_")
            src_segments = OracleGraphSource_anchor_segments(src_scan)
            ref_segments = OracleGraphSource_anchor_segments(ref_scan)
            anchors = load_oracle_anchor_ids(
                src_scan, ref_scan, src_segments, ref_segments
            )
            gt = oracle_gt_transform(  # evaluation only (PLY frame)
                src_scan, ref_scan, src_segments, ref_segments
            )
        else:
            gt = load_gt_transform(pair_id)  # evaluation only
            anchors = load_anchor_ids(pair_id)

        # node metrics against GT anchors (original-id space)
        src_map = data_dict["src_object_id2idx"]
        ref_map = data_dict["ref_object_id2idx"]
        anchor_pairs_idx = {
            (src_map[s], ref_map[r] + src_count)
            for s, r in anchors if s in src_map and r in ref_map
        }
        predicted_pairs = set(
            (data_dict["obj_ids"][a], data_dict["obj_ids"][b])
            for a, b in node_corrs
        )
        predicted_idx = set(node_corrs)
        tp = len(predicted_idx & anchor_pairs_idx)
        node_p = tp / len(predicted_idx) if predicted_idx else 0.0
        node_r = tp / len(anchor_pairs_idx) if anchor_pairs_idx else 0.0
        node_f1 = (
            2 * node_p * node_r / max(node_p + node_r, 1e-12)
        )

        strict = relaxed = False
        rre = rte = None
        if registration is not None and gt is not None:

            t_world = registration["transform"]
            cos_r = (np.trace(t_world[:3, :3].T @ gt[:3, :3]) - 1) / 2
            rre = float(np.degrees(np.arccos(np.clip(cos_r, -1, 1))))
            rte = float(np.linalg.norm(t_world[:3, 3] - gt[:3, 3]))
            strict = rre <= STRICT[0] and rte <= STRICT[1]
            relaxed = rre <= RELAXED[0] and rte <= RELAXED[1]

            features, decision, icp_result = decision_features_full(
                data_dict, registration, node_corrs, pair_id,
                device=device, rule=decision_rule,
            )
            # final transform = ICP-refined (ICP update within bounds is
            # enforced by the rule; rejected pairs never write it)
            final_transform = (
                icp_result.transform
                if decision["usable_for_reconstruction"] else None
            )
            write_decision_files(
                output_dir, decision,
                final_transform if decision["usable_for_reconstruction"]
                else None,
            )
            (output_dir / "registration_result.json").write_text(
                json.dumps({
                    "rre": rre, "rte": rte, "strict": strict,
                    "relaxed": relaxed, "inliers": registration["inliers"],
                    "corrs": registration["corrs"],
                    "checkpoint_epoch": ckpt_epoch,
                    "node_pairs_used": used_pairs,
                    "node_pair_failures": registration.get(
                        "node_pair_failures", []
                    ),
                    "pcl_center": data_dict["pcl_center"].tolist(),
                    "pcl_center_definition":
                        data_dict["pcl_center_definition"],
                }, indent=2) + "\n"
            )
            status.update({
                "status": "ok",
                "strict": strict, "relaxed": relaxed,
                "rre": rre, "rte": rte,
                "node_precision": node_p, "node_recall": node_r,
                "node_f1": node_f1,
                "accepted": decision["usable_for_reconstruction"],
                "rejection_reasons": decision["rejection_reasons"],
            })
        else:
            (output_dir / "failure.json").write_text(json.dumps({
                "stage": "registration",
                "reason": "no point correspondences survived",
                "node_pair_failures": node_failures,
                "failure_stage_counts": {
                    stage: sum(
                        1 for f in node_failures if f["stage"] == stage
                    )
                    for stage in sorted({
                        f["stage"] for f in node_failures
                    })
                },
            }, indent=2) + "\n")
            status.update({
                "status": "failed", "failed_stage": "registration",
                "node_precision": node_p, "node_recall": node_r,
                "node_f1": node_f1, "strict": False, "relaxed": False,
            })
        return status
    except Exception as exc:  # noqa: BLE001 - structured failure
        import traceback

        (output_dir / "failure.json").write_text(json.dumps({
            "stage": "pipeline",
            "reason": repr(exc),
            "traceback": traceback.format_exc()[-2000:],
        }, indent=2) + "\n")
        return {"pair_id": pair_id, "mode": mode, "status": "failed",
                "failed_stage": "pipeline", "error": repr(exc)}


def run_legacy_baseline(pair_id: str, output_dir: Path) -> dict:
    """Legacy mode: subprocess boundary into the frozen legacy env."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_repo = "/home/aidenwu/Documents/inseg-sgaligner-sgf-context-v1"
    pair_path = (
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        f"delivery_stage1_20260823/training_dataset/pairs/{pair_id}/pair.json"
    )
    ckpt = (
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/phase6_registration_aware_closure/"
        "training/epoch_00055.pt"
    )
    script = (
        "import json, numpy as np, sys;\n"
        "from inseg_sgaligner.data import load_graph_pair;\n"
        "from inseg_sgaligner.inference import infer_pair;\n"
        f"infer_pair({str(pair_path)!r}, {ckpt!r}, {str(output_dir / 'legacy')!r},\n"
        "  embedding_modality='graph', modality_fallback='off',\n"
        "  match_score_margin=0.08, minimum_match_similarity=0.60,\n"
        "  minimum_geometric_support=8, ransac_iterations=3000,\n"
        "  ransac_threshold=0.12, icp_threshold=0.20,\n"
        "  registration_refiner='segment', seed=42, visualize=False);\n"
        "print('legacy_ok')\n"
    )
    result = subprocess.run(
        ["/home/aidenwu/miniconda3/envs/torch113/bin/python", "-c",
         script],
        capture_output=True, text=True,
        cwd=legacy_repo,
        env={
            "PATH": "/home/aidenwu/miniconda3/envs/torch113/bin:/usr/bin:/bin",
            "OMP_NUM_THREADS": "1",
        },
        timeout=1800,
    )
    if result.returncode != 0:
        (output_dir / "failure.json").write_text(json.dumps({
            "stage": "legacy_subprocess",
            "stderr": result.stderr[-1500:],
        }, indent=2) + "\n")
        return {"pair_id": pair_id, "mode": "legacy_geometry_baseline",
                "status": "failed", "failed_stage": "legacy_subprocess"}
    metrics_path = output_dir / "legacy" / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text())
        if metrics_path.exists() else {}
    )
    return {
        "pair_id": pair_id, "mode": "legacy_geometry_baseline",
        "status": "ok",
        "strict": bool(metrics.get("strict")),
        "relaxed": bool(metrics.get("relaxed")),
        "rre": metrics.get("rre"), "rte": metrics.get("rte"),
        "node_precision": metrics.get("node_precision"),
        "node_recall": metrics.get("node_recall"),
        "node_f1": metrics.get("node_f1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "official_oracle", "official_sgf_predicted",
            "legacy_geometry_baseline",
        ),
        required=True,
    )
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.mode == "official_sgf_predicted":
        audit = GTAccessAudit(str(Path(__file__)))
        # the predicted branch itself must not import GT loaders at
        # inference time; only the evaluation section may
        if audit.violations:
            print(
                json.dumps({
                    "status": "blocked", "reason": "GT access audit",
                    "violations": audit.violations,
                })
            )
            sys.exit(2)
    status = run_pair(
        args.pair_id, args.mode, Path(args.output), args.device
    )
    (Path(args.output) / "status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    print(json.dumps(status))


if __name__ == "__main__":
    main()
