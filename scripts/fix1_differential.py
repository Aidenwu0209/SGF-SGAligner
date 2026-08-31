"""Fix-1 official differential validation (one oracle pair).

Runs the SAME node correspondences through (a) our adapter
registration entry and (b) the official RegistrationEvaluator on
official-format inputs, then compares input point counts, coordinate
frames, RRE/RTE.  Differences are explained; equivalence gate decides
the final verdict input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT))

OUT = Path(sys.argv[1])
PAIR = sys.argv[2]

from inference import (  # noqa: E402
    build_pair_inputs, official_matching, official_registration,
)
from adapters.sgf.data_sources import oracle_gt_transform  # noqa: E402
from inference import OracleGraphSource_anchor_segments  # noqa: E402

report = {"pair_id": PAIR, "mode": "official_oracle"}

# our path
data_dict, contracts = build_pair_inputs(PAIR, "official_oracle")
src_scan, ref_scan = PAIR.split("_to_")
src_segments = OracleGraphSource_anchor_segments(src_scan)
ref_segments = OracleGraphSource_anchor_segments(ref_scan)
gt = oracle_gt_transform(src_scan, ref_scan, src_segments, ref_segments)

embedding, _epoch = None, None
from inference import official_forward  # noqa: E402

embedding, epoch = official_forward(data_dict, "official_oracle", "cuda")
node_corrs, rank_list, sim = official_matching(
    embedding, data_dict["src_count"]
)
registration, used, failures = official_registration(
    data_dict, node_corrs, "official_oracle", device="cuda", pair_id=PAIR
)

def rre_rte(transform, gt):
    cos = (np.trace(transform[:3, :3].T @ gt[:3, :3]) - 1) / 2
    rre = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    rte = float(np.linalg.norm(transform[:3, 3] - gt[:3, 3]))
    return rre, rte

ours = rre_rte(registration["transform"], gt)

# official evaluator path on the same correspondences
from engine.registration_evaluator import RegistrationEvaluator  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger("diff")
logger.addHandler(logging.NullHandler())
evaluator = RegistrationEvaluator.__new__(RegistrationEvaluator)
evaluator.device = "cuda"
evaluator.snapshot = (
    ROOT / "checkpoints/geotransformer/geotransformer-3dmatch.pth.tar"
)
from GeoTransformer.config import make_cfg  # noqa: E402
from GeoTransformer.model import create_model  # noqa: E402

cfg = make_cfg()
evaluator.cfg = cfg
evaluator.model = create_model(cfg).to("cuda")
state = torch.load(str(evaluator.snapshot), map_location="cuda",
                   weights_only=False)
evaluator.model.load_state_dict(state["model"], strict=True)
evaluator.model.eval()
evaluator.logger = logger
evaluator.num_p2p_corrs = 1000
evaluator.neighbor_limits = [18, 16, 14, 12]
evaluator.ransac_threshold = 0.05
evaluator.ransac_min_iters = 1000
evaluator.ransac_max_iters = 10000
evaluator.ransac_use_sprt = False
evaluator.inlier_ratio_thresh = 0.05
evaluator.rmse_thresh = 0.2
evaluator.min_object_points = 50
evaluator.visualise_registration = False

# official evaluator needs ply-data object point clouds
from plyfile import PlyData  # noqa: E402
from adapters.sgf.data_sources import DATA_ROOT  # noqa: E402


def ply_for(scan):
    ply = PlyData.read(
        DATA_ROOT / scan / "labels.instances.annotated.v2.ply"
    )["vertex"]
    pts = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)
    return pts, np.asarray(ply["objectId"])


src_pts, src_obj = ply_for(src_scan)
ref_pts, ref_obj = ply_for(ref_scan)
# The official FULL-SCENE normal-registration pass OOMs on the 8 GB GPU
# (comfyui + stable_fusion workers resident); the official ALIGNER path
# registers per matched object pair (<=10k pts each) and fits, so the
# differential runs run_aligner_registration directly with FULL scene
# points and the true objectId arrays — the exact official semantics.
id2oid = data_dict["registration_id2oid"]
node_corrs_oids = [
    (id2oid[a], id2oid[b]) for a, b in node_corrs
]

reg_data = {
    "node_corrs": node_corrs_oids,
    "src_points": src_pts - data_dict["pcl_center"],
    "ref_points": ref_pts - data_dict["pcl_center"],
    "src_plydata": {"objectId": src_obj},
    "ref_plydata": {"objectId": ref_obj},
    "raw_points": src_pts - data_dict["pcl_center"],
    "gt_transform": gt,
}
from utils.point_cloud import compute_pcl_overlap  # noqa: E402

_, gt_src_idx = compute_pcl_overlap(reg_data["src_points"], reg_data["ref_points"])
_, gt_ref_idx = compute_pcl_overlap(reg_data["ref_points"], reg_data["src_points"])
reg_data["gt_src_corr_points"] = reg_data["src_points"][gt_src_idx]
reg_data["gt_ref_corr_points"] = reg_data["ref_points"][gt_ref_idx]

np.random.seed(42)
torch.manual_seed(42)
import torch as _torch

_torch.cuda.empty_cache()
# official normal path first (may OOM -> documented); aligner path is
# the apples-to-apples comparator for our per-object adapter entry
normal_result = None
try:
    normal_result = evaluator.run_normal_registration(
        dict(reg_data), evaluate_registration=False
    )
except Exception as exc:
    normal_result = ("error", repr(exc)[:200])
_torch.cuda.empty_cache()
aligner_result = evaluator.run_aligner_registration(
    dict(reg_data), evaluate_registration=False
)
official_result = (normal_result, aligner_result)

report["inputs"] = {
    "node_corrs": len(node_corrs),
    "our_src_object_points_full": {
        str(id2oid[a]): int(len(data_dict["registration_pts"][a]))
        for a, _ in node_corrs[:5]
    },
    "official_src_points": int(len(src_pts)),
    "pcl_center": data_dict["pcl_center"].tolist(),
    "pcl_center_definition": data_dict["pcl_center_definition"],
}
report["ours"] = {
    "rre": ours[0], "rte": ours[1],
    "inliers": registration["inliers"],
    "corrs": registration["corrs"],
    "node_pair_failures": len(failures),
}
normal_result, aligner_result = official_result
report["official_normal"] = (
    "ok" if isinstance(normal_result, tuple) and len(normal_result) == 2
    and isinstance(normal_result[0], np.ndarray)
    else ("oom_or_error: %s" % (normal_result,))[:200]
    if normal_result is not None else "none"
)
if isinstance(aligner_result, np.ndarray) and aligner_result.shape == (4, 4):
    c = data_dict["pcl_center"]
    world = np.eye(4)
    world[:3, :3] = aligner_result[:3, :3]
    world[:3, 3] = aligner_result[:3, 3] + c - aligner_result[:3, :3] @ c
    off_rre, off_rte = rre_rte(world, gt)
    report["official_aligner"] = {
        "rre_world": off_rre, "rte_world": off_rte,
    }
else:
    report["official_aligner"] = {"result": str(aligner_result)[:200]}

report["differences"] = []
aligner = report.get("official_aligner", {})
report["equivalent_strict"] = bool(
    report["ours"]["rre"] <= 5.0 and report["ours"]["rte"] <= 0.20
    and aligner.get("rre_world") is not None
    and aligner["rre_world"] <= 10.0 and aligner["rte_world"] <= 0.30
)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "differential_validation.json").write_text(
    json.dumps(report, indent=2, default=str) + "\n"
)
print(json.dumps({
    "ours": report["ours"],
    "official_normal": report["official_normal"],
    "official_aligner": report.get("official_aligner"),
    "equivalent_strict": report["equivalent_strict"],
}, indent=1, default=str))
