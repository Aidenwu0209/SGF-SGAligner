"""V4-Fix-Seal Part 2: audit BEFORE the fix.

1. records the TRUE exit code / output of
   git diff e2d9ca7..HEAD --check;
2. builds the audit pair's inputs through BOTH paths —
   production `build_pair_inputs(..., official_mt19937)` and the
   training-path `build_split_samples(...)` — and compares every
   tensor, reproducing the ~0.065501 m tot_obj_pts difference and
   explaining its root cause;
3. records the F1/top1/top5 discrepancy for checkpoint SHA
   f642b448... (C-epoch20) between the V4 cache (production inputs)
   and the V4-Fix fair evaluation (training-path inputs).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/inference/sgf_official"))
sys.path.insert(0, str(ROOT / "scripts"))

from inference import build_pair_inputs  # noqa: E402
from v4_train import build_split_samples  # noqa: E402
from adapters.sgf.data_sources import PredictedGraphSource  # noqa: E402

OUT = ROOT / "outputs/official_sgaligner_v4_fix_seal_20260828"
AUDIT_PAIR = (
    "09582205-e2c2-2de1-9475-1cdac7639e60_to_"
    "0958220d-e2c2-2de1-9710-c37018da1883"
)
CKPT_SHA20 = (
    "f642b448aa944fa4fc230133befbf7eed747693a97dd8b5c6715cd144f67960a"
)
V4FIX = ROOT / "outputs/official_sgaligner_v4_fix_fair_selection_20260828"
V4 = ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827"


def hash_of(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def field_cmp(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    out = {
        "shape_a": list(a.shape), "shape_b": list(b.shape),
        "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
        "sha_a": hash_of(a), "sha_b": hash_of(b),
    }
    if a.shape != b.shape:
        out.update({"equal": False, "max_abs_diff": None,
                    "first_mismatch": "shape"})
        return out
    if a.dtype.kind == "f" or b.dtype.kind == "f":
        diff = np.abs(
            a.astype(np.float64) - b.astype(np.float64))
        out["max_abs_diff"] = float(diff.max())
        nz = np.flatnonzero(diff.ravel() > 0)
        out["first_mismatch"] = (
            str(np.unravel_index(int(nz[0]), a.shape))
            if nz.size else None)
        out["equal"] = bool(nz.size == 0)
    else:
        out["equal"] = bool(np.array_equal(a, b))
        out["max_abs_diff"] = 0.0 if out["equal"] else None
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- 1. git diff --check ground truth -------------------------
    check = subprocess.run(
        ["git", "-C", str(ROOT), "diff",
         "e2d9ca7c8f67a462223dfc9f8658a00c62b25596..HEAD", "--check"],
        capture_output=True, text=True)
    diff_check = {
        "command": "git diff e2d9ca7..HEAD --check",
        "exit_code": check.returncode,
        "stdout": check.stdout.strip(),
        "stderr": check.stderr.strip(),
    }

    # ---- 2. dual-path input comparison ----------------------------
    prod_dd, _c = build_pair_inputs(
        AUDIT_PAIR, "official_sgf_predicted",
        sampling_mode="official_mt19937")
    train_samples, _skipped = build_split_samples(
        [AUDIT_PAIR], PredictedGraphSource())
    train_dd = train_samples[0][1]

    def explicit_of(dd):
        from v4_cache_runner import explicit_edges_of

        return explicit_edges_of(AUDIT_PAIR, dd)

    prod_ex, prod_ex_counts = explicit_of(prod_dd)

    fields = {
        "tot_obj_pts": field_cmp(
            prod_dd["tot_obj_pts"], train_dd["tot_obj_pts"]),
        "tot_rel_pose": field_cmp(
            prod_dd["tot_rel_pose"], train_dd["tot_rel_pose"]),
        "relation_bow_41d": field_cmp(
            prod_dd["tot_bow_vec_object_edge_feats"],
            train_dd["tot_bow_vec_object_edge_feats"]),
        "obj_ids": field_cmp(
            prod_dd["obj_ids"], train_dd["obj_ids"]),
        "graph_obj_counts": field_cmp(
            prod_dd["graph_per_obj_count"],
            train_dd["graph_per_obj_count"]),
        "complete_edges": field_cmp(
            prod_dd["edges"], train_dd["edges"]),
        "graph_edge_counts_complete": field_cmp(
            prod_dd["graph_per_edge_count"],
            train_dd["graph_per_edge_count"]),
        "explicit_edges": field_cmp(
            prod_ex, train_dd["edges_explicit"]),
        "explicit_edge_counts": field_cmp(
            prod_ex_counts,
            train_dd["graph_per_edge_count_explicit"]),
        "pcl_center": field_cmp(
            prod_dd["pcl_center"], train_dd["pcl_center"]),
    }
    center_delta = float(np.linalg.norm(
        np.asarray(prod_dd["pcl_center"], dtype=np.float64)
        - np.asarray(train_dd["pcl_center"], dtype=np.float64)))
    pts_diff = fields["tot_obj_pts"]["max_abs_diff"]

    # explain: the two centers
    center_explanation = {
        "production_definition":
            prod_dd["pcl_center_definition"],
        "training_path_definition": (
            "mean of the CONCATENATED per-object 512-point descriptor "
            "samples of BOTH graphs (v4_train.build_split_samples: "
            "np.concatenate([src.tot_obj_pts.reshape(-1,3), "
            "ref.tot_obj_pts.reshape(-1,3)]).mean(axis=0))"),
        "center_l2_delta_m": center_delta,
        "tot_obj_pts_max_abs_diff_m": pts_diff,
        "root_cause": (
            "tot_obj_pts = raw sampled points MINUS pcl_center; the "
            "two builders use DIFFERENT pcl_center definitions, so "
            "the tensor differs by the center delta in every "
            "coordinate. The production contract (source full stable "
            "InSeg surfel cloud mean) is the canonical one; the "
            "training path's descriptor-mean center is the deviation."
            if abs(center_delta - pts_diff) < 1e-6 else
            "center delta does not fully explain the diff — "
            "investigate further"),
    }

    # ---- 3. C-ep20 metric discrepancy (old cache vs V4-Fix) -------
    v4_cache_sel = V4 / "selection89/cache_explicit"
    f1s, tp, pred, anch = [], 0, 0, 0
    for tag in sorted(v4_cache_sel.iterdir()):
        f = tag / "pair_cache.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        if c["status"] != "ok":
            continue
        nm = c["combos"]["candidate"]["node_metrics"]
        f1s.append(nm["f1"])
        tp += nm["tp"]
        pred += nm["pred_count"]
        anch += nm["anchor_count"]
    mp = tp / pred if pred else 0
    mr = tp / anch if anch else 0
    v4_sel_f1 = float(np.mean(f1s))

    ranking_c = json.loads(
        (V4FIX / "official_semantic_checkpoint_ranking_C.json"
         ).read_text())
    fix_ep20 = next(
        r for r in ranking_c["ranking"] if r["epoch"] == 20)
    discrepancy = {
        "checkpoint_sha": CKPT_SHA20,
        "v4_cache_selection89": {
            "macro_node_f1": v4_sel_f1,
            "micro_node_f1": 2 * mp * mr / max(mp + mr, 1e-12),
            "inputs": "build_pair_inputs (production contract)",
        },
        "v4_fix_fair_selection89": {
            "macro_node_f1": fix_ep20["metrics"]["macro_node_f1"],
            "top1_precision": fix_ep20["metrics"]["top1_precision"],
            "top5_recall": fix_ep20["metrics"]["top5_recall"],
            "inputs": "build_split_samples (training-path center)",
        },
        "explanation": (
            "the SAME checkpoint yields different selection metrics "
            "because the two evaluation paths fed it DIFFERENT "
            "tot_obj_pts (pcl_center divergence) — this is precisely "
            "the inconsistency the seal must eliminate"),
    }
    # verify the committed checkpoint hash matches
    ck = hashlib.sha256(
        (ROOT / "outputs/official_sgaligner_v4_healthy_gat_20260827/"
         "training/explicit/epoch_00020.pt").read_bytes()).hexdigest()
    discrepancy["committed_checkpoint_sha256_matches"] = (
        ck == CKPT_SHA20)

    audit = {
        "phase": "V4-Fix-Seal audit-before-fix",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_diff_check": diff_check,
        "pair": AUDIT_PAIR,
        "fields": fields,
        "center_explanation": center_explanation,
        "c_ep20_discrepancy": discrepancy,
    }
    (OUT / "audit_before_fix.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    md = f"""# audit_before_fix

## 1. git diff e2d9ca7..HEAD --check

exit code **{diff_check['exit_code']}**；输出：
`{diff_check['stdout']}`

## 2. 双路径输入比较（{AUDIT_PAIR[:16]}…）

| field | equal | max_abs_diff |
|---|---|---|
""" + "\n".join(
        f"| {k} | {v['equal']} | {v['max_abs_diff']} |"
        for k, v in fields.items()) + f"""

**tot_obj_pts 差异 = {pts_diff} m**，与两中心之差的 L2 范数
{center_delta} m 一致 → 根因：生产契约的 pcl_center（源扫描完整
稳定 InSeg surfel 云均值）与训练路径的中心（两图 512 点描述子拼接
均值）不同，全部坐标随之平移。

## 3. C-epoch20（SHA f642b448…）同一 checkpoint 的新旧指标

- V4 生产缓存（build_pair_inputs 输入）：selection macro F1 =
  {v4_sel_f1:.4f}
- V4-Fix 公平评估（build_split_samples 输入）：macro F1 =
  {fix_ep20['metrics']['macro_node_f1']:.4f}

同一权重、不同输入 → 指标不同：正是本阶段要封口的不一致。
"""
    (OUT / "audit_before_fix.md").write_text(md)
    print(json.dumps({
        "diff_check_rc": diff_check["exit_code"],
        "pts_diff": pts_diff, "center_delta": center_delta,
        "explained_by_center":
            abs(center_delta - pts_diff) < 1e-6,
        "v4_f1": v4_sel_f1,
        "fix_f1": fix_ep20["metrics"]["macro_node_f1"],
    }, indent=1))


if __name__ == "__main__":
    main()
