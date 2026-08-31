"""V2T-Fix3-Seal stage 2: checkpoint weight health audit.

Audits four models:
  A current official checkpoint   (checkpoints/release/...)
  B re-downloaded official        (checkpoints/redownload_verify/...)
  C Fix1 epoch-21 training ckpt   (v2tfix1/training_B/epoch_00021.pt)
  D random-initialisation control (seeded MultiModalEncoder)

Per tensor: dtype/shape/min/max/mean/std/L1/L2/zero-frac/subnormal-frac/
NaN/Inf, plus requires_grad from a live reference model and the
missing/unexpected key report of a strict=False load.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/aidenwu/Documents/sgaligner-sgf-official")
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix3_seal"

A_PATH = ROOT / "checkpoints/release/sgaligner_pct_gat_rel_attr.pth.tar"
B_PATH = (
    ROOT / "checkpoints/redownload_verify/"
    "sgaligner_pct_gat_rel_attr.pth.tar"
)
C_PATH = (
    ROOT / "outputs/official_sgaligner_migration_fix2_v2tfix1/"
    "training_B/epoch_00021.pt"
)

FOCUS_MODULES = [
    "object_encoder", "object_embedding",
    "structure_encoder.layer_stack.0", "structure_encoder.layer_stack.1",
    "structure_encoder", "structure_embedding",
    "meta_embedding_rel", "meta_embedding_attr", "fusion",
]
TINY = np.finfo(np.float32).tiny  # subnormal threshold


def tensor_stats(t: torch.Tensor) -> dict:
    a = t.detach().cpu().numpy()
    finite_mask = np.isfinite(a)
    out = {
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "min": None, "max": None, "mean": None, "std": None,
        "l1_norm": None, "l2_norm": None,
        "zero_fraction": None, "subnormal_fraction": None,
        "nan_count": int(np.isnan(a).sum()) if t.dtype.is_floating_point else 0,
        "inf_count": int(np.isinf(a).sum()) if t.dtype.is_floating_point else 0,
    }
    if t.dtype.is_floating_point:
        finite = a[finite_mask]
        out["min"] = float(a.min()) if finite.size else None
        out["max"] = float(a.max()) if finite.size else None
        out["mean"] = float(a.mean()) if a.size else None
        out["std"] = float(a.std()) if a.size > 1 else 0.0
        out["l1_norm"] = float(np.abs(finite).sum())
        out["l2_norm"] = float(np.sqrt((finite.astype(np.float64) ** 2).sum()))
        out["zero_fraction"] = float((a == 0).mean())
        out["subnormal_fraction"] = float(
            ((np.abs(a) > 0) & (np.abs(a) < TINY)).mean()
        )
    return out


def load_state(path: Path, key: str = "model") -> OrderedDict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return OrderedDict(state[key])


def audit(state: OrderedDict, live: torch.nn.Module | None,
          label: str, path: Path) -> dict:
    requires = {}
    if live is not None:
        requires = {
            name: bool(p.requires_grad)
            for name, p in live.named_parameters()
        }
        buffers = {name for name, _ in live.named_buffers()}
    load_report = {"missing_keys": [], "unexpected_keys": []}
    if live is not None:
        report = live.load_state_dict(state, strict=False)
        load_report = {
            "missing_keys": list(report.missing_keys),
            "unexpected_keys": list(report.unexpected_keys),
        }
    per_tensor = {}
    for key, tensor in state.items():
        stats = tensor_stats(tensor)
        stats["requires_grad"] = requires.get(key)
        stats["is_buffer"] = key in buffers if live is not None else None
        per_tensor[key] = stats

    # module-level aggregation for the focus list
    module_summary = {}
    for module in FOCUS_MODULES:
        keys = [k for k in state if k == module
                or k.startswith(module + ".")]
        if not keys:
            continue
        grads = [
            float(np.abs(
                state[k].detach().cpu().numpy().astype(np.float64)
            ).max())
            for k in keys if state[k].dtype.is_floating_point
        ]
        nonzero = []
        for k in keys:
            arr = np.abs(
                state[k].detach().cpu().numpy().astype(np.float64)
            ).ravel()
            nz = arr[arr > 0]
            if nz.size:
                nonzero.append(float(nz.min()))
        module_summary[module] = {
            "num_tensors": len(keys),
            "max_abs_value": max(grads) if grads else None,
            "l2_norms": {
                k: per_tensor[k]["l2_norm"] for k in keys
            },
            "min_abs_nonzero": min(nonzero) if nonzero else None,
        }
    return {
        "label": label,
        "path": str(path),
        "num_tensors": len(state),
        "load_report": load_report,
        "module_summary": module_summary,
        "per_tensor": per_tensor,
    }


def diff(label: str, sa: OrderedDict, sb: OrderedDict) -> dict:
    rows = {}
    for k in sorted(set(sa) | set(sb)):
        if k not in sa or k not in sb:
            rows[k] = {"in_a": k in sa, "in_b": k in sb, "identical": False}
            continue
        ta, tb = sa[k], sb[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            rows[k] = {
                "shape_a": list(ta.shape), "shape_b": list(tb.shape),
                "identical": False,
            }
            continue
        if not ta.dtype.is_floating_point:
            rows[k] = {"identical": bool(torch.equal(ta, tb)),
                       "max_abs_diff": 0.0 if torch.equal(ta, tb) else None}
            continue
        d = float((ta.double() - tb.double()).abs().max())
        rows[k] = {"identical": d == 0.0, "max_abs_diff": d}
    changed = [k for k, v in rows.items() if not v.get("identical")]
    return {
        "comparison": label,
        "compared": len(rows),
        "changed_keys": changed,
        "max_abs_diff_overall": max(
            (v.get("max_abs_diff") or 0.0) for v in rows.values()
        ) if rows else 0.0,
        "per_tensor": rows,
    }


def main() -> None:
    from aligner.sg_aligner import MultiModalEncoder

    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260827)

    state_a = load_state(A_PATH)
    state_b = load_state(B_PATH)
    state_c = load_state(C_PATH)

    live_full = MultiModalEncoder(
        modules=["pct", "gat", "rel", "attr"], rel_dim=41, attr_dim=164,
    )
    live_three = MultiModalEncoder(
        modules=["pct", "gat", "rel"], rel_dim=41, attr_dim=164,
    )
    live_random = MultiModalEncoder(
        modules=["pct", "gat", "rel", "attr"], rel_dim=41, attr_dim=164,
    )

    audits = OrderedDict()
    audits["A_current_official"] = audit(
        state_a, live_full, "A_current_official", A_PATH)
    audits["B_redownloaded_official"] = audit(
        state_b, live_full, "B_redownloaded_official", B_PATH)
    audits["C_fix1_epoch21"] = audit(
        state_c, live_three, "C_fix1_epoch21", C_PATH)
    audits["D_random_init_control"] = audit(
        OrderedDict(live_random.state_dict()), live_random,
        "D_random_init_control", Path("(in-memory random init)"))

    diffs = [
        diff("A_vs_B_current_vs_redownload", state_a, state_b),
        diff("A_vs_C_official_vs_fix1_epoch21", state_a, state_c),
    ]

    (OUT / "checkpoint_weight_health.json").write_text(
        json.dumps(audits, indent=2) + "\n"
    )
    (OUT / "checkpoint_tensor_diff.json").write_text(
        json.dumps({
            d["comparison"]: d for d in diffs}, indent=2) + "\n"
    )

    # console digest: GAT branch health across the four models
    for name, a in audits.items():
        gat_l2 = {
            k: v["l2_norm"]
            for k, v in a["per_tensor"].items()
            if k.startswith("structure_encoder")
        }
        gat_max = {
            k: v["max"] for k, v in a["per_tensor"].items()
            if k.startswith("structure_encoder")
        }
        print(f"[{name}] GAT L2 norms:")
        for k in sorted(gat_l2):
            print(f"   {k:55s} l2={gat_l2[k]:.6g} max={gat_max[k]:.6g}")
    for d in diffs:
        print(d["comparison"], "compared", d["compared"],
              "changed", len(d["changed_keys"]),
              "max_diff", d["max_abs_diff_overall"])


if __name__ == "__main__":
    main()
