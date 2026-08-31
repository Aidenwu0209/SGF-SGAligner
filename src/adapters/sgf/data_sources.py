"""Data sources bridging local 3RScan/3DSSG/SGF caches to the adapter.

Two sources:

- OracleGraphSource: GT 3DSSG scene graphs.  Objects from
  ``labels.instances.annotated.v2.ply`` (objectId grouping), attributes
  from ``objects.json``, relationships from ``relationships.json``.
- PredictedGraphSource: SGF replay segmentation (inseg_cloud.npz next
  to the legacy scan cache) + GraphPredictor relations from the
  Phase-13 prediction cache (immutable, keyed).

Both produce (segments, directed_pairs, relation_triples, attributes)
in ORIGINAL object-id space; the graph adapter then builds the official
contract.  GT transforms come from the legacy pair records and are used
ONLY for evaluation after inference, never as input features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path(
    "/home/aidenwu/Documents/SceneGraphFusion/data/3RScan_full"
)
PREDICTION_CACHE = Path(
    "/home/aidenwu/Documents/inseg-sgaligner-sgf-context-v1/outputs/"
    "delivery_stage1_20260825/phase13_sgf_context_generalization/cache"
)
LEGACY_PAIR_ROOTS = [
    Path(
        "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
        "delivery_stage1_20260823/training_dataset/pairs"
    ),
    Path(
        "/home/aidenwu/Documents/inseg-sgaligner-v2/outputs/"
        "3rscan_native_graph_dataset_full/pairs"
    ),
    Path(
        "/home/aidenwu/Documents/inseg-sgaligner-v2/outputs/"
        "3rscan_native_graph_dataset_pilot/pairs"
    ),
]
OBJ_ATTR_PKL = Path(
    "/home/aidenwu/Documents/sgaligner-sgf-official/checkpoints/release/"
    "obj_attr.pkl"
)


@dataclass
class GraphSourceResult:
    segments: dict[int, np.ndarray]
    directed_pairs: list[tuple[int, int]]
    relation_triples: list[tuple[int, int, str]]
    attributes_per_object: dict[int, list[str] | None]
    metadata: dict


def _ply_segments(scan_id: str, min_points: int = 50):
    from plyfile import PlyData

    scan_dir = DATA_ROOT / scan_id
    ply = PlyData.read(scan_dir / "labels.instances.annotated.v2.ply")
    vertex = ply["vertex"]
    points = np.stack(
        [vertex["x"], vertex["y"], vertex["z"]], axis=1
    ).astype(np.float64)
    labels = np.asarray(vertex["objectId"])
    segments = {}
    for label in np.unique(labels):
        if int(label) == 0:
            continue
        mask = labels == label
        if int(mask.sum()) >= min_points:
            segments[int(label)] = points[mask]
    return segments


class OracleGraphSource:
    """GT 3DSSG scene graph for one scan (oracle mode)."""

    def __init__(self):
        self.objects_json = json.loads(
            (DATA_ROOT / "objects.json").read_text()
        )["scans"]
        self.relationships_json = json.loads(
            (DATA_ROOT / "relationships.json").read_text()
        )["scans"]
        self._objects_by_scan = {
            entry["scan"]: entry["objects"] for entry in self.objects_json
        }
        self._relations_by_scan = {
            entry["scan"]: entry["relationships"]
            for entry in self.relationships_json
        }

    def _segments_for(self, scan_id: str) -> dict[int, np.ndarray]:
        if not hasattr(self, "_segments_cache"):
            self._segments_cache = {}
        if scan_id not in self._segments_cache:
            self._segments_cache[scan_id] = _ply_segments(scan_id)
        return self._segments_cache[scan_id]

    def official_object_order(self, scan_id: str) -> list[int]:
        """Official preprocessing iteration order for a scan.

        preprocess.py iterates ``objects.json`` entries in file order,
        keeping those whose PLY point count reaches min_obj_points —
        exactly the segments the adapter already extracted (same ply,
        same >=50 filter).  Used ONLY to order the MT19937 draw
        sequence in official_mt19937 sampling mode.
        """
        segments = self._segments_for(scan_id)
        order = []
        for obj in self._objects_by_scan.get(scan_id, []):
            oid = int(obj["id"])
            if oid in segments:
                order.append(oid)
        return order

    def load(self, scan_id: str) -> GraphSourceResult:
        segments = self._segments_for(scan_id)
        objects = self._objects_by_scan.get(scan_id, [])
        attributes = {}
        for obj in objects:
            oid = int(obj["id"])
            if oid in segments:
                attributes[oid] = [
                    item
                    for sublist in obj.get("attributes", {}).values()
                    for item in sublist
                ]
        triples = []
        pairs = []
        seen_pairs = set()
        for sub, obj, _rid, name in self._relations_by_scan.get(scan_id, []):
            sub, obj = int(sub), int(obj)
            if sub in segments and obj in segments:
                triples.append((sub, obj, name))
                # official preprocess deduplicates (sub, obj) pairs
                # ("if triple[:2] not in pairs") before degree counting
                if (sub, obj) not in seen_pairs:
                    seen_pairs.add((sub, obj))
                    pairs.append((sub, obj))
        return GraphSourceResult(
            segments=segments,
            directed_pairs=pairs,
            relation_triples=triples,
            attributes_per_object=attributes,
            metadata={"scan_id": scan_id, "source": "3dssg_gt"},
        )


class PredictedGraphSource:
    """SGF-predicted scene graph for one scan (predicted mode).

    Geometry segments come from the SGF replay's InSeg cloud (same
    label space as the matcher universe); relations come from the
    GraphPredictor prediction cache (Phase 13).  Attributes are None —
    never fabricated.
    """

    def __init__(self, cache_dir: Path = PREDICTION_CACHE):
        self.cache_dir = cache_dir

    def load(self, scan_id: str, inseg_cloud: Path | None = None) -> GraphSourceResult:
        if inseg_cloud is None:
            candidates = [
                Path(
                    "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
                    "delivery_stage1_20260823/training_dataset/cache"
                )
                / scan_id
                / "inseg_cloud.npz",
                Path(
                    "/home/aidenwu/Documents/sgaligner-sgf-official/outputs/"
                    "official_sgaligner_migration_20260825_235139/"
                    "supplementary_scan_cache"
                )
                / scan_id
                / "inseg_cloud.npz",
            ]
            inseg_cloud = next(
                (path for path in candidates if path.exists()), candidates[0]
            )
        with np.load(inseg_cloud) as data:
            # snapshot_inseg xyz is already METRES (verified: training
            # cache ranges 0-2.4 m) — no unit conversion here
            xyz = np.asarray(data["xyz"], dtype=np.float64)
            labels = np.asarray(data["labels"])
        segments = {}
        for label in np.unique(labels):
            if int(label) == 0:
                continue
            mask = labels == label
            if int(mask.sum()) >= 50:
                segments[int(label)] = xyz[mask]

        triples, pairs = [], []
        cache_entry = self.cache_dir / f"{scan_id}.npz"
        if cache_entry.exists():
            with np.load(cache_entry) as data:
                meta = json.loads(data["meta_json"].tobytes())
                edges = data["relation_edges"]
                for (s, r), name in zip(edges, meta["relation_labels"]):
                    s, r = int(s), int(r)
                    if name == "none":
                        continue
                    if s in segments and r in segments:
                        triples.append((s, r, name))
                        pairs.append((s, r))
        return GraphSourceResult(
            segments=segments,
            directed_pairs=pairs,
            relation_triples=triples,
            attributes_per_object=None,
            metadata={
                "scan_id": scan_id,
                "source": "sgf_predicted",
                "attribute_available": False,
            },
        )


def load_pair_record(pair_id: str) -> dict:
    for root in LEGACY_PAIR_ROOTS:
        path = root / pair_id / "pair.json"
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"pair record not found for {pair_id}")


def load_gt_transform(pair_id: str) -> np.ndarray | None:
    payload = load_pair_record(pair_id)
    gt = payload.get("gt_transform")
    if gt is None:
        return None
    return np.asarray(gt, dtype=np.float64).reshape(4, 4)


def load_anchor_ids(pair_id: str) -> list[tuple[int, int]]:
    """Legacy InSeg-label anchors (predicted/legacy modes)."""
    payload = load_pair_record(pair_id)
    anchors = payload.get("anchor_pairs") or []
    return [(int(a), int(b)) for a, b in anchors]


def load_oracle_anchor_ids(
    src_scan: str, ref_scan: str,
    src_segments: dict, ref_segments: dict,
) -> list[tuple[int, int]]:
    """Oracle anchors: object pairs sharing the 3DSSG global id."""
    objects_json = json.loads((DATA_ROOT / "objects.json").read_text())["scans"]
    by_scan = {entry["scan"]: entry["objects"] for entry in objects_json}
    def global_map(scan, segments):
        return {
            int(obj["global_id"]): int(obj["id"])
            for obj in by_scan.get(scan, [])
            if int(obj["id"]) in segments
        }
    g_src = global_map(src_scan, src_segments)
    g_ref = global_map(ref_scan, ref_segments)
    shared = set(g_src) & set(g_ref)
    return [(g_src[g], g_ref[g]) for g in sorted(shared)]


def legacy_anchor_ids(pair_id: str) -> list[tuple[int, int]]:
    payload = load_pair_record(pair_id)
    anchors = payload.get("anchor_pairs") or []
    return [(int(a), int(b)) for a, b in anchors]


def attribute_vocab_164() -> dict[str, int]:
    import pickle

    with OBJ_ATTR_PKL.open("rb") as handle:
        vocab = pickle.load(handle)
    if len(vocab) < 164:
        raise ValueError(
            f"attribute vocabulary must have >=164 entries, got {len(vocab)}"
        )
    return {name: int(index) for name, index in vocab.items()}


def oracle_gt_transform(
    src_scan: str,
    ref_scan: str,
    src_segments: dict,
    ref_segments: dict,
) -> np.ndarray:
    """Evaluation-only GT for the official PLY frame.

    Deterministic derivation from 3DSSG GT anchor objects: barycentre
    Procrustes initialisation + point-level ICP over the full anchor
    point sets (~2.4cm median residual on the reference pair).  Never
    used as an input feature.
    """
    from scipy.spatial import cKDTree

    anchors = load_oracle_anchor_ids(
        src_scan, ref_scan, src_segments, ref_segments
    )
    if len(anchors) < 3:
        raise ValueError("fewer than 3 oracle anchors for GT derivation")
    src_c = np.asarray(
        [src_segments[s].mean(axis=0) for s, _ in anchors]
    )
    ref_c = np.asarray(
        [ref_segments[r].mean(axis=0) for _, r in anchors]
    )
    u, _, vt = np.linalg.svd(
        (src_c - src_c.mean(0)).T @ (ref_c - ref_c.mean(0))
    )
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    init = np.eye(4)
    init[:3, :3] = rotation
    init[:3, 3] = ref_c.mean(0) - rotation @ src_c.mean(0)

    src_all = np.concatenate([src_segments[s] for s, _ in anchors])
    ref_all = np.concatenate([ref_segments[r] for _, r in anchors])
    transform = init.copy()
    tree = cKDTree(ref_all)
    for _ in range(50):
        moved = src_all @ transform[:3, :3].T + transform[:3, 3]
        distances, indices = tree.query(moved)
        mask = distances <= 0.10
        if int(mask.sum()) < 10:
            break
        a = src_all[mask]
        b = ref_all[indices[mask]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = cb - rotation @ ca
    return transform


def _source_inseg_cloud(scan_id: str) -> Path:
    """First existing InSeg cloud for a scan (legacy + supplementary)."""
    candidates = [
        Path(
            "/home/aidenwu/Documents/inseg-sgaligner-stage1/outputs/"
            "delivery_stage1_20260823/training_dataset/cache"
        )
        / scan_id
        / "inseg_cloud.npz",
        Path(
            "/home/aidenwu/Documents/sgaligner-sgf-official/outputs/"
            "official_sgaligner_migration_20260825_235139/"
            "supplementary_scan_cache"
        )
        / scan_id
        / "inseg_cloud.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no InSeg cloud for {scan_id}")
