"""SGF object extraction -> official SGAligner object fields.

Project adaptation code (namespace ``adapters.sgf``); the official
repository sources are never modified.  Each SGF object (InSeg segment
with >= 50 stable surfels) contributes:

- 512 local XYZ points in METRES (deterministic FPS; deterministic
  sampling-with-replacement below 512 — matching the official
  ``pcl_farthest_sample`` semantics with a fixed seed);
- barycentre (convex-hull mean, exactly like the official
  preprocessing);
- original object id (= persistent InSeg segment label) and a
  continuous 0..N-1 index mapping;
- per-object point provenance (stable surfel count, whether
  replacement sampling was used).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectProvenance:
    original_id: int
    stable_surfel_count: int
    used_replacement: bool
    unique_sampled_points: int
    full_point_count: int = 0        # raw points before dedup
    unique_point_count: int = 0      # after metre-grid dedup


@dataclass
class ObjectAdapterResult:
    obj_pts: np.ndarray              # [N,512,3] float32 descriptor (PCT)
    registration_pts: dict           # idx -> [K,3] float64 full unique (GeoT)
    obj_ids: np.ndarray              # [N] int64 original ids
    object_id2idx: dict              # original id -> 0..N-1
    idx_to_object_id: dict           # 0..N-1 -> original id
    barycenters: np.ndarray          # [N,3] float64
    provenance: list[ObjectProvenance] = field(default_factory=list)
    sampling_mode: str = "deterministic_pcg64"
    sampling_iteration_order: list[int] = field(default_factory=list)


def pcl_farthest_sample_official(point: np.ndarray, npoint: int):
    """VERBATIM algorithm of the official ``utils/point_cloud.py::
    pcl_farthest_sample`` (sayands/sgaligner@51cd572) — global numpy
    MT19937 semantics, identical draw order, identical tie behaviour,
    identical float32 dtype flow.  ``point`` MUST be the float32 view
    of the raw object cloud so every arithmetic op matches the
    official preprocessing bit-for-bit.  Returns (sampled, replaced).

    Draw sequence per object (N = raw point count):
      N <  npoint: one np.random.choice(N, npoint) call
                   (default replace=True);
      N >= npoint: one np.random.randint(0, N) draw for the first
                   point, then a pure-numpy FPS loop (np.argmax takes
                   the FIRST maximum — identical tie behaviour).
    """
    N, D = point.shape
    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint)
        return point[indices], True
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)], False


def farthest_point_sample_deterministic(
    points: np.ndarray, npoint: int, seed: int = 42
) -> np.ndarray:
    """Deterministic FPS; replacement sampling below npoint.

    Mirrors the official ``utils.point_cloud.pcl_farthest_sample``
    behaviour (np.random.choice with replacement when N < npoint) but
    seeds a dedicated generator so the same input + seed reproduces
    byte-identically.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        raise ValueError("cannot sample an empty object")
    if n < npoint:
        rng = np.random.default_rng((seed, n))
        indices = rng.choice(n, npoint, replace=True)
        return points[indices].astype(np.float32), True
    rng = np.random.default_rng(seed)
    # candidate cap keeps FPS cost bounded exactly like the legacy
    # graph builder; determinism is preserved
    max_candidates = 4096
    if n > max_candidates:
        candidates = points[rng.choice(n, max_candidates, replace=False)]
    else:
        candidates = points
    centroids = np.empty(npoint, dtype=np.int64)
    first = int(rng.integers(0, len(candidates)))
    centroids[0] = first
    min_distance = np.full(len(candidates), np.inf)
    for index in range(1, npoint):
        distance = np.sum(
            (candidates - candidates[centroids[index - 1]]) ** 2, axis=1
        )
        min_distance = np.minimum(min_distance, distance)
        centroids[index] = int(np.argmax(min_distance))
    sampled = candidates[centroids]
    return sampled.astype(np.float32), False


def hull_barycenter(points: np.ndarray) -> np.ndarray:
    """Convex-hull-vertex mean, matching the official preprocessing."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    return hull.points[hull.vertices].mean(axis=0)


def adapt_objects(
    segments: dict[int, np.ndarray],
    *,
    min_stable_surfels: int = 50,
    npoint: int = 512,
    seed: int = 42,
    metres: bool = True,
    scale: float = 1.0,
    sampling_mode: str = "deterministic_pcg64",
    scan_seed: int = 0,
    iteration_order: list[int] | None = None,
) -> ObjectAdapterResult:
    """Convert per-label point clouds into official object tensors.

    ``segments`` maps original object id -> [K,3] points (metres when
    ``metres`` else millimetres scaled by ``scale``).

    sampling_mode:
      - "deterministic_pcg64" (production default): per-object seeded
        PCG64 FPS — unchanged behaviour;
      - "official_mt19937": byte-exact reproduction of the official
        preprocessing sampler.  The global numpy MT19937 state is
        SEEDED to ``scan_seed`` once, objects are sampled in
        ``iteration_order`` (the official objects.json order for
        oracle scans; caller-defined for predicted graphs), and the
        global RNG state is saved before / restored after so nothing
        else in the process observes the draws.  The canonical
        sorted-object-id row order is applied AFTER sampling.
    """
    keep_ids = sorted(
        oid for oid, pts in segments.items()
        if len(pts) >= min_stable_surfels
    )
    if len(keep_ids) < 1:
        raise ValueError(
            "no object reaches min_stable_surfels; refusing empty graph"
        )
    if sampling_mode not in ("deterministic_pcg64", "official_mt19937"):
        raise ValueError(f"unknown sampling_mode {sampling_mode}")
    if sampling_mode == "official_mt19937":
        if iteration_order is None:
            iteration_order = list(keep_ids)  # adapter-defined order
        missing = [oid for oid in iteration_order
                   if oid not in segments]
        if missing:
            raise ValueError(
                f"iteration_order ids missing from segments: {missing[:5]}"
            )
        # official draws happen in iteration_order regardless of the
        # canonical output ordering
        sampled_by_oid: dict[int, np.ndarray] = {}
        replaced_by_oid: dict[int, bool] = {}
        rng_state = np.random.get_state()
        try:
            np.random.seed(scan_seed)
            for oid in iteration_order:
                raw32 = np.asarray(
                    segments[oid], dtype=np.float64
                ).astype(np.float32)
                sampled, replaced = pcl_farthest_sample_official(
                    raw32, npoint
                )
                sampled_by_oid[int(oid)] = sampled
                replaced_by_oid[int(oid)] = replaced
        finally:
            np.random.set_state(rng_state)
    obj_pts = np.empty((len(keep_ids), npoint, 3), dtype=np.float32)
    barycenters = np.empty((len(keep_ids), 3), dtype=np.float64)
    registration_pts: dict[int, np.ndarray] = {}
    provenance = []
    for index, oid in enumerate(keep_ids):
        raw = np.asarray(segments[oid], dtype=np.float64)
        if not metres:
            raw = raw * scale
        # registration points: full object, deduplicated on a 1 mm grid,
        # kept in the ORIGINAL world frame (metres)
        unique = np.unique(np.round(raw, 3), axis=0)
        registration_pts[index] = np.ascontiguousarray(unique)
        if sampling_mode == "official_mt19937":
            sampled = sampled_by_oid[int(oid)]
            replaced = replaced_by_oid[int(oid)]
        else:
            # descriptor points: deterministic 512-sample from the RAW set
            # (replacement only below 512) — PCT-only, never GeoT input
            sampled, replaced = farthest_point_sample_deterministic(
                raw, npoint, seed=(seed, oid)
            )
        obj_pts[index] = sampled
        barycenters[index] = hull_barycenter(raw)
        provenance.append(
            ObjectProvenance(
                original_id=int(oid),
                stable_surfel_count=int(len(raw)),
                used_replacement=bool(replaced),
                unique_sampled_points=int(min(len(raw), npoint)),
                full_point_count=int(len(raw)),
                unique_point_count=int(len(unique)),
            )
        )
    return ObjectAdapterResult(
        obj_pts=obj_pts,
        registration_pts=registration_pts,
        obj_ids=np.asarray(keep_ids, dtype=np.int64),
        object_id2idx={int(oid): i for i, oid in enumerate(keep_ids)},
        idx_to_object_id={i: int(oid) for i, oid in enumerate(keep_ids)},
        barycenters=barycenters,
        provenance=provenance,
        sampling_mode=sampling_mode,
        sampling_iteration_order=(
            [int(o) for o in iteration_order]
            if sampling_mode == "official_mt19937" else
            [int(o) for o in keep_ids]
        ),
    )
