# GT-free ColorPCR correspondence / PointDSC shadow contract

Contract version: `colorpcr-pointdsc-shadow/v1`

Status: **preregistered interface specification only**. This document does not
authorize training, solver selection, production registration, checkpoint
promotion, refusion, pose-graph writes, or `official92` access.

## 1. Canonical geometry ABI

All components exchange meters in a right-handed frame and one transform:

```text
x_ref = R @ x_src + t
T_src_to_ref = [[R, t], [0, 0, 0, 1]]
```

Array application: `X_ref = X_src @ R.T + t`.

Required transform assertions:

```text
shape(T) == (4, 4)
isfinite(T).all()
max_abs(T[3] - [0, 0, 0, 1]) <= 1e-7
max_abs(R.T @ R - I) <= preregistered_rotation_tol
abs(det(R) - 1) <= preregistered_det_tol
T @ inv(T) ~= I
```

`pygcransac==0.1.1` is the only current exception at its direct API boundary:
it returns a row-homogeneous `T_raw`; normalize with `T = T_raw.T` before any
metric, comparison, serialization, ICP, or decision.

## 2. Immutable pair input

Each pair is materialized once before solvers run. Required manifest fields:

```json
{
  "schema": "colorpcr-pointdsc-shadow/v1",
  "pair_id": "opaque-stable-id",
  "src_scan_id": "...",
  "ref_scan_id": "...",
  "units": "meter",
  "handedness": "right",
  "transform_direction": "source_to_reference",
  "xyz_dtype": "float32",
  "rgb_dtype": "uint8",
  "rgb_range": [0, 255],
  "gt_available_to_process": false,
  "input_sha256": "...",
  "generator_commit": "d579a80d71c3d6ae37ee58ba5a3943fe81e8427d",
  "generator_weight_sha256": "b4900863c86629c24386189094691f159c1ff437b5623510a11c9468bc8cb814"
}
```

Required arrays:

- `src_xyz[Ns,3]`, `ref_xyz[Nr,3]`
- `src_rgb[Ns,3]`, `ref_rgb[Nr,3]`
- stable source/reference point IDs or original indices
- the exact sampling/voxel trajectory and its seed

XYZ and RGB must share identical point indices at every stage. RGB comes from
sensor/InSeg color, never from semantic palette or GT annotation.

## 3. GT sentinel and taint contract

The production process is launched with no GT paths, GT environment variables,
or GT-bearing cache fields. Static and runtime checks reject imports/reads that
contain known GT pose, identity, correspondence, scene-pair, or annotation
sources.

Because pinned official ColorPCR unconditionally reads the `transform` key for a
diagnostic branch, the wrapper may insert a **synthetic non-GT sentinel** only if
all of the following pass:

1. sentinel A and sentinel B are generated without scene data and are different;
2. model is in `eval()` and gradient tracking is off;
3. predicted `src_corr_points`, `ref_corr_points`, `corr_scores`, and
   `estimated_transform` are byte-identical for A and B;
4. all `gt_*` and diagnostic GT-correspondence outputs are dropped before cache;
5. a sentinel hash and invariance receipt are included in the manifest.

Any difference is `GT_SENTINEL_DEPENDENCE` and stops the pair. Identity fallback
is forbidden. A solver exception is a typed rejection, never a transform.

## 4. ColorPCR correspondence ABI

Pinned source and weight:

```text
source = mujc2021/ColorPCR@d579a80d71c3d6ae37ee58ba5a3943fe81e8427d
weight = weights.pth.tar
weight_size = 40246600
weight_sha256 = b4900863c86629c24386189094691f159c1ff437b5623510a11c9468bc8cb814
```

ColorPCR runs in an isolated, pinned runtime. The shared SGAligner environment
must not be mutated. The adapter consumes the immutable pair input and writes:

```text
src_corr[M,3] float32 meters
ref_corr[M,3] float32 meters
scores[M] float32 finite
src_point_ids[M] int64
ref_point_ids[M] int64
rigid_group_ids[M] int64
corr_sha256
```

`rigid_group_ids` are derived from GT-free model/geometry information only.
No real GT transform may be passed to the official model. Output dictionary ABI
is required; tuple-shaped custom Jojo output is rejected.

Correspondences are sorted deterministically by descending finite score and a
stable ID tie-breaker. Ambiguous correspondences remain explicitly marked and
cannot be reclassified using post-hoc GT. A pair with no valid correspondences
is a typed rejection.

## 5. Shared solver cache

Both arms read the exact same read-only correspondence cache. Required fields:

```json
{
  "correspondence_schema": "colorpcr-corr/v1",
  "pair_id": "...",
  "count_before_filter": 0,
  "count_after_filter": 0,
  "cap": 256,
  "ordering": "score_desc_then_stable_ids",
  "source": "colorpcr",
  "corr_sha256": "...",
  "input_sha256": "..."
}
```

The cap, order, finite/duplicate rules, and rigid-group arm are frozen and shared.
Solvers cannot regenerate, resample, reorder, merge, or see different inputs.
Multiple incompatible rigid groups are never pooled.

## 6. Arm A: PointDSC

Pinned source and weight:

```text
source = XuyangBai/PointDSC@b009d536ac10b570853833f2178397c154745da9
weight = snapshot/PointDSC_3DMatch_release/models/model_best.pkl
weight_size = 4380238
weight_sha256 = 20662778fca1a7d2c4e2f79f381d4be6cb891834d7bb4bd91ade9d89b0d13bd4
```

Loading uses `strict=False` only to permit the one known unexpected legacy key
`gamma`. Missing keys or any other unexpected key stop the run.

Preconditions:

- at least 10 finite unique correspondence pairs after cap;
- one rigid group only;
- no GT-bearing model inputs;
- stable input and weight hashes.

Input construction:

```python
corr_pos = np.concatenate([src_corr, ref_corr], axis=1)
corr_pos = corr_pos - corr_pos.mean(axis=0, keepdims=True)
data = {
    "corr_pos": corr_pos[None, ...],
    "src_keypts": src_corr[None, ...],
    "tgt_keypts": ref_corr[None, ...],
    "testing": True,
}
```

The official model's output transform is normalized and validated as canonical
`T_src_to_ref`. PointDSC is a shadow solver; it cannot write a production pose.
Its pinned tree has no explicit license file, so source/weight redistribution is
not authorized by this contract.

## 7. Arm B: pyGCRANSAC

Arm B consumes the identical `src_corr`, `ref_corr`, and deterministic order.
It may consume `scores` only where the preregistered API explicitly supports
weights; otherwise scores only define the shared cap.

```text
T_raw = pygcransac.findRigidTransform(...)
T_src_to_ref = T_raw.T
```

The synthetic transform receipt proving the transpose rule is mandatory. All
RANSAC thresholds, confidence, sampler, neighborhood, minimum inliers, and seed
are preregistered. Failure returns a typed rejection, not identity.

## 8. Solver-independent gates

Each arm independently records:

- input/correspondence/weight/code hashes;
- SE(3) validity and transform direction;
- inlier count/ratio and residual quantiles;
- bidirectional independent-solve rotation/translation disagreement;
- surface overlap and symmetric Chamfer using raw surfaces;
- deterministic repeat result;
- typed failure reason.

A pair is eligible for later offline comparison only if its arm passes every
frozen gate. Arm disagreement does not select the lower-residual answer; it
causes `SOLVER_DISAGREEMENT` and fail-closed rejection until an independent
preregistered rule resolves it.

ICP is excluded from the primary comparison. If a later protocol adds ICP, it
must use the same initialization-independent settings for both arms, record raw
and refined transforms separately, and never turn a primary rejection into an
acceptance without a new preregistration.

## 9. Separation, selection, and writes

- `selection89` is development-only. It may diagnose fixed alternatives but
  cannot choose the final arm, cap, threshold, checkpoint, or pilot winner.
- Calibration and a fixed pilot pair list are frozen before evaluation.
- `official92` stays closed.
- GT is post-hoc evaluation only after decisions and artifacts are frozen.
- Both arms are shadow-only: no transform file in a production output location,
  no refusion, no pose graph, no default checkpoint change.
- Only a separate authorization may promote a later candidate.

## 10. Required artifacts

For every run, write a new evidence directory containing:

```text
protocol.md
environment.json
input_manifest.json
gt_sentinel_invariance.json
correspondences.npz
correspondence_manifest.json
pointdsc_result.json
pygcransac_result.json
solver_comparison.json
failures.json
commands.sh
git_state.txt
artifact_manifest.json
```

`artifact_manifest.json` is written last and independently verified. Every
material artifact carries byte count and SHA-256. The evidence directory is
append-only after sealing.

## 11. Minimum safe execution sequence

```text
0. verify pinned code, weights, input list, environment, and no-GT policy
1. build immutable XYZ/RGB pair inputs
2. run ColorPCR twice with two non-GT sentinels
3. require predicted-output invariance; discard all gt_* values
4. seal one correspondence cache
5. replay PointDSC and pyGCRANSAC from that cache only
6. normalize transform conventions and run frozen safety gates
7. repeat for determinism
8. seal manifests
9. stop; do not select, reconstruct, open blind data, or promote checkpoint
```

A V11.3-only preliminary smoke may begin at step 4 using existing
`geot_corrs.npz`, with generator recorded as `geotransformer`. Such a run can
test PointDSC/pyGCRANSAC plumbing, but it is not evidence that ColorPCR is
integrated and cannot satisfy the final pilot.

## 12. Authorization state

```text
pointdsc_on_v11_3_corr_shadow = allowed
pygcransac_on_same_v11_3_corr_shadow = allowed
colorpcr_in_shared_sgaligner_env = forbidden
colorpcr_shadow = blocked_on_isolated_runtime_and_sentinel_invariance
training = forbidden
production_registration = forbidden
checkpoint_promotion = forbidden
official92 = forbidden
```
