# ColorPCR / PointDSC read-only integration audit

Status: **PointDSC shadow-runtime ready; ColorPCR adapter not runtime-ready**

Audit base: `8e247076a9a5882cb5d71efb176cc46740197ffe`

This report is evidence for a future, GT-free shadow pilot. It does not authorize
training, checkpoint replacement, production selection, refusion, pose-graph
writes, or opening `official92`. No upstream source or default checkpoint was
modified during this audit.

## 1. Pinned upstreams and immutable receipts

| Component | Pinned source | Weight receipt | License finding |
|---|---|---|---|
| ColorPCR | `mujc2021/ColorPCR@d579a80d71c3d6ae37ee58ba5a3943fe81e8427d` | `weights.pth.tar`, 40,246,600 bytes, local SHA-256 `b4900863c86629c24386189094691f159c1ff437b5623510a11c9468bc8cb814` | MIT; preserve upstream license/notice |
| PointDSC | `XuyangBai/PointDSC@b009d536ac10b570853833f2178397c154745da9` | `snapshot/PointDSC_3DMatch_release/models/model_best.pkl`, 4,380,238 bytes, SHA-256 `20662778fca1a7d2c4e2f79f381d4be6cb891834d7bb4bd91ade9d89b0d13bd4` | No `LICENSE`, `COPYING`, or `NOTICE` in the pinned tree; do not redistribute until clarified |
| Official SGAligner release | current project receipt | `sgaligner_pct_gat_rel_attr.pth.tar`, 6,014,535 bytes, SHA-256 `b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e` | project/upstream terms apply |
| Jojo archive SGAligner | archive-only comparison | `sgaligner.pth.tar`, 6,014,118 bytes, SHA-256 `54fff852c8b2e5b99bf5b4a56af2fbef771c8a770b7249620bf31edea94c0e0d` | not an official-checkpoint substitute |

The ColorPCR GitHub release API exposes the asset size but no publisher digest.
Therefore its SHA above is a local receipt, not a publisher-attested checksum.
The PointDSC weight is byte-identical between the pinned official clone and the
Jojo archive. The Jojo SGAligner checkpoint is not byte-identical to the official
release checkpoint and must never silently replace it.

## 2. Jojo glue is not an official ColorPCR adapter

The archived glue script
`colorpcr_pointdsc/global_as_ref_pcd_with_sphere_filtering_pairwise_matched_node_via_colorpcr_pointdsc_with_wall_floor_retain.py`
has four material deviations:

1. It imports a custom `sia_model_for_point_corr_v2` that is absent from the
   archived official ColorPCR tree and expects a five-element tuple. Official
   ColorPCR returns a dictionary.
2. Official ColorPCR reads `data_dict["transform"]` even in evaluation because
   that value feeds a diagnostic GT-correspondence branch. The archived glue
   does not provide that key. An unmodified official call therefore fails before
   predicted correspondences are returned.
3. A broad exception handler returns identity. A later fallback may read a GT
   relative camera pose. This violates the no-GT inference boundary and hides
   ABI/runtime errors as apparently valid transforms.
4. The glue pre-transforms source points by a cumulative transform, estimates a
   delta, then left-composes the delta. That sequence needs an explicit frame
   proof before reuse; it is not evidence of a correct source-to-reference ABI.

Conclusion: archive code is a method reference only. Reusing its backend idea is
allowed; importing the glue or its GT/identity fallbacks is forbidden.

## 3. Official API and transform conventions

### 3.1 ColorPCR

In official evaluation mode, predicted matching returns the dictionary fields:

- `src_corr_points`
- `ref_corr_points`
- `corr_scores`
- `estimated_transform`

The solver computes a transform from source correspondences to reference
correspondences. The canonical contract is column-vector SE(3):

```text
x_ref = R @ x_src + t
T_src_to_ref = [[R, t], [0, 0, 0, 1]]
```

For an `N x 3` row-major array, application is `X_ref = X_src @ R.T + t`.

The official model's required `transform` argument must not become a GT channel.
A wrapper may temporarily provide a synthetic sentinel solely to satisfy the
diagnostic branch, but authorization requires two different non-GT sentinels to
produce byte-identical predicted outputs. All `gt_*` outputs must be discarded.
A failed invariance test is a hard stop.

### 3.2 PointDSC

PointDSC consumes precomputed source/reference point correspondences. Its output
also uses the canonical source-to-reference column-vector convention above.
The 3DMatch release weight loads under the current Torch version using the
official `strict=False` behavior with one allowlisted unexpected legacy key,
`gamma`, and no missing keys. Any other missing/unexpected key is a hard stop.

### 3.3 pyGCRANSAC

In installed `pygcransac==0.1.1`, `findRigidTransform` returns a row-homogeneous
matrix: rotation is transposed and translation is in the last row. The canonical
matrix is therefore `T_canonical = T_raw.T`. A synthetic known-transform test
recovered rotation and translation to floating-point precision after transpose.
Using the raw result as a column transform is a P0 frame-convention bug.

## 4. Current SGAligner environment compatibility

Read-only environment inventory:

- Python `3.11.15`
- Torch `2.7.1+cu128`; Torch build CUDA `12.8`
- Open3D `0.19.0`
- NumPy `1.26.4`, SciPy `1.17.1`, scikit-learn `1.9.0`
- `pygcransac 0.1.1`, `torch_geometric 2.8.0.post1`
- `easydict` and `plyfile` present; `skimage` absent

ColorPCR upstream documents Python 3.8, Torch 1.7.1/CUDA 11.0, Open3D 0.11.2,
and a custom extension build. The inspected ColorPCR checkout is dirty: its
setup switches `CUDAExtension` to `CppExtension`, build outputs are untracked,
and the existing extension is CPython 3.10. Under the SGAligner Python, the
resolved `geotransformer` is the project's separate copy; its CPython 3.11
extension lacks ColorPCR's `grid_subsampling_dps` entry point. `skimage` is also
missing.

**ColorPCR must not be installed into or imported through the shared SGAligner
environment.** It needs a pinned isolated runtime or a clean, reviewed adapter
build. No install or environment mutation was performed.

PointDSC's model core does not require its historical full FCGF/Open3D/Minkowski
stack when correspondences are already available. A CPU smoke test under Torch
2.7 loaded the pinned weight, ran twice on a stable subset of a V11.3
`geot_corrs.npz`, returned finite `det(R) ~= 1`, and produced deterministic
output. This establishes runtime viability only, not scientific gain.

## 5. Can PointDSC run directly on V11.3 correspondence NPZ?

**Yes, as a GT-free shadow solver arm. No, not as a ColorPCR result and not as a
production winner.**

V11.3 `geot_corrs.npz` files expose per-arm arrays such as `src_corr_0`,
`ref_corr_0`, and `scores_0` in meter-scale coordinates. The minimum safe
PointDSC invocation is:

1. choose one rigid-group arm; never pool mutually inconsistent groups;
2. deterministically sort/cap by stable score and correspondence index;
3. require at least 10 finite, unique correspondences;
4. form `corr_pos = concatenate([src, ref], axis=1)` and subtract its column
   mean, matching official test preprocessing;
5. pass `src_keypts`, `tgt_keypts`, `corr_pos`, and `testing=True`;
6. convert/validate output against the canonical transform contract;
7. record that the input generator is GeoTransformer, not ColorPCR.

The pilot must retain a separate same-input pyGCRANSAC arm. The PointDSC config
supports at most 1000 nodes; the initial deterministic shadow cap may be 256.
PointDSC does not consume GeoTransformer confidence scores as model features;
scores may select the deterministic cap but cannot silently alter its ABI.

## 6. RGB availability and ColorPCR provenance

Raw InSeg data contains usable color:

- cached `inseg_cloud.npz` has `colors` with shape `N x 4`, dtype `uint8`, range
  0--255;
- current 3RScan `inseg.ply` files contain `red`, `green`, `blue` channels.

The V12 predicted graph loader currently reads only `xyz` and `labels`; it drops
colors. A future adapter must carry the first three color channels through the
same point indices, filtering, voxelization, and sampling trajectory as XYZ,
then normalize to `[0,1]` and derive HSV. It must not substitute a semantic-label
palette such as `node_RGB.ply` for sensor color.

## 7. Required GT-free two-arm shadow pilot

The correspondence generator must write one immutable cache. Both solvers read
the same source/reference arrays, order, deterministic cap, and scores:

- Arm A: ColorPCR correspondences -> PointDSC -> canonical `T_src_to_ref`
- Arm B: identical ColorPCR correspondences -> pyGCRANSAC -> transpose raw
  result -> canonical `T_src_to_ref`

ColorPCR's native local-to-global registration may be retained as a diagnostic
arm, not as the selected winner. ICP, if used, runs only after the primary arm
comparison and is recorded as a separate, identical post-solver refinement.

Both arms are fail-closed on nonfinite data, insufficient/duplicate
correspondences, an invalid rotation, solver exception, frame ambiguity,
PointDSC checkpoint-key drift, or transform disagreement outside preregistered
bounds. Rejection writes no transform and cannot reach refusion or pose graph.

## 8. Data separation and decision boundary

- `selection89` is development-only and may compare fixed candidates; it cannot
  select the final pilot winner.
- Calibration thresholds and the fixed pilot list must be frozen before final
  evaluation.
- `official92` remains closed.
- GT is allowed only in post-hoc metrics after all transforms, decisions, and
  hashes are frozen. GT is forbidden from import paths, environment variables,
  cache construction, sampling, matching, solving, vetoes, and fallback.
- This audit does not authorize checkpoint promotion or default-path changes.

## 9. Blocking items

1. Build a clean isolated ColorPCR runtime at the pinned source, with an SBOM and
   license record; do not reuse the dirty checkout as evidence.
2. Implement and pass the dual non-GT sentinel invariance test around official
   ColorPCR evaluation.
3. Add RGB/HSV to the adapter while proving exact XYZ/color index alignment.
4. Freeze the shared correspondence cache schema and transformation contract.
5. Clarify PointDSC code/weight redistribution permission.
6. Run a preregistered shadow pilot; no production writes and no blind set.

Until items 1--4 pass, the system is **not ready for ColorPCR integration**.
PointDSC is **ready only for a V11.3-correspondence shadow pilot** under the
contract in the companion document.
