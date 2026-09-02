# G-CVO RGB-D local refinement experiment

Branch: `exp/pose-gcvo-rgbd-refinement`

Base: `develop@f3d1adbd14d0bdd444f7ca9436aaa5d2d23cb326`

## Hypothesis

Generalized-CVO can refine consecutive metric RGB-D motion in feature-sparse
or locally planar regions more reliably than retaining the DPV delta alone.
It is a local refinement arm, not a loop-closure or full-SLAM replacement.

## Contract

- Input remains the same GT-free RGB-D manifest and complete DPV
  `T_world_camera` trajectory.
- External G-CVO rows use `p_source = T_source_target @ p_target`.
- Every consecutive pair must be present. Missing pairs fail closed; they are
  never replaced by identity.
- A transform that disagrees too strongly with DPV or violates per-step motion
  gates is rejected. If fewer than 80% of pairs survive, the complete DPV
  baseline is retained.
- The existing pose graph and full-frame refusion are unchanged.

## Acceptance

Run baseline and candidate from the identical manifest and fused-frame list.
Promotion requires no pose-coverage loss, no non-finite transform, no identity
fallback, and final PLY improvement under occupied-voxel, extent, matched-plane
thickness/tilt, and layer-conflict checks. A lower local residual or ATE alone
is not sufficient.

## Adapter

```bash
export PYTHONPATH="$PWD/src"
python scripts/apply_gcvo_refinement.py \
  --manifest /path/manifest.json \
  --baseline-trajectory /path/dpv_trajectory.json \
  --gcvo-result /path/gcvo_relative.json \
  --output-trajectory /new/output/candidate.json \
  --output-audit /new/output/gcvo_audit.json
```

The official G-CVO executable remains an isolated external dependency. Its
exact commit, build configuration, and output hash must be recorded in the
external result before an authoritative run.

## 2026-09-02 RTX 4060 fail-fast result

Official commit `d1268382faf8853499e6f72abe1096ff396a1ae0` compiled with
CUDA 12.6 for SM 8.9 after applying the recorded build-only patch for a missing
upstream test source. The official transformed-bunny test passed with SE(3) log
error `0.007010`.

The same-input ScanNet fail-fast window used tracked frames 92--101 from
`scene0000_00`, DPV relative poses as initializers, and the official
`gcvo_eth3d_rgbd_v3.yaml`. Across nine pairs:

- translation RMSE: DPV `0.0108564 m`, G-CVO `0.0108600 m`;
- rotation RMSE: DPV `0.267472 deg`, G-CVO `0.358520 deg`;
- G-CVO won translation on 6/9 pairs, rotation on 1/9, and both on only 1/9.

The local arm therefore fails the fail-fast promotion condition: aggregate
translation is unchanged/slightly worse and rotation is 34% worse. A full
213-pair run and SGF refusion were intentionally not started. The safe default
remains disabled; this result is evidence against promoting the current
parameterization, not a claim that G-CVO cannot help after further tuning.
