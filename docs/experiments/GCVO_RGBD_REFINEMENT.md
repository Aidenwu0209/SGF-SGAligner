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
