# DROID-W shadow frontend

This branch evaluates DROID-W as an independent pose frontend. It does not
replace the existing pose graph or RGB-D refusion backend.

## Locked provider

- repository: `https://github.com/MoyangLi00/DROID-W`
- commit: `c3414af6047d06bafc1a6645d09f23247f3b2cdc`
- official raw output: `traj/est_poses_full.txt`

The official evaluator performs a ground-truth Sim(3) alignment. That aligned
trajectory is forbidden here. The accepted artifact is the full trajectory
written by `save_traj()` before `align_full_traj()`, from a no-pose dataset
stream. Metric scale must come from the inference-side Metric3D-v2 depth
regularizer and its checkpoint digest must be recorded.

## Run contract

1. Use exactly the RGB frames in `rgbd_sequence_manifest.v1`; RGB-D depth is
   retained for SGF refusion and may be exposed through an isolated no-pose
   dataset adapter, but dataset pose files are forbidden.
2. Disable Gaussian mapping and save the raw, full trajectory.
3. Record provider/config/checkpoint hashes in a provenance JSON with
   `gt_consumed=false` and `sim3_alignment_used=false`.
4. Import with `scripts/import_droid_w_shadow.py`. Missing/reordered frames,
   non-metric motion, evaluation alignment, or GT provenance fail closed.
5. Run unchanged pose graph/refusion on the same fused frames and compare final
   geometry with DPV-SLAM. A DROID-W rendering is not an SGF success metric.

Current RTX 4060 feasibility and same-input results are recorded only after an
actual provider run; this adapter alone is not evidence of improvement.
