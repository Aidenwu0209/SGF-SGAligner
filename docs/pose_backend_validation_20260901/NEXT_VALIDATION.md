# Next pose validation protocol

This document is the entry point for the validation after the rejected
high-recall matrix. It prevents another expensive full run before the known
front-end and correction-propagation blockers are removed.

## Frozen starting point

- Branch: `develop`.
- Failed high-recall evidence commit: `5cc5712ab8e9a7122e882f8e4dab208948c7885a`.
- Safe defaults remain `configs/pose/robust_backend.yaml`.
- `configs/pose/experimental_high_recall.yaml` is a failure control, not a
  candidate for promotion.
- The machine-readable stage contract is recorded in
  `configs/pose/next_validation.yaml`.

The full result that motivates this protocol is under `high_recall_full/`.
The run completed all 16 ScanNet scenes and all five Orbbec sequences, but
3RScan produced only 139 valid poses from 41,487 input frames. ScanNet ATE
improved while final geometry improved in only one scene; Orbbec improved in
zero of five scenes.

## Required implementation before a new full matrix

1. Repair the 3RScan metric-scale/input adapter until a frozen smoke set
   reaches at least 80% valid-pose coverage without scale jumps or GT access.
2. Add a pre-commit pose-graph correction audit. It must expose local
   correction translation/rotation derivatives, maximum anchor correction,
   and post-optimization residuals. A rejected correction retains the complete
   DPV trajectory.
3. Replace point-count-only map completeness with occupied-voxel and robust
   map-extent evidence. Match the same physical plane before comparing tilt.
4. Keep full PLY A/B geometry checks as release QA. They must not be presented
   as a realtime per-frame production gate.

## Stage 0: checkout and focused tests

On the authoritative host, use a clean checkout and create-only output root:

```bash
cd /home/aidenwu/Documents/SGF-SGAligner-highrecall-full-20260901-src
git fetch origin develop
git switch develop
git pull --ff-only origin develop
git status --short --branch

export PYTHONPATH="$PWD/src"
python -m unittest \
  tests.test_pose_pipeline_contracts \
  tests.test_robust_pose_backend \
  tests.test_pose_graph_backend \
  tests.test_adapter_contract \
  tests.test_geometry_metrics \
  tests.test_pose_runner_fail_closed \
  tests.test_rgbd_refusion_contract \
  tests.test_public_pose_matrix_driver
```

Stop if the checkout is dirty, a focused test fails, or the resolved commit is
not recorded in the new output `environment.json`.

## Stage 1: 3RScan front-end fail-fast

Rebuild a transform-free selection in a new output directory. The inference
runner must receive the DPV metric profile explicitly:

```bash
python scripts/build_scan3r_pose_selection.py \
  --metadata /home/aidenwu/Documents/SceneGraphFusion/data/3RScan/3RScan.json \
  --data-root /home/aidenwu/Documents/SceneGraphFusion/data/3RScan_full \
  --development-groups 8 \
  --sentinel 4acaebcc-6c10-2a2a-858b-29c7e4fb410d \
  --output /home/aidenwu/Documents/SGF-SGAligner-next-scan3r-selection-20260902.json

python scripts/run_public_pose_matrix.py \
  --dataset 3rscan \
  --data-root /home/aidenwu/Documents/SceneGraphFusion/data/3RScan_full \
  --scan3r-selection /home/aidenwu/Documents/SGF-SGAligner-next-scan3r-selection-20260902.json \
  --output /home/aidenwu/Documents/SGF-SGAligner-next-scan3r-smoke-20260902 \
  --dpv-python /home/aidenwu/miniconda3/envs/torch113/bin/python3.10 \
  --dpv-worker /home/aidenwu/Documents/SceneGraphFusion_DPVSLAM/pose_frontend/dpvslam_pose_worker.py \
  --dpvo-root /home/aidenwu/Documents/SceneGraphFusion_DPVSLAM/third_party/DPVO \
  --dpvo-network /home/aidenwu/Documents/SceneGraphFusion_DPVSLAM/third_party/DPVO/dpvo.pth \
  --dpvo-config /home/aidenwu/Documents/SceneGraphFusion_DPVSLAM/config/dpvslam_live.yaml \
  --dpv-metric-config /home/aidenwu/Documents/SceneGraphFusion_DPVSLAM/config/experiments/dpv_metric_posefix_v3.json \
  --only 0988ea72-eb32-2e61-8344-99e2283c2728 \
  --only 2451c048-fae8-24f6-9043-f1604dbada2c \
  --only 095821f9-e2c2-2de1-9707-8f735cd1c148 \
  --only 4acaebcc-6c10-2a2a-858b-29c7e4fb410d
```

Do not start pair registration or a full 179-scan matrix unless every frozen
smoke sequence has finite `T_world_camera`, no identity fallback, and at least
80% valid-pose coverage.

## Stage 2: ScanNet correction regression set

Use the five fixed scenes in `configs/pose/next_validation.yaml`. They cover
map shrinkage (`scene0000_00`), the tuning scene (`scene0030_00`), two tilt
failures, and the only full-matrix geometry improvement (`scene0046_00`). Run
baseline and candidate from the same replayed DPV trajectory and refusion
frame list. Do not tune on `scene0046_00` after inspecting its result.

The stage stops on any catastrophic/unevaluable edge, pose coverage loss over
one percentage point, missing refusion frame, non-finite transform, or
correction-audit failure. The PLY comparison must report occupied-voxel
coverage, robust extent, matched-plane tilt/thickness, and layer conflict.

## Stage 3: five complete Orbbec sequences

Run all five cached no-GT DPV trajectories with identical admitted frames.
Rejected loops retain the baseline trajectory. Full promotion still requires
at least four of five primary geometry improvements and no sequence worsening
over ten percent. Point count alone is not sufficient to accept or reject a
map.

## Stage 4: held-out matrix

Only after stages 0-3 pass may a new create-only ScanNet/3RScan/Orbbec matrix
start. Freeze the candidate config and its SHA-256 before reading held-out or
Orbbec results. GT remains evaluation-only. Store large PLYs and raw logs on
`100.72.138.33`; commit only compact summaries, hashes, fixed views, and a
failure index.
