# Robust no-GT pose backend

This branch keeps DPV-SLAM as the continuous RGB-D camera tracker. PAGOR,
G3Reg and TEASER++ are used only as inspiration or witnesses for sparse
submap constraints; none of them is presented as 30 FPS odometry.

The production boundary is:

`RGB-D manifest -> DPV T_world_camera -> local submaps -> SGAligner point
correspondences -> compatibility/pyGCRANSAC/TEASER hypotheses -> unique
cross-family consensus -> dense fail-closed gates -> sparse pose graph ->
all-frame correction -> TSDF refusion`.

pyGCRANSAC is repeated five times with a four-run quorum and is recorded as a
non-voting witness. This is intentional: its process-level randomness must
not change whether a loop is released. The voting families are the
deterministic compatibility graph, deterministic bounded triangle RANSAC, and
TEASER++ when available.

TEASER++ has no PyPI wheel in the tested environment. Build the official
MIT-SPARK source binding in a sibling checkout with:

```bash
python -m pip install pybind11
scripts/build_teaserpp_binding.sh /path/to/TEASER-plusplus /path/to/python
```

The authoritative 2026-09-01 server build used source commit
`52a9c52ee7d4c838c5e8a75458c33178be5bfb70`; its fixed-scale GNC-TLS binding
was checked with a known rigid transform containing outliers before data runs.

`official_top3` and the old inference defaults are unchanged. The new path is
enabled only with `--robust-pose-backend`. Add `--no-gt-evaluation` for a
process that never opens GT transforms or anchor labels.

## Sequence CLI

Set `PYTHONPATH=src`, then create a dataset manifest:

```bash
python -m pose_pipeline manifest --dataset scannet --input /data/scene0000_00 \
  --output /results/scene0000_00.manifest.json
```

Replay it through an already running DPV socket worker:

```bash
python -m pose_pipeline replay --manifest /results/scene0000_00.manifest.json \
  --socket /tmp/dpvslam.sock --output /results/scene0000_00_dpv
```

Run the same-input arms and refusion:

```bash
python -m pose_pipeline run --arm baseline \
  --manifest /results/scene0000_00_dpv/tracked_manifest.json \
  --trajectory /results/scene0000_00_dpv/trajectory.json \
  --output /results/scene0000_00_baseline

python -m pose_pipeline run --arm candidate \
  --manifest /results/scene0000_00_dpv/tracked_manifest.json \
  --trajectory /results/scene0000_00_dpv/trajectory.json \
  --output /results/scene0000_00_candidate

python -m pose_pipeline refuse \
  --manifest /results/scene0000_00_dpv/tracked_manifest.json \
  --trajectory /results/scene0000_00_candidate/trajectory.json \
  --output /results/scene0000_00_candidate_refusion
```

An experimental high-recall preset can be reproduced without changing the
default solver gates:

```bash
python -m pose_pipeline run --arm candidate \
  --manifest /results/scene0000_00_dpv/tracked_manifest.json \
  --trajectory /results/scene0000_00_dpv/trajectory.json \
  --output /results/scene0000_00_candidate_high_recall \
  --maximum-loop-pairs 120 \
  --high-leverage-loop-min-span-fraction 1.0 \
  --high-leverage-loop-weight-cap 0.3
```

This preset is development-only and remains opt-in. It was selected on one
ScanNet scene using no-GT refusion safety; it is not a promoted default. The
loop evidence records both proposal and weighting configs. Public-matrix runs
with this setting must pass `--development-sequence scene0030_00` to keep the
parameter-selection scene out of held-out aggregation.

The sequence runner labels FPFH correspondences as `geometry_bootstrap_fpfh`.
It must not be reported as SGAligner evidence. SGAligner inference feeds its
GeoTransformer correspondences directly into the same robust hypothesis
backend.

All outputs are create-only. Dataset pose and mesh paths are rejected by the
inference manifest. Evaluation uses a separate command and output directory.

## Frozen validation drivers

The public-data runner restarts DPV for every sequence and hashes every RGB-D
file visible to inference:

```bash
python scripts/run_public_pose_matrix.py --dataset scannet \
  --data-root /data/scannet-rgbd --output /results/scannet \
  --dpv-python /envs/dpv/bin/python --dpv-worker /dpv/pose_worker.py \
  --dpvo-root /dpvo --dpvo-network /dpvo/dpvo.pth \
  --dpvo-config /dpv/config.yaml \
  --dpv-metric-config /dpv/metric_posefix.json --refuse
```

`--dpv-metric-config` is optional only for deliberate worker-default runs. If
it is omitted, local recovery may be disabled by the DPV worker. The runner
records both the resolved path and SHA-256 of a supplied metric config in
`environment.json`; matched experiments must compare these fields before
comparing backend results.

3RScan uses a transform-free selection file. Its builder is deliberately a
separate process: the inference runner never opens `3RScan.json` because that
file also carries evaluation transforms.

```bash
python scripts/build_scan3r_pose_selection.py \
  --metadata /data/3RScan.json --data-root /data/3RScan_full \
  --development-groups 8 --sentinel 4acaebcc-6c10-2a2a-858b-29c7e4fb410d \
  --output /results/scan3r-selection.json

python scripts/run_public_pose_matrix.py --dataset 3rscan \
  --data-root /data/3RScan_full --scan3r-selection /results/scan3r-selection.json \
  --output /results/scan3r-sequences \
  --dpv-python /envs/dpv/bin/python --dpv-worker /dpv/pose_worker.py \
  --dpvo-root /dpvo --dpvo-network /dpvo/dpvo.pth \
  --dpvo-config /dpv/config.yaml \
  --dpv-metric-config /dpv/metric_posefix.json --refuse
```

Reference/rescan registration is also split into inference and evaluation
processes. The `run` phase consumes only reconstructed clouds and the sanitized
selection; only `evaluate` opens group transforms and camera poses:

```bash
python scripts/run_scan3r_pair_matrix.py run \
  --selection /results/scan3r-selection.json \
  --sequence-results /results/scan3r-sequences \
  --output /results/scan3r-pair-inference

python scripts/run_scan3r_pair_matrix.py evaluate \
  --inference /results/scan3r-pair-inference \
  --metadata /data/3RScan.json --data-root /data/3RScan_full \
  --sequence-results /results/scan3r-sequences \
  --output /results/scan3r-pair-evaluation
```

The five-sequence Orbbec runner treats a rejected sparse loop as a verified
no-op: it retains the complete DPV trajectory, never creates missing poses,
and still refuses every admitted frame. A no-op is not an identity-pose
fallback.

3RScan portrait frames are padded on the right and bottom to the next multiple
of 16 before DPV replay. The padding operation and dimensions are recorded in
the replay audit; no image resize, crop, or hidden principal-point shift is
performed. Fewer than two valid DPV poses and sparse-submap construction
failures produce an explicit fail-closed no-op rather than an exception or an
identity trajectory.

## 2026-09-01 validation outcome

The full ScanNet, 3RScan, and five-sequence Orbbec validation did not meet the
promotion gates. The backend remains opt-in and the safe default is unchanged.
Compact summaries, hashes, fixed-view comparisons, and the failure index are
in `pose_backend_validation_20260901/`. Large PLYs and raw logs remain in the
create-only result roots on the authoritative server.

A later causal rerun found that the original public-data command omitted the
DPV metric config used by the earlier demo. See
`pose_backend_validation_20260901/POSEFIX_CAUSAL_RERUN.md`. This corrects the
interpretation of the original ScanNet coverage result but is not sufficient
to promote the backend from a single scene.
