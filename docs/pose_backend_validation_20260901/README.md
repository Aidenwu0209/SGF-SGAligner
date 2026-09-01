# No-GT pose backend validation, 2026-09-01

## Decision

The implementation is retained as an opt-in experiment on `develop`. It does
not meet the promotion gates and does not change `official_top3`, `main`, or
the default pose path. DPV-SLAM remains the continuous per-frame front end;
the new PAGOR/G3Reg/TEASER++-inspired code is only a sparse constraint backend.

Important addendum: the original ScanNet public-data run below used worker
metric defaults and did not reproduce the earlier demo's DPV `posefix_v3`
local-recovery profile. A controlled `scene0030_00` rerun restored valid-pose
coverage from 238 to 2277 frames and produced one safe loop. See
[`POSEFIX_CAUSAL_RERUN.md`](POSEFIX_CAUSAL_RERUN.md). The original matrix is
retained as a historical result of its recorded configuration, not as a fair
demo-versus-backend comparison.

Implementation commit: `481707fdb5ba31f79247449ca011244d99d39d46`  
Base commit: `84434efbbd19ea3c0914b7cf57c62a6f4e02000b`  
Frozen config SHA-256: `1040a4fa32174922fa66d2d6761214624811192c59ec1c46611f88f42bb6d641`

## Result matrix

| Dataset | Scope | Result | Promotion gate |
|---|---|---|---|
| ScanNet | 16 complete sequences, 12 held-out | translation ATE improved 12.52%, rotation ATE 15.62%; both 95% CIs touch zero | Fail |
| ScanNet geometry | 15/16 meshes | Chamfer improved 1.51%, F-score error 2.23% | Fail: below 10% |
| 3RScan | 47 validation groups, 156 present scans | 0.687% mean valid-pose coverage; zero corrected sequences | Fail |
| 3RScan pairs | 109 official pairs | 35 completed inference, 0 recall; one catastrophic accepted validation edge | Fail |
| Orbbec | five complete sequences | 0/5 improved by 10%; 3/5 pass safety | Fail |

There is one useful local result: ScanNet `scene0011_00` improved from
0.3847 m / 10.43 degrees to 0.2288 m / 4.27 degrees with two accepted sparse
loops. The paired and geometry aggregates do not support promotion.

## Contract evidence

- Estimation manifests contain RGB, depth, intrinsics, depth scale, and
  timestamps only. Input files are bound by SHA-256.
- Public-dataset GT is opened only by the separate `evaluate` commands.
- Orbbec runs contain no GT.
- Trajectories use metre-scale `T_world_camera`; relative hypotheses use
  `T_reference_source`.
- Official SGAligner no-GT smoke wrote `registration_decision.v2`, seven robust
  hypotheses, and a unique compatibility-graph plus TEASER++ consensus. Its
  status has `evaluation_enabled=false`, `gt_consumed=false`, and null RRE/RTE.
- Missing fused frames, non-finite poses, unavailable reverse evidence, and
  missing transforms fail closed. A rejected loop is a no-op over the original
  DPV trajectory, not an identity-pose fallback.

## Authoritative artifacts

Large PLYs and raw logs remain on `100.72.138.33`:

- ScanNet pose: `/home/aidenwu/Documents/SGF-SGAligner-develop-scannet16-authoritative-teaser-20260901`
- ScanNet reconstruction evaluation: `/home/aidenwu/Documents/SGF-SGAligner-develop-scannet-reconstruction-eval-teaser-20260901`
- 3RScan sequence: `/home/aidenwu/Documents/SGF-SGAligner-develop-scan3r-authoritative-teaser-v4-20260901`
- 3RScan pair inference: `/home/aidenwu/Documents/SGF-SGAligner-develop-scan3r-pair-inference-teaser-v4-20260901`
- 3RScan pair evaluation: `/home/aidenwu/Documents/SGF-SGAligner-develop-scan3r-pair-evaluation-teaser-v4c-20260901`
- Orbbec: `/home/aidenwu/Documents/SGF-SGAligner-develop-orbbec5-authoritative-teaser-20260901`
- Official SGAligner no-GT smoke: `/home/aidenwu/Documents/SGF-SGAligner-develop-official-nogt-robust-v2-smoke-20260901`

Exact compact source summaries are under `raw_summaries/`. `summary.json` and
`summary.csv` contain the promotion-level result. `HASHES.sha256` binds the
config, summaries, and five Orbbec fixed-view comparisons. See `FAIL_INDEX.md`
for failure classification and retained failed-run roots.

## ARM compatibility smoke

Host `100.105.135.18` shallow-cloned `develop` at
`e8fcb7cea61362025728752ee6d17692b1508f7e`. Twenty pure CPU tests covering
SE(3), pose graph, compatibility consensus, fail-closed gates, manifest
contracts, and RGB-D padding passed on aarch64. The current-clone Orbbec adapter
then rebuilt and hashed a 41-frame manifest with `gt_consumed=false`.

- manifest payload SHA-256: `43706be6c949cbbd1b84fa80842ed22d4f5250708d3c3aaffa38d395ca2a50e0`
- input-record SHA-256: `c3c8e9466c8706850ef3a7e4c39fd2aded6198a9e9cc898da550ebd525be2516`
- disk available before/after: 5,071,836 / 5,050,032 KiB

No public full dataset, Docker image, or large dependency was installed on the
ARM host. The 3 GiB stop threshold was not reached.

## Reproduction

Use the commands in `../ROBUST_POSE_BACKEND.md`. Freeze
`../../configs/pose/robust_backend.yaml` before selecting held-out or Orbbec
results. Run baseline and candidate from the same replayed DPV trajectory and
admitted-frame list. The inference phase must not receive ScanNet/3RScan pose,
mesh, metadata-transform, or evaluation directories. Run the separate
evaluator only after both arms are sealed.
