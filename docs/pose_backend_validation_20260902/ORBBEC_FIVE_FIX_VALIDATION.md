# Orbbec five-fix validation — 2026-09-02

Base: `develop@f3d1adbd14d0bdd444f7ca9436aaa5d2d23cb326`

All changes and outputs were isolated from the existing dirty DPV-SLAM tree.
Inference used RGB-D only, consumed no GT, and used no identity fallback.

## Decision

The current Orbbec candidate is:

- DPV-SLAM with proximity loop closure disabled for this capture profile;
- `configs/pose/orbbec_15fps_metric_v2.json`;
- create-only finalized-pose sidecar and offline warm-up/recovery backfill;
- exact full-frame trajectory binding before refusion.

Do not enable proximity loop closure for this Orbbec path yet. The atomic guard
fails closed correctly, but the loop-on probe still has two deliberately
rejected frames and therefore does not satisfy the complete-trajectory gate.

## Five controlled validations

1. **No-loop causal control.** With the old v3 metric config, `scene_004`
   changed from 644/676 online poses with a post-warm-up gap to a complete
   676/676 finalized trajectory. `scene_002` still missed frames 260 and 274,
   proving that loop closure explains the 004 event but not all six sequences.
2. **Independent 15 FPS motion profile.** The final v2 limits are 0.10 m,
   10 degrees, 0.9 m/s, and 150 degrees/s. Depth consistency thresholds are
   unchanged. The v1 full matrix found the last two boundary frames in
   `scene_006`; both had 97.7% or better depth inliers and at most 2.7 cm depth
   RMSE. v2 then passed the repeated full six-sequence matrix.
3. **Atomic gauge/anchor transaction.** In the loop-on `scene_004` probe,
   epoch 78 produced a 1.212 m anchor correction. The worker logged
   `GAUGE_ANCHOR_TRANSACTION_ROLLED_BACK` and did not commit that epoch's scale.
   A later safe epoch committed scale and a 0.026 m anchor together. The probe
   remained fail-closed at 674/676, so loop-on is not promoted.
4. **Real pose backfill.** Warm-up frames are resolved from DPVO raw poses and
   transformed into the committed metric origin. Local recovery buffers actual
   DPVO pose candidates and releases them only after five verified RGB-D steps.
   There is no interpolation and no identity fill. The final matrix added 321
   poses absent from online valid responses. Source labels contain 317 warm-up
   and 12 recovery records; eight of those labels replace frames that were also
   valid in the online response stream.
5. **30 FPS + IMU capture/replay.** The capture path enables 200 Hz accelerometer
   and gyroscope streams, stores SI-unit samples in `imu.csv`, imports them into
   the signed GT-free manifest, partitions them by RGB-D timestamp, and sends
   them through the existing socket protocol. Synthetic end-to-end protocol
   tests pass. Physical capture is not yet validated because the Mac probe
   returned `No device found`; this is an explicit remaining hardware gate.

## Final pose matrix

Evidence root:

`/home/aidenwu/Documents/SGF-SGAligner-orbbec-five-fixes-runs-20260902/no_loop_15fps_v2_full6`

| Sequence | Input | Final poses | Online valid | Additional backfill |
|---|---:|---:|---:|---:|
| scene_001 | 524 | 524 | 491 | 33 |
| scene_002 | 278 | 278 | 218 | 60 |
| scene_003 | 1207 | 1207 | 1168 | 39 |
| scene_004 | 676 | 676 | 649 | 27 |
| scene_005 | 467 | 467 | 332 | 135 |
| scene_006 | 366 | 366 | 339 | 27 |
| **Total** | **3518** | **3518** | **3197** | **321** |

Independent audit: `independent_audit.json`

- exact manifest/trajectory frame IDs: pass;
- all transforms finite, proper SE(3): pass;
- no GT and no identity fallback: pass;
- GPU: RTX 4060 Laptop, driver 595.84;
- worker SHA256: `103e7cee41709bda58c93ce65b27abc95ce69080ca1c2eab8f77c231025f1e4f`;
- v2 config SHA256: `77605944b34e6102c14948bcbf3091a5ae612462102e18250e390988296de64b`.

CUDA/DPVO output hashes are not bitwise reproducible across repeated runs even
with the same seed, so the validation relies on exact input hashes, structural
gates, SE(3) checks, and receiving-side refusion rather than hash equality
between separate executions.

## Full receiving-side refusion

Evidence root:

`/home/aidenwu/Documents/SGF-SGAligner-orbbec-five-fixes-runs-20260902/no_loop_15fps_v2_full6_refusion`

All six sequences completed exact full-frame TSDF refusion:

- integrated frames: 3518/3518;
- finite PLY point total: 1,110,778;
- one create-only `refusion_result.json`, `geometry.json`, and hashed
  `refused.ply` per sequence;
- no GT and no identity fallback.

## Tests and hardware stop condition

- Remote focused suite: 43 tests, all passed.
- Local focused additions: 15 tests passed, one environment-dependent refusion
  test skipped.
- Capture failure path: cleanly writes `capture_status.json` with
  `state=failed` and `error=No device found`.
- Do not call IMU capture physically validated until a connected camera produces
  non-zero accel and gyro counts and the replay summary reports non-zero
  `imu_samples_sent`.

## ScanNet / 3RScan promotion check

The Orbbec-specific candidate was also tested against the public-dataset
promotion gates before any `develop` submission.  Both arms used the same
GT-free RGB-D manifest, v3 metric config, loop-closure setting, seed, DPVO
network, and DPVO config.  Dataset poses were opened only after both inference
arms exited.  Accuracy was compared on the exact baseline/candidate frame
intersection, so newly backfilled frames could not make the common-frame ATE or
RPE comparison easier.

The 3RScan fail-fast sequence
`2451c048-fae8-24f6-9043-f1604dbada2c` failed twice:

| Run | Baseline poses | Candidate poses | Candidate online | Common ATE m | Common ATE deg |
|---|---:|---:|---:|---:|---:|
| first | 9/150 | 76/150 | 1 | 0.173 -> 2.204 | 6.532 -> 26.356 |
| repeat | 9/150 | 75/150 | 2 | 0.160 -> 2.054 | 6.745 -> 29.529 |

The shortest ScanNet regression sequence, `scene0030_02`, also failed:

- coverage increased from 1717/1922 (89.33%) to 1911/1922 (99.43%), but the
  required complete trajectory still missed 11 frames;
- common-frame translation ATE improved from 0.455 m to 0.428 m;
- common-frame translation/rotation RPE improved slightly;
- common-frame rotation ATE regressed from 6.125 degrees to 9.361 degrees;
- full candidate ATE was 0.604 m / 18.324 degrees.

Evidence roots:

- `/home/aidenwu/Documents/SGF-SGAligner-orbbec-five-fixes-runs-20260902/3rscan_ab_smoke_150`
- `/home/aidenwu/Documents/SGF-SGAligner-orbbec-five-fixes-runs-20260902/3rscan_ab_smoke_150_repeat`
- `/home/aidenwu/Documents/SGF-SGAligner-orbbec-five-fixes-runs-20260902/scannet_ab_smoke_scene0030_02`

Global-default promotion decision: **rejected**.  The remaining three 3RScan
and four ScanNet sequences were not run after the repeated fail-fast gate.  No
full refusion was attempted because neither public-dataset candidate produced
the required exact all-frame trajectory.

The validated Orbbec path may be integrated only as an explicit opt-in
experiment.  That integration may include the backward-compatible IMU manifest
and replay contract, finalized-pose sidecar support, the create-only capture and
validation tools, the Orbbec v2 no-loop profile, and this evidence report.  It
must not replace the default DPV worker or default ScanNet/3RScan configuration,
and it must not be described as a public-dataset accuracy improvement.
