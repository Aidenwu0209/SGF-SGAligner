# V13 ColorPCR + PointDSC Shadow Research Protocol

Status: Phase-1 adapter pre-registration, GT-free, shadow-only. The isolated
ColorPCR source/weight/runtime audit and one sentinel-invariance smoke are
complete; fixed4 evaluation is not complete and no production use is allowed.

## Frozen inputs

- Base: sealed V11.3 commit `4848eec8b425188ebeec1e2b1e585918755f1dc6`.
- Pilot: immutable positions 0/44/88 plus immutable KNOWN_BAD.
- The sealed V11.3 shadow NPZs provide pair/geometry provenance; their prior
  transform and decision results are not inputs.
- Raw InSeg `xyz`, segmentation `labels`, and `colors` remain row-aligned.
  Segmentation labels are object masks, not pair/evaluation labels. XYZ is
  metres without rescaling. Colors remain uint8 RGB/RGBA and normalize only at
  an audited solver boundary. Stable exact-XYZ dedup is allowed only when label
  and color agree; conflicts fail closed.
- Arms are frozen as `sgf_selected_union` and `fullscan`. The selected union is
  the sealed V11.3 source/reference surface point set mapped back to raw InSeg
  rows by unique 1 mm grid membership. It uses only predicted/frozen shadow
  inputs, never GT or post-hoc pair labels. Mapping coverage must be 100%; a
  missing or multiply mapped grid key fails closed. `labels>0` is explicitly
  forbidden as a matched-object surrogate because fixed4 labels cover the
  entire full scan.
- Arm filtering happens before downsampling. Each filtered arm is aggregated on
  a world-origin 0.10 m voxel grid using `floor(xyz/0.10)`: XYZ is a float64
  centroid cast to float32, RGB is a float64 mean cast to float32 in [0,255],
  and exact source rows are retained as CSR flat indices plus offsets. Labels
  are never majority-voted and never enter ColorPCR. If the sealed selected
  union equals fullscan after mapping/voxelization, the arms are explicitly
  marked identical and are not counted as two independent pieces of evidence.
- Forbidden: GT transforms, identity/pose fallback, selection/calibration
  labels, posthoc, official92, relaxed fitness, official-source edits, default
  checkpoint change, or output-driven threshold tuning.

## Frozen future estimator

ColorPCR is only a correspondence provider. Each arm runs PointDSC and the
existing pyGCRANSAC independently, each forward/reverse with five repeats.
Each solver/direction requires a unique q4 mode (4/5) within unchanged 5 degrees
/ 0.10 m. Forward and inverse-reverse agree at the same gate; PointDSC and
pyGCRANSAC medoids also agree. Both pass fixed-trace ICP and unchanged Rule-B.
Known-bad is an unconditional veto. Missing dependency/SHA, fallback, multiple
modes, solver disagreement, or unsafe Rule-B rejects.

Resource ceiling: fixed4 x 2 arms x 2 solvers x 2 directions x 5 = 160 workers.
The ColorPCR stack is additionally capped at 512 total points at its coarsest
level. Allocation is deterministic proportional largest-remainder across the
two scans and each scan uses FPS with start index 0. All neighborhoods and
subsample/upsample maps are rebuilt after the cap in adapter code. Neither the
official repository nor checkpoint is edited.

## Existing GT-free PointDSC-only diagnostic

The sealed V11.3 official-GeoT correspondences were replayed through the audited
3DMatch PointDSC checkpoint on CPU, five seeds and both directions. Only pair
0958 met cross-direction stability (1.047 degrees / 0.039 m). Pair 68ba failed
(116.49 degrees / 0.982 m), f381 failed (20.82 degrees / 0.142 m), and known-bad
6a36 was vetoed (4.707 degrees / 0.182 m). This is a diagnosis, not a V13 result:
PointDSC-only cannot advance and the ColorPCR front-end audit is required.

Current status is `PHASE1_ISOLATED_RUNTIME_AUDITED_PILOT_NOT_RUN`. PointDSC has
no license file in the pinned tree, so redistribution remains forbidden. Even
a pilot pass does not authorize selection89, calibration, fixed12,
reconstruction, promotion, or official92.

## Phase-1 isolated runtime addendum

The audited ColorPCR candidate is upstream commit `d579a80...` with weight SHA
`b4900863...`, executed only in the separate `jojo2026` subprocess. Its upstream
runtime is additionally pinned by Python-tree SHA `26f73274...`, empty tracked
diff SHA `e3b0c442...`, and clean-build extension SHA `33160b28...`. A separate
detached worktree at exact d579a80 built the upstream CUDAExtension unchanged
under jojo2026 and imported after Torch load (clean-build receipt SHA
`08616d42...`, artifact-manifest SHA `25e97b90...`). The earlier dirty audit checkout
(CppExtension and CUDA-header build changes) is retained for provenance but is
forbidden as the V13 runtime. Its upstream
forward reads `transform` to build GT-node diagnostics even in evaluation, so
V13 never trusts a single forward. Every arm/direction is run twice with the
same RGB/HSV/stack-mode input: identity sentinel and a frozen nonzero proper
SE(3) sentinel (90 degree z rotation, translation [0.123,-0.071,0.049] m).
`ref_corr_points`, `src_corr_points`, `corr_scores`, and `estimated_transform`
must keep identical shape and remain within frozen absolute tolerances
1e-6/1e-6/1e-7/1e-6. Any dependency on the sentinel fails closed. Only those
four arrays enter the corr cache; upstream GT-node arrays and input transform
are discarded. Audited stack-mode neighbor limits are mandatory input, never
guessed. The sgaligner environment may send an invariant cache to PointDSC and
pyGCRANSAC independently, but cannot run until both dependency audits close.

The isolated `jojo2026` runtime (Python 3.10, Torch 1.13, Open3D 0.17) has
strict-loaded the pinned checkpoint. On pair 0958, identity and a second
non-GT 90-degree-plus-translation sentinel produced byte-identical 4550
correspondences, scores, native transform, and node indices; peak GPU memory
was about 2.6 GiB. This observed run authorizes only the isolated sentinel
mechanism. It is not shared-Torch-2.7 compatibility evidence and remains
provisional until its files and hashes are copied into the V13 evidence root.

## Resource-safety addendum frozen after raw-input OOM

An exploratory raw-full forward on f381 (about 29k points) failed allocating
2.16 GiB at roughly 7.6 GiB total usage. The Jojo-reference 0.05 m input voxel
plus coarsest cap 850 also OOMed on the same 8 GB host. Before any fixed4 result,
a resource preflight froze the only runnable configuration at 0.10 m
color-preserving input voxels and a total coarsest cap of 512. F381 completed
both directions at levels 4514/4444/3192/512, peak about 1.53 GiB, with
2552/2455 correspondences and native cross disagreement 0.613 deg/0.0097 m.
These are resource parameters, not accuracy thresholds, and may not be changed
again. The first three raw-full native cross-direction receipts are retained
only as diagnostics: 0958 = 0.241 deg/0.0051 m, 68ba = 0.393 deg/0.0975 m,
KNOWN_BAD = 0.112 deg/0.0045 m. KNOWN_BAD remains an unconditional veto despite
its apparent stability. All four pairs, both directions, and both non-GT
sentinels must now be replayed at the single frozen 0.10 m/512 configuration;
raw and 0.05 m outputs cannot be mixed into the fixed4 decision. No other pair,
voxel size, cap, threshold, or checkpoint may be introduced based on outcomes.
