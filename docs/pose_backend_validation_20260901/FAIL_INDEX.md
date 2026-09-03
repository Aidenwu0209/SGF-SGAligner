# FAIL index

The candidate is not promoted. This index separates failures by stage instead
of treating every failed gate as a registration error.

## Matching and solver disagreement

- ScanNet: 12 of 16 sequences produced a verified no-op because no sparse loop
  survived the cross-family and dense gates. This is safe rejection, not proof
  that the DPV trajectory is correct.
- 3RScan sequence runs: all 84 completed sequences were no-ops. The candidate
  produced no corrections, so the pose metric improvement was zero.
- Official SGAligner no-GT smoke did produce a unique compatibility-graph plus
  TEASER++ cluster. pyGCRANSAC repeat consensus was unavailable for that pair;
  it did not become a hidden required fallback.

## Incorrect sparse constraint

- The candidate accepted one non-official-reference validation pair:
  `10b1792e-3938-2467-8bb3-172148ae5a67_to_bf9a3d9e-45a5-2e80-83c6-4e427c5586a2`.
  Independent evaluation measured 99.3067 degrees RRE and 1.16905 m RTE. The
  all-validation catastrophic-edge gate therefore fails even though the error
  is outside the 109 official reference/rescan rows.

## Optimization over-correction and map shrinkage

- Orbbec `leave_and_return` improved layer conflict by 24.39%, but retained
  only 71.32% of the baseline points. It fails the 80% map-content gate.
- ScanNet corrected four sequences, but no corrected sequence passed the full
  geometry-improvement decision. Two corrected sequences failed geometry
  safety in the sequence summary.

## DPV front-end coverage and drift

- 3RScan: 95 of 179 present sequences failed before candidate correction, and
  the 84 completed sequences had only 0.687% mean valid-pose coverage. The RGB-D
  sampling contains inter-frame motion outside the tested DPV continuity
  regime; a sparse backend cannot recover missing full-frame trajectories.
- ScanNet mean pose coverage was 49.67%. For example, `scene0000_00` retained
  213 of 5578 input frames. The backend preserved coverage but cannot solve the
  upstream tracking loss.

## Refusion

- No accepted output used an identity fallback, and full-frame refusion checks
  did not report a missing admitted frame.
- One of 16 ScanNet scenes lacked the evaluation mesh, so reconstruction
  evaluation completed on 15 scenes. This is recorded as missing evaluation
  data, not converted into a successful geometry result.

## Retained failed create-only runs on the authoritative host

- `/home/aidenwu/Documents/SGF-SGAligner-develop-scan3r-authoritative-teaser-20260901`:
  15 samples exposed RGB-D dimensions not divisible by 16.
- `/home/aidenwu/Documents/SGF-SGAligner-develop-scan3r-authoritative-teaser-v2-20260901`:
  10 samples exposed an empty RPE aggregate for a single valid pose.
- `/home/aidenwu/Documents/SGF-SGAligner-develop-scan3r-authoritative-teaser-v3-20260901`:
  48 samples exposed an uncaught sparse-submap point-count exception.
- `/home/aidenwu/Documents/SGF-SGAligner-develop-scan3r-pair-evaluation-teaser-v4b-20260901`:
  evaluation was deliberately stopped when the compact data root lacked one
  pose file; v4c uses the full evaluation-only root.

The first three implementation defects were fixed in `develop`; their
directories remain unchanged as failure evidence. The final v4 sequence and
v4c pair results are the authoritative 3RScan results.
