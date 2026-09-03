# High-recall parameter sweep: `scene0030_00`

Date: 2026-09-01

Authoritative host: `100.72.138.33`

This is a development-scene diagnostic, not held-out evidence. Parameter
selection makes `scene0030_00` ineligible for future held-out claims.
All inference choices below were made from proposal evidence and no-GT
refusion gates; GT poses were opened only after each inference output was
sealed.

## Finding

The current top-36 proposal budget included only two of the eight nearest
anchor equivalents of the earlier demo loops. Six were outside the budget,
at ranks 42, 45, 46, 60, 90, and 112. The dominant limitation was therefore
proposal recall, not the cross-solver consensus thresholds.

| Setting | Accepted loops | Translation ATE gain | Rotation ATE gain | Layer-conflict gain | Ground-tilt delta | Safety |
|---|---:|---:|---:|---:|---:|---|
| 36 pairs, default weights | 1 | 21.4% | 41.6% | 6.15% | +1.66 deg | pass |
| 72 pairs, default weights | 3 | 56.4% | 36.9% | 12.90% | +0.51 deg | pass |
| 120 pairs, default weights | 4 | 61.9% | 59.4% | 18.88% | +2.66 deg | **fail** |
| 120 pairs, full-span cap 0.7 | 4 | not selected | not selected | 19.24% | +2.20 deg | **fail** |
| 120 pairs, full-span cap 0.3 | 4 | 61.0% | 48.3% | 19.23% | +1.18 deg | **pass** |

The 120/default and 120/cap-0.7 variants were rejected before GT evaluation
because their no-GT ground-tilt regression exceeded 2 degrees. The retained
experimental setting keeps all registration acceptance gates unchanged and
only caps the weight of a loop spanning the entire anchor sequence.

## Retained opt-in result

The four accepted loops were `526->1838`, `116->1663`, `526->1933`, and
`116->2497`. Their evaluation-only transform errors were respectively
`0.76 deg/0.019 m`, `1.47 deg/0.042 m`, `1.32 deg/0.044 m`, and
`2.60 deg/0.110 m`; none crossed the catastrophic `20 deg/0.5 m` boundary.

Against the same 2277-pose DPV trajectory, translation ATE RMSE changed from
`0.6076 m` to `0.2367 m`, and rotation ATE RMSE from `7.483 deg` to
`3.870 deg`. In no-GT refusion, layer conflict improved 19.23%, dominant-plane
thickness improved 5.05%, point count remained 83.75%, and every bounding-box
axis remained above 85% of baseline. The independent GT-control geometry
metric improved symmetric mean distance from `0.2377 m` to `0.1221 m`
(48.6%) and median horizontal spread from `0.5034 m` to `0.2054 m` (59.2%).

## Reproduce the candidate arm

```bash
PYTHONPATH=src python -m pose_pipeline run --arm candidate \
  --manifest /results/scene0030_00/frontend/tracked_manifest.json \
  --trajectory /results/scene0030_00/frontend/trajectory.json \
  --output /results/scene0030_00/candidate_high_recall \
  --maximum-loop-pairs 120 \
  --high-leverage-loop-min-span-fraction 1.0 \
  --high-leverage-loop-weight-cap 0.3
```

The public and Orbbec matrix drivers accept the same three parameter flags.
Defaults remain 36 pairs with no high-leverage cap. Any public matrix using
this preset must also pass `--development-sequence scene0030_00`, so the scene
used for parameter selection cannot enter held-out aggregation.

The compact machine-readable result is
[`parameter_sweep_scene0030_summary.json`](parameter_sweep_scene0030_summary.json).
Large PLYs and logs remain in the create-only server result root
`/home/aidenwu/Documents/SGF-SGAligner-posefix-causal-scene0030-20260901/scene0030_00/parameter_sweeps`.
