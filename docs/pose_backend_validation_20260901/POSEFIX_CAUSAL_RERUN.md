# DPV posefix causal rerun: `scene0030_00`

Date: 2026-09-01

Branch: `develop`

Authoritative host: `100.72.138.33`

## Question

The earlier PAGOR-inspired demo looked much better than the full validation.
The full-validation driver was found to omit the DPV worker's
`--metric-config`. Its worker log therefore reported `local_recovery=disabled`
and stopped producing valid poses after frame 353. This rerun changes only
that input to the front end and keeps the current robust backend unchanged.

## Reproduction boundary

- RGB-D sequence: ScanNet `scene0030_00`, 2498 frames.
- DPVO network/config and seed are unchanged.
- DPV metric config: `dpv_metric_posefix_v3.json`, SHA-256
  `cd8f4fd4e163e00546d3855ad0e7e2ad62edf795784d4ab8940f89c2469eebe1`.
- Input-record audit SHA-256:
  `d9867383e872d0a42ec725907c9c23cb582e1d648fa93d80886628a6adb04a4a`.
- GT pose was unavailable to both inference arms and was opened only after
  both trajectories were sealed.
- The create-only result root is
  `/home/aidenwu/Documents/SGF-SGAligner-posefix-causal-scene0030-20260901`.

## Causal result

| Measure | Original default-metric run | Posefix causal rerun | Earlier PAGOR demo |
|---|---:|---:|---:|
| Valid DPV poses | 238 | 2277 | 2275 |
| Valid-pose coverage | 9.53% | 91.15% | 91.07% |
| Submap anchors | 4 | 30 | 22 |
| Loop proposals | 0 | 36 | 36 |
| Accepted loops | 0 | 1 | 8 |

The posefix run logged a committed local recovery and remained valid through
frame 2497. This establishes that the earlier 238-pose failure was a driver
configuration regression rather than a sparse-backend rejection.

## Current backend result

The only released loop was frame `526 -> 1838`. Evaluation-only error was
`0.763 degrees / 0.0186 m`; it was not catastrophic.

| Pose metric, all 2277 valid frames | Baseline | Candidate | Improvement |
|---|---:|---:|---:|
| Translation ATE RMSE | 0.6076 m | 0.4773 m | 21.4% |
| Rotation ATE RMSE | 7.483 deg | 4.370 deg | 41.6% |
| Translation ATE median | 0.5838 m | 0.4757 m | 18.5% |
| Rotation ATE median | 7.847 deg | 3.586 deg | 54.3% |

No-GT refusion safety passed. Layer conflict improved by 6.15%, dominant-plane
thickness by 10.94%, point count remained at 92.60%, and ground tilt increased
by 1.66 degrees. The predeclared 10% layer-conflict improvement gate therefore
did not pass.

Using the earlier demo's evaluation-only GT-control geometry metric, symmetric
mean distance improved from 0.2377 m to 0.1815 m (23.6%) and median horizontal
cell spread improved from 0.5034 m to 0.2546 m (49.4%). These are meaningful
gains, but remain below the earlier PAGOR demo's 78.4% and 66.6% improvements.

The fixed-view comparison is stored as
[`posefix_scene0030_fixed_views.png`](posefix_scene0030_fixed_views.png).

## Interpretation

The large demo/full-run discrepancy had two independent causes:

1. The public validation omitted the DPV metric profile, destroying the
   continuous trajectory before the sparse backend had enough submaps.
2. After fixing coverage, the current fail-closed backend released one loop,
   while the earlier looser PAGOR shadow released eight. The old result also
   used SGF's 1676-frame selected refusion, while this causal run performed
   unified refusion over all 2277 valid frames.

The metric-config pass-through is retained as a reproducibility fix. The
robust backend remains opt-in because this is one favorable ScanNet scene and
the full 3RScan/Orbbec matrices have not been rerun with the corrected front
end.
