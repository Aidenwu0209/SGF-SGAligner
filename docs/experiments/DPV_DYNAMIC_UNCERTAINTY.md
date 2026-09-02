# DPV dynamic uncertainty and depth-prior weighting experiment

Branch: `exp/pose-dpv-dynamic-uncertainty`

Base: `develop@f3d1adbd14d0bdd444f7ca9436aaa5d2d23cb326`

## Hypothesis

Per-pixel dynamic uncertainty from DROID-W or pi3-mos can keep moving regions
out of DPV metric-scale estimation, RGB-D consistency tests, and differentiable
BA patch weights. This should improve pose robustness without replacing the
continuous DPV frontend.

## Design boundary

- Confidence is a sidecar. RGB pixels are not blacked out or rewritten.
- The sidecar must cover every manifest frame in order and declare
  `gt_consumed=false`.
- Optional predicted depth is compared to sensor depth and only reduces
  confidence; it never replaces metric RGB-D depth.
- The existing pose graph and refusion backend remain unchanged.

## Implemented integration API

`DynamicUncertaintyStore.patch_weights()` returns aligned static confidence at
DPVO patch coordinates. `uncertainty_weighted_metric_scale()` provides the
corresponding robust metric-scale estimator. The same confidence map is the
intended multiplier for the DPVO BA information weights and RGB-D consistency
samples in the isolated worker environment.

```bash
export PYTHONPATH="$PWD/src"
python scripts/audit_dpv_dynamic_uncertainty.py \
  --manifest /path/manifest.json \
  --artifact /path/droidw_or_pi3mos_uncertainty.npz \
  --output /new/output/uncertainty_audit.json
```

## Acceptance

The candidate must use the same raw RGB-D and fused-frame list as baseline,
retain full metric `T_world_camera` coverage, and pass trajectory plus final
PLY gates. An uncertainty visualization or improved masked residual alone is
not evidence of a better SGF-SGA integration.
