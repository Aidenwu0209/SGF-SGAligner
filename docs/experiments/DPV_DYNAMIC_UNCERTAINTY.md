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

## Fail-fast result (2026-09-02)

DROID-W produced a real, complete 32-frame uncertainty sidecar for
`scene0000_00`. Its audit reports all 32 frame IDs in order,
`gt_consumed=false`, and static-confidence coverage at the configured
threshold. Before modifying the production DPV worker, the uncertainty method
was tested in its native DROID-W BA on the same RGB stream and seed:

| DROID-W metric RMSE | uncertainty off | uncertainty on |
| --- | ---: | ---: |
| relative translation | 0.010462 m | 0.010838 m |
| relative rotation | 0.251421 deg | 0.258181 deg |
| absolute translation | 0.070412 m | 0.080883 m |
| absolute rotation | 0.308286 deg | 0.337376 deg |

All four metrics worsened. The official provider also needed three isolated
initialization-only patches to make its `activate=false` ablation runnable;
these patches are recorded under `provider_patches/` and do not alter the
uncertainty-on candidate. This negative precursor gate stopped the experiment
before mutating or rerunning the production DPV worker. Consequently, the
branch supplies the fail-closed sidecar/integration API and reproducible
evidence, but does **not** claim a DPV pose or final-refusion improvement.

Remote sidecar evidence root:
`/home/aidenwu/Documents/DPV-uncertainty-shadow-20260902-results`.
