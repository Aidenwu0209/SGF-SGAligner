# LingBot-Map-inspired adaptive anchors

This experiment adapts one bounded LingBot-Map idea: geometric-flow keyframe
selection. It deliberately does not import LingBot-Map's learned model,
Sim(3) stitching, or pose convention.

The selector samples real sensor depth, unprojects it with per-frame
intrinsics, transforms it with the existing complete metric
`T_world_camera`, and reprojects it into the last accepted anchor. It records
median/p90 pixel displacement, in-bounds overlap, translation, and rotation.
Missing or unusable depth rejects the adaptive path and retains the original
DPV trajectory. The existing fixed-stride policy remains the default.

Default experimental policy:

- minimum gap 20 frames, aligned with the submap half-window;
- maximum gap 80 frames, equal to the fixed production stride;
- 24 px median geometric-flow trigger;
- 0.35 minimum projected in-bounds fraction;
- 0.25 m translation or 12 degree rotation trigger;
- no GT, no identity fallback, metric SE(3) only.

The 2026-09-03 Orbbec five-scene validation completed all 4,126 frames and all
five final TSDF refusions. Anchor schedule coverage passed 5/5, but final
geometry improvement passed only 1/5 and safety passed 4/5. Therefore the
feature remains CLI opt-in; the unchanged production configuration continues
to use fixed anchors.

Reproduce the stages with:

```bash
PYTHONPATH=src python scripts/validate_adaptive_anchors.py \
  --manifest MANIFEST.json \
  --trajectory DPV_TRAJECTORY.json \
  --output ANCHOR_AB.json

PYTHONPATH=src python scripts/validate_adaptive_anchor_backend.py \
  --manifest MANIFEST.json \
  --trajectory DPV_TRAJECTORY.json \
  --baseline-trajectory BASELINE_TRAJECTORY.json \
  --baseline-refusion-result BASELINE_REFUSION_RESULT.json \
  --output CREATE_ONLY_OUTPUT
```
