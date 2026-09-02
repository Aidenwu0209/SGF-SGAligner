# SGF-SGA depth denoise A/B

The production default remains `off`. All profiles consume the same original
RGB-D files and produce `uint16` depth with the original shape and invalid-pixel
mask. No profile performs temporal filtering or hole filling.

Profiles:

- `off`: byte-exact existing behavior.
- `range_v1`: retain depths in the inclusive range 0.30–4.50 m.
- `bilateral_light_v1`: range filter plus OpenCV bilateral `d=5`,
  `sigmaColor=0.015 m`, `sigmaSpace=2 px`.
- `bilateral_medium_v1`: range filter plus OpenCV bilateral `d=5`,
  `sigmaColor=0.030 m`, `sigmaSpace=2 px`.

The same profile can be selected for each consumer:

```bash
PYTHONPATH=src python -m pose_pipeline replay \
  --manifest MANIFEST --socket DPV_SOCKET --output CREATE_ONLY_OUTPUT \
  --depth-filter-profile bilateral_light_v1

PYTHONPATH=src python -m pose_pipeline run \
  --arm candidate --manifest TRACKED_MANIFEST --trajectory TRAJECTORY \
  --output CREATE_ONLY_OUTPUT \
  --depth-filter-profile bilateral_light_v1

PYTHONPATH=src python -m pose_pipeline refuse \
  --manifest TRACKED_MANIFEST --trajectory TRAJECTORY \
  --output CREATE_ONLY_OUTPUT \
  --depth-filter-profile bilateral_light_v1
```

Stage 1 fixes one complete metric `T_world_camera` trajectory and varies only
the depth profile:

```bash
PYTHONPATH=src python scripts/run_depth_denoise_ab.py \
  --manifest TRACKED_MANIFEST \
  --trajectory TRAJECTORY \
  --profiles off,range_v1,bilateral_light_v1,bilateral_medium_v1 \
  --output CREATE_ONLY_OUTPUT
```

The runner hashes every inference-visible RGB-D input, records the profile
parameter hash, OpenCV version, filter timing, valid-pixel changes and filtered
depth rolling hash, then writes each final PLY, fixed views, geometry metrics,
acceptance decisions, environment, and an append-only event log. The optional
SOR arm is final-PLY-only (`nb_neighbors=20`, `std_ratio=2.0`) and never feeds
TSDF, DPV, or submaps. `source_sha256.json` seals the implementation files and
`MANIFEST.sha256` seals every completed artifact.

An existing output directory is fatal. Missing poses are fatal and are never
replaced by identity. Dataset ground truth is forbidden from filtering,
tracking, submaps, and fusion; it may only be consumed later by a separate
ScanNet evaluator.
