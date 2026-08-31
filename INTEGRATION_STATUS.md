# SGF-SGAligner v0.1.0 research status

Snapshot date: 2026-08-31
Product version: `v0.1.0-research-preview`
Source commit: `2bd1bbf7f280bd65edcad427fd0840e09c39f6dc`

## Current backend scope

```text
prepared SGF/InSeg graph + local point cloud
  -> multimodal graph embeddings and node matching
  -> GeoTransformer / RANSAC / ICP registration
  -> fail-closed RegistrationDecision
  -> fused PLY candidate
```

Pose estimation, SLAM and raw RGB-D replay are deliberately deferred. The
backend accepts prepared graph/cloud inputs; it is not yet a complete raw-RGB-D
application.

## Verified in the source environment

- The graph-alignment adapter and registration worker execute successfully for
  the current authorized single-node pilot.
- `production_attempt.json` reports `succeeded`.
- `adapter_validation.json` reports `PASS`.
- The full V16 fixed4 test suite passed before this snapshot (`156 passed`).
- No result is released when the parent runtime-input contract fails.

## Open blocker

The outer execution audit currently fails closed on exact lazy runtime reads.
The latest pilot recorded 930 violation events across 778 unique paths, all
classified as `undeclared read`. Most are Matplotlib/fontconfig font discovery
caused by a fresh production `MPLCONFIGDIR`; smaller groups include lazy Python
codec/zip imports, metadata reads and empty ColorPCR output directories.

This blocker does not change the matching or registration mathematics, but it
does prevent the candidate transform and PLY from being released as a sealed
production result. The next closure step is to make the preregistration probe
mirror the production scratch environment and to remove unnecessary plotting
imports from the headless inference path.

## Repository contents

Included:

- SGF-to-official-SGAligner adapter and canonical input builders
- matching and registration code
- fail-closed safety contracts and sealed-execution scripts
- unit/integration tests
- preregistration protocols and manifests
- checkpoint hashes and upstream download instructions

Excluded:

- ScanNet/3RScan/3DSSG datasets
- generated outputs and evidence bundles
- trained checkpoint binaries
- local caches and logs
- authorization private keys or credentials

## Portability note

Some research and audit scripts retain source-machine default paths for exact
historical reproduction. Prefer CLI/config overrides and inspect defaults before
running on another machine. Those paths are not credentials, but they are not a
portable installation contract.
