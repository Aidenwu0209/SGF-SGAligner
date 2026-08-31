# V13 frozen-correspondence dual-solver runtime

This branch adds only a GT-free **solver shadow** for a frozen ColorPCR-style
correspondence cache.  It does not run ColorPCR, fixed4, selection89,
calibration90, reconstruction, or official92, and it cannot authorize a
`RegistrationDecision`.

## Frozen input contract

The NPZ contains exactly `src_corr`, `ref_corr`, and `scores`. Coordinates are
metres and rows describe source-to-reference correspondences. The cache is
hashed before and after reading. Scores are sorted descending with original row
index as the tie break and capped at 1,000 rows. At least 40 rows are required
by the sealed PointDSC 3DMatch configuration. Any extra GT/label field is a
fail-closed schema error.

## Solvers and direction

- PointDSC source is pinned to commit
  `b009d536ac10b570853833f2178397c154745da9`; the official 3DMatch checkpoint
  must hash to `20662778fca1a7d2c4e2f79f381d4be6cb891834d7bb4bd91ade9d89b0d13bd4`.
- pyGCRANSAC is pinned to version 0.1.1. Its raw row-vector rigid transform is
  transposed into the public column-vector convention. A nonzero synthetic
  rotation/translation regression proves this conversion.
- Public direction is `reference = R @ source + t`. Reverse workers swap the
  frozen rows; their medoid is inverted before forward/reverse comparison.
- Five deterministic input permutations are derived from the frozen input
  hash. No identity or alternative solver fallback exists.

Each worker atomically writes JSON containing input/array/runtime/dependency/
checkpoint bindings, fixed permutation seed, unit and direction, transform,
inlier/residual diagnostics, an explicit empty GT-input list, and a typed
failure. Failed workers carry `transform: null`.

## Consensus and scope

Each solver/direction needs a unique complete-linkage 4-of-5 cluster within
5 degrees and 0.10 m. Forward and inverted reverse medoids must agree at the
same thresholds; PointDSC and pyGCRANSAC forward medoids must also agree.
Passing this returns `dual_solver_consensus_only`, never a production
registration decision. Rule-B, geometry vetoes, known-bad behavior, ColorPCR
provenance, and fixed4 evidence remain mandatory downstream gates.
The CLI `--known-bad` role is an unconditional veto even when both solvers
agree, and is covered by a contract test.

## Command

```bash
PYTHONPATH=src /home/aidenwu/miniconda3/envs/sgaligner/bin/python \
  scripts/v13_dual_solver_runtime.py \
  --cache /absolute/frozen_corr_cache.npz \
  --output-dir /absolute/output \
  --pointdsc-root /home/aidenwu/Documents/SceneGraphFusion_RGBDPointDSC/upstream/PointDSC \
  --pointdsc-checkpoint /home/aidenwu/Documents/SceneGraphFusion_RGBDPointDSC/upstream/PointDSC/snapshot/PointDSC_3DMatch_release/models/model_best.pkl \
  --device cpu
```

Exit 0 means only that the frozen correspondence set achieved dual-solver
consensus. Exit 2 means a gate failed. A dependency or input contract error
fails without a transform.
