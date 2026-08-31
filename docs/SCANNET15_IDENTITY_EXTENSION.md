# ScanNet15 exact15 identity extension

This candidate adds an unsigned, hash-bound identity path for the 15 prepared
ScanNet pairs without changing the reviewed fixed4 converter, V13 gate, V14
builder, V14 strict runner, identities, thresholds, or algorithms.

The extension is deliberately implemented beside the frozen path:

- `v16_b716_scannet15_corr_cache_converter.py` validates exact36 prepared NPZs
  and the exact15 preregistration before converting a ColorPCR worker result.
- `v16_b716_scannet15_v14_identity.py` validates builder identity or strict
  preflight closure without importing or executing a model, solver, or ICP.
- `v16_b716_scannet15_v13_gate_bridge.py` validates and exposes only the raw
  current-coordinate surfaces. It contains no decision or registration logic.
- `v16_b716_scannet15_identity_preflight.py` snapshots and hashes the extension,
  the unchanged fixed4 sources, raw/prepared evidence, official source/checkpoint,
  ColorPCR, PointDSC, and interpreters.

All generated preregistrations set every execution, GPU, GT, threshold change,
selection, reconstruction, refusion, and official92 policy field to `false`.
No result from this layer is a registration success or a final PLY.

Remaining production blockers are intentional: the active production adapter
does not yet route color tasks through the sibling converter plus preregister;
there is no reviewed execution authorization; V14 real pilot remains disabled;
and no ColorPCR parent-result SHA exists.
