# b716 fixed4 execution contract fix2

## Status

This commit closes the severe metadata-integrity gaps found in the independent
review of `63f6152ae6d5309006eb0d926399f094bedd830e`. It remains execution-disabled:
the hard-bound stage registry contains fail-closed adapters and does not launch
ColorPCR, PointDSC, pyGCRANSAC, ICP, reconstruction, or refusion.

## Closed boundaries

- The public execution API no longer accepts a caller-provided runner. The four
  stage callables are selected only from
  `src/safety/v16_b716_fixed4_stage_runners.py`; its file SHA and registry
  descriptor are bound by preregistration, preflight, authorization, and every
  attempt receipt.
- Every one of the 6,091 evidence DAG nodes has exactly one operational owner:
  four nodes per directional task, 171 per pilot, and one per pair/aggregate
  task. Construction fails on a missing, duplicate, or reordered node.
- Operational results must consumer-rehash one immutable expanded evidence
  receipt per owned node. A 107-task success therefore requires the complete
  6,091-receipt closure.
- Each pilot requires exactly 20 file-backed attempts covering
  `2 solvers x 2 directions x 5 seeds`. Empty, duplicate, missing, writable, or
  SHA-drifted attempts fail closed. The remaining preregistered candidate-slot
  solver nodes stay visible as expanded evidence and may only be explicit
  `typed_not_generated` receipts.
- Authorization binds repository root, Git HEAD/tree, output root, preflight,
  full task manifest, all 107 task IDs, runner/source closures, evidence
  mapping, exact191/prepared34 identities, and an independent guard-audit
  receipt. TTL is positive and at most one hour.
- Results, attempts, evidence, and artifacts are exact-schema and file-backed.
  Wrapper receipts are `O_EXCL`; runner artifacts must be immutable, live under
  the canonical task root, and match the complete task inventory. Any partial
  state makes retry fail rather than overwrite.
- While reconstruction remains unauthorized, every `.ply` or path suggesting
  reconstruction/refusion is rejected, regardless of self-reported booleans.
- Typed failures require `transform=null`; pair results recompute all upstream
  receipts; the known-bad pair must replay all 12 hypotheses and return the
  permanent veto; aggregate acceptance recomputes three normal accepts plus the
  known-bad veto.
- GT, official92, threshold changes, checkpoint replacement, and result-based
  selection are exact false fields on every result, expanded evidence receipt,
  solver/sentinel attempt, replay receipt, and operational attempt.

## Re-review stop checks

Do not issue an execution authorization until a separate commit replaces the
disabled registry adapters with reviewed stage implementations, updates the
runner source SHA, and obtains a new independent audit receipt. A successful
metadata test run is not authorization and must not be presented as solver or
registration evidence.
