# V8 stage-order consensus research protocol

## Status and scientific boundary

V8 is a research protocol for the SGF-predicted adapter connected to the
official SGAligner source/checkpoint path. It changes registration aggregation
order; it does not change the official SGAligner checkpoint, GeoTransformer,
Rule-B thresholds, default checkpoint, or any historical V6/V7 evidence.

The frozen 12-pair V7 development set has already been inspected. A posthoc
ablation observed 7 correct and 0 erroneous accepted outcomes in both outer
runs under this ordering. That number is **development evidence only**. It is
not a blind, confirmatory, calibration, fixed12, or official92 result.

## Frozen candidate

There is exactly one V8 candidate:

- five independent forward and five true reverse workers;
- complete-linkage radius: rotation `5 degrees`, translation `0.10 m`;
- directional quorum: `4/5`;
- cross-direction final-transform quorum: `4/5`;
- observed medoids only; transforms are never averaged;
- the official frozen Rule-B evaluator is applied to the forward and reverse
  final-transform medoids only after geometric clustering;
- the fixed-correspondence ICP trace of each medoid must be non-increasing and
  its final update must remain within the frozen stability limits;
- all raw-transform consensus results are diagnostics and cannot veto or make
  an otherwise rejected candidate usable.

No one-run Rule-B result may remove a finite final transform before clustering.
The winning directional cluster must be the unique largest complete-linkage
cluster and contain at least four members. Forward/reverse agreement is maximum
bipartite matching between the two winning clusters using the same fixed radius.
Both observed medoids must independently pass unchanged Rule-B.

## Legacy V7 replay

The offline replay command reads an already frozen V7 batch. It revalidates:

1. the manifest file SHA and ordered pair set;
2. the batch receipt and its source-snapshot hash;
3. every pair receipt and aggregate file/evidence SHA;
4. all worker file/evidence/transform/permutation hashes;
5. cache and checkpoint bindings carried by the V7 validator.

The replay has no GT/label imports and runs no registration. Historical V7
workers produced before the trace correction lack the fixed-correspondence
fields; their stage-order result can be reported only as development evidence
and `fresh_v8_qualified=false`. Newly generated workers may set
`fresh_v8_qualified=true` only when both selected medoids carry and pass the
corrected trace. The old dynamic-NN trace is never treated as a V8 gate. The
receipt itself is always marked `development_split_exposed=true` and
`qualifies_as_blind_gate=false` because this runner targets the exposed V7 set.

Labels may be loaded only by `scripts/v8_stage_order_posthoc.py` in a separate
process after the GT-free receipt has been atomically frozen. The posthoc result
also remains development evidence.

## Split order and stopping rules

1. `selection89` is development. It may compare the single V8 candidate with a
   preregistered incumbent/control using identical frozen worker caches, but it
   may not be described as confirmatory.
2. The complete code/config/checkpoint/manifest hashes and a unique candidate
   must be frozen before any calibration labels are opened.
3. `calibration90` is the first V8 blind gate. It is run exactly once with the
   unique frozen candidate. A failure cannot be followed by changing the radius,
   quorum, threshold, arm, or checkpoint and rerunning the same gate.
4. Only after calibration passes may `fixed12` run as a safety regression gate.
5. `official92` remains forbidden until all prior gates pass. Nothing in this
   protocol authorizes changing the default checkpoint or production path.

Minimum safety requirements at every gated split are zero accepted-strict-error,
zero exception/nonfinite/cache/hash mismatch, complete coverage accounting, and
deterministic pair-level outcomes under the preregistered repeat count. Split-
specific coverage/strict/accepted-correct thresholds must be frozen separately
before their labels are loaded.

## Required reporting

Report completed/exception/nonfinite/hash-mismatch counts; usable coverage;
strict, relaxed, accepted-correct and accepted-error; pair-level repeatability;
directional clique sizes and medoid replicate; cross-direction delta rotation
and translation; unchanged Rule-B reasons; corrected fixed-correspondence trace;
raw-transform diagnostics; and every source/cache/checkpoint/receipt SHA.
