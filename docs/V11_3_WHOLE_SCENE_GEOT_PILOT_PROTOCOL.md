# V11.3 Whole-Scene Official GeoTransformer Pilot Protocol

Status: pre-registered before any transform, RANSAC, ICP, Rule-B, or GT
accuracy result from the whole-scene diagnostic caches is read.

## Question

Can a graph-independent, balanced whole-scene surface arm recover a safe,
repeatable scan transform when the V11.2 local multi-object proper-SE(3)
hypothesis arm is structurally unreachable?

This is a diagnostic pilot. It does not replace the official SGAligner matcher,
does not change the default checkpoint, and is not eligible for reconstruction
or official92 until all frozen gates below pass.

## Frozen code and model inputs

- Code base: integration commit `33450bd`.
- Official SGAligner checkpoint: unchanged release checkpoint; its SHA-256 is
  validated from the current checkout. The offline whole-scene arm does not
  consume its matcher embeddings. The `checkpoint_sha256` stored by historical
  V11 caches is the frozen historical matcher-selection checkpoint provenance,
  not the release-checkpoint identity; it must agree across all four caches and
  is recorded under that truthful name.
- Official GeoTransformer checkpoint: unchanged release checkpoint.
- Canonical inputs: `scripts/canonical_inputs.py`,
  `sampling_mode=official_mt19937`, scan seed 0.
- Historical V11 whole-scene caches are reused only when their file SHA-256,
  payload hash, protocol SHA, checkpoint SHA, pair id, surface hashes, and
  canonical input surface hashes all validate. They contain only SGF-predicted
  inputs and official GeoTransformer correspondences. No transform or decision
  result from them has been inspected for this protocol.
- Forbidden inputs: GT transforms, selection/calibration labels, post-hoc
  outcomes, official92, and any threshold selected after seeing results.

## Frozen pilot set

Use the exact V11 fixed positions 0, 44, 88 from the immutable selection89
order plus the immutable known-bad pair. Deduplicate only if identical.

## Frozen surface arm

For each direction, concatenate every canonical `registration_pts` object on
that graph in canonical object order. The rebuilt union's float32 SHA-256 and
point count must exactly equal the historical cache. No object correspondence
or selection label is consumed. This is the `whole_scene_balanced_union` arm.

## Frozen estimator and repeats

For each pair and direction:

1. Validate the cached official GeoTransformer correspondence arrays and SHA.
2. Stable-sort by descending score and cap at 1000 correspondences.
3. Run five independent deterministic row permutations. The permutation seed
   is derived only from protocol SHA, pair id, direction, and repeat index.
4. Use the unchanged official pyGCRANSAC composition and constants.
5. Run the unchanged fixed-trace segment ICP implementation.
6. Compute unchanged Rule-B features on the complete directional surface
   unions. Spatial support uses all canonical source-object barycentres;
   `successful_node_pairs=1`, `failed_node_pairs=0` denotes one successful
   whole-scene registration arm, not fabricated object-pair support.
7. Evaluate the unchanged V8 q4 stage-order consensus: at least 4/5 repeats,
   maximum within-direction dispersion 5 degrees / 0.10 m, both medoid Rule-B
   decisions safe, and final forward/reverse transforms consistent under the
   existing cross-direction gate.

The known-bad pair is always vetoed regardless of numeric quality.

## Pilot success gate

The arm is `PILOT_PASS` only if all are true:

- all fixed non-known-bad pilot pairs have a unique q4-stable, Rule-B-safe,
  forward/reverse-consistent transform;
- known-bad is rejected by the explicit veto;
- zero unknown errors, zero malformed/tampered/resume mismatches;
- immutable caches, canonical surfaces, row permutations, and input hashes are
  bitwise reproducible; because the official `pygcransac` binding exposes no
  seed API, an independent offline replay must reproduce the same accept/reject
  verdicts and keep corresponding transforms within the already-frozen
  5 degree / 0.10 m stability bound. Exact RANSAC transform bytes are recorded
  but are not falsely claimed deterministic.

Any failure yields `PILOT_FAILED`. In that case no selection89 run, calibration,
fixed12, reconstruction, checkpoint promotion, official92, push, or merge is
authorized.

## Evidence

The run must atomically write per-repeat workers, per-pair gates, a pilot
summary, input and artifact manifests, exact source/model/cache hashes, commands,
environment, resource usage, and an independent SHA-256 verification receipt.
All outcomes, including failures, are retained.

## Pre-run evidence-closure addendum

This addendum was committed before reading any cached transform/decision result
or running pyGCRANSAC. It changes no arm, estimator, threshold, or success gate.

- Each cache's historical V11 protocol SHA must equal the on-disk V11 protocol.
- The release SGAligner checkpoint, release GeoTransformer checkpoint, canonical
  builder, RANSAC composition, ICP/Rule-B, and q4 source hashes are recorded.
- Every run writes an immutable NPZ shadow-solver interface containing the exact
  balanced-union XYZ surfaces and both directional cached GeoTransformer
  correspondence arrays. Canonical SGF `registration_pts` has no RGB contract;
  `color_available=false` is recorded and RGB is never fabricated. This keeps a
  future GT-free ColorPCR (after separately sealing RGB) or PointDSC/pyGCRANSAC
  shadow comparison pluggable without making it eligible for this pilot.
- A separate verifier rehashes every declared artifact before primary/replay
  comparison. Comparison rejects non-passing verification receipts.
