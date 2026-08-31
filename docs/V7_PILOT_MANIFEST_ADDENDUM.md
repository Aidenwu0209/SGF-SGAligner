# V7 12-pair pilot manifest addendum

Status: **pre-registered and frozen before the 12-pair V7 pilot**.

This addendum seals the pair population and clarifies one audit term in
`V7_REGISTRATION_CONSENSUS_PROTOCOL.md`. It does not change registration,
Rule-B, consensus, veto, gate or evaluation thresholds.

## Frozen batch contract

The batch runner consumes
`outputs/v7_pilot_manifest_seal_20260830/v7_pilot_manifest.json` with:

- schema `v7-registration-veto-batch-manifest-v1`;
- status `FROZEN`;
- exactly 12 unique opaque pair identifiers in listed order;
- checkpoint B, SHA-256
  `89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200`;
- original V7 protocol SHA-256
  `399ec014689f1bb5e0128b77f65c461c07e548f7ffe0cc7d0fd77f8debfaf477`;
- a raw-file SHA-256 for every immutable B/selection cache.

The runner must verify the schema, status, count, uniqueness,
`pair_ids_sha256`, checkpoint/protocol hashes and every cache hash before it
starts a worker. A mismatch is a fail-closed stop.

Formal mode is additionally bound to the committed default manifest path and
the exact manifest file SHA above. A copied, regenerated or alternative
manifest can run only with the explicit `NON_PREREGISTERED_RESEARCH`
classification. Such evidence is useful diagnostically but the formal pilot
gate must return `INDETERMINATE`; it can never authorise selection89.

## GT-free mechanical selection

Pair identifiers are opaque lookup keys. They are not split, grouped or
decoded into a scene, scan or reference identity. The sole label exception is
the previously and publicly declared near-miss pair. Only its opaque pair ID is
inserted; no RRE, RTE, strict/relaxed status or correctness value is consumed
by the selector or used to set a threshold.

The remaining strata are derived from three sealed V6 B/F selection repeats:

1. insert the predeclared opaque near-miss ID;
2. include every other pair whose final Rule-B usable decision changes across
   the three repeats, ordered by transform dispersion descending;
3. include the three stable passes nearest an unchanged Rule-B boundary;
4. from the remaining stable passes, include the three largest worst-case
   Rule-B-margin controls;
5. include the stable reject with the largest transform dispersion.

Transform dispersion is
`max(max_pairwise_SO3_angle / 5 degrees, max_pairwise_translation / 0.20 m)`.
Lower-bound Rule-B margins use `(value-threshold)/threshold`; upper-bound
margins use `(threshold-value)/threshold`. Stable pair IDs break exact ties.
No post-hoc label breaks a tie.

## Whitelist and leakage boundary

The selector accepts only the frozen materialized projection:

- opaque pair ID;
- immutable source, checkpoint, protocol, selector and cache hashes;
- repeat index and provenance signatures;
- flat hypothesis node-correspondence count;
- final RANSAC/ICP transform;
- unchanged Rule-B decision and GT-free Rule-B input features;
- three-repeat decision and transform stability.

Unknown fields fail closed at every projection level. Runtime path guards
reject label-sidecar and downstream evaluation trees. The selector AST may
not import or call a GT transform or anchor loader. The cache is not
deserialized by this selector: only raw bytes, byte count and SHA-256 are
checked.

Inference and selection forbid anchors, GT transforms, RRE, RTE,
strict/relaxed labels, accepted-correct/error labels, scene IDs and any scene
identity inferred from the pair string. They also forbid post-selection
calibration, fixed-set and official benchmark result trees. The frozen
projection contains none of those keys.

## Outer-repeatability clarification

For this V7 protocol, outer repeatability means repeatability of the final
per-policy `usable_for_reconstruction`/veto decisions and of the input and
evidence structure: cache identity, worker count and direction, permutation
provenance, schema/status, policy grid, exception/non-finite state, and the
presence of all required evidence.

In short, the invariant is the final per-policy `usable_for_reconstruction`/veto
decision plus the complete input/evidence structure.

The evidence structure includes a direction/replicate-keyed binding for all
240 workers. Both `permutation_provenance_sha256` and
`permutation_sha256` are recomputed from pair, direction, replicate,
correspondence count and the frozen protocol SHA, then compared across the two
outer repeats. Merely presenting a hash-shaped field is insufficient.

It does not require byte-identical raw/final transform hashes from the
unseeded `pygcransac` backend. Floating transforms may differ while the final
policy decision and sealed evidence structure repeat. Such variation remains
visible and is still constrained by the pre-registered consensus and safety
gates.

Therefore the historical `PILOT_FAILED_REPEATABILITY` label in
`V7_NEAR_MISS_PILOT_RESULT.md` is an **audit-semantic misclassification** of
the frozen protocol term. The historical evidence and file are immutable and
are not rewritten; this addendum records the correction prospectively.

No radius, quorum, Rule-B threshold, ICP trace threshold, policy ordering,
selection gate, calibration gate or fixed-set gate changes. In particular,
the policy grid remains 2.5/5.0 degrees, 0.05/0.10 m and quorum 4/5 or 5/5.

## Formal pilot gate baseline semantics

The formal evaluator pre-binds the raw SHA-256 of all three V6 B/F selection
control records. For the ambiguous phrase "existing strict and accepted pilot
pairs", the conservative frozen interpretation is:

- a pair is `majority-existing` only when it is both raw-strict in at least
  two of three controls and accepted-correct in at least two of three controls;
- exactly seven of the frozen twelve pairs meet that intersection;
- counts are compared against the median of the three control repeats, not a
  union, best repeat or post-hoc-selected repeat;
- every outer repeat may lose at most one majority-existing raw-strict pair
  and at most one majority-existing accepted-correct pair;
- every policy must also lose at most one from the control median count, veto
  the known pair twice, repeat every pair decision, and have zero accepted
  error.

All eight policies must pass independently and all 240 worker records must be
complete before selection89 is authorised. If the pre-registration identity,
control hashes or evidence bindings cannot be established, the result is
`INDETERMINATE` or `FAIL`, never `PASS`.

## Stops

- Pilot or selection failure stops before calibration.
- Calibration failure stops before the fixed 12-pair safety gate.
- Fixed12 still requires at least 4 distinct strict pairs, at least 4 distinct
  correct accepted pairs, and zero error/failure/unknown over all 36 runs.
- This addendum does not authorize official92, a default-checkpoint change,
  reconstruction deployment, push or merge.
