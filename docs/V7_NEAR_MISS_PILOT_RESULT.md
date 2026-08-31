# V7 known-pair pilot result

Status: **PILOT_FAILED_REPEATABILITY**. The registration-consensus veto is
research-only and is not authorised for selection89, calibration90, fixed12,
official92, reconstruction, or a default-checkpoint change.

## Frozen run

- Pair: `6a36052f...63077_to_6a36052d...cc2b5`, the B/F selection near miss.
- Code: `e16123a5b5849bbf5a4eae9944f8c6ee1cb4beea`.
- Immutable B/selection cache SHA-256:
  `b7e65b74c45f42e2a5d6c45cf434dabddd35bebd189fb0979f3ea7c6cc7b7abd`.
- Checkpoint SHA-256:
  `89eddb50b19fd44a24778877a445b4ad72488936711eea317675d338bf6c4200`.
- Protocol SHA-256:
  `399ec014689f1bb5e0128b77f65c461c07e548f7ffe0cc7d0fd77f8debfaf477`.
- Two outer runs, each with five isolated forward and five isolated reverse
  worker processes. All 20 workers completed with zero exceptions and zero
  non-finite transforms.

The GT-free receipt SHA-256 is
`4b69ae4ddde5aa01916f199d9e21f89e2dfada090e3c88bf86471cb129d13bb3`.
The separately produced posthoc-label file SHA-256 is
`2b77f5d3ef4534e32cbeb15000865df2f812520fd37673b990c1a9e9c24aaa78`.

## GT-free outcome

All eight pre-registered radius/quorum policies vetoed the pair in both outer
runs. The first run had zero Rule-B passes in either direction. The second had
one of five forward and one of five reverse Rule-B passes, still far below
quorum. No policy produced a usable reconstruction transform.

This satisfies the narrow safety observation, but the protocol also requires
repeatability. For every direction/replicate, the cache and row-permutation
hashes were identical between outer runs; nevertheless all ten raw and all ten
final transform hashes changed. The direct cause is the upstream
`pygcransac.findRigidTransform` interface, which exposes no seed and is still
nondeterministic in a fresh isolated process. Therefore the pilot fails and
must stop before expanding to the 12-pair pilot.

The medoid ICP traces in outer run 1 also failed the pre-registered monotonic
RMSE condition, although their last updates were within the stability bound.

## Independent posthoc labels

Posthoc labels were generated only after both GT-free aggregate files and the
receipt were atomically frozen. Outer run 0 had no Rule-B medoid. Outer run 1's
observed forward medoid had:

- official raw transform: 7.8352 degrees / 0.1563 m, relaxed but not strict;
- final ICP transform: 4.3997 degrees / 0.1054 m, strict;
- consensus decision: vetoed, hence no accepted-strict error.

These labels did not participate in RANSAC, ICP, Rule B, consensus, medoid
selection, or veto.

## Next permitted work

Do not adjust consensus radii or quorum to force a pass. The next minimal
research task is to replace or wrap the unseeded RANSAC solver with a genuinely
deterministic, independently tested rigid-transform sampler while preserving
the same 5 cm hypothesis threshold and 10 cm reporting inlier rule. The known
pair must then be rerun from a new frozen protocol/output root; old evidence is
immutable.
