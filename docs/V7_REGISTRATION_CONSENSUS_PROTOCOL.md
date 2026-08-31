# V7 Registration Consensus Protocol

Status: research-only protocol, frozen before the first V7 pilot.

## Scope

- Treat the SGF to official SGAligner adapter, official checkpoint,
  four-modal contract and GeoTransformer cache as immutable inputs.
- Production candidate is checkpoint B plus exact-flat F plus consensus veto.
  C0/C1 are excluded and D is diagnostic only.
- Existing RANSAC, ICP and Rule-B thresholds remain unchanged.
- Ground truth is forbidden from inference, consensus, veto and transform
  selection. It is loaded only by the post-hoc evaluator.

## Replicates

- Exactly K=5 forward and K=5 independent reverse replicates per pair.
- Every replicate consumes one immutable node and GeoTransformer cache.
- Forward runs source-to-reference RANSAC and ICP. Reverse swaps the cached
  correspondence sides, independently runs RANSAC and ICP, then is inverted
  into the source-to-reference frame.
- A SHA-256-derived row permutation for pair, hypothesis, direction,
  replicate and protocol hash is recorded.
- Exceptions, missing or non-finite transforms, unknown states and cache
  mismatches are fail-closed rejects.

## Fixed policy grid

- rotation radius in 2.5 or 5.0 degrees;
- translation radius in 0.05 or 0.10 metres;
- quorum in 4 of 5 or 5 of 5.

These values are fixed fractions of the strict target boundary and are not
fitted to the known error pair. Forward raw, forward ICP, inverted reverse raw
and inverted reverse ICP all use complete linkage. Every two transforms in a
clique must satisfy both radii. The largest clique must be unique and any rival
clique of size two or more is an ambiguity reject.

The selected output is an observed forward medoid, never an average. It
minimises the sum of rotation-distance divided by 5 degrees and
translation-distance divided by 0.20 metres. Stable hash breaks a tie.

## Fail-closed veto

For a pair to be usable:

1. forward and reverse each have at least quorum original Rule-B passes;
2. every Rule-B pass belongs to the unique winning clique;
3. at least quorum forward and inverted-reverse solves have a one-to-one
   cross-direction match within the same radii;
4. raw-RANSAC and final-ICP transforms both satisfy consensus;
5. winning ICP traces have non-increasing fixed-surface RMSE and last updates
   no greater than 0.25 degrees and 0.005 metres;
6. the observed forward medoid independently passes unchanged Rule-B.

No singleton success can be accepted.

## Pilot and gates

The pilot uses exactly 12 selection pairs: the known near-miss pair, four
smallest positive Rule-B-margin pairs, four largest transform-dispersion
pairs and three stable high-margin accepted pairs. It runs twice and does not
choose a policy. It passes only if the known pair is vetoed twice, inputs and
decisions repeat, existing post-hoc strict and accepted pilot pairs lose at
most one, and there are zero exceptions, NaNs, GT leaks or cache mismatches.

Only after pilot passes are all eight policies replayed on selection89. The
winner order is: zero accepted error and zero failed or unknown in every
repeat and hash fold; maximise the minimum accepted-correct; maximise median
accepted-correct; maximise median raw strict; minimise vetoes; prefer larger
quorum and smaller radii for exact ties.

Selection requires three complete repeats, median raw strict at least 11,
median accepted-correct at least 9, every repeat accepted-correct at least 8,
zero candidate not worse than the B/F control, and byte-identical B node
evidence. Failure stops before calibration.

After an atomic config and SHA freeze, calibration90 runs once with a
same-cache B/F control. Veto strict and accepted-correct may each lose at most
one, while error, failure and unknown remain zero. Failure stops before
fixed12.

Only after calibration passes, fixed12 runs 12 distinct pairs by three outer
repeats. Required: distinct strict at least 4 of 12, distinct
accepted-correct at least 4 of 12, all 36 runs error/failure/unknown zero, and
each successful pair direction-consistent in at least two repeats.

No official92 run is authorised by this protocol.

## Limits

The veto targets random multi-modal convergence and a single lucky pass. If a
wrong symmetric pose is consistently reproduced in at least four of five
solves and also passes bidirectional and trajectory checks, GT-free consensus
can still miss it. Therefore the zero-error calibration and fixed12 gates may
not be removed.
