# V11 GT-free matched-region / multi-object protocol

Status: pre-registered research adapter.  This document and its code/tests
must be committed before any selection89 execution.  V11 cannot promote a
checkpoint, authorize `official92`, or alter the production/default route.

## Why V11 exists

V9 found no three-member cross-direction rigid mode in the original node-pair
GeoT cache.  V10 expanded the frozen official rank list to 4,230 mutual
cross-graph top-5 candidates and still found zero modes on 89/89 pairs.  The
mechanical mutual-top1 and top2 subsets also yielded zero.  More independent
single-object GeoT calls therefore have no supported coverage mechanism.

V11 changes the unit of point registration, not the checkpoint, matcher, or
safety threshold: several mutually plausible matched objects form a local
surface region; official GeoTransformer sees the union once and can use
cross-object geometry to disambiguate partial object surfaces.

## Frozen candidate and grouping policy

1. Read only V10's SHA-verified mutual cross-graph top-5 rank records and the
   canonical full `registration_pts`.  Labels, GT transforms, posthoc,
   calibration/fixed12 labels, and `official92` are forbidden.
2. Build a local graph independently on each scan as the union of canonical
   explicit graph edges and centroid 4-nearest-neighbour edges.
3. A combination is one-to-one: no source or reference object repeats.  New
   members must be locally linked on both scans to at least one member already
   present.  All pairwise object-centroid distances must agree within
   `max(0.10 m, 20% * max(d_src,d_ref))`.
4. Enumerate deterministically from the frozen reciprocal-rank ordering with a
   beam width of 64.  Keep combinations of 3 through 6 objects.  Rank by
   larger member count, then lower summed worst reciprocal rank, lower summed
   reciprocal-rank sum, and canonical node-pair tuple.  Keep at most 12 per
   scan pair.  No output-dependent expansion or regrouping is allowed.
5. Concatenate each member's complete, deduplicated, world-frame stable surface
   in canonical node-pair order.  Run the unchanged official GeoTransformer
   once on the source union and reference union.  Its existing deterministic
   10,000-point cap remains authoritative; no point duplication is allowed.
6. Run a second, independent official GeoTransformer call with scan sides
   swapped.  Never substitute the inverse of the forward result.

## Frozen downstream route and safety gate

Each region arm separately follows the already frozen path: official GeoT
correspondences -> unchanged pyGCRANSAC -> fixed-trace ICP -> unchanged Rule-B
-> q4 stability -> 5 degree / 0.10 metre forward/reverse agreement.  Exactly
one safe hypothesis is required.  Zero modes, multiple inconsistent safe
modes, a missing reverse run, any untyped/runtime error, or the registered
known-bad pair is a rejection.  Error-accepted must remain zero.  Rule-B,
thresholds, checkpoint, official source, and default inference routing are
immutable.

The direct full-scene object-surface union is a diagnostic arm only.  It runs
once per direction under the same official GeoT resource contract, is reported
separately, and cannot be selected or used to tune grouping.

## Frozen resource bound and run order

- At most 12 selectable hypotheses x 2 directions + 1 diagnostic x 2
  directions = 26 official GeoT calls per scan pair, 2,314 over selection89.
- Cache one immutable record per `(pair, hypothesis, direction)` with input,
  candidate, surface, checkpoint, code, and output SHA-256 values.
- Before any label access: require 89/89 structural completion, zero
  unknown/untyped errors, independent repeat payload equality, known-bad veto,
  deterministic candidate/hypothesis hashes, and zero error-accepted.
- First execute a small GT-free resource pilot.  Full selection89 execution
  requires explicit review of this pre-registration commit; no result from the
  pilot may change thresholds or the frozen grouping policy.

## Pilot addendum (frozen before execution)

The first pilot is the de-duplicated ordered set consisting of canonical V10
pair-list positions `0`, `44`, and `88`, followed by the registered known-bad
pair.  Pair choice cannot depend on a structural, GeoT, registration, or gate
result.  Before this pilot, the structural generator must run on all 89 pairs
and report the zero-hypothesis count without changing the grouping policy.

Every selectable hypothesis cache records independent forward and reverse
GeoT outputs.  Five deterministic row-order pyGCRANSAC workers per direction
each record raw transform, fixed-correspondence ICP trace, unchanged Rule-B
features/verdict, and artifact SHA.  The existing V8 q4 gate records
directional and cross-direction complete-linkage status.  The pair selector
receives only hypothesis gates; its API has no whole-scene diagnostic input.

Pilot continuation requires zero unknown/untyped errors and a known-bad veto.
Only engineering faults (schema, path, serialization, typed exception
handling) may be repaired.  Thresholds, grouping, pair list, checkpoint,
official source, Rule-B, q4, and candidate ranks cannot change after this
addendum.  A scientific failure is reported and stops the full run.
