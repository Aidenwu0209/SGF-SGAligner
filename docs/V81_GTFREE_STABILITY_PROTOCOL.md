# V8.1 GT-free stability protocol

Status: implementation candidate.  This document fixes the policy before any
selection89 label/posthoc process is allowed to run.  It changes no official
SGAligner source, checkpoint, Rule-B threshold, RANSAC radius, or ICP setting.

## Inputs

For every pair, consume exactly two independently executed outer runs.  Each
outer contains five forward and five true-reverse workers, for 20 immutable
worker receipts total.  Every receipt, transform, cache, checkpoint, protocol,
source snapshot, and manifest SHA must be revalidated by the caller.

This layer imports no labels or GT loader.  It may consume only worker status,
final/raw transforms, Rule-B features and recorded decision, corrected
fixed-correspondence ICP trace, replicate identity, and evidence hashes.

## Frozen candidate

1. Pool both outers into ten forward and ten inverse-normalized reverse final
   transforms.
2. Use the existing 5 degree / 0.10 m complete-linkage relation.  Require one
   unique component with at least 9/10 members in each direction.
3. Require at least nine one-to-one forward/reverse final-transform matches.
4. Select the observed component medoid with evidence SHA as the deterministic
   tie breaker.  Both directional medoids must pass unchanged Rule-B and the
   corrected fixed-correspondence trace gate.
5. At least five members of each winning component must independently pass the
   same Rule-B plus fixed-trace safety gate.  This prevents one lucky medoid
   from representing an otherwise unsafe component.
6. Run all twenty single-worker deletions.  Every deletion must retain unique
   q=8 directional components, q=8 cross-final agreement, and safe medoids.
   The jackknife medoids may move by at most 1 degree and 0.02 m from the full
   pooled medoids.
7. Accept only when every condition above passes.  Return an actually observed
   forward medoid; never average SE(3) matrices and never emit a transform from
   a rejected result.

The constants are fixed as one policy.  q=8, q=9, dual-outer unanimity and
larger member-vote variants are audit comparisons only and cannot be selected
after posthoc results are known.

## Pre-label whole-run gate

Before loading selection89 labels:

- the immutable batch must contain 89 pairs, 178 outers and 1,780 valid workers;
- exception, nonfinite, cache mismatch, source mismatch and duplicate evidence
  binding counts must all be zero; identical worker content hashes are allowed
  when independent outer/path bindings prove genuinely repeated executions;
- two independent offline executions of this policy must be byte-identical;
- all 89 pair decisions must be present and deterministic;
- the predeclared known-bad pair
  `6a36052f-fa53-2915-9400-831b60c63077_to_6a36052d-fa53-2915-9764-30d81b2cc2b5`
  must remain vetoed;
- the aggregate receipt and source/config bindings must be atomically frozen.

If any item fails, freeze `GTFREE_PRELABEL_GATE_FAIL` and stop.  Do not load
labels, calibration90, fixed12, or official92.

## Posthoc boundary

Passing this protocol establishes deterministic, label-free registration
stability only.  It does not establish correct pose, selection coverage, or an
official benchmark result.  The already pre-registered selection posthoc gates
remain unchanged: zero accepted errors, accepted-correct and raw-strict floors,
and frozen V6 B/F comparison bindings.  Failure cannot be repaired by changing
this policy after labels are revealed.

Calibration90 remains a separate, once-only blind validation using a freshly
generated cache bound to the selected checkpoint.  fixed12 and official92 stay
locked until every earlier gate passes.
