# V16 b716 safe matched-region execution protocol (closed)

This stage is disabled. It defines and unit-tests the decision mathematics
that may be used after the separately reviewed CPU-only b716 builder has frozen
all region inputs. It does not authorize ColorPCR, registration solvers,
fixed4 claims, reconstruction, selection89, calibration90, or official92.

## Checkpoint provenance

The frozen ranks and structural hypotheses are produced by the official
release checkpoint, SHA-256
`b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e`.
They are bound to the `v16-b716-candidate-plan-manifest-v1` and the sealed
`v16-b716-exact191-merged-manifest-v1`. Legacy B_ep20/89ed paths are rejected.

## Fixed experiment shape

The primary arm is the matched-region surface arm encoded under the inherited
name `sgf_selected_union`. The four frozen pairs contain exactly 12, 8, 2, and
12 structural hypotheses, in that order. Every expected hypothesis must have a
unique absolute prepared-input path and a unique deterministic output path.
Missing, duplicate, colliding, or malformed evidence aborts the row.

The `fullscan` arm remains a frozen V15 diagnostic control. It is not expanded
per region, cannot rescue a failed primary row, and cannot select thresholds or
hypotheses.

## Two complete-linkage levels

Each region hypothesis first uses the unchanged V15 rule over V13-strict
candidates. A safe candidate contributes four canonical final poses:

1. PointDSC forward;
2. inverse PointDSC reverse;
3. pyGCRANSAC forward;
4. inverse pyGCRANSAC reverse.

Two candidates are equivalent only if all sixteen cross-realization distances
are within both inclusive thresholds: 5 degrees and 0.10 metres. The V15 rule
must produce one maximal clique covering every safe candidate.

At the hypothesis level, only V15-accepted hypotheses are eligible. Two such
hypotheses are equivalent only if every realization in one accepted cluster is
compatible with every realization in the other. V16 enumerates all maximal
cliques in SHA order over at most twelve eligible hypotheses. A row is accepted
only when the eligible set is nonempty and one maximal clique covers the whole
eligible set.

Unsafe hypotheses remain in the evidence but do not vote. Consequently, one
safe and eleven unsafe hypotheses may be accepted, while eleven mutually
compatible safe hypotheses plus one incompatible safe hypothesis must be
rejected. There is no majority vote, largest-cluster choice, best score,
fitness selection, or first-result fallback.

The final pose is an observed canonical final transform. For every observation
T, define its distance to U as the maximum of rotation(T,U)/5 degrees and
translation(T,U)/0.10 metres. Choose lexicographically by minimum maximum
distance, minimum summed distance, hypothesis SHA, candidate SHA, and
realization ordinal. Averaging and synthesized transforms are forbidden.

## Known-bad and fixed4 semantics

When execution is later authorized, all twelve known-bad hypotheses must run
so pre-veto evidence remains auditable. The row is then rejected
unconditionally. Reports distinguish candidate-level
`strict_geometry_safe_candidate_count_before_veto` from
`hypotheses_with_pre_veto_geometry_safe_evidence`; neither field may be reset
to zero merely because the final veto fired.

The fixed4 aggregate passes only when all three normal primary rows are safe
and the known-bad primary is vetoed. Control results are reported separately
and never affect this decision.

## Future execution boundary

A later authorization must name one exact reviewed b716 builder manifest and SHA,
then freeze formal source hashes. Each region NPZ requires independent forward
and reverse ColorPCR direction jobs, and each direction retains the two V13
sentinel processes. With 34 hypotheses this means 136 official ColorPCR worker
processes, 68 sentinel-invariant direction caches, and 68 exact-three caches.
Every V14 candidate then runs the unchanged V13 2x2x5 solver matrix and strict
gate. No current file or test launches those jobs.
