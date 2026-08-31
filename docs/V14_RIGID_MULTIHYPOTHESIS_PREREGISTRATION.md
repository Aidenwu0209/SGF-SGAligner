# V14 rigid-compatibility multi-hypothesis shadow protocol

Status: **PRE-REGISTERED / CODE-AND-TESTS ONLY / REAL PILOT NOT AUTHORIZED**

The machine gate is `allow_real_pilot=false` and `allow_gpu_pilot=false`.
Although the downstream PointDSC and pyGCRANSAC matrix is CPU-only, no real
fixed4 cache may be consumed until a separate reviewed authorization commit
changes only the former flag and explicitly records that approval.  The
current code commit is therefore mechanically unable to start a real pilot.

Base: V13 sealed source commit
`c71f86ac860752cc36d446c86f6950e74982ab2e`.  V13's formal fixed4
result is retained as immutable evidence: the SGF-selected primary arm passed
one of three normal pairs, rejected the frozen known-bad pair in both arms,
and failed the other two normal pairs.  V14 is a research shadow.  It does not
change SGAligner's official source, official ColorPCR or PointDSC source or
weights, the default checkpoint, Rule-B, q4, ICP, or any registration
threshold.

## Question fixed before execution

Does deterministic separation of one ColorPCR correspondence cache into
several mutually incompatible rigid hypotheses remove the mixed-rigid-mode
failure observed in V13 while retaining the frozen known-bad veto?

No label, GT transform, identity fallback, post-hoc metric, official92 result,
or RegistrationDecision output may enter hypothesis construction or ranking.
The diagnostic Kabsch transforms produced by this stage are not passed as an
initial transform to either downstream solver.

## Frozen input and identity

The fixed four pair identities and order are inherited verbatim from
`manifests/v13_colorpcr_pointdsc_fixed4_preregister.json` (SHA-256
`16a8a4165e68c3fceddb9bf0a07e6e2571a7ac587c4941221900071ee4452538`).
The prepared input closure is inherited from the V13 preflight manifest
(SHA-256
`d2ac0e5fccf98b905110b4f98de68f4fcfd7918b6998e42a50c84d6c7ba197d4`).
V14 consumes only the direction-specific, sentinel-invariant ColorPCR
`src_corr/ref_corr/scores` caches regenerated from those frozen inputs.

`sgf_selected_union` is the only primary arm.  `fullscan` remains a negative
control and cannot rescue, rank, or select the primary result.

## Frozen candidate construction

Each direction is processed independently.  The exact score-descending,
original-index-ascending V13 top-1000 contract is reused.  Fewer than 40 input
correspondences fail closed.

1. Use the top 64 frozen rows as the seed pool.
2. Build the complete rigid-compatibility graph.  An edge exists only when
   both endpoint separations are at least 2 cm and their source/reference
   distance discrepancy is at most
   `max(0.05 m, 0.05 * max(source_distance, reference_distance))`.
3. Enumerate all graph triangles in the seed pool, order them by SHA-256 of
   `(source-cache SHA, pair, arm, direction, original row indices, config
   SHA)`, and retain the first 256.
4. Fit a proper-SE(3) Kabsch pose to each non-degenerate triangle.  A
   correspondence belongs to its support only if its transform residual is
   at most 0.10 m and it is graph-compatible with at least two triangle
   members.  Refit proper-SE(3) Kabsch once on that complete support, then
   recompute every residual and the final support against the refitted pose
   with the same 0.10 m and seed-compatibility conditions.  Reject if the
   recomputed support falls below 40; no iterative threshold adjustment is
   permitted.
5. Require at least 40 support correspondences.  Rank without labels by
   support count descending, score sum descending, median residual ascending,
   then original indices.  Suppress only candidates that are both within
   5 degrees/0.10 m and have support Jaccard at least 0.80.  Retain at most
   eight per direction.
6. Pair forward and reverse candidates only when forward and inverse(reverse)
   differ by at most 5 degrees/0.10 m.  Retain at most eight bidirectional
   candidate pairs, ordered only by geometric discrepancy, support, and frozen
   indices.

Every direction must preserve its source-cache SHA, selected original row
indices and SHA, graph-edge SHA/count, configuration SHA, seed triple,
support-array SHA, exact-three-key candidate cache SHA, and candidate receipt.
Every retained hypothesis has at least 40 correspondences.  Empty candidate
sets are a typed failure, never identity.

Before a future pilot, the exact ordered 4x2 input manifest must bind the V14
preregistration, V13 preregistration, V13 preflight manifest, prepared inputs,
both direction manifests, original direction caches, support-index arrays,
candidate caches, and candidate receipts by absolute path and SHA-256.  Every
load rehashes these dependencies; a same-ID replacement or one-byte mutation
fails closed.  It also deterministically regenerates each direction manifest
from the sealed three-key cache and frozen configuration, then compares the
complete hypothesis core and every candidate array.  A coordinated rewrite of
a source-valid subset plus all downstream receipts therefore cannot mint a new
trusted candidate.  Forward and reverse source caches must have distinct paths
and SHA-256 values.  This recursive verification remains mandatory when the
candidate count is zero.

The same manifest also carries the centralized
`scripts/v14_formal_source_manifest.py` exact map: all eleven sealed V13
formal runtime sources plus the V14 source manifest, hypothesis module,
builder, fixed4 input builder, strict candidate runner, and fixed4 orchestrator.
Candidate strict
summaries and the final aggregate repeat that exact map.  Mutation of any one
listed source invalidates the run before a candidate is executed.

Authorization is deliberately a separate future commit.  It must bind the
reviewed source commit and the exact 17-source hash map.  This preregistration
keeps those bindings null and `allow_real_pilot=false`, so regenerating a
self-consistent input manifest after a code change can never authorize a run.

## Frozen downstream and decision

Every bidirectional candidate pair receives its own isolated output directory
and independently executes the already sealed V13 matrix:

`PointDSC CPU 5 repeats + pyGCRANSAC CPU 5 repeats -> q4 -> fixed-trace ICP -> unchanged Rule-B -> bidirectional/cross-solver consistency`.

The candidate cache contains exactly `src_corr/ref_corr/scores`; no diagnostic
Kabsch pose is passed downstream.  A candidate is safe only when its sealed
V13 strict summary is safe and is bound to both candidate cache SHAs and both
candidate receipts.  The pair decision is:

- frozen known-bad pair: hard veto regardless of candidate results;
- zero safe candidates: reject (`no_safe_candidate`);
- exactly one safe candidate: accept the unique candidate in research shadow;
- two or more safe candidates: reject (`ambiguous_multiple_safe_candidates`),
  even when their final transforms are geometrically close.

The fixed4 research gate requires all three normal primary rows to accept a
unique candidate and the known-bad primary and control rows to remain vetoed.
Control results are reported but never rescue primary.  The bounded top-level
orchestrator requires exactly the frozen 4 pairs x 2 arms, runs at most eight
candidates per row (64 CPU candidate runs total), calls the sole exactly-one
selector and fixed4 aggregate, and atomically seals commands, environment,
resources, rows, summary, artifact manifest, and closure.  Real execution
remains unauthorized until the code/tests/manifest commit is reviewed.

## Stop conditions

Stop without GPU if any source/preflight/preregister SHA differs; any cache has
fewer than 40 rows; any candidate uses GT/labels/identity fallback; candidate
index or hash closure cannot be recomputed; a downstream strict receipt lacks
the unchanged V13 authority; known-bad is accepted; or more than one safe
candidate survives.
