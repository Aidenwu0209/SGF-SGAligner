# V10 GT-free cross-graph candidate protocol

Status: pre-registered research adapter. The default official path remains
unchanged. V10 cannot promote a checkpoint or authorize `official92`.

## Frozen rationale

`official_matching` ranks the complete two-scan graph, takes global top-3,
then discards same-graph neighbours. A source node can therefore produce zero
cross-graph candidates even when its first cross-graph neighbours are ranked
immediately afterward. V9 proved that the resulting cached node-pair
transforms contain no three-member forward/reverse rigid mode on selection89.

## Unique candidate policy

1. Consume the byte-verified B/selection `rank_list` produced by the unchanged
   official checkpoint and embeddings. Never read labels, GT or posthoc.
2. For each source node, filter to reference-graph nodes first and retain the
   first `k=5`. For each reference node, independently filter to source-graph
   nodes and retain the first `k=5`.
3. Keep only mutual top-5 candidates. Order by `(worst reciprocal rank,
   rank sum, forward rank, reverse rank, source index, reference index)` and
   keep at most 48 per scan pair. This is a resource bound, not a learned
   score. No output-dependent expansion is allowed.
4. Reuse a GeoTransformer result only when it is present in the immutable
   sealed cache with matching entry SHA. Run the unchanged official
   GeoTransformer once for each genuinely new candidate; keep typed failures.
5. Feed the candidate cache to the V9 frozen node-pair rigid estimator and
   multi-hypothesis gate without changing its 5 cm, six-inlier, 64-trial,
   three-member, 5 degree / 0.10 metre, Rule-B, fixed-trace, or q4 thresholds.

## Resource and safety gates

- Maximum candidate pairs: `89 * 48 = 4,272`; in the frozen manifest
  preflight, 1,919 require new GeoTransformer execution.
- A cache is written atomically per scan pair and can be resumed only after
  its SHA and candidate fingerprint validate.
- Required before any label read: 89/89 complete, zero unknown/untyped errors,
  two structural passes have identical payload hashes, the known-bad pair is
  vetoed, and every accepted candidate has exactly one Rule-B-safe rigid mode.
- Error-accepted must remain zero. Threshold, checkpoint, official source,
  and default inference routing are immutable.
