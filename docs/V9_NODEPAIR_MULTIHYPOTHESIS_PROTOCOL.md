# V9 GT-free node-pair rigid-subset protocol

Status: pre-registered research diagnostic. It cannot promote a checkpoint or
authorize `official92`.

## Frozen inputs and exclusions

V9 reads only the immutable B/selection inference cache and canonical object
surfaces. Each cached `(node_src,node_ref)` GeoTransformer
`src_corr/ref_corr/scores` entry is treated independently. Selection labels,
GT transforms, posthoc files and accepted/rejected outcomes are forbidden.
Official SGAligner source, checkpoint, Rule-B and all thresholds remain byte
unchanged.

## Unique policy

1. Keep the highest-scored 256 correspondences of each node pair. Run 64
   deterministic three-point trials, score at 5 cm, require at least six
   inliers, and refine by Kabsch on its inliers. Malformed or degenerate input
   fails closed.
2. In each direction, connect node-pair transforms only inside the existing
   5 degree / 0.10 metre radius. Deterministically extract mutually exclusive
   complete-linkage cliques; a candidate requires at least three distinct
   node pairs. A chain is never treated as one rigid mode.
3. Match forward cliques to independently estimated reverse cliques only after
   inversion, under the same radius. Keep every matched candidate; do not rank
   one using labels or GT.
4. Each candidate separately pools only its member node pairs under the
   official total correspondence cap, then runs five forward and five reverse
   pooled pyGCRANSAC solves, the existing segment ICP with fixed-correspondence
   trace, unchanged Rule-B, q4 complete-linkage final consensus, and cross
   direction agreement.
5. Exactly one candidate must be safe. Zero safe candidates are rejected.
   Multiple safe candidates are rejected as an ambiguous scene transform.

## Required pre-label evidence

The full selection89 manifest and cache hashes must validate; two runs must
give identical structural partitions; the declared known-bad pair must remain
vetoed. Report structural coverage separately from final Rule-B-safe coverage.
The old all-node-pair pool is a comparison baseline only. Increasing repeated
RANSAC counts over the same polluted pool is not a recovery method.

## Complexity

For node-pair `i` with at most `m=256` retained correspondences, estimation is
`O(64*m)`. Pairwise transform compatibility is `O(n^2)` and deterministic
greedy complete-linkage extraction is worst-case `O(n^3)` Boolean work for
`n` estimated node pairs. Expensive pooled RANSAC/ICP is run only for
cross-direction candidate modes.
