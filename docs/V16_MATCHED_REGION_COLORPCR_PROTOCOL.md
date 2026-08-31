# V16 matched-region ColorPCR builder protocol

This pre-registration is a **disabled builder-only stage**. It is not an
independent result and cannot authorize fixed4, reconstruction, selection89,
calibration90, or official92.

The only admissible membership source is the authenticated V10 candidate cache
and V11 deterministic structural hypotheses. The builder resolves their stored
node index/object identity into the unique raw InSeg object rows and RGB in
metres. Semantic class labels, GT transforms, selection labels, scene labels,
post-hoc choices, identity fallbacks, and any oracle mode are prohibited.

The current V10/V11 ranks belong to research checkpoint **B**
`89eddb50...`; the target downstream official release checkpoint is
`b716c7d8...`. These are deliberately separate provenance domains. Therefore
all current V16 products are diagnostic builder artifacts only and are
prohibited as ColorPCR inputs. End-to-end official-release work must first
regenerate and freeze V10 ranks and V11 hypotheses from `b716c7d8...`; this
protocol does not authorize that reranking or any solver run.

The raw InSeg `labels` array is narrowly allowed as an **instance membership
key**: `(scan_id, side, object_id) -> raw row set`. It is hashed and recorded,
never compared across scans, never treated as semantic/global identity, and
never consumed by candidate ranking, ColorPCR, RANSAC, ICP, or Rule-B. An
object normally maps to many rows; ambiguity means missing side/scan, a
non-unique node-to-object mapping, no rows, conflicting duplicate coordinates,
or failure to reconstruct the canonical surface from those rows.

For every hypothesis the builder must write member rank records, object IDs,
surface files and hashes, row offsets and exact row indices, per-member surface
hashes, union hashes, and a recursive provenance closure. The closure must
verify the V10 source cache and its source hashes, bind current canonical
`obj_ids`, `src_count`, and every registration-surface hash to V10 provenance,
and exactly recompute V11 `canonical_input_sha256`.

Filtering happens before the frozen V13 world-origin 0.10 m voxel aggregation.
The builder saves raw unions and voxel10 prepared arrays only. It must **not**
write `fps512_*` arrays: inspection of the frozen V13 official worker confirms
that cap512 occurs after its repeated multi-level `grid_subsample`, at the
final coarsest stage. A future, separately authorized pilot must leave that
worker behavior unchanged and launch forward/reverse ColorPCR independently.

Rule-B, ICP, 0.10 m residual, 40-support, and 5 degree/0.10 m pose-cluster gates
are unchanged. Any ambiguity or missing provenance fails closed.
