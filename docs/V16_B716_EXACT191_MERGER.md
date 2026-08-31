# V16 b716 exact191 merger contract

`scripts/v16_b716_exact191_merger.py` is a post-execution evidence merger. It
does not import SGAligner, GeoTransformer, CUDA, a selector, GT, or official92.
It can seal an output only when all of the following are true:

- the frozen b716 fixed4 candidate manifest is exactly 191 ordered entries;
- the original distribution remains `48/48/48/47`, with exactly 119 immutable
  existing cache entries and missing distribution `2/21/21/28`;
- the authorized preflight is exact72, every task and source closure still
  matches, and the authorization is bound to that candidate/missing closure;
- all 72 attempt receipts bind the same authorization and all 72 result files
  are successful, foreign-field-free, correspondence-SHA-closed results;
- every existing entry is byte-compared with both the sealed immutable NPZ and
  its official V3 GeoTransformer cache row;
- the original 34 hypotheses remain byte-frozen in distribution `12/8/2/12`.

The sealed top-level schema is
`v16-b716-exact191-merged-manifest-v1`. Per-pair entries use
`v16-b716-exact191-pair-v1`; hypothesis allowlists use
`v16-b716-frozen-hypothesis-allowlist-v1`. The allowlists prohibit candidate,
result, and hypothesis selection and require replay of all 34 frozen
hypotheses. Existing typed failures remain visible; they are not rewritten as
success. Any failed new result blocks the entire merger.

## Deliberate downstream P0 boundary

The older V13/V16 matched-region builders are not silently reused: they contain
historical B/`89ed` bindings, different NPZ key conventions, and hard-coded
`12/6/1/12` or other non-b716 arm counts. A downstream adapter must explicitly
consume the exact191 pair schema plus each frozen allowlist and must prove it
replays `12/8/2/12` without selecting by registration outcome. Until that
adapter is reviewed, the exact191 artifact is sealed evidence, not an automatic
ColorPCR/PointDSC authorization.

The CLI requires explicit SHA-256 values for the candidate, preflight,
preregister, authorization, and batch result. Output creation is create-only:
an identical replay is accepted, while a different existing artifact is
rejected.

## Hardened execution-receipt boundary

The merger consumes the receipt schemas frozen by backfill commit `8986a2a`.
It validates the complete authorization binding (preregister, preflight file
and payload, recursive source and artifact closures, task closure, immutable
runtime source bundle, runtime module entrypoints, CUDA UUID), derives and
byte-compares every authorized-task view, and follows the batch row through
the attempt receipt and result receipt to the correspondence artifact.

Every `src_corr`, `ref_corr`, and `scores` declaration is checked separately
for its exact field set, shape, dtype, finite values, and array SHA-256.  The
batch and every nested receipt reject outcome/selection fields.  Regression
tests preserve the three independent-audit exploits: a forged attempt
`preregister_sha256`, a batch row containing `selected=true`, and a forged
per-array `src_corr` SHA.  All three must fail even when their outer JSON
payload and receipt SHA chain has been recomputed.

The candidate plans are also read independently to reconstruct all 191 keys in
their original order.  That list must exactly equal the preflight
`future_merge_contract.expected_candidate_keys` and its expected/existing/new
closure hashes; the frozen ordered-key SHA is
`572634917937d79b88a1ba4e99ea34e68e3fa5b0e5401567fec308d3b48ef6b4`.
The batch result top level is an exact whitelist, so even a self-signed benign
extra field is rejected.

Sixteen of the 119 frozen existing entries are typed failures.  They remain as
visible ordered entries, and all 34 hypotheses remain in their allowlists;
eight hypotheses explicitly record typed-failure members.  Consumers must not
filter on `all_members_ok` or any registration outcome.
