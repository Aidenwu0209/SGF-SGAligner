# V16 b716 candidate and matched-region plan

This adapter repairs the V10/V11 checkpoint-domain mismatch without changing
the official SGAligner checkpoint or source.  It rebuilds `official_matching`
rank lists from the frozen official-release `joint_model` embeddings whose
checkpoint SHA-256 is
`b716c7d81b70274f98c7b4bd894c40534bac007ab71050713e39a67c5964a17e`.

The fixed candidate rule remains V10 `cross_graph_k=5`, mutual matching and a
48-pair cap.  The matched-region rule remains the V11 structural generator.
`src_count` is accepted only when `graph_per_obj_count`, both side-specific
object-index maps and `registration_id2oid` agree.  Every registration surface
is rebuilt from a unique raw InSeg row set and byte-bound to its raw file.

`pair_cache.json` is read with a top-level whitelist.  The `combos` subtree is
skipped lexically, so its GT-derived `node_metrics` are never decoded.  GT,
selection/evaluation labels, posthoc evidence, official92 and fallbacks are
forbidden inputs.

This first frozen stage executes zero new GeoTransformer jobs.  Existing
candidate keys are copied into deterministic immutable entries, including
their original failure states.  Missing keys are recorded as
`disabled_missing_geotransformer`; they are not silently dropped.  Fixed4 is
expected to contain 191 candidates, 34 hypotheses, 119 existing GeoT entries
and 72 missing entries.  Downstream ColorPCR authorization remains false until
the missing official GeoTransformer work runs under a separately reviewed,
isolated-GPU and clean-service protocol.

Build fixed4 on CPU only:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/aidenwu/miniconda3/envs/sgaligner/bin/python \
  scripts/v16_b716_candidate_plan.py --scope fixed4 \
  --output-root outputs/v16_b716_candidate_plan_fixed4_20260830
```

`--scope selection89` exists for a later development-only expansion.  It does
not run or open official92 and is not an authorization stage.
