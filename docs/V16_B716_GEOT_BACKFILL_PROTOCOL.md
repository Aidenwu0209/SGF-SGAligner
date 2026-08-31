# V16 b716 GeoTransformer backfill protocol

This stage is the execution boundary for the exact 72 candidate keys recorded
as missing by commit `9acef62f130298048285dc4b3d1dfb3f15c7cbef`.
The default and currently only authorized action is a CPU dry-run that rebuilds
and hashes 72 independent task records. No pair/key selector exists.

Before task creation the runner verifies the frozen candidate manifest and all
four structural-plan hashes. It does not accept their self-attestation alone:
the pinned V3 `artifact_manifest.json` must bind SHA and size for all 16 fixed4
cache artifacts, the pinned V3 `cache_manifest.json` must bind the exact 89-row
selection digest and each fixed4 inner cache key, and the pinned formal V13
preregister must bind all four full pair IDs (including the appended known-bad
pair). The emitted preflight contains the read-only selection89 count/digest
summary, but does not copy the large upstream products.

The runner then compares the observed missing keys and order with the complete
preregistered table. It recursively rechecks every recorded source file, the
b716 epoch-6 official checkpoint, code head `98df603`, official GeoTransformer
checkpoint, canonical cached tensors, `joint_model`, graph/object `src_count`
boundary and every raw InSeg object surface. It also revalidates the inner
`cache_key` pair/input/checkpoint/sampling/model-config/code-head fields.
`pair_cache.json` is read through the existing top-level whitelist, so
`combos/node_metrics` are never decoded.

Each key gets an immutable `task.json` in its own directory. Future execution
does not mutate that task: the preregistration must change coherently from
`disabled=true/real_execution_allowed=false` to
`disabled=false/real_execution_allowed=true`, after which a SHA-bound
`authorized_task_view.json` may be derived. The planned task remains
`planned_disabled` with `execution_authorized=false` in both modes.

Execution will create one atomic attempt receipt, deterministic correspondence
NPZ and hash-bound result JSON per key. The attempt binds the planned task,
authorized task view, authorization receipt SHA, preregistration SHA, preflight
manifest file/payload SHA, recursive source/artifact/task closures, immutable
runtime source bundle, CUDA UUID and exact resource snapshot. The source bundle
includes the runner and its local safety/adapter/inference modules plus the
official GeoTransformer, engine and utility Python trees. The preflight also
freezes the exact `inference.py`, GeoTransformer config/model/data, evaluator
and torch utility entrypoints. After the initial clean gate, their imported
`__file__` paths, bytes and hashes must resolve inside that source bundle;
every project module loaded transitively must remain inside the same bundle.
Those source bytes are rechecked at every execution boundary. The result
repeats the bindings and adds the attempt-receipt SHA. Resume requires both
artifacts and
fully revalidates their schemas, payload hashes and every binding; a result
without an attempt is invalid, while an attempt without a result is ambiguous
and cannot be rerun automatically. Each batch row carries its attempt-receipt
SHA and the batch carries their ordered closure SHA. The batch always traverses
the full ordered 72-key table; it never ranks, selects or suppresses results.
Successful resumed NPZ files must have exactly `src_corr`, `ref_corr`, and
`scores`, with frozen shapes, float32 dtypes, finite values and per-array SHA.
Only the two registered failure statuses are accepted.
Deterministic NPZ outputs are create-only: an identical replay is accepted,
while a stale/tampered or input-divergent existing file is never overwritten.

Real CUDA execution has four independent hard gates:

1. A reviewed preregistration must change `real_execution_allowed` to true.
2. A SHA-pinned, unexpired authorization receipt must bind the exact candidate
   manifest, 72-key closure, preregistration, preflight manifest, output root
   and GPU UUID. The receipt file and expiry are re-read before every key.
3. A SHA-pinned clean-service receipt and both environment sentinels must exist.
4. Dynamic `nvidia-smi` checks must show one visible GPU, at most 256 MiB used,
   at most 5 percent utilization and zero compute processes before model
   import. Before every key, a second dynamic gate permits at most the current
   authorized runner PID (never a foreign process), at most 8192 MiB resident
   memory and any transient utilization up to the device maximum. This
   distinction keeps the hard
   initial clean gate enforceable after the runner's own CUDA context exists.
   The clean-service receipt SHA and expiry are also re-read before every key.

Immediately before each new GeoTransformer call, the canonical pair and raw
InSeg bindings are rebuilt. The source/reference registration arrays must match
the task's object identity, raw path/SHA, point count and canonical surface SHA;
the runner does not reuse a pair-level in-memory cache across keys.

The preflight also freezes a deterministic downstream ledger contract covering
all 119 immutable official cache entries plus all 72 backfill keys, for exactly
191 candidate entries in original fixed4 order. It cannot drop failures or
select results, and remains `downstream_authorized=false` until all 72 completed
receipt/result chains exist and a later reviewed stage explicitly authorizes it.

The current preregistration deliberately fails gate 1, so `--execute` cannot
reach a GeoTransformer import or GPU call before a reviewed follow-up commit.
GT, evaluation/selection labels, pair `combos`, posthoc evidence, official92,
fallbacks and result-based key selection remain prohibited.

Dry-run command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /home/aidenwu/miniconda3/envs/sgaligner/bin/python \
  scripts/v16_b716_geot_backfill.py \
  --output-root outputs/v16_b716_geot_backfill_preflight_20260830
```
