# V8 calibration90 locked gate

## Boundary

`calibration90` is the first locked V8 gate.  It may be opened only after
`selection89` has frozen one unique candidate, checkpoint, configuration,
source inventory and cache inventory.  The only candidate is:

- final ICP transforms clustered before any Rule-B filtering;
- five forward and five true-reverse workers;
- unique complete-linkage quorum 4/5 at 5 degrees and 0.10 m;
- observed forward/reverse medoids, both passing unchanged Rule-B;
- cross-direction final-transform quorum 4/5;
- fixed-correspondence ICP trace and stable final update;
- raw-transform consensus is diagnostic only.

This phase never changes the official SGAligner source/checkpoint, Rule-B
thresholds, the V8 policy or the production default.  `fixed12` and
`official92` are forbidden before this gate passes.

## Existing input audit

The canonical pair list contains exactly 90 unique opaque pair IDs and has SHA
`0d82256933c9e6dbac55a4ccc85a902ec49941f5dc5b600f3ff9150ef51a88fd`.

The existing 90-pair V3 `final_inference_cache/calibration90` tree is complete
for its historical purpose, but its checkpoint SHA is the official epoch-6
release (`b716c7d8...`), not the frozen SGF-domain winner B
(`89eddb50...`).  It is therefore **not** a valid V8 worker cache.  No array or
label is opened by this audit.  Before the manifest can be frozen, a fresh
label-free B/calibration cache must be prepared by the already-audited
`v6fix-inference-cache-v2` builder and contain exactly the 90 pair files.

## Gate-0: fresh B/calibration90 cache

Gate-0 reuses `v6fix_consistency_audit.build_or_load_cache`; it does not
reimplement SGAligner, GeoTransformer or inference.  Before selection89 has
passed, only freeze the label-free preparation plan:

```bash
PYTHONPATH=.:src:scripts /home/aidenwu/miniconda3/envs/sgaligner/bin/python \
  scripts/v8_calibration90_cache_prepare.py \
  --freeze-plan \
  --plan outputs/v8_calibration90_gate0_plan_20260830.json
```

Commit that plan before any cache generation.  It binds the canonical ordered
90-pair list, checkpoint B (`89eddb50...`), reachable-function AST audit and
every source SHA.  It explicitly authorises no labels, workers, posthoc,
fixed12 or official92.

Only after the frozen selection89 receipt passes, execute exactly once with
the committed plan file SHA (placeholder below):

```bash
PYTHONPATH=.:src:scripts /home/aidenwu/miniconda3/envs/sgaligner/bin/python \
  scripts/v8_calibration90_cache_prepare.py \
  --execute \
  --plan outputs/v8_calibration90_gate0_plan_20260830.json \
  --plan-sha256 <COMMITTED_PLAN_FILE_SHA256> \
  --out /home/aidenwu/Documents/sgaligner-sgf-v8-calibration90-B-cache
```

The final cache root is published by one same-filesystem rename only after
90/90 files validate.  Its receipt binds source, checkpoint, pairlist, every
input/cache/embedding/similarity SHA, and proves that no labels, workers or
posthoc ran.  A failed build leaves no final root.  Epoch-6 cache reuse is
forbidden.

## One-shot order

1. Freeze the Gate-0 cache plan before opening any calibration labels.
2. Freeze a `v8-selection89-winner-freeze-v1` receipt after selection89 passes.
3. Prepare the 90 label-free B/calibration cache files with the frozen plan.
   Do not load GT.
4. Freeze the calibration manifest, binding every cache file SHA, ordered pair
   list, winner receipt, code inventory, checkpoint, configuration and gate.
5. Execute the GT-free batch once.  A durable `O_EXCL` claim is written before
   the first worker and is never removed, including after failure.
6. Freeze and validate 1,800 worker records and the GT-free batch receipt.
7. Run the independent posthoc program once.  It writes a second durable
   `O_EXCL` claim before its first GT load; the claim also survives failure.
   Only that program may import GT.
8. On PASS, `fixed12` becomes authorised as a safety regression.  `official92`
   remains forbidden.  On FAIL, neither rerun nor retuning on calibration90 is
   authorised.

## Thresholds frozen before labels

The historical sealed calibration Rule-A floor is used mechanically rather
than tuned from the new labels: both outer runs must have 90 completed, at
least 6 strict, at least 8 relaxed, at least 5 accepted-correct, zero
accepted-error, zero exception/nonfinite/cache mismatch and 90/90 repeatable
pair outcomes.  The posthoc program has no threshold CLI.

These are minimal safety/fairness floors, not a production-quality claim.
Passing them permits only the preregistered fixed12 regression; it does not
authorise official92 or replacement of the default checkpoint.
