# V8 selection89 development worker batch

This stage is **development evidence only**.  It must not be described as a
blind, confirmatory, or preregistered result.

## Reusable inputs

The canonical `selection.txt` has exactly 89 unique pair IDs.  The immutable
V6-Fix B/selection cache root has exactly one validated `.pt` cache for each
pair.  The cache checkpoint is B (`89eddb50...`) and contains node matches plus
GeoTransformer point correspondences, but no registration ground-truth label.

The three historical `selection/B/repeat_*.json` files are not worker caches.
They contain posthoc-labelled final F/C0/C1 path rows and no five-forward plus
five-reverse records.  The freezer only hashes and explicitly excludes them;
it never parses or replays them.

## Execution contract

For every one of the 89 pairs:

1. run two independent outer repeats;
2. in each outer run five forward and five true-reverse workers;
3. persist and validate the resulting 1,780 worker JSON files once;
4. let all V8 policies replay the identical frozen workers offline;
5. freeze the GT-free batch receipt before any label process is started.

The V8 decision order is fixed by `V8_STAGE_ORDER_CONSENSUS_PROTOCOL.md`:
cluster all finite final ICP transforms, select an observed medoid, apply the
unchanged Rule-B checks to that medoid, and then require forward/reverse final
consensus at `q=4`, `rotation=5deg`, `translation=0.10m`.  Raw RANSAC consensus
is diagnostic only.

## Commands after the V8 protocol commit is integrated

Freeze once:

```bash
PYTHONPATH=.:src:scripts python scripts/v8_selection89_development.py \
  --freeze-manifest
sha256sum outputs/v8_selection89_manifest_seal_v2_20260830/v8_selection89_manifest.json
```

Validate all 89 caches and write the no-compute plan (replace `<sha>` with the
printed manifest file SHA):

```bash
PYTHONPATH=.:src:scripts python scripts/v8_selection89_development.py \
  --dry-run --manifest-sha256 <sha>
```

Only after the manifest and protocol are committed, start the GT-free workers
in a fresh output directory:

```bash
PYTHONPATH=.:src:scripts python scripts/v8_selection89_development.py \
  --execute-workers --manifest-sha256 <sha> \
  --out outputs/v8_selection89_gt_free_workers_20260830 \
  --pair-concurrency 2
```

This command does not load labels or apply a winner policy.  A separate V8
offline replay freezes its GT-free decision receipt:

```bash
PYTHONPATH=.:src:scripts python scripts/v8_selection89_replay.py \
  --batch-receipt outputs/v8_selection89_gt_free_workers_20260830/gt_free_worker_batch_receipt.json \
  --manifest outputs/v8_selection89_manifest_seal_v2_20260830/v8_selection89_manifest.json \
  --manifest-sha256 <sha> \
  --out outputs/v8_selection89_gt_free_workers_20260830/v8_replay.json
```

Only after `v8_replay.json` exists and verifies may the distinct label process
report selection89 development metrics:

```bash
PYTHONPATH=.:src:scripts python scripts/v8_stage_order_posthoc.py \
  --receipt outputs/v8_selection89_gt_free_workers_20260830/v8_replay.json \
  --out outputs/v8_selection89_gt_free_workers_20260830/posthoc.json
```

## Stop checks

Stop and preserve evidence if any cache hash changes, the pair order changes,
the V8 protocol hash changes, a worker is non-finite, an outer is partial, or
the repository has any change outside the designated fresh output root.

Do not run calibration90, fixed12, or official92 from this stage.  Calibration
is the next locked verification gate only after a selection89 development
winner and its complete configuration are frozen.
