# ABot-Recon official no-loop smoke on Orin

This began as a local preparation record and minimal **create-only** execution
runbook.  The final section now records the completed Orin smoke.  That result
does not authorize a frontend replacement, a quality comparison or refusion.

## Frozen boundary

The official source was inspected from two already-present, clean shallow
clones without fetching.  In both clones, `HEAD`, local `main`, and the saved
`origin/main` ref resolved to:

```text
195cb9240ffc6300e008d2b70e54d281dd7caf4b
```

This is therefore the **last locally available official-main snapshot**, not a
claim that the live remote main was refreshed on 2026-09-02.  Do not run a
different commit under this receipt.

Freeze the released weight independently:

```text
Hugging Face snapshot: c69c26ca1853afc9c9212014459e729d7339ecdf
checkpoint filename:  abot_recon.safetensors
checkpoint SHA-256:   ea41a7659f6087069e6b3aac8830cc1c62d7c4a5c27a7d2679b51ba97cabcd2e
official input size:  504 x 280
official local window: 12 frames
precision:            BF16 default
```

The official CLI defaults to loop closure **enabled** and saves with ordinary
overwrite-capable NumPy/Torch calls.  This protocol consequently requires both
an explicit `--no-loop-closure` and a previously nonexistent run root.

## Why the eight-frame `scene0030_00` smoke

Use the already sealed first-eight ScanNet manifest as a model-load and
end-to-end contract smoke:

```text
docs/model_replacement_validation_20260902/scene0030_00_first8.json
payload SHA-256: c8a7baba10c5b98adeb6d66016f1c79f7de2a5e512d799c8367d7313b6f0c837
frame IDs: 0..7
```

The remote inference tree contains RGB, depth, and intrinsics only; ScanNet
poses were physically isolated under a sibling `eval_only/` tree.  The smoke
stages only the eight RGB files.  Metric-depth files are first read *after* a
successful official run, solely to estimate metres/model-unit for the adapter.

Eight frames are not an ABot quality or long-horizon validation.  They are the
smallest currently sealed input that can test checkpoint load, causal no-loop
inference, complete output coverage, metric-scale adaptation, and the SGF-SGA
trajectory contract.  A pass must be followed by a separately sealed longer
run (for example a preregistered `sgf_parameter_control` slice or sequence) on
the same frozen source/checkpoint.  Do not reinterpret the previous 8/41 OOM
internal progress as a valid eight-pose result.

## Known failure evidence

The exact official weight already failed on the 41-frame Orbbec
`slow_table_loop` contract on an RTX 4060 Laptop GPU with 8,188 MiB:

- SDPA: CUDA OOM after 8/41 internal streaming steps, 7,768 MiB observed peak,
  85.06 s wall time;
- paged KV cache with FlashInfer 0.6.18: CUDA OOM in the first-frame decoder,
  7,144 MiB observed peak, 24.03 s wall time;
- both attempts produced 0/41 valid poses, 0% coverage, no trajectory, no GT
  consumption, and no identity fallback.

This means an 8 GiB GPU is ineligible for another official-weight comparison.
On Orin, do not force `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the
separate MapAnything audit found that allocator unsuitable on this host.  Keep
the first ABot Orin smoke on the official SDPA path and preserve the failure as
zero coverage if it exits nonzero or the host reboots.

## Preflight and create-only staging

Edit only the four host paths below.  `RUN_ROOT` must name a new directory.
The checkpoint must already be present: offline mode deliberately prevents a
download or an implicit revision change.

```bash
set -euo pipefail
umask 027

SGF_SRC=/home/ai3d/Documents/SGF-SGAligner-model-validation-src
ABOT_SRC=/home/ai3d/Documents/ABot-Recon
ABOT_CHECKPOINT=/mnt/ssd/checkpoints/abot-recon-c69c26ca/abot_recon.safetensors
RUN_ROOT=/mnt/ssd/model-validation-runs-20260902/abot_scene0030_first8_noloop_sdpa

ABOT_COMMIT=195cb9240ffc6300e008d2b70e54d281dd7caf4b
ABOT_CHECKPOINT_SHA256=ea41a7659f6087069e6b3aac8830cc1c62d7c4a5c27a7d2679b51ba97cabcd2e
MANIFEST="$SGF_SRC/docs/model_replacement_validation_20260902/scene0030_00_first8.json"

test ! -e "$RUN_ROOT"
test -f "$ABOT_CHECKPOINT"
test "$(git -C "$ABOT_SRC" rev-parse HEAD)" = "$ABOT_COMMIT"
test "$(git -C "$ABOT_SRC" rev-parse refs/heads/main)" = "$ABOT_COMMIT"
test "$(git -C "$ABOT_SRC" rev-parse refs/remotes/origin/main)" = "$ABOT_COMMIT"
test -z "$(git -C "$ABOT_SRC" status --porcelain)"
test "$(sha256sum "$ABOT_CHECKPOINT" | awk '{print $1}')" = "$ABOT_CHECKPOINT_SHA256"

mkdir -m 0750 "$RUN_ROOT"
mkdir -m 0750 "$RUN_ROOT/input_rgb"
```

Stage only manifest-declared RGB and write a create-only receipt.  The loader
rejects pose/GT/mesh path components and verifies the signed manifest before
any image is copied.

```bash
PYTHONPATH="$SGF_SRC/src" python - "$MANIFEST" "$RUN_ROOT/input_rgb" \
  "$RUN_ROOT/input_receipt.json" "$ABOT_COMMIT" "$ABOT_CHECKPOINT" \
  "$ABOT_CHECKPOINT_SHA256" <<'PY'
from pathlib import Path
import json
import shutil
import sys

from pose_pipeline.contracts import load_manifest, sha256_file, stable_json_sha256

manifest_path, rgb_dir, receipt_path, commit, checkpoint, expected_checkpoint = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4],
    Path(sys.argv[5]), sys.argv[6],
)
manifest = load_manifest(manifest_path)
payload = manifest.as_dict()
assert manifest.dataset == "scannet"
assert manifest.sequence_id == "scene0030_00"
assert [frame.frame_id for frame in manifest.frames] == list(range(8))
assert payload["payload_sha256"] == (
    "c8a7baba10c5b98adeb6d66016f1c79f7de2a5e512d799c8367d7313b6f0c837"
)
assert sha256_file(checkpoint) == expected_checkpoint

rows = []
for ordinal, frame in enumerate(manifest.frames):
    suffix = frame.color_path.suffix.lower()
    destination = rgb_dir / f"{ordinal:06d}{suffix}"
    with frame.color_path.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    rows.append({
        "ordinal": ordinal,
        "frame_id": frame.frame_id,
        "timestamp_us": frame.timestamp_us,
        "source_path": str(frame.color_path),
        "source_sha256": sha256_file(frame.color_path),
        "staged_path": str(destination),
        "staged_sha256": sha256_file(destination),
    })
assert all(row["source_sha256"] == row["staged_sha256"] for row in rows)

unsigned = {
    "schema": "abot_official_input_receipt.v1",
    "sequence_id": manifest.sequence_id,
    "model_commit": commit,
    "checkpoint_sha256": expected_checkpoint,
    "manifest_payload_sha256": payload["payload_sha256"],
    "ordered_frames": rows,
    "gt_consumed": False,
    "identity_fallback_used": False,
}
signed = {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}
with receipt_path.open("x", encoding="utf-8") as stream:
    json.dump(signed, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY
```

Stop here if any assertion fails.  Do not repair a missing frame with a copied
neighbor, an identity pose, an interpolated pose, or a shorter manifest.

## Official no-loop execution

Run only after confirming enough unified memory for the checkpoint, local
point maps, confidence maps and logs.  On the storage-constrained Orin used for
this smoke, the checkpoint was held in `/dev/shm` and outputs alone were kept
under `Documents`; no multi-GB checkpoint was written to eMMC.  This command
neither uses loop assets nor contacts Hugging Face.

```bash
set +e
(
  cd "$ABOT_SRC"
  env -u PYTORCH_CUDA_ALLOC_CONF \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/time -f 'elapsed_s=%e\nmax_rss_kb=%M\nexit_status=%x' \
      -o "$RUN_ROOT/resource_time.txt" \
      python demo.py \
        --image-dir "$RUN_ROOT/input_rgb" \
        --checkpoint "$ABOT_CHECKPOINT" \
        --output-dir "$RUN_ROOT/official_noloop" \
        --device cuda \
        --attention-backend sdpa \
        --no-loop-closure \
        --dense-stride 1 \
        --max-frames 64
) >"$RUN_ROOT/official_stdout_stderr.log" 2>&1
ABOT_EXIT_CODE=$?
set -e
printf '%s\n' "$ABOT_EXIT_CODE" >"$RUN_ROOT/exit_code.txt"

if test "$ABOT_EXIT_CODE" -ne 0; then
  echo "FAIL: preserve this run root; coverage is 0/8 and no trajectory may be adapted" >&2
  exit "$ABOT_EXIT_CODE"
fi
```

If the process exits nonzero, the host reboots, or any expected output is
missing, stop.  Preserve the run root and record a failed
`model_runtime_report.v1` with `output_pose_count=0`; never serialize whatever
internal frame count appeared in the log.

## Output audit and metric adapter

The official run is complete only if all eight finite SE(3) poses exist, loop
outputs are absent, and metadata says `loop_closure=false`.  The first identity
pose is the model's world-frame gauge and is not a gap fill; this audit does not
fabricate or replace any pose.

```bash
PYTHONPATH="$SGF_SRC/src" python - "$MANIFEST" "$RUN_ROOT/official_noloop" \
  "$RUN_ROOT/official_output_audit.json" <<'PY'
from pathlib import Path
import json
import sys

import numpy as np
from pose_pipeline.contracts import load_manifest, sha256_file, stable_json_sha256, validate_se3

manifest = load_manifest(Path(sys.argv[1]))
output = Path(sys.argv[2])
audit_path = Path(sys.argv[3])
required = [
    "camera_poses.npy", "camera_poses_noloop.npy",
    "relative_poses.npy", "relative_poses_noloop.npy",
    "local_points.pt", "confidence.pt", "metadata.json",
]
assert all((output / name).is_file() for name in required)
assert not (output / "camera_poses_loop.npy").exists()
assert not (output / "relative_poses_loop.npy").exists()
metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
assert metadata["loop_closure"] is False
assert metadata["pose_outputs"] == ["noloop"]

poses = np.load(output / "camera_poses_noloop.npy", mmap_mode="r")
selected = np.load(output / "camera_poses.npy", mmap_mode="r")
assert poses.shape == (len(manifest.frames), 4, 4)
assert np.array_equal(poses, selected)
for index, pose in enumerate(poses):
    validate_se3(pose, f"ABot official pose {index}")

unsigned = {
    "schema": "abot_official_output_audit.v1",
    "sequence_id": manifest.sequence_id,
    "expected_pose_count": len(manifest.frames),
    "observed_pose_count": int(len(poses)),
    "complete": True,
    "official_loop_mode": False,
    "gt_consumed": False,
    "identity_fallback_used": False,
    "artifacts": {name: sha256_file(output / name) for name in required},
}
signed = {**unsigned, "payload_sha256": stable_json_sha256(unsigned)}
with audit_path.open("x", encoding="utf-8") as stream:
    json.dump(signed, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
PY

PYTHONPATH="$SGF_SRC/src" python -m pose_pipeline abot-scale-evidence \
  --manifest "$MANIFEST" \
  --local-points "$RUN_ROOT/official_noloop/local_points.pt" \
  --confidence "$RUN_ROOT/official_noloop/confidence.pt" \
  --maximum-frames 8 \
  --sample-stride 8 \
  --output "$RUN_ROOT/abot_scale_pairs.npz"

PYTHONPATH="$SGF_SRC/src" python -m pose_pipeline adapt-abot \
  --manifest "$MANIFEST" \
  --poses "$RUN_ROOT/official_noloop/camera_poses_noloop.npy" \
  --mode noloop \
  --scale-evidence "$RUN_ROOT/abot_scale_pairs.npz" \
  --model-commit "$ABOT_COMMIT" \
  --checkpoint-sha256 "$ABOT_CHECKPOINT_SHA256" \
  --output "$RUN_ROOT/abot_noloop.trajectory.json"
```

The adapter is fail-closed: it requires the exact `camera_poses_noloop.npy`
filename, exact commit/checkpoint hashes, finite SE(3), manifest-bound scale
evidence, and exactly one pose for every manifest frame.  Its output is written
with create-only mode.

## Acceptance and stop checks

The smoke passes only when all of the following are true:

1. source and checkpoint hashes equal the frozen values;
2. the official command exits zero without host reboot or CUDA OOM;
3. `metadata.json` states `loop_closure=false` and no loop output exists;
4. official no-loop output has 8/8 finite SE(3) poses;
5. scale evidence is bound to the same manifest and produces a positive finite
   metres/model-unit estimate;
6. `abot_noloop.trajectory.json` has 8/8 metric `T_world_camera` records;
7. no GT pose, mesh, evaluation artifact, identity fill, interpolation, or
   partial-output promotion was used.

Even a pass is only `scene0030_00` first-eight smoke evidence.  It does not
establish long-horizon stability, Orbbec quality, runtime competitiveness,
complete-scene refusion, or eligibility to replace DPV-SLAM.

## Local preparation verification (no GPU)

The preparation host ran only static and contract checks:

```text
SGF-SGAligner model replacement adapter/contracts: 9 passed
official ABot release-hygiene tests:                 4 passed
official ABot Python AST parse:                      77 files passed
```

The official Torch-dependent ABot unit tests were attempted but could not be
collected in the preparation host's base Python because `torch` is not
installed there.  This is an environment limitation, not a passing runtime
result.  The Orin execution environment must run the official CPU/unit suite
before the checkpoint smoke; no CUDA inference was performed while preparing
this document.

## Executed Orin result

The official no-loop SDPA smoke completed after switching the Jetson to
`MODE_30W`.  The input still contained exactly 8 RGB frames.  `--max-frames`
had to remain 64 because ABot also uses that value as a RoPE spatial bound;
setting it to 8 failed closed with an invalid reshape before producing any
promotable output.

The successful create-only run is:

`/home/ai3d/Documents/sgf_sga_model_validation_20260902/runs/abot_scene0030_first8_30w_sdpa_attempt2`

It produced 8/8 finite no-loop SE(3) poses plus frame-complete local points and
confidence maps.  The depth-backed robust scale adapter used 13,576 valid
samples and estimated `2.8006115587` metres/model-unit; its 5th-to-95th
relative spread was `0.0927798235`.  The resulting
`pose_trajectory.v1.json` passed the SGF-SGAligner loader with payload SHA-256
`8ace563e175b6ffffd8639f80879d244677ac9ac300ececa8f06c27d0257e623`.
The scale-evidence SHA-256 is
`6464e943b8d1540bb3b78f90b09ec2aacfe3836cf8a6797e807b80f421d847f4`.

This establishes runtime and output-contract feasibility only.  The metric
steps are finite but very small over this short nearly-static prefix; accuracy
must be judged on a longer sealed slice and final refusion, not inferred from
this smoke.
