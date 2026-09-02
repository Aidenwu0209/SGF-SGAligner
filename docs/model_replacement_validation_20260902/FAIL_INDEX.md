# Model replacement validation failure index

This index records real official-weight execution failures separately from
adapter/unit-test results. A failed run has zero valid trajectory coverage;
internal frames processed before failure are not promoted to a partial
`pose_trajectory.v1`.

## ABot-Recon / Orbbec 41-frame contract smoke

- Status: **FAIL — CUDA out of memory**
- Role/arm: continuous full-frame frontend, official no-loop output
- SGF-SGAligner adapter commit: `29a6d36445848b7cb6c0d51dddab1ad34e5b6cb1`
- ABot-Recon commit: `195cb9240ffc6300e008d2b70e54d281dd7caf4b`
- Hugging Face snapshot: `c69c26ca1853afc9c9212014459e729d7339ecdf`
- Checkpoint SHA-256:
  `ea41a7659f6087069e6b3aac8830cc1c62d7c4a5c27a7d2679b51ba97cabcd2e`
- Input: first 41 admitted frames from authoritative
  `slow_table_loop` Orbbec manifest; frame IDs 303 through 367 with the
  manifest's original gaps and timestamps preserved.
- Input manifest payload SHA-256:
  `ddbfdab84c3b14a729ae1586198f1f310b13279526c3622ecd1ecb7b20f51ce8`
- Resolution: official checkpoint setting, 504 x 280
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB reported total
- GT consumed: no
- Identity fallback: no
- Valid trajectory written: no

### Official SDPA backend

- Failure point: after 8/41 internal streaming steps
- Wall time: 85.06 s
- Observed peak framebuffer usage: 7768 MiB
- Valid output pose count / coverage: 0 / 0%
- Signed runtime report SHA-256:
  `fb96d899ba1cbe1c189ff62901440e8cad504ce6874a4ce9f3634793b67bf7f9`
- Runtime payload SHA-256:
  `76c17bccf8f9e008be0f80c5de264252d6fbdbb9fd41bfdbbad9cd67fc85af69`

### Official recommended paged KV-cache attempt

- FlashInfer: 0.6.18; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Failure point: first frame decoder
- Wall time: 24.03 s
- Observed peak framebuffer usage: 7144 MiB
- Valid output pose count / coverage: 0 / 0%
- Signed runtime report SHA-256:
  `6d9c48c41873ef2701217e6cec9717ba68ccbb974224cc500b7708cf278e799d`

### Decision

ABot-Recon remains an implemented opt-in research candidate, but this 8 GiB
host is not eligible for its official-weight comparison. Do not reduce the
sequence, serialize the first eight internal states, or insert identity poses
to turn this failure into apparent coverage. Re-run the exact input/checkpoint
on a higher-memory GPU before evaluating metric scale, trajectory, refusion or
promotion.

Authoritative raw artifacts remain create-only at:

`/home/aidenwu/Documents/model-validation-runs-20260902/abot_orbbec41_slow_table_loop`

## MapAnything / ScanNet scene0030 first-eight-frame smoke

- Status: **FAIL — Jetson host rebooted during official model load**
- Role/arm: background 8-frame RGB + intrinsics + metric-depth refinement
- MapAnything commit: `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`
- Hugging Face snapshot: `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`
- Checkpoint size: 4,914,062,480 bytes, stored only in `/dev/shm`
- Checkpoint SHA-256:
  `981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0`
- Input: ScanNet `scene0030_00`, frame IDs `0..7`, 518 x 392 official
  preprocessing, memory-efficient inference, minibatch size 1 and BF16
- Input manifest payload SHA-256:
  `c8a7baba10c5b98adeb6d66016f1c79f7de2a5e512d799c8367d7313b6f0c837`
- Host: Jetson Orin, JetPack 6.2.1, 61 GiB unified memory
- GT consumed: no; ScanNet GT poses were physically isolated under
  `eval_only/` and absent from the inference root
- Identity fallback: no
- Valid output pose count / coverage: 0 / 0%

The first attempt used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
failed at `model.to(cuda)` with a CUDA driver OOM. Diagnostics then showed that
the Orin could cumulatively allocate 20 GiB using the default allocator, while
the expandable-segment setting failed on a single 1 GiB allocation. The runner
was corrected not to force that allocator.

The second attempt passed the previous failure point and reached roughly 14
GiB system memory use, but the entire host rebooted at `2026-09-02 08:49:52`
before inference. `last -x` marked both model/tegrastats tmux sessions as
`crash`; `/dev/shm` was cleared. No persistent previous-boot kernel log or
pstore record was available, so the exact kernel/power cause is unknown and
must not be reported as a confirmed OOM.

The signed failed runtime report has payload SHA-256
`d354ed8cabc1288717f266c082e88cc77c3b053649c8f990a4d459a4d72140fb`.
Do not retry this 4.9 GB checkpoint on the eMMC-only Orin until a persistent
external SSD and a Jetson-specific stability plan are available.

Persistent remote artifacts are under:

`/home/ai3d/Documents/sgf_sga_model_validation_20260902`

### Superseding controlled 30 W retry

The failure above remains valid evidence for the original power-mode attempt,
but it is **not the current model verdict**.  After an authorized reboot into
Jetson `MODE_30W` (nvpmodel ID 2), the same frozen source, checkpoint and eight
frames completed with the default allocator and official BF16,
memory-efficient inference settings.

The first 30 W process reached inference and exposed an adapter-only error:
the runner supplied `is_metric_scale` as a one-element NumPy array, whereas the
official preprocessing path expects the missing-field scalar default.  That
attempt produced no promoted output.  The runner now omits that optional field
and a focused regression test seals the input shape.

The create-only retry then produced 8/8 finite camera poses and the full
official depth, confidence and point-map output:

- inference: `8.775879409` s; total model-process wall: `51.620335979` s;
- peak CUDA allocated / reserved: `8,318,545,408` / `8,797,552,640` bytes;
- output SHA-256:
  `0136f2510a2695c2060080a343f7ca81e457bbe075f03cd2c02329dc0077bbb9`;
- rotation determinants: `0.9999999548` through `1.0000000243`;
- GT consumed: no; identity gap filling: no.

Authoritative artifacts are create-only under:

`/home/ai3d/Documents/sgf_sga_model_validation_20260902/runs/mapanything_scene0030_8_independent_30w_original_attempt2`

This supersedes only the host-eligibility failure.  An eight-frame smoke is not
pose-accuracy, full-scene refusion or promotion evidence.
