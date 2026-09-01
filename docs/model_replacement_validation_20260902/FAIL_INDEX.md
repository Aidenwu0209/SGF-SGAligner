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
