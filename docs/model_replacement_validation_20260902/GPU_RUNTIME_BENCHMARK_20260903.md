# MapAnything RTX 5070 Ti and RTX 4060 runtime benchmark

Date: 2026-09-03

This benchmark uses the existing `scene0030_00` meaningful-motion window and
does not expose ScanNet ground-truth poses to inference.  The frozen inputs are
frames `2334..2349`; the 8-frame runs use `2334..2341`.  All comparable runs
use MapAnything commit `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`,
checkpoint SHA-256
`981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0`,
official `518 x 392` preprocessing, BF16 autocast,
`memory_efficient_inference=True`, and minibatch size 1.

## RTX 5070 Ti 16 GB

The host used PyTorch `2.7.1+cu128` under WSL2.  Runtime JSON measures model
loading and CUDA-synchronized inference separately.

| Mode | Frames | Inference samples | Steady median | Input frames/s | Peak allocated / reserved |
| --- | ---: | --- | ---: | ---: | ---: |
| independent RGB + intrinsics + metric depth | 8 | 1.8655, 1.5531, 1.4347 s | 1.4939 s | 5.36 | 8.341 / 8.760 GB |
| DPV-conditioned refinement | 8 | 1.8922, 1.6221, 1.6043 s | 1.6132 s | 4.96 | 8.341 / 8.760 GB |
| independent RGB + intrinsics + metric depth | 16 | 3.0158, 2.8894 s | 2.9526 s | 5.42 | 8.998 / 9.341 GB |

The steady 8-frame median excludes the first run after environment and model
cache setup.  The independent 8-frame output SHA-256 was identical across all
three runs (`fe5e5ea5cf8961158672834da56fc8b4210f86869b92ce0e53253d63a439b486`).
The conditioned output was also identical across all three runs
(`9288a77f9a331b85d790b1d77ec8b204033cc89971a60aeed48465554e8e5f86`).
The two 16-frame outputs shared SHA-256
`f4a89d9ba4a5749928438d08a311cdb0b30de19d0a396a88be89d1e58b39bb58`.

The first process reported 171.74 s model load because it also populated the
DINOv2 code cache.  Later fresh processes loaded the model in 11.76--12.87 s.
This load time is not included in inference time.  A production refinement
worker should keep the model resident while the option is enabled; this
benchmark did not yet measure multiple windows inside one persistent process.

Compared with the matched Orin 30 W measurements, the RTX 5070 Ti is 5.37x
faster for independent 8-frame inference, 5.10x faster for DPV-conditioned
8-frame inference, and 5.50x faster for independent 16-frame inference.

## RTX 4060 Laptop 8 GB

The host reports 8,188 MiB VRAM, while PyTorch exposes 7.60 GiB usable capacity.
It used PyTorch `2.5.1+cu121`.  No valid 8-frame runtime exists because the
official configuration failed during the model forward pass:

- default allocator: OOM while requesting another 130 MiB;
- `expandable_segments:True`: OOM while requesting another 66 MiB;
- adapted `504 x 378` diagnostic: still OOM while requesting another 62 MiB;
- adapted six-frame `518 x 392` diagnostic: still OOM while requesting another
  50 MiB.

A BF16-weight diagnostic was not a valid speed result: it reached a mixed-dtype
assignment in MapAnything's geometric-input path and failed closed.  These
adapted diagnostics are not part of the official-quality comparison.

The correct conclusion for this 8 GB RTX 4060 is therefore **capacity failure,
not a slower successful run**.  Do not report an estimated FPS for it and do
not silently lower resolution or window size in the production arm.

## Online interpretation

The intended DPV-conditioned 8-frame refinement takes about 1.61 s per window
on the RTX 5070 Ti.  If capture continues at 15 FPS, a single serialized worker
must submit no more often than about every 25 new frames to avoid queue growth;
at 30 FPS the corresponding interval is about 49 frames.  Running refinement
synchronously on the capture thread would visibly block it for roughly 1.6 s.

This result supports an opt-in quasi-online worker or a pause/end post-process,
not per-frame MapAnything refinement.  Combined DPV plus MapAnything latency
and peak memory on one GPU remain unmeasured and must be profiled before
enabling concurrent inference.

Raw create-only artifacts remain on the test hosts at:

- RTX 5070 Ti: `/home/aidenwu/Documents/mapanything-benchmark-20260903`
- RTX 4060 Laptop: `/home/aidenwu/Documents/mapanything-benchmark-20260903`
