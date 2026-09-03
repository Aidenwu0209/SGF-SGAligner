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
| DPV-conditioned, selective `info_sharing` BF16 storage | 8 | 1.8469 s | 1.8469 s | 4.33 | 6.488 / 6.963 GB |

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

The selective-storage row is an opt-in low-memory diagnostic, not a replacement
for the full-weight timing rows.  On the same RTX 5070 Ti it reduced peak
allocated memory by 1.853 GB (22.2%) and peak reserved memory by 1.797 GB.  A
same-input tensor comparison against the full-weight conditioned output found
mean camera translation and rotation deltas of 0.433 mm and 0.0184 degrees.
Within the valid 0--4.5 m TSDF depth range, mean depth and 3D-point deltas were
0.317 mm and 0.655 mm respectively (95th percentile 1.196 mm and 1.924 mm).

Compared with the matched Orin 30 W measurements, the RTX 5070 Ti is 5.37x
faster for independent 8-frame inference, 5.10x faster for DPV-conditioned
8-frame inference, and 5.50x faster for independent 16-frame inference.

## RTX 4060 Laptop 8 GB

The host reports 8,188 MiB VRAM, while PyTorch exposes 7.60 GiB usable capacity.
It used PyTorch `2.5.1+cu121`.  The official full-weight configuration failed
during the model forward pass:

- default allocator: OOM while requesting another 130 MiB;
- `expandable_segments:True`: OOM while requesting another 66 MiB;
- adapted `504 x 378` diagnostic: still OOM while requesting another 62 MiB;
- adapted six-frame `518 x 392` diagnostic: still OOM while requesting another
  50 MiB.

A whole-model BF16-weight diagnostic was not valid: it reached a mixed-dtype
assignment in MapAnything's geometric-input path and failed closed.  Resolution
diagnostics down to `448 x 336` also remained OOM because stored weights, not
only activation resolution, dominate this boundary.

An opt-in selective cast resolves the capacity failure: only the
`model.info_sharing` module is stored in BF16 after CUDA loading.  This module
already executes inside the official BF16 autocast path; the geometric input
path and all other stored weights remain FP32.

| Mode | Frames | Inference samples | Median | Input frames/s | Peak allocated / reserved |
| --- | ---: | --- | ---: | ---: | ---: |
| independent, selective `info_sharing` BF16 | 8 | 3.9750, 3.4879 s | 3.7314 s | 2.14 | 6.438 / 6.604 GB |
| DPV-conditioned, selective `info_sharing` BF16 | 8 | 3.5804, 3.5431 s | 3.5618 s | 2.25 | 6.489 / 7.032 GB |

The two independent outputs were byte-identical.  The conditioned run retained
8/8 poses, consumed no GT, and completed full TSDF refusion with all eight
frames and no identity fallback.  Its metric SE(3) ATE was 5.495 mm versus
5.490 mm for the full-weight reference run.  After the same first-frame
ScanNet-world alignment, its mean surface error against the GT-pose refusion
was 10.327 mm versus 10.325 mm for full weights (+0.021%); F-score was 0.986709
versus 0.986876.  The low-memory and full-weight refusions differ by only
0.868 mm symmetric mean surface distance.

The runner exposes this as
`--model-storage-dtype info-sharing-bf16`; omitting the option preserves FP32
storage.  The correct conclusion is therefore: the official FP32-storage arm still fails
the 8 GB capacity gate, but the explicit `info-sharing-bf16` storage arm runs at
the official input resolution without a material short-window quality loss.
It remains opt-in and must not silently replace the default arm.  Fresh-process
model loading took about 22.7 s, so an enabled implementation must keep the
model resident rather than reload it for every window.

## Online interpretation

The intended DPV-conditioned 8-frame refinement takes about 1.61 s per window
on the RTX 5070 Ti.  If capture continues at 15 FPS, a single serialized worker
must submit no more often than about every 25 new frames to avoid queue growth;
at 30 FPS the corresponding interval is about 49 frames.  Running refinement
synchronously on the capture thread would visibly block it for roughly 1.6 s.

This result supports an opt-in quasi-online worker or a pause/end post-process,
not per-frame MapAnything refinement.  The 4060 can execute an 8-frame window,
but 3.58 s per conditioned window is still too slow for synchronous capture.
Combined DPV plus MapAnything latency and peak memory on one GPU remain
unmeasured and must be profiled before enabling concurrent inference.  A
16-frame 8 GB run and longer multi-window scene gate also remain outstanding.

Raw create-only artifacts remain on the test hosts at:

- RTX 5070 Ti: `/home/aidenwu/Documents/mapanything-benchmark-20260903`
- RTX 4060 Laptop: `/home/aidenwu/Documents/mapanything-benchmark-20260903`
