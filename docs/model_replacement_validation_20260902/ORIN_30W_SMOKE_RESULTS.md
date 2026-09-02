# Jetson Orin 30 W model smoke results

The follow-up meaningful-motion evaluation is documented in
[`ORIN_30W_MOTION16_RESULTS.md`](ORIN_30W_MOTION16_RESULTS.md). It supersedes
the first-eight-frame prefix for accuracy interpretation, while this document
remains the original compatibility-smoke record.

Remote create-only root:

`/home/ai3d/Documents/sgf_sga_model_validation_20260902`

All inference used the sealed `scene0030_00` frames 0 through 7.  The inference
tree exposed RGB, raw depth and intrinsics only; evaluation poses remained in a
physically separate `eval_only` tree.  No model consumed GT and no missing pose
was replaced by an identity.

| Candidate | Frozen source | Result | Coverage | Important runtime evidence |
| --- | --- | --- | ---: | --- |
| MapAnything, independent RGB-D window | `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | PASS | 8/8 | 8.776 s inference; 8.32 GB peak allocated |
| ABot-Recon, official no-loop SDPA | `195cb9240ffc6300e008d2b70e54d281dd7caf4b` | PASS | 8/8 | about 7.7 s official run; 1.04 input frames/s |
| SLAM-Former, V1.1-long at 224 | `0071ca9e6c53aec55572a5557c5fcf3a23cdba5d` | PASS with `kf_th=0` | 8/8 keyframes | 8.239 s internal total; 0.971 input frames/s |

## MapAnything

The original checkpoint is Hugging Face snapshot
`a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa`, SHA-256
`981f060c64664dff3272b5f5a823d350abe71a2f144444db4cfc325f3ed5a3a0`.
The successful invocation used the default CUDA allocator, original FP32
weights on CUDA, official BF16 autocast, `memory_efficient_inference=True`,
minibatch 1 and official 518 x 392 preprocessing.  It accepted RGB,
intrinsics and metric depth and ignored input poses.

The result contains poses `[8,4,4]`, depth `[8,392,518,1]`, confidence
`[8,392,518]`, points `[8,392,518,3]` and metric scale `[8,1]`.  Its output
SHA-256 is
`0136f2510a2695c2060080a343f7ca81e457bbe075f03cd2c02329dc0077bbb9`.
Model load took 34.656 s, inference 8.776 s and wall time 51.620 s.  Peak CUDA
allocated/reserved memory was 8,318,545,408 / 8,797,552,640 bytes.

## ABot-Recon

The official checkpoint snapshot is
`c69c26ca1853afc9c9212014459e729d7339ecdf`, SHA-256
`ea41a7659f6087069e6b3aac8830cc1c62d7c4a5c27a7d2679b51ba97cabcd2e`.
The successful run used official no-loop mode and SDPA.  `max_frames=64` is a
model/positional-encoding capacity setting, not an input-frame count; setting
it to 8 failed closed before output.

Eight finite model-space poses and all eight dense maps were produced.  The
depth-backed scale was `2.8006115587` metres/model-unit.  The adapted metric
trajectory contains all frame IDs 0 through 7 and passed the
`pose_trajectory.v1` loader.

## SLAM-Former

The V1.1-long checkpoint SHA-256 is
`5375d5cfdf2423327d71bbd38351f1219a66c68e0a1fce34ee63b058d28ccfd1`.
The official default `kf_th=0.1` processed all eight images without OOM but
selected only one keyframe on this almost-static prefix; the official
termination path then failed closed because its map was still `None`.  A new
create-only attempt used `kf_th=0.0`, preserving every observed frame as a
real keyframe rather than fabricating non-keyframe poses.

That attempt produced 8/8 finite TUM trajectory rows, a 255,897-vertex PLY and
eight framewise point maps.  The trajectory and PLY SHA-256 values are
`f55353cb048740750272c054ca92393f8a648d8ddfb6417c444df7edcb3fd109`
and `dcd926673813a9aff03cd24117003c75968b33a4a2064e2f9e50249b5c15f595`.
The create-only verification payload SHA-256 is
`0fc78e09ad71d3ff2f2c4507c97f90a98fd14ba828d560e44a75c93b1ad2f9e3`.

SLAM-Former reported 8.239 s internal total time and 0.971 input frames/s;
the tegrastats span including process/model load was 38 s.  System RAM rose by
about 5.5 GB (9,447 to 14,942 MB used) and the host remained stable in 30 W
mode.  Metric RGB-D scale recovery paired 218,593 valid samples, estimated
`2.5205524262` metres/model-unit and had a 5th-to-95th relative spread of
`0.0718177923`.  The evidence SHA-256 is
`075444614d6eb0e9d901d187f6f518259c66ec8053a41884236b3d8c6efafd17`.

This Orin run is a compatibility port, not the repository's exact declared
software environment: it used NVIDIA Jetson Torch 2.5.0/CUDA 12.6 and NumPy
1.26.3 instead of declared Torch 2.9.1 and NumPy 2.2.6, and the CUDA RoPE2D
extension was unavailable so the official PyTorch fallback ran.  Those
deviations must stay visible in any comparison.

## Interpretation boundary

These passes answer whether the original weights can execute on this Orin and
cross the SGF-SGAligner data-contract boundary.  They do not yet show that
either model is more accurate than DPV-SLAM.  Promotion still requires a
longer sealed scene0030 development run, metric ATE/RPE evaluated only after
inference, complete TSDF refusion and fixed-view geometry QA.

As a post-inference check, the physically isolated ScanNet poses were exposed
only to the evaluator.  On this prefix MapAnything obtained metric relative
translation/rotation RMSE of `0.0030078 m / 0.08913 deg`; ABot-Recon obtained
`0.0038138 m / 0.09452 deg`; SLAM-Former with RGB-D scale obtained
`0.0058951 m / 0.09728 deg`.  These numbers are useful contract diagnostics,
not a quality ranking: the eight GT positions span only `0.01011 m`, making
global alignment and scale estimates ill-conditioned.  In particular, the
large aligned absolute rotation figures from this prefix must not be used to
reject any candidate.  The next accuracy run needs a sealed interval with
meaningful translation and rotation.

The multi-GB checkpoints were kept in `/dev/shm`; persistent eMMC received
only small source trees, manifests, logs and output evidence.  A session-created
pip-cache burst was removed by exact timestamped-file selection, returning the
cache to its pre-run baseline and leaving about 2.1 GB free on `/`.
