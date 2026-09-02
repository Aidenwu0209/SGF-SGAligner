# Remote Orin data integrity record

Remote create-only root:

`/home/ai3d/Documents/sgf_sga_model_validation_20260902`

The ScanNet inference root contains RGB, depth and intrinsics only. Ground-truth
poses were physically moved to the sibling `eval_only/` tree before model
execution. The Orbbec copy is a dereferenced regular-file copy of
`sgf_parameter_control`; it contains no symlinks.

| Payload | Files | Source and remote tree SHA-256 |
| --- | ---: | --- |
| ScanNet `scene0030_00/color` | 2,498 | `a9d7b1c5cd62bab55acc7ce4fe6b606a3a0655bdd5ee02c90a0e6d07c5ca7e9f` |
| ScanNet `scene0030_00/depth` | 2,498 | `3c40359dc0ea6d84826dee29e0f0f95e0eab7ecd9d1fa328df7e9f54beee28e4` |
| ScanNet `scene0030_00/intrinsic` | 4 | `396fd9e684cb8d9b69a6af71950c398b3d88f0e487cb4000957cbef8813e0a25` |
| ScanNet evaluation-only pose | 2,498 | `fb5a7c9b3e16e56ecd8e2108adc6edae19fdbf0083c1b69dec44a8ab0af37c47` |
| Orbbec `sgf_parameter_control` | 3,048 | `e428673c9bb58d1bd6e3710579828776449f9396a35f8cbb35979d8b9d35eec7` |

The persistent remote test root occupied approximately 1.3 GiB after transfer
and source checkout. The eMMC root had approximately 2.9 GiB free, so the
4,914,062,480-byte MapAnything checkpoint was downloaded to `/dev/shm` only
and was cleared by the subsequent host reboot. No model checkpoint was written
to the eMMC test root.
