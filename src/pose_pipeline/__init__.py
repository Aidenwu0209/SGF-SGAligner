"""GT-isolated RGB-D pose backend for SGF-SGAligner."""

from .contracts import (
    MANIFEST_SCHEMA,
    TRAJECTORY_SCHEMA,
    FrameRecord,
    ImuSample,
    PoseRecord,
    SequenceManifest,
    load_manifest,
    load_legacy_tcw_mm,
    load_trajectory,
    write_manifest,
    write_trajectory,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "FrameRecord",
    "ImuSample",
    "PoseRecord",
    "SequenceManifest",
    "load_manifest",
    "load_legacy_tcw_mm",
    "load_trajectory",
    "write_manifest",
    "write_trajectory",
]
