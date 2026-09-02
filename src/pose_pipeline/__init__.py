"""GT-isolated RGB-D pose backend for SGF-SGAligner."""

from .contracts import (
    MANIFEST_SCHEMA,
    TRAJECTORY_SCHEMA,
    FrameRecord,
    PoseRecord,
    SequenceManifest,
    load_manifest,
    load_legacy_tcw_mm,
    load_trajectory,
    write_manifest,
    write_trajectory,
)
from .depth_filter import (
    DEPTH_FILTER_PROFILES,
    DepthFilterAccumulator,
    DepthFilterConfig,
    DepthFilterStats,
    apply_depth_filter,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "FrameRecord",
    "PoseRecord",
    "SequenceManifest",
    "load_manifest",
    "load_legacy_tcw_mm",
    "load_trajectory",
    "write_manifest",
    "write_trajectory",
    "DEPTH_FILTER_PROFILES",
    "DepthFilterAccumulator",
    "DepthFilterConfig",
    "DepthFilterStats",
    "apply_depth_filter",
]
