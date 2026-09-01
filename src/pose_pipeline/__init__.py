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
from .model_contracts import (
    EXTERNAL_ARTIFACT_SCHEMA,
    MODEL_RUNTIME_SCHEMA,
    SPARSE_PROPOSAL_SCHEMA,
    TRAJECTORY_REVISION_SCHEMA,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "MODEL_RUNTIME_SCHEMA",
    "TRAJECTORY_REVISION_SCHEMA",
    "SPARSE_PROPOSAL_SCHEMA",
    "EXTERNAL_ARTIFACT_SCHEMA",
    "FrameRecord",
    "PoseRecord",
    "SequenceManifest",
    "load_manifest",
    "load_legacy_tcw_mm",
    "load_trajectory",
    "write_manifest",
    "write_trajectory",
]
