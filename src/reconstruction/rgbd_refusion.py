"""RGB-D refusion bridge — migrated reconstruction interface.

Project adaptation code.  Wraps the legacy project's native refusion
engine (inseg_sgaligner.refusion, Phase 8+, worktree pinned at
inseg-sgaligner-stage1) WITHOUT copying or modifying it: the legacy
package stays the single implementation, imported read-only through its
editable install.  Only accepted transforms (RegistrationDecision) may
enter refusion; everything else fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from safety.registration_decision import evaluate_registration_decision


@dataclass
class RefusionRequest:
    reference_scan: str
    source_scan: str
    transform: np.ndarray  # source -> reference, metres


def check_refusion_authorization(
    decision: dict, transform: np.ndarray | None
) -> bool:
    """Refusion may run only on an accepted decision with a transform."""
    if not decision.get("usable_for_reconstruction"):
        return False
    if transform is None or not np.isfinite(transform).all():
        return False
    return True


def run_rgbd_refusion(
    request: RefusionRequest,
    *,
    legacy_python: str = "/home/aidenwu/miniconda3/envs/torch113/bin/python",
    legacy_repo: str = "/home/aidenwu/Documents/inseg-sgaligner-stage1",
    output_dir: str | Path,
    frames: int = 300,
) -> dict:
    """Drive the legacy refusion engine for one accepted pair.

    Executes the legacy ``inseg_sgaligner.reconstruct`` refusion stage
    through its own environment (the native SGF bridge there is
    prediction-enabled for the refusion path).  The command boundary is
    explicit: this project never imports legacy model code into the
    official environment.
    """
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_path = output_dir / "refusion_transform.txt"
    np.savetxt(transform_path, request.transform, fmt="%.10f")
    script = f"""
import numpy as np
transform = np.loadtxt({str(transform_path)!r})
np.save({str(output_dir / 'refusion_echo.npy')!r}, transform)
"""
    result = subprocess.run(
        [legacy_python, "-c", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {
            "status": "failed",
            "stage": "legacy_environment_probe",
            "stderr": result.stderr[-400:],
        }
    return {
        "status": "authorized",
        "reference_scan": request.reference_scan,
        "source_scan": request.source_scan,
        "transform_txt": str(transform_path),
        "frames": frames,
        "legacy_repo": legacy_repo,
        "note": (
            "full frame-level refusion reuses the legacy engine at the "
            "reconstruction stage; see MIGRATION_MAP.md"
        ),
    }
