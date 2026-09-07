"""Licensed, optional RGB appearance descriptors for loop proposal only.

The descriptor is never allowed to create a pose.  It only expands the fixed
proposal budget; the GT-free geometric backend remains authoritative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import FrameRecord, PoseRecord, sha256_file, stable_json_sha256


@dataclass(frozen=True)
class AppearanceConfig:
    provider: str = "none"
    model_name: str = "ViT-B/32"
    device: str = "auto"
    download_root: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in {"none", "openai_clip"}:
            raise ValueError("appearance provider must be none or openai_clip")
        if self.provider == "openai_clip" and not self.download_root:
            raise ValueError("openai_clip requires an explicit download_root")


def _checkpoint_path(clip_module, model_name: str, download_root: Path) -> Path | None:
    url = getattr(clip_module, "_MODELS", {}).get(model_name)
    if not url:
        return None
    return download_root / url.rsplit("/", 1)[-1]


def extract_anchor_descriptors(
    bound: Sequence[tuple[FrameRecord, PoseRecord]],
    anchors: Sequence[int],
    config: AppearanceConfig,
    output_path: Path,
) -> tuple[np.ndarray, dict]:
    if config.provider != "openai_clip":
        raise ValueError("descriptor extraction requires openai_clip")
    try:
        import clip
        import cv2
        from PIL import Image
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "openai_clip appearance runtime is unavailable"
        ) from exc
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    download_root = Path(str(config.download_root)).resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    model, preprocess = clip.load(
        config.model_name, device=device, jit=False,
        download_root=str(download_root),
    )
    model.eval()
    tensors = []
    frame_ids = []
    input_hashes = []
    for ordinal in anchors:
        frame = bound[int(ordinal)][0]
        image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"missing appearance frame {frame.frame_id}")
        if frame.rotate_ccw:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensors.append(preprocess(Image.fromarray(rgb)))
        frame_ids.append(int(frame.frame_id))
        input_hashes.append(sha256_file(frame.color_path))
    with torch.no_grad():
        batch = torch.stack(tensors).to(device)
        encoded = model.encode_image(batch).float()
        encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    descriptors = np.ascontiguousarray(encoded.cpu().numpy(), dtype=np.float32)
    checkpoint = _checkpoint_path(clip, config.model_name, download_root)
    checkpoint_sha = (
        sha256_file(checkpoint) if checkpoint is not None and checkpoint.is_file()
        else None
    )
    metadata = {
        "schema": "appearance_descriptor_cache.v1",
        "provider": config.provider,
        "model_name": config.model_name,
        "device": device,
        "anchor_frame_ids": frame_ids,
        "input_color_sha256": input_hashes,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "config": asdict(config),
        "config_sha256": stable_json_sha256(asdict(config)),
        "descriptor_sha256": hashlib.sha256(descriptors.tobytes()).hexdigest(),
        "gt_consumed": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            descriptors=descriptors,
            frame_ids=np.asarray(frame_ids, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(
                metadata, sort_keys=True, separators=(",", ":"),
            )),
        )
    return descriptors, metadata
