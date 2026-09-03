"""Dataset adapters that never open dataset pose or mesh files."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from .contracts import FrameRecord, SequenceManifest


def scannet_manifest(scene: Path, *, frame_period_us: int = 33_333) -> SequenceManifest:
    scene = Path(scene).resolve()
    intrinsic = np.loadtxt(scene / "intrinsic" / "intrinsic_depth.txt")
    if intrinsic.shape != (4, 4):
        raise ValueError("ScanNet intrinsic_depth.txt must be 4x4")
    values = tuple(float(value) for value in (
        intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2],
    ))
    ids = sorted(int(path.stem) for path in (scene / "depth").glob("*.png"))
    frames = tuple(FrameRecord(
        frame_id=frame_id,
        timestamp_us=1_000_000 + ordinal * frame_period_us,
        color_path=scene / "color" / f"{frame_id}.jpg",
        depth_path=scene / "depth" / f"{frame_id}.png",
        intrinsics=values,
    ) for ordinal, frame_id in enumerate(ids))
    return SequenceManifest(
        dataset="scannet", sequence_id=scene.name, root=scene,
        depth_scale=1000.0, frames=frames, source="raw_rgbd_intrinsics_only",
    ).validate()


def _parse_3rscan_info(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            values[key.strip()] = value.strip()
    return values


def scan3r_manifest(
    sequence: Path, *, frame_period_us: int = 33_333, rotate_ccw: bool = True,
) -> SequenceManifest:
    sequence = Path(sequence).resolve()
    info = _parse_3rscan_info(sequence / "_info.txt")
    raw = [float(value) for value in info["m_calibrationDepthIntrinsic"].split()]
    if len(raw) != 16:
        raise ValueError("3RScan depth intrinsic must contain 16 values")
    intrinsics = (raw[0], raw[5], raw[2], raw[6])
    pattern = re.compile(r"frame-(\d{6})\.depth\.pgm$")
    ids = sorted(
        int(match.group(1)) for path in sequence.glob("frame-*.depth.pgm")
        if (match := pattern.match(path.name))
    )
    frames = tuple(FrameRecord(
        frame_id=frame_id,
        timestamp_us=1_000_000 + ordinal * frame_period_us,
        color_path=sequence / f"frame-{frame_id:06d}.color.jpg",
        depth_path=sequence / f"frame-{frame_id:06d}.depth.pgm",
        intrinsics=intrinsics,
        rotate_ccw=rotate_ccw,
    ) for ordinal, frame_id in enumerate(ids))
    return SequenceManifest(
        dataset="3rscan", sequence_id=sequence.parent.name, root=sequence,
        depth_scale=float(info.get("m_depthShift", "1000")), frames=frames,
        source="raw_rgbd_info_only",
    ).validate()


def orbbec_manifest(journal: Path) -> SequenceManifest:
    journal = Path(journal).resolve()
    root = journal.parent
    frames = []
    sequence_id = root.name
    for line_number, line in enumerate(journal.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("storage_status") != "complete":
            continue
        if row.get("admitted_to_fusion") is not True:
            continue
        camera = row["camera"]
        color = (root / row["color_path"]).resolve()
        depth = (root / row["depth_path"]).resolve()
        frames.append(FrameRecord(
            frame_id=int(row["frame_id"]),
            timestamp_us=int(row["color_timestamp_us"]),
            color_path=color,
            depth_path=depth,
            intrinsics=(float(camera["fx"]), float(camera["fy"]),
                        float(camera["cx"]), float(camera["cy"])),
        ))
    return SequenceManifest(
        dataset="orbbec", sequence_id=sequence_id, root=root,
        depth_scale=1000.0, frames=tuple(frames),
        source="frame_journal_complete_admitted_only",
    ).validate()
