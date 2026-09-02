#!/usr/bin/env python3
"""Offline, fail-closed preparation and verification for a SLAM-Former smoke.

This wrapper never clones, fetches, downloads a checkpoint, or invents poses.
It stages only RGB frames and intrinsics from an SGF-SGAligner no-GT manifest,
records an exact official command without invoking it during ``prepare``, and
validates the official ``final_traj.txt``/``final.ply`` artifacts afterwards.

Example (the command after ``--`` must match the pinned official checkout)::

    python scripts/run_slamformer_smoke.py prepare \
      --official-repo /opt/SLAM-Former \
      --checkpoint /models/V1.1-long-224.pth \
      --checkpoint-sha256 <64-hex-sha256> \
      --manifest /results/scene0030_00/frontend/tracked_manifest.json \
      --output-root /results/slamformer_scene0030_smoke_attempt_001 \
      --frame-limit 8 -- \
      python <official-entrypoint> --images '{input_root}/rgb' \
      --checkpoint '{checkpoint}' --output '{output_root}'

Run ``execute`` only in the separately installed official environment on an
eligible GPU host.  Preparation, execution and verification remain separate
so an environment or model failure cannot be mistaken for accepted output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence


OFFICIAL_REPOSITORY = "https://github.com/Tsinghua-MARS-Lab/SLAM-Former"
FROZEN_MODEL_COMMIT = "0071ca9e6c53aec55572a5557c5fcf3a23cdba5d"
FROZEN_MODEL_VARIANT = "V1.1-long@224"
EXPECTED_OUTPUTS = ("final_traj.txt", "final.ply")
FORBIDDEN_PARTS = frozenset({
    "pose", "poses", "gt", "ground_truth", "ground-truth", "evaluation",
    "evaluations", "mesh", "meshes",
})
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
DEPENDENCY_FILES = (
    "requirements.txt", "requirements-dev.txt", "environment.yml",
    "environment.yaml", "pyproject.toml", "setup.py", "setup.cfg",
)
MODULE_PROBES = ("torch", "torchvision", "numpy", "cv2", "scipy", "open3d", "yaml")


class SmokeContractError(RuntimeError):
    """Raised when a frozen/no-GT/create-only contract is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_create_only(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run_read_only(argv: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(
        list(argv), cwd=cwd, check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SmokeContractError(f"command failed ({' '.join(argv)}): {detail}")
    return result.stdout.strip()


def _audit_official_checkout(path: Path) -> dict[str, Any]:
    repository = path.resolve()
    if not (repository / ".git").exists():
        raise SmokeContractError(f"official repository is not a git checkout: {repository}")
    head = _run_read_only(("git", "rev-parse", "HEAD"), cwd=repository)
    if head != FROZEN_MODEL_COMMIT:
        raise SmokeContractError(
            f"SLAM-Former HEAD {head} does not match frozen commit {FROZEN_MODEL_COMMIT}"
        )
    dirty = _run_read_only(("git", "status", "--porcelain"), cwd=repository)
    if dirty:
        raise SmokeContractError("official SLAM-Former checkout must be clean")
    dependency_files = []
    for name in DEPENDENCY_FILES:
        candidate = repository / name
        if candidate.is_file():
            dependency_files.append({
                "path": str(candidate),
                "sha256": _sha256_file(candidate),
                "size_bytes": candidate.stat().st_size,
            })
    if not dependency_files:
        raise SmokeContractError(
            "official checkout has no recognized dependency declaration at repository root"
        )
    return {
        "repository": OFFICIAL_REPOSITORY,
        "checkout": str(repository),
        "commit": head,
        "clean": True,
        "dependency_declarations": dependency_files,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "module_probes": {
                name: importlib.util.find_spec(name) is not None for name in MODULE_PROBES
            },
        },
    }


def _path_forbidden(path: Path) -> set[str]:
    return {part.lower() for part in path.resolve().parts} & FORBIDDEN_PARTS


def _load_no_gt_manifest(
    path: Path,
    frame_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "rgbd_sequence_manifest.v1":
        raise SmokeContractError("input must use rgbd_sequence_manifest.v1")
    if payload.get("gt_at_inference") is not False:
        raise SmokeContractError("manifest must explicitly declare gt_at_inference=false")
    unsigned = dict(payload)
    expected_hash = unsigned.pop("payload_sha256", None)
    if expected_hash != _stable_json_sha256(unsigned):
        raise SmokeContractError("input manifest payload SHA-256 mismatch")
    if payload.get("sequence_id") != "scene0030_00":
        raise SmokeContractError("this smoke runner is restricted to scene0030_00")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise SmokeContractError("manifest must contain at least two frames")
    if frame_limit < 2 or frame_limit > len(frames):
        raise SmokeContractError(f"frame-limit must be in [2, {len(frames)}]")
    selected = frames[:frame_limit]
    seen_ids: set[int] = set()
    previous_timestamp = -1
    for row in selected:
        allowed = {
            "frame_id", "timestamp_us", "color_path", "depth_path",
            "intrinsics", "rotate_ccw",
        }
        unknown = set(row) - allowed
        if unknown:
            raise SmokeContractError(f"frame contains undeclared fields: {sorted(unknown)}")
        frame_id = int(row["frame_id"])
        timestamp = int(row["timestamp_us"])
        if frame_id in seen_ids or timestamp < previous_timestamp:
            raise SmokeContractError("frame ids must be unique and timestamps monotonic")
        intrinsics = row.get("intrinsics")
        if not isinstance(intrinsics, list) or len(intrinsics) != 4:
            raise SmokeContractError(f"frame {frame_id} needs fx,fy,cx,cy")
        if not all(math.isfinite(float(value)) for value in intrinsics):
            raise SmokeContractError(f"frame {frame_id} has non-finite intrinsics")
        if float(intrinsics[0]) <= 0 or float(intrinsics[1]) <= 0:
            raise SmokeContractError(f"frame {frame_id} has non-positive focal length")
        for kind in ("color_path", "depth_path"):
            source = Path(row[kind])
            forbidden = _path_forbidden(source)
            if forbidden:
                raise SmokeContractError(
                    f"frame {frame_id} {kind} contains forbidden GT part: {sorted(forbidden)}"
                )
            if not source.is_file():
                raise SmokeContractError(f"frame {frame_id} missing {kind}: {source}")
        seen_ids.add(frame_id)
        previous_timestamp = timestamp
    return payload, selected


def _stage_rgb_create_only(
    input_root: Path,
    frames: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rgb_root = input_root / "rgb"
    rgb_root.mkdir(parents=True, exist_ok=False)
    staged = []
    for ordinal, row in enumerate(frames):
        source = Path(row["color_path"]).resolve()
        suffix = source.suffix.lower() or ".jpg"
        # SLAM-Former parses the numeric image stem into final_traj.txt.  Use
        # the manifest frame ID rather than a window-local ordinal so a smoke
        # starting after frame zero remains traceable to the source sequence.
        destination = rgb_root / f"{int(row['frame_id']):06d}{suffix}"
        os.symlink(source, destination)
        staged.append({
            "ordinal": ordinal,
            "frame_id": int(row["frame_id"]),
            "timestamp_us": int(row["timestamp_us"]),
            "source_color_path": str(source),
            "staged_color_path": str(destination),
            "color_sha256": _sha256_file(source),
            "intrinsics": [float(value) for value in row["intrinsics"]],
            "rotate_ccw": bool(row.get("rotate_ccw", False)),
        })
    return staged


def _expand_command(command: Sequence[str], *, plan: dict[str, Any]) -> list[str]:
    substitutions = {
        "{input_root}": plan["paths"]["input_root"],
        "{output_root}": plan["paths"]["official_output_root"],
        "{checkpoint}": plan["checkpoint"]["path"],
    }
    expanded = []
    for argument in command:
        value = argument
        for token, replacement in substitutions.items():
            value = value.replace(token, replacement)
        expanded.append(value)
    return expanded


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if not HASH_RE.fullmatch(args.checkpoint_sha256):
        raise SmokeContractError("checkpoint-sha256 must be exactly 64 lowercase hex characters")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SmokeContractError(f"checkpoint not found: {checkpoint}")
    observed_checkpoint_hash = _sha256_file(checkpoint)
    if observed_checkpoint_hash != args.checkpoint_sha256:
        raise SmokeContractError(
            "checkpoint SHA mismatch: "
            f"expected {args.checkpoint_sha256}, observed {observed_checkpoint_hash}"
        )
    official = _audit_official_checkout(args.official_repo)
    manifest, selected = _load_no_gt_manifest(args.manifest, args.frame_limit)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    input_root = output_root / "input_no_gt"
    input_root.mkdir(exist_ok=False)
    staged = _stage_rgb_create_only(input_root, selected)
    official_output_root = output_root / "official_output"
    command = list(args.official_command)
    if command and command[0] == "--":
        command = command[1:]
    plan = {
        "schema": "slamformer_scene0030_smoke_plan.v1",
        "status": "prepared_not_executed",
        "model": {
            "provider": "SLAM-Former",
            "variant": FROZEN_MODEL_VARIANT,
            "official": official,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": observed_checkpoint_hash,
        },
        "input": {
            "source_manifest": str(args.manifest.resolve()),
            "source_manifest_payload_sha256": manifest["payload_sha256"],
            "sequence_id": "scene0030_00",
            "frame_count": len(staged),
            "frame_ids": [row["frame_id"] for row in staged],
            "timestamp_us": [row["timestamp_us"] for row in staged],
            "gt_at_inference": False,
            "staged_rgb_sha256": [row["color_sha256"] for row in staged],
        },
        "paths": {
            "attempt_root": str(output_root),
            "input_root": str(input_root),
            "official_output_root": str(official_output_root),
        },
        "official_command_template": command,
        "expected_outputs": list(EXPECTED_OUTPUTS),
        "output_policy": "create_only",
        "notes": [
            "prepare performs no inference and makes no network request",
            "only RGB symlinks and intrinsics/frame mapping are staged",
            "final_traj.txt is sparse/keyframe output, not a complete per-frame trajectory",
            "final.ply is visual geometry evidence, not SGF-SGAligner refusion acceptance",
        ],
    }
    plan["expanded_official_command"] = _expand_command(command, plan=plan) if command else []
    plan["payload_sha256"] = _stable_json_sha256(plan)
    _write_json_create_only(input_root / "frames.json", staged)
    _write_json_create_only(output_root / "run_plan.json", plan)
    return plan


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    expected = plan.pop("payload_sha256", None)
    if plan.get("schema") != "slamformer_scene0030_smoke_plan.v1":
        raise SmokeContractError("run plan schema mismatch")
    if expected != _stable_json_sha256(plan):
        raise SmokeContractError("run plan payload SHA-256 mismatch")
    plan["payload_sha256"] = expected
    return plan


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args.plan)
    command = plan.get("expanded_official_command", [])
    if not command:
        raise SmokeContractError(
            "plan has no official command; prepare again with a command after --"
        )
    official = _audit_official_checkout(Path(plan["model"]["official"]["checkout"]))
    if official["commit"] != plan["model"]["official"]["commit"]:
        raise SmokeContractError("official checkout changed after preparation")
    checkpoint = Path(plan["checkpoint"]["path"])
    if _sha256_file(checkpoint) != plan["checkpoint"]["sha256"]:
        raise SmokeContractError("checkpoint changed after preparation")
    output_root = Path(plan["paths"]["official_output_root"])
    output_root.mkdir(parents=False, exist_ok=False)
    environment = os.environ.copy()
    environment.update({
        "SLAMFORMER_INPUT_ROOT": plan["paths"]["input_root"],
        "SLAMFORMER_OUTPUT_ROOT": str(output_root),
        "SLAMFORMER_CHECKPOINT": str(checkpoint),
        "SGF_GT_AT_INFERENCE": "false",
    })
    completed = subprocess.run(
        command,
        cwd=Path(plan["model"]["official"]["checkout"]),
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise SmokeContractError(f"official command failed with exit code {completed.returncode}")
    missing = [name for name in EXPECTED_OUTPUTS if not (output_root / name).is_file()]
    if missing:
        raise SmokeContractError(f"official command omitted required outputs: {missing}")
    return {"status": "executed_unverified", "output_root": str(output_root)}


def _trajectory_contract(path: Path, plan: dict[str, Any], identifier_mode: str) -> dict[str, Any]:
    rows: list[list[float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) != 8:
            raise SmokeContractError(f"final_traj.txt line {line_number} must have 8 fields")
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise SmokeContractError(f"final_traj.txt line {line_number} is not numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise SmokeContractError(f"final_traj.txt line {line_number} is non-finite")
        quaternion_norm = math.sqrt(sum(value * value for value in values[4:8]))
        if quaternion_norm <= 1e-12:
            raise SmokeContractError(f"final_traj.txt line {line_number} has zero quaternion")
        rows.append(values)
    if len(rows) < 2:
        raise SmokeContractError("final_traj.txt needs at least two keyframe rows")
    identifiers = [row[0] for row in rows]
    if identifiers != sorted(set(identifiers)):
        raise SmokeContractError("final_traj.txt identifiers must be unique and increasing")
    allowed: set[float]
    if identifier_mode == "frame_id":
        allowed = {float(value) for value in plan["input"]["frame_ids"]}
    elif identifier_mode == "timestamp_us":
        allowed = {float(value) for value in plan["input"]["timestamp_us"]}
    else:
        allowed = {float(value) / 1_000_000.0 for value in plan["input"]["timestamp_us"]}
    identifiers_admitted = all(
        any(abs(identifier - value) < 1e-5 for value in allowed)
        for identifier in identifiers
    )
    if not identifiers_admitted:
        raise SmokeContractError(
            f"final_traj.txt contains identifiers outside the admitted {identifier_mode} set"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "row_count": len(rows),
        "identifier_mode": identifier_mode,
        "first_identifier": identifiers[0],
        "last_identifier": identifiers[-1],
        "finite": True,
        "zero_quaternion": False,
    }


def _ply_contract(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header_bytes = bytearray()
        for _ in range(10000):
            line = stream.readline()
            if not line:
                break
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
    if not header_bytes.startswith(b"ply\n") and not header_bytes.startswith(b"ply\r\n"):
        raise SmokeContractError("final.ply does not start with a PLY magic header")
    if b"end_header" not in header_bytes:
        raise SmokeContractError("final.ply has no end_header")
    try:
        header = header_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SmokeContractError("final.ply header is not ASCII") from exc
    format_match = re.search(
        r"^format\s+(ascii|binary_little_endian|binary_big_endian)\s+1\.0$",
        header,
        re.M,
    )
    vertex_match = re.search(r"^element\s+vertex\s+(\d+)$", header, re.M)
    if not format_match or not vertex_match:
        raise SmokeContractError("final.ply needs format 1.0 and an element vertex declaration")
    vertex_count = int(vertex_match.group(1))
    if vertex_count <= 0:
        raise SmokeContractError("final.ply has no vertices")
    for coordinate in ("x", "y", "z"):
        if not re.search(rf"^property\s+\S+\s+{coordinate}$", header, re.M):
            raise SmokeContractError(f"final.ply is missing vertex property {coordinate}")
    if path.stat().st_size <= len(header_bytes):
        raise SmokeContractError("final.ply has a header but no vertex payload")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "format": format_match.group(1),
        "vertex_count": vertex_count,
        "xyz_properties_present": True,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_plan(args.plan)
    output_root = Path(plan["paths"]["official_output_root"])
    trajectory = output_root / "final_traj.txt"
    cloud = output_root / "final.ply"
    if not trajectory.is_file() or not cloud.is_file():
        raise SmokeContractError("official_output must contain final_traj.txt and final.ply")
    receipt = {
        "schema": "slamformer_scene0030_smoke_verification.v1",
        "status": "official_artifacts_contract_valid",
        "plan_payload_sha256": plan["payload_sha256"],
        "model_commit": FROZEN_MODEL_COMMIT,
        "model_variant": FROZEN_MODEL_VARIANT,
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
        "gt_consumed": False,
        "trajectory": _trajectory_contract(trajectory, plan, args.identifier_mode),
        "geometry": _ply_contract(cloud),
        "limitations": [
            "keyframe trajectory coverage is not complete per-frame T_world_camera coverage",
            "PLY structure validation is not geometric quality or refusion acceptance",
            "promotion still requires scale evidence, DPV-anchor revision, "
            "complete refusion, and fixed-view QA",
        ],
    }
    receipt["payload_sha256"] = _stable_json_sha256(receipt)
    receipt_path = output_root.parent / "verification.json"
    _write_json_create_only(receipt_path, receipt)
    return {"verification": str(receipt_path), **receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="offline audit and create-only staging")
    prepare_parser.add_argument("--official-repo", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint-sha256", required=True)
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--frame-limit", type=int, default=8)
    prepare_parser.add_argument("official_command", nargs=argparse.REMAINDER)
    prepare_parser.set_defaults(handler=prepare)

    execute_parser = subparsers.add_parser("execute", help="run the recorded official command")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.set_defaults(handler=execute)

    verify_parser = subparsers.add_parser("verify", help="validate official outputs create-only")
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument(
        "--identifier-mode", choices=("frame_id", "timestamp_us", "timestamp_s"),
        default="frame_id",
    )
    verify_parser.set_defaults(handler=verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (OSError, ValueError, SmokeContractError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
