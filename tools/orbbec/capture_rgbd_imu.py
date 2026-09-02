#!/usr/bin/env python3
"""Capture a synchronized, D2C-aligned Orbbec RGB-D image sequence on macOS."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBAccelFullScaleRange,
    OBAccelSampleRate,
    OBFrameAggregateOutputMode,
    OBGyroFullScaleRange,
    OBGyroSampleRate,
    OBSensorType,
    OBStreamType,
    Pipeline,
)


STOP_REQUESTED = False


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def finish_without_macos_sdk_teardown(status: int) -> None:
    """Avoid the known macOS SDK stop/destructor delay after files are sealed."""
    if sys.platform != "darwin":
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save synchronized color PNG, uint16-mm depth PNG, timestamps, "
            "intrinsics, and a completion manifest. Press Ctrl+C to save and stop."
        )
    )
    parser.add_argument("label", nargs="?", default="scan")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "Documents" / "orbbec_dataset",
    )
    parser.add_argument("--duration", type=int, default=86400)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.label or any(not (c.isalnum() or c in "_-") for c in args.label):
        raise SystemExit("label may contain only letters, numbers, underscore, and dash")
    if min(args.duration, args.width, args.height, args.fps) <= 0:
        raise SystemExit("duration, width, height, and fps must be positive")


def make_scan_dir(root: Path, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}"
        suffix += 1
    (candidate / "color").mkdir(parents=True)
    (candidate / "depth").mkdir()
    return candidate


def profile_description(profile: Any) -> Dict[str, Any]:
    return {
        "width": int(profile.get_width()),
        "height": int(profile.get_height()),
        "fps": int(profile.get_fps()),
        "format": str(profile.get_format()),
    }


def select_video_profile(
    pipeline: Pipeline,
    sensor: OBSensorType,
    width: int,
    height: int,
    fps: int,
    formats: Iterable[OBFormat],
) -> Any:
    profiles = pipeline.get_stream_profile_list(sensor)
    for frame_format in formats:
        try:
            return profiles.get_video_stream_profile(width, height, frame_format, fps)
        except Exception:
            pass

    # Some Femto Mega macOS profile tables do not expose the requested format
    # through get_video_stream_profile even when a nearby, usable profile is
    # available. Inspect the full table and choose the closest usable entry
    # instead of silently falling back to the 1920x1080 default.
    allowed_formats = set(formats)
    candidates = []
    for index in range(int(profiles.get_count())):
        profile = profiles.get_stream_profile_by_index(index)
        try:
            profile = profile.as_video_stream_profile()
        except Exception:
            pass
        try:
            if profile.get_format() not in allowed_formats:
                continue
            profile_width = int(profile.get_width())
            profile_height = int(profile.get_height())
            profile_fps = int(profile.get_fps())
        except Exception:
            continue
        score = (
            100000 * abs(profile_fps - fps)
            + 1000 * abs(profile_width - width)
            + 1000 * abs(profile_height - height)
            + abs(profile_width * profile_height - width * height)
        )
        candidates.append((score, index, profile))
    if candidates:
        return min(candidates, key=lambda item: (item[0], item[1]))[2]
    return profiles.get_default_video_stream_profile()


def color_to_bgr(frame: Any) -> Optional[np.ndarray]:
    width = int(frame.get_width())
    height = int(frame.get_height())
    frame_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if frame_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame_format == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if frame_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if frame_format == OBFormat.YUYV:
        yuyv = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    if frame_format == OBFormat.UYVY:
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if frame_format == OBFormat.NV12:
        nv12 = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    if frame_format == OBFormat.NV21:
        nv21 = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
    if frame_format == OBFormat.I420:
        i420 = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
    return None


def depth_to_mm(frame: Any) -> np.ndarray:
    height = int(frame.get_height())
    width = int(frame.get_width())
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(height, width)
    scale = float(frame.get_depth_scale())
    millimeters = np.rint(raw.astype(np.float32) * scale)
    return np.clip(millimeters, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def public_fields(value: Any, names: Iterable[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name in names:
        if hasattr(value, name):
            result[name] = json_compatible(getattr(value, name))
    return result


def json_compatible(value: Any) -> Any:
    """Convert SDK/NumPy values into types accepted by json.dumps."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return [json_compatible(item) for item in value]
    except TypeError:
        return str(value)


def camera_parameters(pipeline: Pipeline) -> Dict[str, Any]:
    params = pipeline.get_camera_param()
    intrinsic_names = ("width", "height", "fx", "fy", "cx", "cy")
    distortion_names = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
    transform_names = ("rot", "transform")
    return {
        "rgb_intrinsic": public_fields(params.rgb_intrinsic, intrinsic_names),
        "depth_intrinsic": public_fields(params.depth_intrinsic, intrinsic_names),
        "rgb_distortion": public_fields(params.rgb_distortion, distortion_names),
        "depth_distortion": public_fields(params.depth_distortion, distortion_names),
        "depth_to_rgb": public_fields(params.transform, transform_names),
        "is_mirrored": bool(params.is_mirrored),
    }


def device_information(pipeline: Pipeline) -> Dict[str, Any]:
    info = pipeline.get_device().get_device_info()
    getters = {
        "name": "get_name",
        "serial_number": "get_serial_number",
        "firmware_version": "get_firmware_version",
        "hardware_version": "get_hardware_version",
        "connection_type": "get_connection_type",
        "vid": "get_vid",
        "pid": "get_pid",
    }
    result: Dict[str, Any] = {}
    for key, getter_name in getters.items():
        try:
            result[key] = getattr(info, getter_name)()
        except Exception as error:
            result[key] = f"unavailable: {error}"
    return result


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_compatible(payload), ensure_ascii=False, indent=2) + "\n"
    )


def capture(args: argparse.Namespace) -> int:
    scan_dir = make_scan_dir(args.output_root.expanduser().resolve(), args.label)
    status_path = scan_dir / "capture_status.json"
    write_json(status_path, {"state": "starting", "scan_dir": str(scan_dir)})

    pipeline = None
    config = None
    started = False
    completed = False
    frame_count = 0
    skipped_count = 0
    imu_count = 0
    accel_count = 0
    gyro_count = 0
    started_wall = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    capture_finished_monotonic = None
    color_profile = None
    depth_profile = None
    frame_queue: queue.Queue = queue.Queue(maxsize=64)
    imu_queue: queue.Queue = queue.Queue()

    def on_frames(frames: Any) -> None:
        """Keep high-rate IMU frames separate from synchronized RGB-D sets."""
        nonlocal skipped_count
        if frames is None:
            return
        accel = frames.get_accel_frame()
        if accel is not None:
            value = accel.get_value()
            imu_queue.put((
                0,
                int(accel.get_timestamp_us()),
                int(accel.get_system_timestamp_us()),
                float(value.x),
                float(value.y),
                float(value.z),
            ))
        gyro = frames.get_gyro_frame()
        if gyro is not None:
            value = gyro.get_value()
            imu_queue.put((
                1,
                int(gyro.get_timestamp_us()),
                int(gyro.get_system_timestamp_us()),
                float(value.x),
                float(value.y),
                float(value.z),
            ))
        if frames.get_color_frame() is None or frames.get_depth_frame() is None:
            return
        try:
            frame_queue.put_nowait(frames)
        except queue.Full:
            skipped_count += 1

    try:
        pipeline = Pipeline()
        config = Config()
        color_profile = select_video_profile(
            pipeline,
            OBSensorType.COLOR_SENSOR,
            args.width,
            args.height,
            args.fps,
            (
                OBFormat.MJPG,
                OBFormat.RGB,
                OBFormat.BGR,
                OBFormat.YUYV,
                OBFormat.UYVY,
                OBFormat.NV12,
                OBFormat.NV21,
                OBFormat.I420,
            ),
        )
        depth_profile = select_video_profile(
            pipeline,
            OBSensorType.DEPTH_SENSOR,
            args.width,
            args.height,
            args.fps,
            (OBFormat.Y16,),
        )
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.enable_accel_stream(
            OBAccelFullScaleRange.ACCEL_FS_4g,
            OBAccelSampleRate.SAMPLE_RATE_200_HZ,
        )
        config.enable_gyro_stream(
            OBGyroFullScaleRange.FS_1000dps,
            OBGyroSampleRate.SAMPLE_RATE_200_HZ,
        )
        config.set_frame_aggregate_output_mode(
            OBFrameAggregateOutputMode.ANY_SITUATION
        )
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        pipeline.enable_frame_sync()
        pipeline.start(config, on_frames)
        started = True

        calibration = {
            "schema": "orbbec_rgbd_calibration.v1",
            "alignment": "software_depth_to_color",
            "device": device_information(pipeline),
            "selected_color_profile": profile_description(color_profile),
            "selected_depth_source_profile": profile_description(depth_profile),
            "selected_accel_profile": {
                "sample_rate_hz": 200,
                "full_scale": "4g",
                "units": "m/s^2",
            },
            "selected_gyro_profile": {
                "sample_rate_hz": 200,
                "full_scale": "1000dps",
                "units": "rad/s",
            },
            "camera_parameters": camera_parameters(pipeline),
            "stored_depth_pixel_model": (
                "software D2C output; use rgb_intrinsic for saved depth pixels"
            ),
        }
        write_json(scan_dir / "calibration.json", calibration)
        # Measure only the actual recording interval. Device/profile startup and
        # calibration probing are intentionally excluded.
        started_wall = datetime.now().astimezone()
        started_monotonic = time.monotonic()
        write_json(
            status_path,
            {
                "state": "recording",
                "scan_dir": str(scan_dir),
                "started_at": started_wall.isoformat(),
            },
        )
        print(f"[record] START {scan_dir}")
        print("[record] Move the camera now. Press Ctrl+C once to save and stop.")

        with (
            (scan_dir / "frames.csv").open("w", newline="") as stream,
            (scan_dir / "imu.csv").open("w", newline="") as imu_stream,
        ):
            writer = csv.writer(stream)
            imu_writer = csv.writer(imu_stream)
            writer.writerow(
                [
                    "frame_index",
                    "color_file",
                    "depth_file",
                    "color_timestamp_us",
                    "depth_timestamp_us",
                    "color_system_timestamp_us",
                    "depth_system_timestamp_us",
                    "timestamp_delta_us",
                    "depth_scale_to_mm",
                    "color_width",
                    "color_height",
                    "depth_width",
                    "depth_height",
                ]
            )
            imu_writer.writerow(
                [
                    "sample_index",
                    "kind",
                    "timestamp_us",
                    "system_timestamp_us",
                    "x",
                    "y",
                    "z",
                    "units",
                ]
            )
            while not STOP_REQUESTED:
                elapsed = time.monotonic() - started_monotonic
                if elapsed >= args.duration:
                    break
                while True:
                    try:
                        kind, timestamp_us, system_timestamp_us, x, y, z = (
                            imu_queue.get_nowait()
                        )
                    except queue.Empty:
                        break
                    imu_writer.writerow(
                        [
                            imu_count,
                            "accel" if kind == 0 else "gyro",
                            timestamp_us,
                            system_timestamp_us,
                            x,
                            y,
                            z,
                            "m/s^2" if kind == 0 else "rad/s",
                        ]
                    )
                    imu_count += 1
                    accel_count += int(kind == 0)
                    gyro_count += int(kind == 1)
                try:
                    frames = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    skipped_count += 1
                    continue
                frames = align_filter.process(frames)
                if frames is None:
                    skipped_count += 1
                    continue
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if color is None or depth is None:
                    skipped_count += 1
                    continue
                color_image = color_to_bgr(color)
                if color_image is None:
                    skipped_count += 1
                    continue
                depth_mm = depth_to_mm(depth)

                filename = f"{frame_count:06d}.png"
                color_relative = Path("color") / filename
                depth_relative = Path("depth") / filename
                if not cv2.imwrite(
                    str(scan_dir / color_relative),
                    color_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                ):
                    raise RuntimeError(f"failed to save {color_relative}")
                if not cv2.imwrite(
                    str(scan_dir / depth_relative),
                    depth_mm,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                ):
                    raise RuntimeError(f"failed to save {depth_relative}")

                color_ts = int(color.get_timestamp_us())
                depth_ts = int(depth.get_timestamp_us())
                writer.writerow(
                    [
                        frame_count,
                        str(color_relative),
                        str(depth_relative),
                        color_ts,
                        depth_ts,
                        int(color.get_system_timestamp_us()),
                        int(depth.get_system_timestamp_us()),
                        abs(color_ts - depth_ts),
                        float(depth.get_depth_scale()),
                        int(color.get_width()),
                        int(color.get_height()),
                        int(depth.get_width()),
                        int(depth.get_height()),
                    ]
                )
                frame_count += 1
                if frame_count % 30 == 0:
                    stream.flush()
                    imu_stream.flush()
                    print(
                        f"[record] elapsed={elapsed:.1f}s frames={frame_count} "
                        f"imu={imu_count} skipped={skipped_count}"
                    )
            while True:
                try:
                    kind, timestamp_us, system_timestamp_us, x, y, z = (
                        imu_queue.get_nowait()
                    )
                except queue.Empty:
                    break
                imu_writer.writerow(
                    [
                        imu_count,
                        "accel" if kind == 0 else "gyro",
                        timestamp_us,
                        system_timestamp_us,
                        x,
                        y,
                        z,
                        "m/s^2" if kind == 0 else "rad/s",
                    ]
                )
                imu_count += 1
                accel_count += int(kind == 0)
                gyro_count += int(kind == 1)
        capture_finished_monotonic = time.monotonic()
        completed = frame_count > 0
    except KeyboardInterrupt:
        capture_finished_monotonic = time.monotonic()
        completed = frame_count > 0
    except Exception as error:
        capture_finished_monotonic = time.monotonic()
        write_json(
            status_path,
            {
                "state": "failed",
                "scan_dir": str(scan_dir),
                "error": str(error),
                "hint": (
                    "On macOS, run the wrapper with sudo if the error contains "
                    "uvc_open failed."
                ),
            },
        )
        print(f"[record] FAILED: {error}", file=sys.stderr)
        print(f"[record] See {status_path}", file=sys.stderr)
        finish_without_macos_sdk_teardown(1)
        return 1
    finally:
        # On macOS/Femto Mega, pipeline.stop() can block for about 30 seconds,
        # report an already-deactivated device, and repeat the error again from
        # the native destructor. All outputs are independent PNG/CSV files, so
        # the macOS process exits after sealing them below; the OS releases USB.
        if started and pipeline is not None and sys.platform != "darwin":
            pipeline.stop()

    if capture_finished_monotonic is None:
        capture_finished_monotonic = time.monotonic()
    elapsed_seconds = capture_finished_monotonic - started_monotonic
    manifest = {
        "schema": "orbbec_rgbd_sequence.v1",
        "state": "complete" if completed else "invalid",
        "label": args.label,
        "scan_dir": str(scan_dir),
        "started_at": started_wall.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "rgbd_frames": frame_count,
        "imu_samples": imu_count,
        "accel_samples": accel_count,
        "gyro_samples": gyro_count,
        "skipped_frame_sets": skipped_count,
        "alignment": "software_depth_to_color",
        "depth_storage": "uint16 PNG in millimeters",
        "color_storage": "BGR-encoded PNG",
        "frames_index": "frames.csv",
        "imu_index": "imu.csv",
        "calibration": "calibration.json",
    }
    write_json(scan_dir / "manifest.json", manifest)
    write_json(status_path, manifest)
    if not completed or accel_count == 0 or gyro_count == 0:
        if completed:
            manifest["state"] = "invalid"
            manifest["reason"] = "RGB-D captured but accel/gyro stream was empty"
            write_json(scan_dir / "manifest.json", manifest)
            write_json(status_path, manifest)
        reason = (
            "no synchronized RGB-D frames were saved"
            if not completed
            else "accelerometer or gyroscope stream was empty"
        )
        print(f"[record] INVALID: {reason}", file=sys.stderr)
        finish_without_macos_sdk_teardown(3)
        return 3
    (scan_dir / "COMPLETE").write_text("capture completed\n")
    print(f"[record] SAVED {scan_dir}")
    print(f"[record] frames={frame_count} elapsed={elapsed_seconds:.1f}s")
    finish_without_macos_sdk_teardown(0)
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    return capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
