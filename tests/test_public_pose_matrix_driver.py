from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.run_public_pose_matrix import _frontend_command


def _args(metric_config: Path | None) -> Namespace:
    return Namespace(
        dpv_python=Path("/opt/dpv/bin/python"),
        dpv_worker=Path("/opt/dpv/worker.py"),
        dpvo_root=Path("/opt/dpvo"),
        dpvo_network=Path("/opt/dpvo/dpvo.pth"),
        dpvo_config=Path("/opt/dpvo/config.yaml"),
        dpv_metric_config=metric_config,
        seed=17,
    )


def test_frontend_command_omits_optional_metric_config() -> None:
    command = _frontend_command(_args(None), Path("/tmp/pose.sock"))

    assert "--metric-config" not in command


def test_frontend_command_passes_metric_config_exactly_once() -> None:
    metric_config = Path("/opt/dpv/posefix_v3.json")
    command = _frontend_command(_args(metric_config), Path("/tmp/pose.sock"))

    assert command.count("--metric-config") == 1
    index = command.index("--metric-config")
    assert command[index + 1] == str(metric_config)
