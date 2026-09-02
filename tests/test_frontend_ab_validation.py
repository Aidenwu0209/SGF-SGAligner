from argparse import Namespace
from pathlib import Path

from scripts.run_frontend_ab_validation import _metric_not_regressed, _worker_command


def _args() -> Namespace:
    return Namespace(
        dpv_python=Path("/venv/python"),
        dpvo_root=Path("/dpvo"),
        network=Path("/dpvo.pth"),
        dpvo_config=Path("/live.yaml"),
        metric_config=Path("/metric.json"),
        seed=7,
        loop_closure=True,
    )


def test_baseline_command_does_not_require_finalized_sidecar():
    command = _worker_command(
        _args(), worker=Path("/baseline.py"), socket_path=Path("/tmp/a.sock"),
        finalized_path=None,
    )
    assert "--finalized-trajectory" not in command
    assert "--loop-closure" in command
    assert "--no-gravity-align" in command


def test_candidate_command_has_create_only_finalized_sidecar():
    command = _worker_command(
        _args(), worker=Path("/candidate.py"), socket_path=Path("/tmp/b.sock"),
        finalized_path=Path("/new/finalized.jsonl"),
    )
    index = command.index("--finalized-trajectory")
    assert command[index + 1] == "/new/finalized.jsonl"


def test_metric_regression_uses_relative_and_absolute_tolerance():
    assert _metric_not_regressed(
        1.0, 1.059, relative_tolerance=0.05, absolute_tolerance=0.01,
    )
    assert not _metric_not_regressed(
        1.0, 1.061, relative_tolerance=0.05, absolute_tolerance=0.01,
    )
