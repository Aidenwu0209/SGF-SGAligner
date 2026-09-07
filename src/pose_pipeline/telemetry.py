"""Process-local runtime and resource telemetry kept outside decision evidence."""

from __future__ import annotations

import resource
import sys
import time
from typing import Mapping


def collect_resource_telemetry(
    started: float,
    *,
    stages: Mapping[str, float] | None = None,
    work_items: int | None = None,
    work_item_name: str | None = None,
) -> dict:
    runtime_s = time.perf_counter() - started
    maximum_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        maximum_rss *= 1024.0
    cuda_peak = None
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            if torch_module.cuda.is_available():
                cuda_peak = (
                    float(torch_module.cuda.max_memory_allocated()) / (1024 ** 2)
                )
        except (AttributeError, RuntimeError):
            cuda_peak = None
    telemetry = {
        "schema": "resource_telemetry.v1",
        "runtime_s": runtime_s,
        "stage_runtime_s": dict(stages or {}),
        "peak_cpu_rss_mb": maximum_rss / (1024 ** 2),
        "peak_cuda_memory_mb": cuda_peak,
    }
    if work_items is not None:
        telemetry["work_items"] = int(work_items)
        telemetry["work_item_name"] = work_item_name or "item"
        telemetry["throughput_items_per_s"] = work_items / max(runtime_s, 1e-12)
    return telemetry
