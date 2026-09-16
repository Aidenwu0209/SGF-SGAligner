#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${SGF_RUNTIME_ROOT:-/home/aidenwu/Documents}"
CPU="${SGF_CPU_PYTHON:-$BASE/candidate-v2-runtime-20260904/bin/python}"
GPU="${SGF_GPU_PYTHON:-$BASE/.envs/droid-w-sgf-sga-shadow-20260902/bin/python}"
CAPTURE="${SGF_CAPTURE_PYTHON:-$BASE/orbbec_jojo_input_adapter/.venv/bin/python}"
PROVIDER="${SGF_DROID_PROVIDER:-$BASE/DROID-W-sgf-sga-shadow-20260902-src}"
RUNTIME="${SGF_SEMANTIC_RUNTIME:-$ROOT/configs/runtime.ssh33.local.json}"
export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$CPU" -u -m pose_pipeline.live_gui --runtime "$RUNTIME" \
 --provider-root "$PROVIDER" --cpu-python "$CPU" --gpu-python "$GPU" \
 --capture-python "$CAPTURE" --output "${SGF_SCAN_OUTPUT:-$BASE/SGF-developnew-GUI-scans}" "$@"
