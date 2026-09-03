#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 TEASER_SOURCE PYTHON [CREATE_ONLY_BUILD_DIR]" >&2
  exit 2
fi

teaser_source=$(cd "$1" && pwd)
pose_python=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
teaser_build=${3:-"${teaser_source}/build-sgf-pose"}

if [[ -e "$teaser_build" ]]; then
  echo "refusing to reuse build directory: $teaser_build" >&2
  exit 3
fi

python_prefix=$($pose_python -c 'import sys; print(sys.prefix)')
python_version=$($pose_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
pybind11_dir=$($pose_python -m pybind11 --cmakedir)

cmake -S "$teaser_source" -B "$teaser_build" \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$pose_python" \
  -DPYTHON_LIBRARY="${python_prefix}/lib/libpython${python_version}.so" \
  -DPYTHON_INCLUDE_DIR="${python_prefix}/include/python${python_version}" \
  -Dpybind11_DIR="$pybind11_dir"
cmake --build "$teaser_build" -j4 --target teaserpp_python

binding=$(find "$teaser_build/python/teaserpp_python" -maxdepth 1 -type f -name '_teaserpp*.so' -print -quit)
if [[ -z "$binding" ]]; then
  echo "TEASER++ binding was not produced" >&2
  exit 4
fi
cp "$binding" "$teaser_source/python/teaserpp_python/"

PYTHONPATH="$teaser_source/python" "$pose_python" -c \
  'import teaserpp_python; print(teaserpp_python.__file__)'
