#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Reproducible build for the _elastic_mmap nanobind extension.
#
# Builds against the mlx wheel installed in the target Python environment
# (default: the repo's .venv) and drops the .so into omlx/elastic/ where
# omlx.elastic.loader imports it.
#
# Usage:
#   scripts/build-elastic-ext.sh [python-executable]
#
# Requirements: cmake >= 3.25, a C++17 toolchain, network access on first
# run (FetchContent pulls nanobind v2.12.0 — pinned to match the mlx wheel).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${1:-$REPO_ROOT/.venv/bin/python}"
NATIVE_DIR="$REPO_ROOT/omlx/elastic/native"
BUILD_DIR="$NATIVE_DIR/build"

if [ ! -x "$PYTHON" ]; then
  echo "error: python not found at $PYTHON" >&2
  exit 1
fi

"$PYTHON" - <<'EOF'
import mlx.core as mx
print(f"building against mlx wheel: {mx.__version__}")
EOF

cmake -S "$NATIVE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYTHON"
cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu)"

SO="$(find "$BUILD_DIR" -maxdepth 1 -name '_elastic_mmap*.so' | head -1)"
if [ -z "$SO" ]; then
  echo "error: build produced no _elastic_mmap*.so" >&2
  exit 1
fi
cp "$SO" "$REPO_ROOT/omlx/elastic/"
echo "installed: omlx/elastic/$(basename "$SO")"

"$PYTHON" - <<EOF
import sys
sys.path.insert(0, "$REPO_ROOT")
from omlx.elastic import loader
assert loader.is_available(), "extension built but import failed"
print("smoke: omlx.elastic native extension imports OK")
EOF
