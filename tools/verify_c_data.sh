#!/usr/bin/env bash
# Import our Arrow C Data Interface export into pyarrow and compare.
# Needs `uv` on PATH; builds a throwaway venv.
set -euo pipefail
cd "$(dirname "$0")/.."
LIB="build/libpqcarrow${SHLIB_EXT:-.so}"
[ -f "$LIB" ] || LIB="build/libpqcarrow.dylib"
[ -f "$LIB" ] || LIB="build/libpqcarrow.so"
VENV="${TMPDIR:-/tmp}/parquet-mojo-fixtures-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26'
"$VENV/bin/python" tools/consume_c_data.py "$LIB"
