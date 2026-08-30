#!/usr/bin/env bash
# pyarrow's single-threaded decode numbers for the same fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${TMPDIR:-/tmp}/parquet-mojo-fixtures-venv"
uv venv --quiet --allow-existing "$VENV" 2>/dev/null || uv venv --quiet "$VENV"
VIRTUAL_ENV="$VENV" uv pip install --quiet 'pyarrow>=21,<26'
"$VENV/bin/python" tools/bench_pyarrow.py --make
"$VENV/bin/python" tools/bench_pyarrow.py
